import argparse
import sys
import time
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.openalex import (  # noqa: E402
    OPENALEX_TIMEOUT,
    clean_doi,
    crossref_mailto_ua,
    openalex_headers,
    openalex_work_url,
    retry_after_seconds,
)
from lib.pipeline import PipelinePaths, add_query_arg, load_dotenv  # noqa: E402

SAVE_INTERVAL = 50

# Written when OpenAlex, Semantic Scholar, CrossRef, Europe PMC, and PubMed all return
# no abstract for this row, so a later run does not burn API quota on the same permanent miss.
NO_EXTERNAL_ABSTRACT_MARKER = "__ABSTRACT_UNAVAILABLE__"

EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"


def _is_unavailable_marker(val) -> bool:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return str(val).strip() == NO_EXTERNAL_ABSTRACT_MARKER


def _abstract_cell_has_text(val) -> bool:
    """True if the cell has human-readable abstract text (not NaN/empty/marker from CSV)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat", "#n/a"):
        return False
    if s == NO_EXTERNAL_ABSTRACT_MARKER:
        return False
    return True


def _should_skip_abstract_fetch(val) -> bool:
    """True if we should not call external APIs (already have text or prior exhaustive miss)."""
    if _is_unavailable_marker(val):
        return True
    return _abstract_cell_has_text(val)


def _configure_output_streams() -> None:
    """Line-buffer stdout/stderr when redirected so logs and tqdm update promptly."""
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except OSError:
                pass


def get_abstract_openalex(doi, session: requests.Session, max_retries=4):
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return None

    url = openalex_work_url(doi_norm)
    headers = openalex_headers()

    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=OPENALEX_TIMEOUT, headers=headers)
            if r.status_code == 200:
                data = r.json()
                inverted = data.get("abstract_inverted_index")
                if not inverted:
                    return None
                words = {}
                for word, positions in inverted.items():
                    for pos in positions:
                        words[pos] = word
                return " ".join(words[i] for i in sorted(words))
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                wait = retry_after_seconds(r) or min(10.0 * (attempt + 1), 60.0)
                time.sleep(wait)
                if attempt < max_retries - 1:
                    continue
                return None
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return None
        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return None
    return None


def get_abstract_semantic(doi, max_retries=3):
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return None

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_norm}"
    params = {"fields": "abstract"}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                abstract = r.json().get("abstract")
                if abstract:
                    return abstract
        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return None
    return None


def get_abstract_crossref(doi, max_retries=3):
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return None

    url = f"https://api.crossref.org/works/{doi_norm}"
    headers = {"User-Agent": crossref_mailto_ua()}

    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if "message" in data:
                    abstract_text = data["message"].get("abstract")
                    if abstract_text:
                        if isinstance(abstract_text, str):
                            return abstract_text
                        if isinstance(abstract_text, list):
                            return " ".join(str(t) for t in abstract_text)
            elif r.status_code == 404:
                return None
        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return None
    return None


def get_abstract_europe_pmc(doi, session: Optional[requests.Session] = None, max_retries=3):
    """Fetch abstract text from Europe PMC by DOI."""
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return None

    http = session if session is not None else requests
    params = {
        "query": f'DOI:"{doi_norm}"',
        "format": "json",
        "resultType": "core",
        "pageSize": 1,
    }

    for attempt in range(max_retries):
        try:
            r = http.get(EUROPE_PMC_SEARCH_URL, params=params, timeout=30)
            if r.status_code == 200:
                results = r.json().get("resultList", {}).get("result", [])
                if not results:
                    return None
                abstract = results[0].get("abstractText")
                if abstract and str(abstract).strip():
                    return str(abstract).strip()
                return None
            if r.status_code == 404:
                return None
            if r.status_code == 429:
                time.sleep(min(10.0 * (attempt + 1), 60.0))
                if attempt < max_retries - 1:
                    continue
                return None
            if attempt < max_retries - 1:
                time.sleep(2**attempt)
                continue
            return None
        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return None
    return None


def get_abstract_pubmed(doi, max_retries=3):
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return None

    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": f"{doi_norm}[DOI]",
        "retmode": "json",
    }

    try:
        r = requests.get(search_url, params=search_params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])

            if not pmids:
                return None

            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids[:1]),
                "retmode": "xml",
                "rettype": "abstract",
            }

            r2 = requests.get(fetch_url, params=fetch_params, timeout=15)
            if r2.status_code == 200:
                import xml.etree.ElementTree as ET

                try:
                    root = ET.fromstring(r2.text)
                    for abstract_text in root.iter("{http://www.ncbi.nlm.nih.gov}AbstractText"):
                        if abstract_text.text:
                            return abstract_text.text
                except ET.ParseError:
                    return None
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        return None

    return None


def _fetch_abstract_waterfall(doi, session: requests.Session, stats: dict):
    """Try sources in order; update stats; return abstract text or None."""
    abstract = get_abstract_openalex(doi, session)
    if abstract:
        stats["openalex"] += 1
        return abstract

    abstract = get_abstract_semantic(doi)
    if abstract:
        stats["semantic"] += 1
        return abstract

    abstract = get_abstract_crossref(doi)
    if abstract:
        stats["crossref"] += 1
        return abstract

    abstract = get_abstract_europe_pmc(doi, session)
    if abstract:
        stats["europe_pmc"] += 1
        return abstract

    abstract = get_abstract_pubmed(doi)
    if abstract:
        stats["pubmed"] += 1
        return abstract

    stats["failed"] += 1
    return None


def run_fetch_abstracts(
    paths: PipelinePaths,
    save_interval: int = SAVE_INTERVAL,
    retry_unavailable: bool = False,
):
    _configure_output_streams()
    input_csv = paths.combined_scopus_api
    output_csv = paths.with_abstracts

    stats = {
        "openalex": 0,
        "semantic": 0,
        "crossref": 0,
        "europe_pmc": 0,
        "pubmed": 0,
        "already_had": 0,
        "failed": 0,
        "marked_unavailable": 0,
        "cleared_unavailable": 0,
    }

    if output_csv.exists():
        print(f"Resuming from existing file: {output_csv}")
        df = pd.read_csv(output_csv)
    else:
        print(f"Starting fresh from: {input_csv}")
        df = pd.read_csv(input_csv)

    if "Abstract" not in df.columns:
        df["Abstract"] = ""

    if retry_unavailable:
        cleared = df["Abstract"].map(_is_unavailable_marker)
        stats["cleared_unavailable"] = int(cleared.sum())
        df.loc[cleared, "Abstract"] = ""
        print(
            f"Retry unavailable: cleared {stats['cleared_unavailable']} "
            f"{NO_EXTERNAL_ABSTRACT_MARKER} markers for re-fetch"
        )

    stats["already_had"] = int(df["Abstract"].map(_should_skip_abstract_fetch).sum())

    processed = 0
    total = len(df)
    needs_abstract = total - stats["already_had"]

    print(f"\nquery_id={paths.query_id}")
    print(f"Starting abstract fetching...")
    print(f"Total papers: {total}")
    print(
        f"Already resolved (readable abstract or {NO_EXTERNAL_ABSTRACT_MARKER}): "
        f"{stats['already_had']}"
    )
    print(f"Need abstract fetch attempts: {needs_abstract}")
    print(
        "\nTrying sources in order: "
        "OpenAlex → Semantic Scholar → CrossRef → Europe PMC → PubMed\n"
    )

    session = requests.Session()
    try:
        for idx, row in tqdm(
            df.iterrows(),
            total=total,
            desc="Fetching abstracts",
            file=sys.stderr,
            mininterval=0.5,
        ):
            if _should_skip_abstract_fetch(row.get("Abstract")):
                continue

            abstract = _fetch_abstract_waterfall(row.get("DOI"), session, stats)

            if df["Abstract"].dtype != "object":
                df["Abstract"] = df["Abstract"].astype(str)
            if abstract:
                df.at[idx, "Abstract"] = abstract
            else:
                df.at[idx, "Abstract"] = NO_EXTERNAL_ABSTRACT_MARKER
                stats["marked_unavailable"] += 1

            processed += 1

            if processed % save_interval == 0:
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_csv, index=False)
                current_found = (
                    stats["openalex"]
                    + stats["semantic"]
                    + stats["crossref"]
                    + stats["europe_pmc"]
                    + stats["pubmed"]
                )
                tqdm.write(
                    f"Progress: {processed}/{total} | Found: {current_found} | "
                    f"OpenAlex: {stats['openalex']}, Semantic: {stats['semantic']}, "
                    f"CrossRef: {stats['crossref']}, EuropePMC: {stats['europe_pmc']}, "
                    f"PubMed: {stats['pubmed']}, "
                    f"Failed: {stats['failed']}, "
                    f"Marked unavailable: {stats['marked_unavailable']}",
                    file=sys.stderr,
                )
                sys.stderr.flush()

            time.sleep(0.2)
    finally:
        session.close()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving final results to {output_csv}...")
    df.to_csv(output_csv, index=False)

    print(f"\n{'='*70}")
    print("ABSTRACT FETCHING SUMMARY")
    print(f"{'='*70}")
    print(f"Total papers processed: {total}")
    print(
        f"Rows already resolved at start (readable + prior marker): {stats['already_had']}"
    )
    print(f"\nAbstracts fetched this run:")
    print(f"  OpenAlex:     {stats['openalex']}")
    print(f"  Semantic Scholar: {stats['semantic']}")
    print(f"  CrossRef:     {stats['crossref']}")
    print(f"  Europe PMC:   {stats['europe_pmc']}")
    print(f"  PubMed:       {stats['pubmed']}")
    print(f"  Failed:       {stats['failed']}")
    print(f"  Marked {NO_EXTERNAL_ABSTRACT_MARKER} (no text from any source): {stats['marked_unavailable']}")

    readable = int(df["Abstract"].map(_abstract_cell_has_text).sum())
    marked = int(df["Abstract"].map(_is_unavailable_marker).sum())
    readable_pct = (readable / total * 100) if total > 0 else 0
    marked_pct = (marked / total * 100) if total > 0 else 0

    print(f"\nFinal readable abstracts: {readable}/{total} ({readable_pct:.1f}%)")
    print(
        f"Marked unavailable (skip future API calls): {marked}/{total} ({marked_pct:.1f}%)"
    )
    print(f"{'='*70}")
    print(f"✓ Complete! Results saved to {output_csv}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch abstracts for combined Scopus API export")
    add_query_arg(parser)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument(
        "--retry-unavailable",
        action="store_true",
        help=(
            f"Clear {NO_EXTERNAL_ABSTRACT_MARKER} markers and re-try the full waterfall "
            "(useful after adding a new source such as Europe PMC)."
        ),
    )
    args = parser.parse_args()
    run_fetch_abstracts(
        PipelinePaths(args.query_id),
        save_interval=args.save_interval,
        retry_unavailable=args.retry_unavailable,
    )
