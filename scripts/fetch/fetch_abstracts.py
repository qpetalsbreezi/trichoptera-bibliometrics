import argparse
import sys
import time
from pathlib import Path

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


def _abstract_cell_has_text(val) -> bool:
    """True if the cell looks like a real abstract (not NaN/empty/placeholder string from CSV)."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat", "#n/a"):
        return False
    return True


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


def run_fetch_abstracts(paths: PipelinePaths, save_interval: int = SAVE_INTERVAL):
    _configure_output_streams()
    input_csv = paths.combined_scopus_api
    output_csv = paths.with_abstracts

    stats = {
        "openalex": 0,
        "semantic": 0,
        "crossref": 0,
        "pubmed": 0,
        "already_had": 0,
        "failed": 0,
    }

    if output_csv.exists():
        print(f"Resuming from existing file: {output_csv}")
        df = pd.read_csv(output_csv)
        if "Abstract" in df.columns:
            stats["already_had"] = int(df["Abstract"].map(_abstract_cell_has_text).sum())
        else:
            stats["already_had"] = 0
    else:
        print(f"Starting fresh from: {input_csv}")
        df = pd.read_csv(input_csv)

    if "Abstract" not in df.columns:
        df["Abstract"] = ""

    processed = 0
    total = len(df)
    needs_abstract = total - stats["already_had"]

    print(f"\nquery_id={paths.query_id}")
    print(f"Starting abstract fetching...")
    print(f"Total papers: {total}")
    print(f"Already have abstracts: {stats['already_had']}")
    print(f"Need abstracts: {needs_abstract}")
    print(f"\nTrying sources in order: OpenAlex → Semantic Scholar → CrossRef → PubMed\n")

    session = requests.Session()
    try:
        for idx, row in tqdm(
            df.iterrows(),
            total=total,
            desc="Fetching abstracts",
            file=sys.stderr,
            mininterval=0.5,
        ):
            if _abstract_cell_has_text(row.get("Abstract")):
                continue

            doi = row.get("DOI")
            abstract = None

            abstract = get_abstract_openalex(doi, session)
            if abstract:
                stats["openalex"] += 1
            else:
                abstract = get_abstract_semantic(doi)
                if abstract:
                    stats["semantic"] += 1
                else:
                    abstract = get_abstract_crossref(doi)
                    if abstract:
                        stats["crossref"] += 1
                    else:
                        abstract = get_abstract_pubmed(doi)
                        if abstract:
                            stats["pubmed"] += 1
                        else:
                            stats["failed"] += 1

            if abstract:
                if df["Abstract"].dtype != "object":
                    df["Abstract"] = df["Abstract"].astype(str)
                df.at[idx, "Abstract"] = abstract

            processed += 1

            if processed % save_interval == 0:
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_csv, index=False)
                current_found = (
                    stats["openalex"] + stats["semantic"] + stats["crossref"] + stats["pubmed"]
                )
                tqdm.write(
                    f"Progress: {processed}/{total} | Found: {current_found} | "
                    f"OpenAlex: {stats['openalex']}, Semantic: {stats['semantic']}, "
                    f"CrossRef: {stats['crossref']}, PubMed: {stats['pubmed']}, "
                    f"Failed: {stats['failed']}",
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
    print(f"Already had abstracts: {stats['already_had']}")
    print(f"\nAbstracts fetched this run:")
    print(f"  OpenAlex:     {stats['openalex']}")
    print(f"  Semantic Scholar: {stats['semantic']}")
    print(f"  CrossRef:     {stats['crossref']}")
    print(f"  PubMed:       {stats['pubmed']}")
    print(f"  Failed:       {stats['failed']}")

    total_fetched = stats["openalex"] + stats["semantic"] + stats["crossref"] + stats["pubmed"]
    total_with_abstract = stats["already_had"] + total_fetched
    coverage = (total_with_abstract / total * 100) if total > 0 else 0

    print(f"\nFinal coverage: {total_with_abstract}/{total} ({coverage:.1f}%)")
    print(f"{'='*70}")
    print(f"✓ Complete! Results saved to {output_csv}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch abstracts for combined Scopus API export")
    add_query_arg(parser)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    args = parser.parse_args()
    run_fetch_abstracts(PipelinePaths(args.query_id), save_interval=args.save_interval)
