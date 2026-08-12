"""Fetch abstracts for a combined Scopus API export.

Cascade: OpenAlex -> Semantic Scholar -> CrossRef -> Europe PMC -> PubMed,
with a title-based fallback for rows that have no usable DOI.

Every fetcher returns (text, status) so that a genuine "this record has no
abstract" is never confused with "the request failed". Only rows where every
source reported a *hard* miss are marked exhausted; rows that hit a rate limit,
auth error, timeout, or network error are left blank so a later run retries them.
"""

import argparse
import html
import os
import re
import sys
import time
import xml.etree.ElementTree as ET
from difflib import SequenceMatcher
from pathlib import Path
from typing import Optional, Tuple

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

EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
CROSSREF_WORKS_URL = "https://api.crossref.org/works"
PUBMED_ESEARCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
PUBMED_EFETCH_URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

# ---------------------------------------------------------------------------
# Provenance / status vocabulary
# ---------------------------------------------------------------------------

SOURCE_ORDER = ["openalex", "semantic_scholar", "crossref", "europe_pmc", "pubmed"]
SOURCE_TITLE_FALLBACK = ["crossref_title", "europe_pmc_title"]
SOURCE_PRESENT = "scopus"  # abstract was already in the input file
SOURCE_NONE = "none"  # every source reported a hard miss

# Legacy sentinel from the previous version of this script. Rows carrying it are
# migrated on load: the marker moves out of the text column into abstract_source.
LEGACY_UNAVAILABLE_MARKER = "__ABSTRACT_UNAVAILABLE__"

# Hard miss: the source was reachable and authoritative, and has no abstract.
ST_OK = "ok"
ST_NO_ABSTRACT = "no_abstract"
ST_NOT_FOUND = "not_found"
ST_NO_DOI = "no_doi"
ST_TITLE_MISMATCH = "title_mismatch"
HARD_MISS_STATUSES = {ST_NO_ABSTRACT, ST_NOT_FOUND, ST_NO_DOI, ST_TITLE_MISMATCH}

# Soft failure: we learned nothing. Never mark a row exhausted on these.
ST_RATE_LIMITED = "rate_limited"
ST_FORBIDDEN = "forbidden"
ST_TIMEOUT = "timeout"
ST_NETWORK = "network_error"
ST_PARSE_ERROR = "parse_error"

TITLE_MATCH_THRESHOLD = 0.90

# One-time warnings so a misconfigured key does not print once per row.
_WARNED = set()


def _warn_once(key: str, message: str) -> None:
    if key not in _WARNED:
        _WARNED.add(key)
        print(f"\n[WARN] {message}\n", file=sys.stderr)
        sys.stderr.flush()


# ---------------------------------------------------------------------------
# Cell helpers
# ---------------------------------------------------------------------------


def _abstract_cell_has_text(val) -> bool:
    """True if the cell holds human-readable abstract text."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    s = str(val).strip()
    if not s or s.lower() in ("nan", "none", "nat", "#n/a"):
        return False
    if s == LEGACY_UNAVAILABLE_MARKER:
        return False
    return True


def _is_exhausted(val) -> bool:
    """True if abstract_source records that every source reported a hard miss."""
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return False
    return str(val).strip() == SOURCE_NONE


def _clean_text(text: str) -> str:
    """Strip JATS/HTML markup, unescape entities, collapse whitespace."""
    if not text:
        return ""
    t = re.sub(r"(?is)<[^>]+>", " ", str(text))
    t = html.unescape(t)
    t = re.sub(r"\s+", " ", t).strip()
    # CrossRef and Europe PMC often prepend a literal "Abstract" heading.
    t = re.sub(r"^(abstract|summary)\b[\s:.\-–—]*", "", t, flags=re.IGNORECASE).strip()
    return t


def _normalize_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(title or "").lower()).strip()


def _titles_match(a: str, b: str) -> bool:
    na, nb = _normalize_title(a), _normalize_title(b)
    if not na or not nb:
        return False
    return SequenceMatcher(None, na, nb).ratio() >= TITLE_MATCH_THRESHOLD


def _configure_output_streams() -> None:
    """Line-buffer stdout/stderr when redirected so logs and tqdm update promptly."""
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# Shared HTTP retry handling
#
# FIX: the previous version fell through to the next loop iteration without
# sleeping on any non-200 that was not a 404, so three "retries" fired back to
# back in milliseconds and all three were rate-limited. Every non-200 now backs
# off, and 401/403 short-circuit because an auth problem will not self-resolve.
# ---------------------------------------------------------------------------


def _handle_non_200(
    response: requests.Response, attempt: int, max_retries: int, source: str
) -> Optional[Tuple[None, str]]:
    """Return a terminal (None, status) result, or None to mean 'retry after backoff'."""
    code = response.status_code

    if code == 404:
        return (None, ST_NOT_FOUND)

    if code in (401, 403):
        _warn_once(
            f"{source}-auth",
            f"{source} returned HTTP {code} (auth/credential problem). "
            f"Abstracts from this source will be silently unavailable until it is fixed. "
            f"For OpenAlex, note an API key has been required since 2026-02-13.",
        )
        return (None, ST_FORBIDDEN)

    if code == 429:
        wait = retry_after_seconds(response) or min(10.0 * (attempt + 1), 60.0)
        if attempt < max_retries - 1:
            time.sleep(wait)
            return None  # retry
        return (None, ST_RATE_LIMITED)

    if attempt < max_retries - 1:
        time.sleep(2**attempt)
        return None  # retry

    return (None, f"http_{code}")


def _handle_exception(
    exc: Exception, attempt: int, max_retries: int
) -> Optional[Tuple[None, str]]:
    if attempt < max_retries - 1:
        time.sleep(2**attempt)
        return None  # retry
    if isinstance(exc, requests.exceptions.Timeout):
        return (None, ST_TIMEOUT)
    return (None, ST_NETWORK)


# ---------------------------------------------------------------------------
# Sources
# ---------------------------------------------------------------------------


def get_abstract_openalex(
    doi, session: requests.Session, max_retries: int = 4
) -> Tuple[Optional[str], str]:
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return (None, ST_NO_DOI)

    url = openalex_work_url(doi_norm)
    headers = openalex_headers()

    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=OPENALEX_TIMEOUT, headers=headers)
            if r.status_code == 200:
                inverted = r.json().get("abstract_inverted_index")
                if not inverted:
                    return (None, ST_NO_ABSTRACT)
                words = {}
                for word, positions in inverted.items():
                    for pos in positions:
                        words[pos] = word
                text = _clean_text(" ".join(words[i] for i in sorted(words)))
                return (text, ST_OK) if text else (None, ST_NO_ABSTRACT)

            terminal = _handle_non_200(r, attempt, max_retries, "OpenAlex")
            if terminal is not None:
                return terminal
        except (ValueError, KeyError):
            return (None, ST_PARSE_ERROR)
        except requests.exceptions.RequestException as exc:
            terminal = _handle_exception(exc, attempt, max_retries)
            if terminal is not None:
                return terminal
    return (None, ST_NETWORK)


def get_abstract_semantic(
    doi, session: requests.Session, max_retries: int = 3
) -> Tuple[Optional[str], str]:
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return (None, ST_NO_DOI)

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi_norm}"
    params = {"fields": "abstract"}
    headers = {}
    api_key = os.environ.get("SEMANTIC_SCHOLAR_API_KEY")
    if api_key:
        headers["x-api-key"] = api_key

    for attempt in range(max_retries):
        try:
            r = session.get(url, params=params, headers=headers, timeout=15)
            if r.status_code == 200:
                text = _clean_text(r.json().get("abstract") or "")
                return (text, ST_OK) if text else (None, ST_NO_ABSTRACT)

            terminal = _handle_non_200(r, attempt, max_retries, "Semantic Scholar")
            if terminal is not None:
                return terminal
        except ValueError:
            return (None, ST_PARSE_ERROR)
        except requests.exceptions.RequestException as exc:
            terminal = _handle_exception(exc, attempt, max_retries)
            if terminal is not None:
                return terminal
    return (None, ST_NETWORK)


def get_abstract_crossref(
    doi, session: requests.Session, max_retries: int = 3
) -> Tuple[Optional[str], str]:
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return (None, ST_NO_DOI)

    url = f"{CROSSREF_WORKS_URL}/{doi_norm}"
    headers = {"User-Agent": crossref_mailto_ua()}

    for attempt in range(max_retries):
        try:
            r = session.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                message = r.json().get("message") or {}
                raw = message.get("abstract")
                if isinstance(raw, list):
                    raw = " ".join(str(t) for t in raw)
                # CrossRef abstracts are JATS XML; strip the markup.
                text = _clean_text(raw or "")
                return (text, ST_OK) if text else (None, ST_NO_ABSTRACT)

            terminal = _handle_non_200(r, attempt, max_retries, "CrossRef")
            if terminal is not None:
                return terminal
        except ValueError:
            return (None, ST_PARSE_ERROR)
        except requests.exceptions.RequestException as exc:
            terminal = _handle_exception(exc, attempt, max_retries)
            if terminal is not None:
                return terminal
    return (None, ST_NETWORK)


def get_abstract_europe_pmc(
    doi, session: requests.Session, max_retries: int = 3
) -> Tuple[Optional[str], str]:
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return (None, ST_NO_DOI)

    params = {
        "query": f'DOI:"{doi_norm}"',
        "format": "json",
        "resultType": "core",
        "pageSize": 1,
    }

    for attempt in range(max_retries):
        try:
            r = session.get(EUROPE_PMC_SEARCH_URL, params=params, timeout=30)
            if r.status_code == 200:
                results = r.json().get("resultList", {}).get("result", [])
                if not results:
                    return (None, ST_NOT_FOUND)
                text = _clean_text(results[0].get("abstractText") or "")
                return (text, ST_OK) if text else (None, ST_NO_ABSTRACT)

            terminal = _handle_non_200(r, attempt, max_retries, "Europe PMC")
            if terminal is not None:
                return terminal
        except ValueError:
            return (None, ST_PARSE_ERROR)
        except requests.exceptions.RequestException as exc:
            terminal = _handle_exception(exc, attempt, max_retries)
            if terminal is not None:
                return terminal
    return (None, ST_NETWORK)


def _pubmed_params(extra: dict) -> dict:
    """Attach NCBI tool/email/api_key identifiers when available."""
    params = dict(extra)
    params["tool"] = os.environ.get("NCBI_TOOL", "abstract-fetch")
    email = os.environ.get("NCBI_EMAIL") or os.environ.get("CROSSREF_MAILTO")
    if email:
        params["email"] = email
    api_key = os.environ.get("NCBI_API_KEY")
    if api_key:
        params["api_key"] = api_key
    return params


def get_abstract_pubmed(
    doi, session: requests.Session, max_retries: int = 3
) -> Tuple[Optional[str], str]:
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return (None, ST_NO_DOI)

    search_params = _pubmed_params(
        {"db": "pubmed", "term": f"{doi_norm}[DOI]", "retmode": "json"}
    )

    pmid = None
    for attempt in range(max_retries):
        try:
            r = session.get(PUBMED_ESEARCH_URL, params=search_params, timeout=15)
            if r.status_code == 200:
                idlist = r.json().get("esearchresult", {}).get("idlist", [])
                if not idlist:
                    return (None, ST_NOT_FOUND)
                pmid = idlist[0]
                break

            terminal = _handle_non_200(r, attempt, max_retries, "PubMed")
            if terminal is not None:
                return terminal
        except ValueError:
            return (None, ST_PARSE_ERROR)
        except requests.exceptions.RequestException as exc:
            terminal = _handle_exception(exc, attempt, max_retries)
            if terminal is not None:
                return terminal

    if not pmid:
        return (None, ST_NETWORK)

    fetch_params = _pubmed_params(
        {"db": "pubmed", "id": pmid, "retmode": "xml", "rettype": "abstract"}
    )

    for attempt in range(max_retries):
        try:
            r = session.get(PUBMED_EFETCH_URL, params=fetch_params, timeout=15)
            if r.status_code == 200:
                try:
                    root = ET.fromstring(r.text)
                except ET.ParseError:
                    return (None, ST_PARSE_ERROR)

                # FIX: efetch XML is NOT namespaced, so the old
                # "{http://www.ncbi.nlm.nih.gov}AbstractText" iterator matched
                # nothing on every record. Match on the local tag name instead,
                # and concatenate ALL sections -- structured abstracts split into
                # BACKGROUND / METHODS / RESULTS / CONCLUSIONS.
                parts = []
                for el in root.iter():
                    if el.tag.split("}")[-1] != "AbstractText":
                        continue
                    body = "".join(el.itertext()).strip()
                    if not body:
                        continue
                    label = el.get("Label") or el.get("NlmCategory")
                    parts.append(f"{label.strip().title()}: {body}" if label else body)

                text = _clean_text(" ".join(parts))
                return (text, ST_OK) if text else (None, ST_NO_ABSTRACT)

            terminal = _handle_non_200(r, attempt, max_retries, "PubMed")
            if terminal is not None:
                return terminal
        except requests.exceptions.RequestException as exc:
            terminal = _handle_exception(exc, attempt, max_retries)
            if terminal is not None:
                return terminal
    return (None, ST_NETWORK)


# ---------------------------------------------------------------------------
# Title-based fallback for rows with no usable DOI
# ---------------------------------------------------------------------------


def get_abstract_crossref_by_title(
    title, session: requests.Session, max_retries: int = 3
) -> Tuple[Optional[str], str]:
    if not title or not str(title).strip():
        return (None, ST_NOT_FOUND)

    params = {"query.bibliographic": str(title)[:300], "rows": 3, "select": "title,abstract"}
    headers = {"User-Agent": crossref_mailto_ua()}

    for attempt in range(max_retries):
        try:
            r = session.get(CROSSREF_WORKS_URL, params=params, headers=headers, timeout=20)
            if r.status_code == 200:
                items = r.json().get("message", {}).get("items", []) or []
                for item in items:
                    candidate = (item.get("title") or [""])[0]
                    if not _titles_match(title, candidate):
                        continue
                    text = _clean_text(item.get("abstract") or "")
                    return (text, ST_OK) if text else (None, ST_NO_ABSTRACT)
                return (None, ST_TITLE_MISMATCH)

            terminal = _handle_non_200(r, attempt, max_retries, "CrossRef")
            if terminal is not None:
                return terminal
        except ValueError:
            return (None, ST_PARSE_ERROR)
        except requests.exceptions.RequestException as exc:
            terminal = _handle_exception(exc, attempt, max_retries)
            if terminal is not None:
                return terminal
    return (None, ST_NETWORK)


def get_abstract_europe_pmc_by_title(
    title, session: requests.Session, max_retries: int = 3
) -> Tuple[Optional[str], str]:
    if not title or not str(title).strip():
        return (None, ST_NOT_FOUND)

    safe = str(title).replace('"', " ")[:300]
    params = {
        "query": f'TITLE:"{safe}"',
        "format": "json",
        "resultType": "core",
        "pageSize": 3,
    }

    for attempt in range(max_retries):
        try:
            r = session.get(EUROPE_PMC_SEARCH_URL, params=params, timeout=30)
            if r.status_code == 200:
                results = r.json().get("resultList", {}).get("result", []) or []
                for item in results:
                    if not _titles_match(title, item.get("title") or ""):
                        continue
                    text = _clean_text(item.get("abstractText") or "")
                    return (text, ST_OK) if text else (None, ST_NO_ABSTRACT)
                return (None, ST_TITLE_MISMATCH)

            terminal = _handle_non_200(r, attempt, max_retries, "Europe PMC")
            if terminal is not None:
                return terminal
        except ValueError:
            return (None, ST_PARSE_ERROR)
        except requests.exceptions.RequestException as exc:
            terminal = _handle_exception(exc, attempt, max_retries)
            if terminal is not None:
                return terminal
    return (None, ST_NETWORK)


# ---------------------------------------------------------------------------
# Waterfall
# ---------------------------------------------------------------------------

DOI_FETCHERS = [
    ("openalex", get_abstract_openalex),
    ("semantic_scholar", get_abstract_semantic),
    ("crossref", get_abstract_crossref),
    ("europe_pmc", get_abstract_europe_pmc),
    ("pubmed", get_abstract_pubmed),
]

TITLE_FETCHERS = [
    ("crossref_title", get_abstract_crossref_by_title),
    ("europe_pmc_title", get_abstract_europe_pmc_by_title),
]


def _fetch_abstract_waterfall(doi, title, session: requests.Session, stats: dict):
    """Try each source in order.

    Returns (text, source, exhausted, statuses):
      text      -- abstract text, or None
      source    -- which source supplied it, or None
      exhausted -- True only if EVERY source reported a hard miss. False when any
                   source soft-failed, so the row is retried on a later run
                   instead of being written off as "no abstract exists".
      statuses  -- {source: status} for diagnostics
    """
    statuses = {}
    all_hard = True

    doi_norm = clean_doi(doi)

    if doi_norm:
        for name, fetcher in DOI_FETCHERS:
            text, status = fetcher(doi, session)
            statuses[name] = status
            stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1
            if text:
                stats[name] += 1
                return text, name, False, statuses
            if status not in HARD_MISS_STATUSES:
                all_hard = False
    else:
        statuses["doi"] = ST_NO_DOI

    # No DOI, or the DOI resolved nowhere: try matching on title.
    for name, fetcher in TITLE_FETCHERS:
        text, status = fetcher(title, session)
        statuses[name] = status
        stats["status_counts"][status] = stats["status_counts"].get(status, 0) + 1
        if text:
            stats[name] += 1
            return text, name, False, statuses
        if status not in HARD_MISS_STATUSES:
            all_hard = False

    stats["failed"] += 1
    return None, None, all_hard, statuses


# ---------------------------------------------------------------------------
# Diagnostic mode
# ---------------------------------------------------------------------------


def run_diagnose(dois) -> None:
    """Probe every source for a handful of DOIs and print status per source."""
    _configure_output_streams()
    session = requests.Session()
    try:
        for doi in dois:
            print(f"\n{'='*74}\nDOI: {doi}\n{'='*74}")
            print(f"{'source':<20} {'status':<16} {'chars':>7}  preview")
            print("-" * 74)
            for name, fetcher in DOI_FETCHERS:
                text, status = fetcher(doi, session)
                n = len(text) if text else 0
                preview = (text[:60].replace("\n", " ") + "...") if text else ""
                print(f"{name:<20} {status:<16} {n:>7}  {preview}")
                time.sleep(0.2)
    finally:
        session.close()
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_fetch_abstracts(
    paths: PipelinePaths,
    save_interval: int = SAVE_INTERVAL,
    retry_unavailable: bool = False,
    log_statuses: bool = False,
):
    _configure_output_streams()
    input_csv = paths.combined_scopus_api
    output_csv = paths.with_abstracts

    stats = {
        "openalex": 0,
        "semantic_scholar": 0,
        "crossref": 0,
        "europe_pmc": 0,
        "pubmed": 0,
        "crossref_title": 0,
        "europe_pmc_title": 0,
        "already_had": 0,
        "failed": 0,
        "marked_none": 0,
        "left_open": 0,
        "cleared_none": 0,
        "migrated_legacy": 0,
        "status_counts": {},
    }

    if output_csv.exists():
        print(f"Resuming from existing file: {output_csv}")
        df = pd.read_csv(output_csv)
    else:
        print(f"Starting fresh from: {input_csv}")
        df = pd.read_csv(input_csv)

    # FIX: Abstract holds text only. Provenance moves to its own column so that
    # downstream consumers (the LLM labellers) never receive a sentinel string
    # as if it were abstract text.
    if "Abstract" not in df.columns:
        df["Abstract"] = ""
    if "abstract_source" not in df.columns:
        df["abstract_source"] = ""
    df["Abstract"] = df["Abstract"].astype("object")
    df["abstract_source"] = df["abstract_source"].astype("object")

    # Migrate the old in-column sentinel.
    legacy = df["Abstract"].map(
        lambda v: str(v).strip() == LEGACY_UNAVAILABLE_MARKER if pd.notna(v) else False
    )
    stats["migrated_legacy"] = int(legacy.sum())
    if stats["migrated_legacy"]:
        df.loc[legacy, "Abstract"] = ""
        df.loc[legacy, "abstract_source"] = SOURCE_NONE
        print(
            f"Migrated {stats['migrated_legacy']} legacy {LEGACY_UNAVAILABLE_MARKER} "
            f"markers into abstract_source='{SOURCE_NONE}'"
        )

    # Backfill provenance for abstracts that came in with the export.
    has_text = df["Abstract"].map(_abstract_cell_has_text)
    missing_src = has_text & ~df["abstract_source"].map(
        lambda v: bool(str(v).strip()) if pd.notna(v) else False
    )
    df.loc[missing_src, "abstract_source"] = SOURCE_PRESENT

    if retry_unavailable:
        cleared = df["abstract_source"].map(_is_exhausted)
        stats["cleared_none"] = int(cleared.sum())
        df.loc[cleared, "abstract_source"] = ""
        print(
            f"Retry unavailable: cleared {stats['cleared_none']} "
            f"abstract_source='{SOURCE_NONE}' rows for re-fetch"
        )

    def _skip(idx) -> bool:
        return _abstract_cell_has_text(df.at[idx, "Abstract"]) or _is_exhausted(
            df.at[idx, "abstract_source"]
        )

    total = len(df)
    stats["already_had"] = sum(1 for idx in df.index if _skip(idx))
    needs_abstract = total - stats["already_had"]

    print(f"\nquery_id={paths.query_id}")
    print("Starting abstract fetching...")
    print(f"Total papers: {total}")
    print(f"Already resolved (has text, or exhausted): {stats['already_had']}")
    print(f"Need abstract fetch attempts: {needs_abstract}")
    print(
        "\nTrying sources in order: "
        "OpenAlex -> Semantic Scholar -> CrossRef -> Europe PMC -> PubMed"
    )
    print("Fallback for rows without a usable DOI: CrossRef title -> Europe PMC title\n")

    processed = 0
    session = requests.Session()
    try:
        for idx in tqdm(
            df.index,
            total=total,
            desc="Fetching abstracts",
            file=sys.stderr,
            mininterval=0.5,
        ):
            if _skip(idx):
                continue

            text, source, exhausted, statuses = _fetch_abstract_waterfall(
                df.at[idx, "DOI"] if "DOI" in df.columns else None,
                df.at[idx, "Title"] if "Title" in df.columns else None,
                session,
                stats,
            )

            if text:
                df.at[idx, "Abstract"] = text
                df.at[idx, "abstract_source"] = source
            elif exhausted:
                # Every source was reachable and none had an abstract. Safe to
                # write off; a rerun will skip this row.
                df.at[idx, "abstract_source"] = SOURCE_NONE
                stats["marked_none"] += 1
            else:
                # At least one soft failure. Leave blank so a rerun retries it.
                df.at[idx, "abstract_source"] = ""
                stats["left_open"] += 1

            if log_statuses and not text:
                tqdm.write(
                    f"  MISS {df.at[idx, 'DOI'] if 'DOI' in df.columns else '(no doi)'} "
                    f"exhausted={exhausted} " + ", ".join(f"{k}={v}" for k, v in statuses.items()),
                    file=sys.stderr,
                )

            processed += 1

            if processed % save_interval == 0:
                output_csv.parent.mkdir(parents=True, exist_ok=True)
                df.to_csv(output_csv, index=False)
                found = sum(stats[s] for s in SOURCE_ORDER + SOURCE_TITLE_FALLBACK)
                tqdm.write(
                    f"Progress: {processed}/{needs_abstract} | Found: {found} | "
                    + ", ".join(
                        f"{s}: {stats[s]}" for s in SOURCE_ORDER + SOURCE_TITLE_FALLBACK
                    )
                    + f" | Exhausted: {stats['marked_none']}"
                    f" | Retryable: {stats['left_open']}",
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
    print(f"Total papers: {total}")
    print(f"Rows already resolved at start: {stats['already_had']}")
    print("\nAbstracts fetched this run:")
    for s in SOURCE_ORDER + SOURCE_TITLE_FALLBACK:
        print(f"  {s:<20} {stats[s]}")
    print(f"  {'no source had one':<20} {stats['failed']}")
    print(f"\nMarked '{SOURCE_NONE}' (all sources reachable, none had an abstract): "
          f"{stats['marked_none']}")
    print(f"Left blank for retry (at least one soft failure): {stats['left_open']}")

    if stats["status_counts"]:
        print("\nStatus tally across all source calls:")
        for status, count in sorted(
            stats["status_counts"].items(), key=lambda kv: -kv[1]
        ):
            flag = "" if status in HARD_MISS_STATUSES or status == ST_OK else "   <-- investigate"
            print(f"  {status:<16} {count}{flag}")

    readable = int(df["Abstract"].map(_abstract_cell_has_text).sum())
    exhausted_n = int(df["abstract_source"].map(_is_exhausted).sum())
    pct = (readable / total * 100) if total else 0
    print(f"\nFinal readable abstracts: {readable}/{total} ({pct:.1f}%)")
    print(f"Exhausted rows: {exhausted_n}/{total}")
    print(f"{'='*70}")
    print(f"Complete. Results saved to {output_csv}")
    print("\nCoverage by source (for Methods):")
    print(df["abstract_source"].fillna("").replace("", "(pending)").value_counts().to_string())


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(
        description="Fetch abstracts for combined Scopus API export"
    )
    add_query_arg(parser)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument(
        "--retry-unavailable",
        action="store_true",
        help=(
            f"Clear abstract_source='{SOURCE_NONE}' rows and re-run the full waterfall. "
            "Required after this patch, since earlier runs marked rows unavailable "
            "on failures that were actually retryable."
        ),
    )
    parser.add_argument(
        "--log-statuses",
        action="store_true",
        help="Print the per-source status for every row that yields no abstract.",
    )
    parser.add_argument(
        "--diagnose",
        nargs="+",
        metavar="DOI",
        help="Probe all sources for the given DOIs and print a status table, then exit.",
    )
    args = parser.parse_args()

    if args.diagnose:
        run_diagnose(args.diagnose)
    else:
        run_fetch_abstracts(
            PipelinePaths(args.query_id),
            save_interval=args.save_interval,
            retry_unavailable=args.retry_unavailable,
            log_statuses=args.log_statuses,
        )
