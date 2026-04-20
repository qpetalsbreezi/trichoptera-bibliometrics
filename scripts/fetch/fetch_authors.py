"""
Fetch full author lists and affiliations from OpenAlex (by DOI).

Scopus/Google `Authors` in the export is often a single name; we do not copy it
into `All_Authors`, so OpenAlex can supply complete multi-author strings.
"""

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
    openalex_headers,
    openalex_work_url,
    retry_after_seconds,
)
from lib.pipeline import PipelinePaths, add_query_arg, load_dotenv  # noqa: E402

SAVE_INTERVAL = 50


def get_authors_openalex(
    doi,
    session: requests.Session,
    max_retries=4,
    headers: dict | None = None,
):
    doi_norm = clean_doi(doi)
    if not doi_norm:
        return None, None, None

    url = openalex_work_url(doi_norm)
    headers = headers or openalex_headers()

    for attempt in range(max_retries):
        try:
            r = session.get(url, timeout=OPENALEX_TIMEOUT, headers=headers)
            if r.status_code == 200:
                data = r.json()

                authors = []
                affiliations = []

                if "authorships" in data:
                    for authorship in data["authorships"]:
                        author = authorship.get("author", {})
                        if author:
                            display_name = author.get("display_name", "")
                            if display_name:
                                authors.append(display_name)

                            author_affiliations = []
                            for inst in authorship.get("institutions", []):
                                inst_name = inst.get("display_name", "")
                                if inst_name:
                                    author_affiliations.append(inst_name)

                            if author_affiliations:
                                affiliations.append("; ".join(author_affiliations))
                            else:
                                affiliations.append("")

                all_authors_str = "; ".join(authors) if authors else None
                author_count = len(authors) if authors else 0
                affiliations_str = " | ".join(affiliations) if affiliations else None

                return all_authors_str, author_count, affiliations_str

            if r.status_code == 404:
                return None, None, None
            if r.status_code == 429:
                wait = retry_after_seconds(r) or min(10.0 * (attempt + 1), 60.0)
                time.sleep(wait)
                if attempt < max_retries - 1:
                    continue
                return None, None, None
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, None, None

        except (requests.exceptions.Timeout, requests.exceptions.RequestException):
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, None, None

    return None, None, None


def _configure_output_streams() -> None:
    """Line-buffer stdout/stderr when redirected so logs and tqdm update promptly."""
    for stream in (sys.stdout, sys.stderr):
        if not stream.isatty() and hasattr(stream, "reconfigure"):
            try:
                stream.reconfigure(line_buffering=True)
            except OSError:
                pass


def run_fetch_authors(
    paths: PipelinePaths,
    save_interval: int = SAVE_INTERVAL,
    request_delay: float = 0.35,
):
    _configure_output_streams()
    input_csv = paths.with_abstracts
    output_csv = paths.with_authors

    if output_csv.exists():
        print(f"Resuming from existing file: {output_csv}")
        df = pd.read_csv(output_csv, low_memory=False)
        start_index = len(df[df["All_Authors"].notna() & (df["All_Authors"] != "")])
    else:
        print(f"Starting fresh from: {input_csv}")
        df = pd.read_csv(input_csv, low_memory=False)
        start_index = 0

    if "All_Authors" not in df.columns:
        df["All_Authors"] = ""
    else:
        df["All_Authors"] = df["All_Authors"].fillna("").astype("string")
    if "Author_Count_Actual" not in df.columns:
        df["Author_Count_Actual"] = 0
    else:
        df["Author_Count_Actual"] = pd.to_numeric(
            df["Author_Count_Actual"], errors="coerce"
        ).fillna(0).astype(int)
    if "Author_Affiliations" not in df.columns:
        df["Author_Affiliations"] = ""
    else:
        df["Author_Affiliations"] = df["Author_Affiliations"].fillna("").astype("string")

    needs_authors = (
        df["All_Authors"].fillna("").astype(str).str.strip().eq("")
        & df["DOI"].notna()
        & df["DOI"].astype(str).str.strip().ne("")
    )
    pending_indices = df.index[needs_authors].tolist()
    already_filled = len(df) - len(pending_indices)

    print(f"\nquery_id={paths.query_id}")
    print(f"\nFetching author data from OpenAlex API...")
    print(f"Starting from index {start_index} of {len(df)} papers")
    print(f"Already have author data for {already_filled} papers")
    print(f"Pending DOI lookups: {len(pending_indices)}\n")

    headers = openalex_headers()
    session = requests.Session()
    processed = 0

    for idx in tqdm(
        pending_indices,
        total=len(pending_indices),
        desc="Fetching authors",
        file=sys.stderr,
        mininterval=0.5,
    ):
        doi = df.at[idx, "DOI"]
        all_authors, author_count, affiliations = get_authors_openalex(
            doi,
            session=session,
            headers=headers,
        )

        if all_authors:
            df.at[idx, "All_Authors"] = all_authors
            df.at[idx, "Author_Count_Actual"] = author_count
            if affiliations:
                df.at[idx, "Author_Affiliations"] = affiliations

        processed += 1
        time.sleep(request_delay)

        if processed % save_interval == 0:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            filled_so_far = (
                df["All_Authors"].fillna("").astype(str).str.strip().ne("").sum()
            )
            tqdm.write(
                f"Saved progress at {processed}/{len(pending_indices)} pending DOI rows "
                f"| filled {filled_so_far}/{len(df)} total papers.",
                file=sys.stderr,
            )
            sys.stderr.flush()

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    session.close()
    print(f"\n✓ Complete! Enriched file saved to {output_csv}")

    filled = df["All_Authors"].notna() & (df["All_Authors"] != "")
    print(f"\nSummary:")
    print(f"  Papers with author data: {filled.sum()} / {len(df)} ({100*filled.sum()/len(df):.1f}%)")
    if filled.sum() > 0:
        print(f"  Average authors per paper: {df[filled]['Author_Count_Actual'].mean():.2f}")
        print(f"  Papers with 3+ authors: {(df[filled]['Author_Count_Actual'] >= 3).sum()}")


if __name__ == "__main__":
    load_dotenv()
    parser = argparse.ArgumentParser(description="Fetch authors from OpenAlex for combined export")
    add_query_arg(parser)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument(
        "--request-delay",
        type=float,
        default=0.35,
        help="Seconds to sleep after each OpenAlex request (default: 0.35)",
    )
    args = parser.parse_args()
    run_fetch_authors(
        PipelinePaths(args.query_id),
        save_interval=args.save_interval,
        request_delay=args.request_delay,
    )
