"""
Fetch full author data from OpenAlex API for papers in the dataset.
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

from lib.pipeline import PipelinePaths, add_query_arg  # noqa: E402

SAVE_INTERVAL = 50


def get_authors_openalex(doi, max_retries=3):
    if pd.isna(doi) or not doi:
        return None, None, None

    url = f"https://api.openalex.org/works/https://doi.org/{doi}"

    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=15)
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


def run_fetch_authors(paths: PipelinePaths, save_interval: int = SAVE_INTERVAL):
    input_csv = paths.with_abstracts
    output_csv = paths.with_authors

    if output_csv.exists():
        print(f"Resuming from existing file: {output_csv}")
        df = pd.read_csv(output_csv)
        start_index = len(df[df["All_Authors"].notna() & (df["All_Authors"] != "")])
    else:
        print(f"Starting fresh from: {input_csv}")
        df = pd.read_csv(input_csv)
        start_index = 0

    if "All_Authors" not in df.columns:
        df["All_Authors"] = ""
    if "Author_Count_Actual" not in df.columns:
        df["Author_Count_Actual"] = 0
    if "Author_Affiliations" not in df.columns:
        df["Author_Affiliations"] = ""

    print(f"\nquery_id={paths.query_id}")
    print(f"\nFetching author data from OpenAlex API...")
    print(f"Starting from index {start_index} of {len(df)} papers\n")

    for idx, row in tqdm(df.iterrows(), total=len(df), initial=start_index):
        if pd.notna(row.get("All_Authors")) and str(row.get("All_Authors")).strip():
            continue

        doi = row.get("DOI")
        all_authors, author_count, affiliations = get_authors_openalex(doi)

        if all_authors:
            df.at[idx, "All_Authors"] = all_authors
            df.at[idx, "Author_Count_Actual"] = author_count
            if affiliations:
                df.at[idx, "Author_Affiliations"] = affiliations

        time.sleep(0.2)

        if (idx + 1) % save_interval == 0:
            output_csv.parent.mkdir(parents=True, exist_ok=True)
            df.to_csv(output_csv, index=False)
            print(f"\nSaved progress at {idx + 1} papers.")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Complete! Enriched file saved to {output_csv}")

    filled = df["All_Authors"].notna() & (df["All_Authors"] != "")
    print(f"\nSummary:")
    print(f"  Papers with author data: {filled.sum()} / {len(df)} ({100*filled.sum()/len(df):.1f}%)")
    if filled.sum() > 0:
        print(f"  Average authors per paper: {df[filled]['Author_Count_Actual'].mean():.2f}")
        print(f"  Papers with 3+ authors: {(df[filled]['Author_Count_Actual'] >= 3).sum()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fetch authors from OpenAlex for combined export")
    add_query_arg(parser)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    args = parser.parse_args()
    run_fetch_authors(PipelinePaths(args.query_id), save_interval=args.save_interval)
