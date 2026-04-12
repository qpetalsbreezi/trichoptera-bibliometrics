"""
Combine multiple Scopus year exports (Publish or Perish) into a single dataset
"""

import argparse
import glob
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.pipeline import PipelinePaths, add_query_arg  # noqa: E402


def combine_scopus_exports(paths: PipelinePaths):
    """Combine all year-specific Scopus exports into one dataset"""

    data_dir = paths.raw_pop
    pattern = f"{paths.query_id}_scopus_raw_*.csv"
    output_file = paths.pop_combined

    pattern_path = str(data_dir / pattern)
    files = sorted(glob.glob(pattern_path))

    files = [f for f in files if Path(f) != output_file and not f.endswith("_2010_2025.csv")]

    if not files:
        print(f"No files found matching pattern: {pattern_path}")
        return None

    print(f"query_id={paths.query_id}")
    print(f"Found {len(files)} year files:")
    for f in files:
        year = Path(f).stem.split("_")[-1]
        print(f"  - {year}")

    all_dataframes = []
    total_papers = 0

    for file in files:
        year = Path(file).stem.split("_")[-1]
        try:
            df = pd.read_csv(file)
            count = len(df)
            total_papers += count
            print(f"  Loaded {year}: {count} papers")
            all_dataframes.append(df)
        except Exception as e:
            print(f"  Error loading {file}: {e}")
            continue

    if not all_dataframes:
        print("No data loaded!")
        return None

    print(f"\nCombining {len(all_dataframes)} files...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)

    print(f"Total papers before deduplication: {len(combined_df)}")

    initial_count = len(combined_df)

    if "DOI" in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=["DOI"], keep="first")
        doi_deduped = initial_count - len(combined_df)
        print(f"  Removed {doi_deduped} duplicates by DOI")

    if "Title" in combined_df.columns:
        before_title_dedup = len(combined_df)
        no_doi = combined_df["DOI"].isna() | (combined_df["DOI"] == "")
        if no_doi.sum() > 0:
            combined_df["Title_Normalized"] = combined_df["Title"].fillna("").str.lower().str.strip()
            mask = ~combined_df.duplicated(subset=["Title_Normalized"], keep="first")
            combined_df = combined_df[mask]
            title_deduped = before_title_dedup - len(combined_df)
            if title_deduped > 0:
                print(f"  Removed {title_deduped} duplicates by Title")
        combined_df = combined_df.drop(columns=["Title_Normalized"])

    print(f"Total papers after deduplication: {len(combined_df)}")

    if "Year" in combined_df.columns:
        print(f"\nYear distribution:")
        year_counts = combined_df["Year"].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count} papers")

    output_file.parent.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving combined dataset to {output_file}...")
    combined_df.to_csv(output_file, index=False)
    print(f"✓ Saved {len(combined_df)} papers to {output_file}")

    print(f"\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(
        f"Years covered: {combined_df['Year'].min() if 'Year' in combined_df.columns else 'N/A'} - "
        f"{combined_df['Year'].max() if 'Year' in combined_df.columns else 'N/A'}"
    )
    print(f"Total unique papers: {len(combined_df)}")
    print(f"Files combined: {len(files)}")
    print(f"Output file: {output_file}")
    print("=" * 60)

    return combined_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Combine PoP Scopus CSVs for one query_id")
    add_query_arg(parser)
    args = parser.parse_args()
    combine_scopus_exports(PipelinePaths(args.query_id))
