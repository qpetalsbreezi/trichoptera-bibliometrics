"""
Combine multiple Scopus year exports into a single dataset
For RQ2: Temporal and Geographic Growth analysis
"""

import pandas as pd
import glob
import os
from pathlib import Path

# Configuration
DATA_DIR = "data"
PATTERN = "trichoptera_scopus_raw_*.csv"
OUTPUT_FILE = "data/trichoptera_scopus_raw_2010_2025.csv"

def combine_scopus_exports():
    """Combine all year-specific Scopus exports into one dataset"""
    
    # Find all matching files
    pattern = os.path.join(DATA_DIR, PATTERN)
    files = sorted(glob.glob(pattern))
    
    # Filter out the generic raw file and combined file if they exist
    files = [f for f in files if f != "data/trichoptera_scopus_raw.csv" 
             and f != OUTPUT_FILE and not f.endswith("_2010_2025.csv")]
    
    if not files:
        print(f"No files found matching pattern: {pattern}")
        return None
    
    print(f"Found {len(files)} year files:")
    for f in files:
        year = Path(f).stem.split('_')[-1]
        print(f"  - {year}")
    
    # Load and combine all files
    all_dataframes = []
    total_papers = 0
    
    for file in files:
        year = Path(file).stem.split('_')[-1]
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
    
    # Combine all dataframes
    print(f"\nCombining {len(all_dataframes)} files...")
    combined_df = pd.concat(all_dataframes, ignore_index=True)
    
    print(f"Total papers before deduplication: {len(combined_df)}")
    
    # Remove duplicates based on DOI (most reliable)
    # Keep first occurrence
    initial_count = len(combined_df)
    
    # First, try to deduplicate by DOI
    if 'DOI' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['DOI'], keep='first')
        doi_deduped = initial_count - len(combined_df)
        print(f"  Removed {doi_deduped} duplicates by DOI")
    
    # Then deduplicate by title (for papers without DOI)
    if 'Title' in combined_df.columns:
        before_title_dedup = len(combined_df)
        # Only check titles for papers without DOI
        no_doi = combined_df['DOI'].isna() | (combined_df['DOI'] == '')
        if no_doi.sum() > 0:
            # Normalize titles for comparison
            combined_df['Title_Normalized'] = combined_df['Title'].fillna('').str.lower().str.strip()
            # Remove duplicates by normalized title (only for no-DOI papers)
            mask = ~combined_df.duplicated(subset=['Title_Normalized'], keep='first')
            combined_df = combined_df[mask]
            title_deduped = before_title_dedup - len(combined_df)
            if title_deduped > 0:
                print(f"  Removed {title_deduped} duplicates by Title")
        combined_df = combined_df.drop(columns=['Title_Normalized'])
    
    print(f"Total papers after deduplication: {len(combined_df)}")
    
    # Verify year distribution
    if 'Year' in combined_df.columns:
        print(f"\nYear distribution:")
        year_counts = combined_df['Year'].value_counts().sort_index()
        for year, count in year_counts.items():
            print(f"  {year}: {count} papers")
    
    # Save combined file
    print(f"\nSaving combined dataset to {OUTPUT_FILE}...")
    combined_df.to_csv(OUTPUT_FILE, index=False)
    print(f"✓ Saved {len(combined_df)} papers to {OUTPUT_FILE}")
    
    # Summary statistics
    print(f"\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    print(f"Years covered: {combined_df['Year'].min() if 'Year' in combined_df.columns else 'N/A'} - {combined_df['Year'].max() if 'Year' in combined_df.columns else 'N/A'}")
    print(f"Total unique papers: {len(combined_df)}")
    print(f"Files combined: {len(files)}")
    print(f"Output file: {OUTPUT_FILE}")
    print("="*60)
    
    return combined_df


if __name__ == "__main__":
    combined_df = combine_scopus_exports()

