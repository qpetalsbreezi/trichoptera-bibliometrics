"""
Combine multiple Scopus API year exports into a single dataset
For use in the full pipeline: fetch_abstracts -> llm_code -> analysis
"""

import pandas as pd
from pathlib import Path

# Get project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
DATA_DIR = PROJECT_ROOT / "data/raw/scopus_api"
OUTPUT_FILE = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_combined_2010_2025.csv"

def combine_scopus_api_years():
    """Combine all year-specific Scopus API exports into one dataset"""
    
    years = list(range(2010, 2026))
    files = []
    
    # Find all year files
    for year in years:
        file = DATA_DIR / f"scopus_api_{year}.csv"
        if file.exists():
            files.append((year, file))
        else:
            print(f"Warning: {file} not found")
    
    if not files:
        print(f"No files found in {DATA_DIR}")
        return None
    
    print(f"Found {len(files)} year files:")
    for year, file in files:
        print(f"  - {year}")
    
    # Load and combine all files
    all_dataframes = []
    total_papers = 0
    
    for year, file in files:
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
    initial_count = len(combined_df)
    
    # First, try to deduplicate by DOI
    if 'DOI' in combined_df.columns:
        combined_df = combined_df.drop_duplicates(subset=['DOI'], keep='first')
        doi_deduped = initial_count - len(combined_df)
        if doi_deduped > 0:
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
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
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
    combined_df = combine_scopus_api_years()
