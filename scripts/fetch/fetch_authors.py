"""
Fetch full author data from OpenAlex API for papers in the dataset.
This addresses the limitation that Publish or Perish only exports the first author.
"""

import pandas as pd
import requests
from tqdm import tqdm
import time
import os
import json
from pathlib import Path

# Get project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
INPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_with_abstracts.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_with_authors.csv"
SAVE_INTERVAL = 50  # Save every N papers

# Load data
if OUTPUT_CSV.exists():
    print(f"Resuming from existing file: {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV)
    start_index = len(df[df['All_Authors'].notna() & (df['All_Authors'] != '')])
else:
    print(f"Starting fresh from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)
    start_index = 0

# Ensure new columns exist
if 'All_Authors' not in df.columns:
    df['All_Authors'] = ''
if 'Author_Count_Actual' not in df.columns:
    df['Author_Count_Actual'] = 0
if 'Author_Affiliations' not in df.columns:
    df['Author_Affiliations'] = ''

def get_authors_openalex(doi, max_retries=3):
    """Fetch full author list from OpenAlex API"""
    if pd.isna(doi) or not doi:
        return None, None, None
    
    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                
                # Extract authors
                authors = []
                affiliations = []
                
                if 'authorships' in data:
                    for authorship in data['authorships']:
                        author = authorship.get('author', {})
                        if author:
                            # Get author name
                            display_name = author.get('display_name', '')
                            if display_name:
                                authors.append(display_name)
                            
                            # Get affiliations
                            author_affiliations = []
                            for inst in authorship.get('institutions', []):
                                inst_name = inst.get('display_name', '')
                                if inst_name:
                                    author_affiliations.append(inst_name)
                            
                            if author_affiliations:
                                affiliations.append('; '.join(author_affiliations))
                            else:
                                affiliations.append('')
                
                all_authors_str = '; '.join(authors) if authors else None
                author_count = len(authors) if authors else 0
                affiliations_str = ' | '.join(affiliations) if affiliations else None
                
                return all_authors_str, author_count, affiliations_str
                
            elif r.status_code == 404:
                return None, None, None  # DOI not found
            else:
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
                    continue
                return None, None, None
                
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return None, None, None
    
    return None, None, None

# Main enrichment loop
print(f"\nFetching author data from OpenAlex API...")
print(f"Starting from index {start_index} of {len(df)} papers\n")

for idx, row in tqdm(df.iterrows(), total=len(df), initial=start_index):
    # Skip if already filled
    if pd.notna(row.get('All_Authors')) and str(row.get('All_Authors')).strip():
        continue
    
    doi = row.get('DOI')
    all_authors, author_count, affiliations = get_authors_openalex(doi)
    
    if all_authors:
        df.at[idx, 'All_Authors'] = all_authors
        df.at[idx, 'Author_Count_Actual'] = author_count
        if affiliations:
            df.at[idx, 'Author_Affiliations'] = affiliations
    
    # Be polite to API
    time.sleep(0.2)
    
    # Save progress periodically
    if (idx + 1) % SAVE_INTERVAL == 0:
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"\nSaved progress at {idx + 1} papers.")

# Final save
df.to_csv(OUTPUT_CSV, index=False)
print(f"\n✓ Complete! Enriched file saved to {OUTPUT_CSV}")

# Summary statistics
filled = df['All_Authors'].notna() & (df['All_Authors'] != '')
print(f"\nSummary:")
print(f"  Papers with author data: {filled.sum()} / {len(df)} ({100*filled.sum()/len(df):.1f}%)")
if filled.sum() > 0:
    print(f"  Average authors per paper: {df[filled]['Author_Count_Actual'].mean():.2f}")
    print(f"  Papers with 3+ authors: {(df[filled]['Author_Count_Actual'] >= 3).sum()}")

