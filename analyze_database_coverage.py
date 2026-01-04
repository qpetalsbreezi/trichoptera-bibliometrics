"""
Analyze database coverage comparison between Scopus and Google Scholar
For RQ1: Database Coverage & Reliability
"""

import pandas as pd
import numpy as np
from difflib import SequenceMatcher
import re

# File paths
SCOPUS_FILE = "data/raw/scopus_years/trichoptera_scopus_raw_2023.csv"
GS_FILE = "data/raw/google_scholar/trichoptera_google_scholar_raw_2023.csv"
OUTPUT_DIR = "analysis/rq1_coverage"

# Create output directory
import os
os.makedirs(OUTPUT_DIR, exist_ok=True)


def normalize_title(title):
    """Normalize title for comparison"""
    if pd.isna(title):
        return ""
    title = str(title).lower()
    # Remove special characters, extra spaces
    title = re.sub(r'[^\w\s]', '', title)
    title = ' '.join(title.split())
    return title


def title_similarity(title1, title2):
    """Calculate similarity between two titles"""
    return SequenceMatcher(None, normalize_title(title1), normalize_title(title2)).ratio()


def find_overlaps(scopus_df, gs_df):
    """Find papers that appear in both databases"""
    overlaps = []
    scopus_matched = set()
    gs_matched = set()
    
    # First, match by DOI (most reliable)
    scopus_dois = {str(row['DOI']).lower(): idx for idx, row in scopus_df.iterrows() 
                   if pd.notna(row.get('DOI')) and str(row['DOI']).strip()}
    gs_dois = {str(row['DOI']).lower(): idx for idx, row in gs_df.iterrows() 
               if pd.notna(row.get('DOI')) and str(row['DOI']).strip()}
    
    for doi, scopus_idx in scopus_dois.items():
        if doi in gs_dois:
            gs_idx = gs_dois[doi]
            overlaps.append({
                'match_type': 'DOI',
                'scopus_idx': scopus_idx,
                'gs_idx': gs_idx,
                'similarity': 1.0
            })
            scopus_matched.add(scopus_idx)
            gs_matched.add(gs_idx)
    
    # Then match by title similarity (for papers without DOI)
    print("Matching by title similarity...")
    for scopus_idx, scopus_row in scopus_df.iterrows():
        if scopus_idx in scopus_matched:
            continue
        
        scopus_title = scopus_row.get('Title', '')
        if pd.isna(scopus_title) or not str(scopus_title).strip():
            continue
        
        best_match = None
        best_similarity = 0.85  # Threshold for title matching
        
        for gs_idx, gs_row in gs_df.iterrows():
            if gs_idx in gs_matched:
                continue
            
            gs_title = gs_row.get('Title', '')
            if pd.isna(gs_title) or not str(gs_title).strip():
                continue
            
            similarity = title_similarity(scopus_title, gs_title)
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = gs_idx
        
        if best_match is not None:
            overlaps.append({
                'match_type': 'Title',
                'scopus_idx': scopus_idx,
                'gs_idx': best_match,
                'similarity': best_similarity
            })
            scopus_matched.add(scopus_idx)
            gs_matched.add(best_match)
    
    return overlaps, scopus_matched, gs_matched


def classify_journal_type(journal_name, publisher):
    """Classify journal as regional or international"""
    if pd.isna(journal_name):
        return "Unknown"
    
    journal_lower = str(journal_name).lower()
    
    # Indicators of regional journals
    regional_indicators = [
        'brasileira', 'brasil', 'chilena', 'chile', 'argentina', 'mexicana',
        'chinese', 'japanese', 'korean', 'indian', 'african', 'asian',
        'revista', 'journal of', 'acta', 'annales', 'bulletin',
        'regional', 'local', 'national'
    ]
    
    # International/major journals
    international_indicators = [
        'nature', 'science', 'plos', 'pnas', 'proceedings',
        'freshwater biology', 'hydrobiologia', 'ecology', 'systematic'
    ]
    
    # Check for international first
    for indicator in international_indicators:
        if indicator in journal_lower:
            return "International"
    
    # Check for regional
    for indicator in regional_indicators:
        if indicator in journal_lower:
            return "Regional"
    
    # Default to unknown if can't determine
    return "Unknown"


def detect_language(title, abstract):
    """Simple language detection (basic - looks for non-English characters)"""
    text = ""
    if pd.notna(title):
        text += str(title)
    if pd.notna(abstract):
        text += " " + str(abstract)
    
    if not text:
        return "Unknown"
    
    # Check for common non-English characters
    if re.search(r'[àáâãäåæçèéêëìíîïñòóôõöøùúûüýÿ]', text, re.IGNORECASE):
        return "Non-English"
    if re.search(r'[α-ωΑ-Ω]', text):  # Greek
        return "Non-English"
    if re.search(r'[一-龯]', text):  # Chinese/Japanese
        return "Non-English"
    
    return "English"


def analyze_database_coverage():
    """Main analysis function"""
    print("Loading datasets...")
    scopus_df = pd.read_csv(SCOPUS_FILE)
    gs_df = pd.read_csv(GS_FILE)
    
    print(f"Scopus: {len(scopus_df)} papers")
    print(f"Google Scholar: {len(gs_df)} papers")
    
    # Add source identifier
    scopus_df['Database'] = 'Scopus'
    gs_df['Database'] = 'Google Scholar'
    
    # Find overlaps
    print("\nFinding overlaps between databases...")
    overlaps, scopus_matched, gs_matched = find_overlaps(scopus_df, gs_df)
    
    print(f"Found {len(overlaps)} overlapping papers")
    print(f"  - Matched by DOI: {sum(1 for o in overlaps if o['match_type'] == 'DOI')}")
    print(f"  - Matched by Title: {sum(1 for o in overlaps if o['match_type'] == 'Title')}")
    
    # Calculate unique papers
    scopus_unique = len(scopus_df) - len(scopus_matched)
    gs_unique = len(gs_df) - len(gs_matched)
    
    print(f"\nUnique papers:")
    print(f"  - Scopus only: {scopus_unique}")
    print(f"  - Google Scholar only: {gs_unique}")
    
    # Analyze journal types
    print("\nAnalyzing journal types...")
    scopus_df['Journal_Type'] = scopus_df.apply(
        lambda row: classify_journal_type(row.get('Source', ''), row.get('Publisher', '')), 
        axis=1
    )
    gs_df['Journal_Type'] = gs_df.apply(
        lambda row: classify_journal_type(row.get('Source', ''), row.get('Publisher', '')), 
        axis=1
    )
    
    # Detect language
    print("Detecting languages...")
    scopus_df['Language'] = scopus_df.apply(
        lambda row: detect_language(row.get('Title', ''), row.get('Abstract', '')), 
        axis=1
    )
    gs_df['Language'] = gs_df.apply(
        lambda row: detect_language(row.get('Title', ''), row.get('Abstract', '')), 
        axis=1
    )
    
    # Generate statistics
    stats = {
        'total_papers': {
            'scopus': len(scopus_df),
            'google_scholar': len(gs_df)
        },
        'overlaps': {
            'total': len(overlaps),
            'by_doi': sum(1 for o in overlaps if o['match_type'] == 'DOI'),
            'by_title': sum(1 for o in overlaps if o['match_type'] == 'Title')
        },
        'unique_papers': {
            'scopus_only': scopus_unique,
            'gs_only': gs_unique
        },
        'journal_types': {
            'scopus': {k: int(v) for k, v in scopus_df['Journal_Type'].value_counts().items()},
            'google_scholar': {k: int(v) for k, v in gs_df['Journal_Type'].value_counts().items()}
        },
        'languages': {
            'scopus': {k: int(v) for k, v in scopus_df['Language'].value_counts().items()},
            'google_scholar': {k: int(v) for k, v in gs_df['Language'].value_counts().items()}
        },
        'document_types': {
            'scopus': {k: int(v) for k, v in scopus_df['Type'].value_counts().items()} if 'Type' in scopus_df.columns else {},
            'google_scholar': {k: int(v) for k, v in gs_df['Type'].value_counts().items()} if 'Type' in gs_df.columns else {}
        },
        'citation_stats': {
            'scopus': {
                'mean': float(scopus_df['Cites'].mean()) if 'Cites' in scopus_df.columns else 0,
                'median': float(scopus_df['Cites'].median()) if 'Cites' in scopus_df.columns else 0,
                'total': int(scopus_df['Cites'].sum()) if 'Cites' in scopus_df.columns else 0
            },
            'google_scholar': {
                'mean': float(gs_df['Cites'].mean()) if 'Cites' in gs_df.columns else 0,
                'median': float(gs_df['Cites'].median()) if 'Cites' in gs_df.columns else 0,
                'total': int(gs_df['Cites'].sum()) if 'Cites' in gs_df.columns else 0
            }
        }
    }
    
    # Generate summary report
    report = f"""
DATABASE COVERAGE ANALYSIS REPORT (RQ1)
========================================

Research Question: Do Scopus and Google Scholar provide comparable coverage of 
Trichoptera literature, or are there significant gaps?

Year: 2023
Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}

BASIC STATISTICS
----------------
Total Papers:
  - Scopus: {stats['total_papers']['scopus']}
  - Google Scholar: {stats['total_papers']['google_scholar']}
  - Ratio (GS/Scopus): {stats['total_papers']['google_scholar'] / stats['total_papers']['scopus']:.2f}

OVERLAP ANALYSIS
----------------
Papers in both databases: {stats['overlaps']['total']}
  - Matched by DOI: {stats['overlaps']['by_doi']}
  - Matched by Title: {stats['overlaps']['by_title']}

Unique Papers:
  - Scopus only: {stats['unique_papers']['scopus_only']} ({stats['unique_papers']['scopus_only']/stats['total_papers']['scopus']*100:.1f}% of Scopus)
  - Google Scholar only: {stats['unique_papers']['gs_only']} ({stats['unique_papers']['gs_only']/stats['total_papers']['google_scholar']*100:.1f}% of Google Scholar)

JOURNAL TYPE DISTRIBUTION
--------------------------
Scopus:
{chr(10).join(f"  - {k}: {v} ({v/stats['total_papers']['scopus']*100:.1f}%)" for k, v in stats['journal_types']['scopus'].items())}

Google Scholar:
{chr(10).join(f"  - {k}: {v} ({v/stats['total_papers']['google_scholar']*100:.1f}%)" for k, v in stats['journal_types']['google_scholar'].items())}

LANGUAGE DISTRIBUTION
---------------------
Scopus:
{chr(10).join(f"  - {k}: {v} ({v/stats['total_papers']['scopus']*100:.1f}%)" for k, v in stats['languages']['scopus'].items())}

Google Scholar:
{chr(10).join(f"  - {k}: {v} ({v/stats['total_papers']['google_scholar']*100:.1f}%)" for k, v in stats['languages']['google_scholar'].items())}

CITATION STATISTICS
-------------------
Scopus:
  - Mean citations: {stats['citation_stats']['scopus']['mean']:.2f}
  - Median citations: {stats['citation_stats']['scopus']['median']:.2f}
  - Total citations: {stats['citation_stats']['scopus']['total']}

Google Scholar:
  - Mean citations: {stats['citation_stats']['google_scholar']['mean']:.2f}
  - Median citations: {stats['citation_stats']['google_scholar']['median']:.2f}
  - Total citations: {stats['citation_stats']['google_scholar']['total']}

KEY FINDINGS
------------
1. Coverage Ratio: Google Scholar has {stats['total_papers']['google_scholar'] / stats['total_papers']['scopus']:.1f}x more papers than Scopus for 2023

2. Overlap: {stats['overlaps']['total']} papers ({stats['overlaps']['total']/min(stats['total_papers']['scopus'], stats['total_papers']['google_scholar'])*100:.1f}% of smaller database) appear in both databases

3. Unique Coverage:
   - Scopus captures {stats['unique_papers']['scopus_only']} papers not in Google Scholar
   - Google Scholar captures {stats['unique_papers']['gs_only']} papers not in Scopus

4. Journal Types: 
   - Scopus Regional: {stats['journal_types']['scopus'].get('Regional', 0)} ({stats['journal_types']['scopus'].get('Regional', 0)/stats['total_papers']['scopus']*100:.1f}%)
   - Google Scholar Regional: {stats['journal_types']['google_scholar'].get('Regional', 0)} ({stats['journal_types']['google_scholar'].get('Regional', 0)/stats['total_papers']['google_scholar']*100:.1f}%)

LIMITATIONS
-----------
- Scopus sample limited to 200 results due to API constraints
- Google Scholar sample limited to 1000 results (export limit)
- Language detection is basic and may miss some non-English papers
- Journal type classification is heuristic-based

"""
    
    with open(f"{OUTPUT_DIR}/coverage_report.txt", 'w') as f:
        f.write(report)
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"\nAnalysis complete! Report saved to {OUTPUT_DIR}/coverage_report.txt")
    
    return stats


if __name__ == "__main__":
    stats = analyze_database_coverage()

