"""
RQ4: Collaboration and Authorship Patterns Analysis

Research Question: How have collaboration patterns changed? Are applied studies 
more collaborative than taxonomic studies?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Get project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
# Use enriched file with full author data if available, otherwise fall back to original
AUTHORS_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_with_authors.csv"
INPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_coded.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis/rq4_collaboration"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_collaboration():
    """Analyze collaboration and authorship patterns"""
    
    # Load data - prefer enriched file with full author data
    if AUTHORS_CSV.exists():
        print(f"Loading enriched data with full author information from: {AUTHORS_CSV}")
        df = pd.read_csv(AUTHORS_CSV)
        has_full_author_data = True
    else:
        print(f"Loading data from: {INPUT_CSV}")
        print("Note: Full author data not available. Using limited data from Publish or Perish export.")
        df = pd.read_csv(INPUT_CSV)
        has_full_author_data = False
    
    # Clean and prepare data
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[df['Year'].between(2010, 2025)]
    
    # Filter out papers with "Not Trichoptera-focused"
    df = df[df['Trichoptera_Relevance'] != 'Not Trichoptera-focused']
    
    # Use accurate author count if available, otherwise try to extract from Authors field
    if has_full_author_data and 'Author_Count_Actual' in df.columns:
        print("Using accurate author counts from OpenAlex API")
        df['AuthorCount'] = df['Author_Count_Actual'].fillna(0).astype(int)
        # Filter out papers with 0 authors
        df = df[df['AuthorCount'] > 0]
    else:
        print("Warning: Using unreliable author count from Publish or Perish export")
        # Fallback: Try to extract author count from Authors field (limited)
        def count_authors(authors_str):
            if pd.isna(authors_str) or not authors_str:
                return 1
            authors_str = str(authors_str).strip()
            if not authors_str or authors_str == 'nan':
                return 1
            # Check for multiple authors (comma, semicolon, 'and', '&')
            if ',' in authors_str or ';' in authors_str or ' and ' in authors_str.lower() or ' & ' in authors_str:
                count = 1
                if ',' in authors_str:
                    count = max(count, authors_str.count(',') + 1)
                if ';' in authors_str:
                    count = max(count, authors_str.count(';') + 1)
                if ' and ' in authors_str.lower():
                    count = max(count, authors_str.lower().count(' and ') + 1)
                if ' & ' in authors_str:
                    count = max(count, authors_str.count(' & ') + 1)
                return count
            return 1
        
        df['AuthorCount'] = df['Authors'].apply(count_authors)
        df = df[df['AuthorCount'] > 0]
    
    print(f"Analyzing {len(df)} papers from 2010-2025")
    
    # Define time periods
    early_period = df[df['Year'].between(2010, 2015)]
    recent_period = df[df['Year'].between(2020, 2025)]
    
    # Categorize papers as applied vs taxonomic
    applied_themes = ['Biomonitoring/Water Quality', 'Applied Ecology', 'Conservation', 'Materials Science (Silk)']
    taxonomic_themes = ['Taxonomy/Systematics']
    
    df['Study_Type'] = df['Research_Theme'].apply(
        lambda x: 'Applied' if x in applied_themes 
        else 'Taxonomic' if x in taxonomic_themes 
        else 'Other'
    )
    
    # Author count statistics
    overall_stats = {
        'mean': df['AuthorCount'].mean(),
        'median': df['AuthorCount'].median(),
        'std': df['AuthorCount'].std(),
        'min': df['AuthorCount'].min(),
        'max': df['AuthorCount'].max()
    }
    
    # Author count by time period
    early_author_stats = {
        'mean': early_period['AuthorCount'].mean(),
        'median': early_period['AuthorCount'].median()
    }
    
    recent_author_stats = {
        'mean': recent_period['AuthorCount'].mean(),
        'median': recent_period['AuthorCount'].median()
    }
    
    # Author count by study type
    applied_df = df[df['Study_Type'] == 'Applied']
    taxonomic_df = df[df['Study_Type'] == 'Taxonomic']
    other_df = df[df['Study_Type'] == 'Other']
    
    applied_author_stats = {
        'mean': applied_df['AuthorCount'].mean(),
        'median': applied_df['AuthorCount'].median(),
        'count': len(applied_df)
    }
    
    taxonomic_author_stats = {
        'mean': taxonomic_df['AuthorCount'].mean(),
        'median': taxonomic_df['AuthorCount'].median(),
        'count': len(taxonomic_df)
    }
    
    # Year-by-year author count trends
    yearly_author = df.groupby('Year')['AuthorCount'].agg(['mean', 'median', 'count'])
    
    # Collaboration categories
    def categorize_collaboration(count):
        if count == 1:
            return 'Single author'
        elif count == 2:
            return '2 authors'
        elif count <= 5:
            return '3-5 authors'
        elif count <= 10:
            return '6-10 authors'
        else:
            return '10+ authors'
    
    df['Collaboration_Category'] = df['AuthorCount'].apply(categorize_collaboration)
    
    # Collaboration distribution
    collab_dist = df['Collaboration_Category'].value_counts()
    collab_dist_props = (collab_dist / len(df) * 100).round(2)
    
    # Collaboration by study type
    collab_by_type = pd.crosstab(df['Study_Type'], df['Collaboration_Category'], normalize='index') * 100
    
    # Collaboration trends over time
    yearly_collab = df.groupby(['Year', 'Collaboration_Category']).size().unstack(fill_value=0)
    yearly_collab_props = yearly_collab.div(yearly_collab.sum(axis=1), axis=0) * 100
    
    # Create comprehensive collaboration distribution table (similar to RQ2/RQ3)
    collab_dist_table = pd.DataFrame()
    collab_categories = ['Single author', '2 authors', '3-5 authors', '6-10 authors', '10+ authors']
    
    for year in sorted(yearly_author.index):
        year_data = {
            'Year': year,
            'Total_Papers': int(yearly_author.loc[year, 'count']),
            'Mean_Authors': yearly_author.loc[year, 'mean'],
            'Median_Authors': yearly_author.loc[year, 'median']
        }
        for category in collab_categories:
            if category in yearly_collab.columns:
                count = int(yearly_collab.loc[year, category])
                prop = yearly_collab_props.loc[year, category]
                year_data[f'{category}_Count'] = count
                year_data[f'{category}_Percent'] = prop
            else:
                year_data[f'{category}_Count'] = 0
                year_data[f'{category}_Percent'] = 0.0
        collab_dist_table = pd.concat([collab_dist_table, pd.DataFrame([year_data])], ignore_index=True)
    
    # International collaboration (papers with authors from multiple countries)
    # Note: This is a simplified analysis - would need better country extraction
    # For now, we'll use Region_Global as a proxy
    df['Has_Multiple_Regions'] = df.groupby('Year')['Region_Global'].transform(
        lambda x: x.nunique() > 1 if len(x) > 1 else False
    )
    
    # Generate report
    report = f"""
COLLABORATION AND AUTHORSHIP PATTERNS ANALYSIS (RQ4)
=====================================================

Research Question: How have collaboration patterns changed? Are applied studies 
more collaborative than taxonomic studies?

Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Dataset: {len(df)} papers (2010-2025)

OVERALL AUTHORSHIP STATISTICS
------------------------------
Mean authors per paper: {overall_stats['mean']:.2f}
Median authors per paper: {overall_stats['median']:.1f}
Standard deviation: {overall_stats['std']:.2f}
Range: {overall_stats['min']:.0f} - {overall_stats['max']:.0f} authors

AUTHORSHIP TRENDS OVER TIME
----------------------------
Early Period (2010-2015):
  Mean: {early_author_stats['mean']:.2f} authors
  Median: {early_author_stats['median']:.1f} authors
  Papers: {len(early_period)}

Recent Period (2020-2025):
  Mean: {recent_author_stats['mean']:.2f} authors
  Median: {recent_author_stats['median']:.1f} authors
  Papers: {len(recent_period)}

Change: Mean {'INCREASED' if recent_author_stats['mean'] > early_author_stats['mean'] else 'DECREASED'} 
        by {abs(recent_author_stats['mean'] - early_author_stats['mean']):.2f} authors

AUTHORSHIP BY STUDY TYPE
-------------------------
Applied Studies ({applied_author_stats['count']} papers):
  Mean: {applied_author_stats['mean']:.2f} authors
  Median: {applied_author_stats['median']:.1f} authors

Taxonomic Studies ({taxonomic_author_stats['count']} papers):
  Mean: {taxonomic_author_stats['mean']:.2f} authors
  Median: {taxonomic_author_stats['median']:.1f} authors

Difference: Applied studies have {'MORE' if applied_author_stats['mean'] > taxonomic_author_stats['mean'] else 'FEWER'} 
            authors on average ({abs(applied_author_stats['mean'] - taxonomic_author_stats['mean']):.2f} difference)

COLLABORATION DISTRIBUTION
---------------------------
"""
    
    for category in ['Single author', '2 authors', '3-5 authors', '6-10 authors', '10+ authors']:
        if category in collab_dist.index:
            count = collab_dist[category]
            prop = collab_dist_props[category]
            report += f"  {category}: {count} papers ({prop:.1f}%)\n"
    
    report += f"""
COLLABORATION BY STUDY TYPE
---------------------------
"""
    
    for study_type in ['Applied', 'Taxonomic', 'Other']:
        if study_type in collab_by_type.index:
            report += f"\n{study_type} Studies:\n"
            for category in ['Single author', '2 authors', '3-5 authors', '6-10 authors', '10+ authors']:
                if category in collab_by_type.columns:
                    prop = collab_by_type.loc[study_type, category]
                    report += f"  {category}: {prop:.1f}%\n"
    
    report += f"""
YEAR-BY-YEAR COLLABORATION DISTRIBUTION TABLE
----------------------------------------------
"""
    
    # Create formatted table
    report += f"{'Year':<6} {'Total':<8} {'Mean':<8} {'Median':<8} "
    for category in collab_categories:
        cat_short = category.replace('Single author', 'Single').replace('2 authors', '2 Auth').replace('3-5 authors', '3-5 Auth').replace('6-10 authors', '6-10 Auth').replace('10+ authors', '10+ Auth')
        report += f"{cat_short[:12]:<14} "
    report += "\n" + "-" * 120 + "\n"
    
    for _, row in collab_dist_table.iterrows():
        report += f"{int(row['Year']):<6} {int(row['Total_Papers']):<8} {row['Mean_Authors']:<8.2f} {row['Median_Authors']:<8.0f} "
        for category in collab_categories:
            count = int(row[f'{category}_Count'])
            pct = row[f'{category}_Percent']
            report += f"{count:>3} ({pct:>5.1f}%)  "
        report += "\n"
    
    report += "\n"
    
    report += f"""
KEY FINDINGS
------------
"""
    
    # Test hypothesis: Applied more collaborative than taxonomic
    applied_more = applied_author_stats['mean'] > taxonomic_author_stats['mean']
    report += f"""
1. Applied vs Taxonomic Collaboration:
   Applied studies: {applied_author_stats['mean']:.2f} authors (mean)
   Taxonomic studies: {taxonomic_author_stats['mean']:.2f} authors (mean)
   Hypothesis {'SUPPORTED' if applied_more else 'NOT SUPPORTED'}: 
   Applied studies are {'more' if applied_more else 'less'} collaborative

2. Temporal Trend:
   Authorship has {'increased' if recent_author_stats['mean'] > early_author_stats['mean'] else 'decreased'} 
   from {early_author_stats['mean']:.2f} to {recent_author_stats['mean']:.2f} authors per paper

3. Collaboration Patterns:
   {collab_dist_props.get('3-5 authors', 0) + collab_dist_props.get('6-10 authors', 0) + collab_dist_props.get('10+ authors', 0):.1f}% 
   of papers have 3+ authors (multi-author collaboration)

LIMITATIONS
-----------
"""
    if has_full_author_data:
        report += """
- Author data from OpenAlex API (full author lists available)
- Author_Count_Actual provides accurate collaboration metrics
- International collaboration analysis may be limited by affiliation data completeness
- Study type classification based on Research_Theme field
"""
    else:
        report += """
- **CRITICAL**: Publish or Perish export only includes FIRST AUTHOR in Authors field
- AuthorCount field in export is unreliable (shows 1 for all papers)
- Collaboration analysis severely limited - cannot accurately determine multi-author papers
- **RECOMMENDATION**: Run fetch_authors.py to get full author data from OpenAlex API
- International collaboration analysis limited by country data availability
- Study type classification based on Research_Theme field
"""
    
    # Save report
    with open(OUTPUT_DIR / "rq4_collaboration_report.txt", 'w') as f:
        f.write(report)
    
    # Save detailed data
    yearly_author.to_csv(OUTPUT_DIR / "yearly_author_stats.csv")
    yearly_collab_props.to_csv(OUTPUT_DIR / "yearly_collaboration_proportions.csv")
    collab_by_type.to_csv(OUTPUT_DIR / "collaboration_by_study_type.csv")
    collab_dist_table.to_csv(OUTPUT_DIR / "collaboration_distribution_by_year.csv", index=False)
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"\nAnalysis complete! Files saved to {OUTPUT_DIR}/")
    print(f"  - rq4_collaboration_report.txt")
    print(f"  - collaboration_distribution_by_year.csv (main table)")
    print(f"  - yearly_author_stats.csv")
    print(f"  - yearly_collaboration_proportions.csv")
    print(f"  - collaboration_by_study_type.csv")


if __name__ == "__main__":
    analyze_collaboration()

