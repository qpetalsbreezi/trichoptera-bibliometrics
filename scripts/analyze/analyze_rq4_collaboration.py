"""
RQ4: Collaboration and Authorship Patterns Analysis

Research Question: How have collaboration patterns changed? Are applied studies 
more collaborative than taxonomic studies?
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.pipeline import PipelinePaths, add_query_arg, normalize_research_theme  # noqa: E402


def analyze_collaboration(paths: PipelinePaths):
    """Analyze collaboration and authorship patterns"""
    authors_csv = paths.with_authors
    input_csv = paths.coded
    output_dir = paths.rq_dir("rq4_collaboration")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"query_id={paths.query_id}")
    print(f"Loading coded data from: {input_csv}")
    df = pd.read_csv(input_csv)
    has_full_author_data = False

    if authors_csv.exists():
        print(f"Merging with author data from: {authors_csv}")
        authors_df = pd.read_csv(authors_csv)
        # Merge on a unique identifier (DOI or Title+Year)
        merge_cols = ['DOI'] if 'DOI' in df.columns and 'DOI' in authors_df.columns else ['Title', 'Year']
        author_cols = ['Author_Count_Actual', 'Author_Affiliations', 'All_Authors']
        if 'Author_Country_Codes' in authors_df.columns:
            author_cols.append('Author_Country_Codes')
        df = df.merge(
            authors_df[merge_cols + author_cols],
            on=merge_cols,
            how='left',
            suffixes=('', '_author')
        )
        has_full_author_data = 'Author_Count_Actual' in df.columns
        if has_full_author_data:
            print("Successfully merged author data")
    else:
        print("Note: Full author data not available. Using limited data from coded file.")
    
    # Clean and prepare data
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[df['Year'].between(2010, 2025)]

    relevance_col = "Taxon_Relevance" if "Taxon_Relevance" in df.columns else "Trichoptera_Relevance"
    not_focused_labels = {"Not target-taxon-focused", "Not Trichoptera-focused"}
    # Filter out papers where the query taxon is not the study focus.
    df = df[~df[relevance_col].isin(not_focused_labels)]
    if "Research_Theme" in df.columns:
        df["Research_Theme"] = df["Research_Theme"].map(normalize_research_theme)
    
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
    applied_themes = ['Biomonitoring/Water Quality', 'Applied Ecology', 'Conservation']
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
    yearly_author = df.groupby('Year')['AuthorCount'].agg(['mean', 'median', 'min', 'max', 'count'])
    
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
    
    # Split data into taxonomic and non-taxonomic for separate tables
    taxonomic_df = df[df['Study_Type'] == 'Taxonomic']
    non_taxonomic_df = df[df['Study_Type'] != 'Taxonomic']
    
    # Year-by-year stats for taxonomic studies
    yearly_author_taxonomic = taxonomic_df.groupby('Year')['AuthorCount'].agg(['mean', 'median', 'min', 'max', 'count'])
    yearly_collab_taxonomic = taxonomic_df.groupby(['Year', 'Collaboration_Category']).size().unstack(fill_value=0)
    yearly_collab_props_taxonomic = yearly_collab_taxonomic.div(yearly_collab_taxonomic.sum(axis=1), axis=0) * 100
    
    # Year-by-year stats for non-taxonomic studies
    yearly_author_non_taxonomic = non_taxonomic_df.groupby('Year')['AuthorCount'].agg(['mean', 'median', 'min', 'max', 'count'])
    yearly_collab_non_taxonomic = non_taxonomic_df.groupby(['Year', 'Collaboration_Category']).size().unstack(fill_value=0)
    yearly_collab_props_non_taxonomic = yearly_collab_non_taxonomic.div(yearly_collab_non_taxonomic.sum(axis=1), axis=0) * 100
    
    # Helper function to create collaboration distribution table
    def create_collab_table(yearly_author_data, yearly_collab_data, yearly_collab_props_data):
        collab_dist_table = pd.DataFrame()
        collab_categories = ['Single author', '2 authors', '3-5 authors', '6-10 authors', '10+ authors']
        
        all_years = sorted(set(yearly_author_data.index) | set(yearly_collab_data.index))
        for year in all_years:
            if year in yearly_author_data.index:
                year_data = {
                    'Year': year,
                    'Total_Papers': int(yearly_author_data.loc[year, 'count']),
                    'Mean_Authors': yearly_author_data.loc[year, 'mean'],
                    'Median_Authors': yearly_author_data.loc[year, 'median'],
                    'Min_Authors': int(yearly_author_data.loc[year, 'min']),
                    'Max_Authors': int(yearly_author_data.loc[year, 'max'])
                }
            else:
                year_data = {
                    'Year': year,
                    'Total_Papers': 0,
                    'Mean_Authors': 0.0,
                    'Median_Authors': 0.0,
                    'Min_Authors': 0,
                    'Max_Authors': 0
                }
            
            for category in collab_categories:
                if year in yearly_collab_data.index and category in yearly_collab_data.columns:
                    count = int(yearly_collab_data.loc[year, category])
                    prop = yearly_collab_props_data.loc[year, category] if year in yearly_collab_props_data.index and category in yearly_collab_props_data.columns else 0.0
                    year_data[f'{category}_Count'] = count
                    year_data[f'{category}_Percent'] = prop
                else:
                    year_data[f'{category}_Count'] = 0
                    year_data[f'{category}_Percent'] = 0.0
            collab_dist_table = pd.concat([collab_dist_table, pd.DataFrame([year_data])], ignore_index=True)
        
        return collab_dist_table
    
    # Create separate tables for taxonomic and non-taxonomic
    collab_dist_table_taxonomic = create_collab_table(yearly_author_taxonomic, yearly_collab_taxonomic, yearly_collab_props_taxonomic)
    collab_dist_table_non_taxonomic = create_collab_table(yearly_author_non_taxonomic, yearly_collab_non_taxonomic, yearly_collab_props_non_taxonomic)
    
    # Also keep overall table for backward compatibility
    collab_dist_table = pd.DataFrame()
    collab_categories = ['Single author', '2 authors', '3-5 authors', '6-10 authors', '10+ authors']
    
    for year in sorted(yearly_author.index):
        year_data = {
            'Year': year,
            'Total_Papers': int(yearly_author.loc[year, 'count']),
            'Mean_Authors': yearly_author.loc[year, 'mean'],
            'Median_Authors': yearly_author.loc[year, 'median'],
            'Min_Authors': int(yearly_author.loc[year, 'min']),
            'Max_Authors': int(yearly_author.loc[year, 'max'])
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
    
    # International collaboration analysis
    # Attempt to detect papers with authors from multiple countries
    # This requires paper-level multi-country data, which may be limited
    
    has_affiliations = has_full_author_data and 'Author_Affiliations' in df.columns
    
    def detect_international_collab(row):
        """Detect if paper likely has international collaboration"""
        codes_raw = str(row.get('Author_Country_Codes', '') or '').strip()
        if codes_raw and codes_raw.lower() != 'nan':
            tokens = [t.strip().upper() for t in codes_raw.replace('|', ';').replace(',', ';').split(';')]
            codes = {t for t in tokens if len(t) == 2 and t.isalpha()}
            if len(codes) > 1:
                return 'International'
            elif len(codes) == 1:
                return 'National'
            else:
                return 'Unknown'

        # Method 1: Check if Author_Affiliations contains multiple country indicators
        if has_affiliations:
            affiliations = str(row.get('Author_Affiliations', ''))
            if pd.isna(affiliations) or not affiliations or affiliations == 'nan':
                return 'Unknown'
            
            # Extract potential countries from affiliation strings
            # This is a heuristic approach - look for country names in affiliations
            country_keywords = {
                'USA': ['United States', 'USA', 'US', 'America'],
                'UK': ['United Kingdom', 'UK', 'England', 'Scotland', 'Wales'],
                'Germany': ['Germany', 'Deutschland'],
                'France': ['France', 'Français'],
                'Brazil': ['Brazil', 'Brasil'],
                'China': ['China', 'Chinese'],
                'Japan': ['Japan', 'Japanese'],
                'Australia': ['Australia', 'Australian'],
                'Canada': ['Canada', 'Canadian'],
                'Italy': ['Italy', 'Italian'],
                'Spain': ['Spain', 'Spanish', 'España'],
            }
            
            countries_found = set()
            affiliations_lower = affiliations.lower()
            for country, keywords in country_keywords.items():
                if any(kw.lower() in affiliations_lower for kw in keywords):
                    countries_found.add(country)
            
            if len(countries_found) > 1:
                return 'International'
            elif len(countries_found) == 1:
                return 'National'
            else:
                return 'Unknown'
        
        # Method 2: If we have multiple regions mentioned (less reliable)
        # This is a fallback - not ideal but better than nothing
        region = str(row.get('Region_Global', ''))
        if region and region != 'Not Specified' and region != 'Global':
            # Single region suggests national collaboration
            return 'National'
        elif region == 'Global':
            return 'International'
        else:
            return 'Unknown'
    
    df['Collaboration_Type'] = df.apply(detect_international_collab, axis=1)
    
    # International collaboration statistics
    intl_collab_dist = df['Collaboration_Type'].value_counts()
    intl_collab_props = (intl_collab_dist / len(df) * 100).round(2)
    
    # International collaboration by study type
    intl_by_type = pd.crosstab(df['Study_Type'], df['Collaboration_Type'], normalize='index') * 100
    
    # International collaboration over time
    yearly_intl = df.groupby(['Year', 'Collaboration_Type']).size().unstack(fill_value=0)
    yearly_intl_props = yearly_intl.div(yearly_intl.sum(axis=1), axis=0) * 100
    
    # Generate report
    report = f"""
COLLABORATION AND AUTHORSHIP PATTERNS ANALYSIS (RQ4)
=====================================================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Dataset: {len(df)} papers (2010-2025)

Research Question: How have collaboration patterns changed? Are applied studies 
more collaborative than taxonomic studies?

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
    
    # Helper function to format table
    def format_collab_table(collab_table, title):
        formatted = f"\n{title}\n"
        formatted += "-" * len(title) + "\n"
        formatted += f"{'Year':<6} {'Total':<8} {'Mean':<8} {'Median':<8} {'Min':<6} {'Max':<6} "
        for category in collab_categories:
            cat_short = category.replace('Single author', 'Single').replace('2 authors', '2 Auth').replace('3-5 authors', '3-5 Auth').replace('6-10 authors', '6-10 Auth').replace('10+ authors', '10+ Auth')
            formatted += f"{cat_short[:12]:<14} "
        formatted += "\n" + "-" * 120 + "\n"
        
        for _, row in collab_table.iterrows():
            formatted += f"{int(row['Year']):<6} {int(row['Total_Papers']):<8} {row['Mean_Authors']:<8.2f} {row['Median_Authors']:<8.0f} {int(row['Min_Authors']):<6} {int(row['Max_Authors']):<6} "
            for category in collab_categories:
                count = int(row[f'{category}_Count'])
                pct = row[f'{category}_Percent']
                formatted += f"{count:>3} ({pct:>5.1f}%)  "
            formatted += "\n"
        
        formatted += "\n"
        return formatted
    
    # Add tables for taxonomic and non-taxonomic
    report += format_collab_table(collab_dist_table_taxonomic, "Taxonomic Studies")
    report += format_collab_table(collab_dist_table_non_taxonomic, "Non-Taxonomic Studies (Applied + Other)")
    
    report += f"""
INTERNATIONAL COLLABORATION ANALYSIS
-------------------------------------
Overall Distribution:
"""
    
    for collab_type in ['International', 'National', 'Unknown']:
        if collab_type in intl_collab_dist.index:
            count = intl_collab_dist[collab_type]
            prop = intl_collab_props[collab_type]
            report += f"  {collab_type}: {count} papers ({prop:.1f}%)\n"
    
    report += f"""
International Collaboration by Study Type:
"""
    
    for study_type in ['Applied', 'Taxonomic', 'Other']:
        if study_type in intl_by_type.index:
            intl_pct = intl_by_type.loc[study_type, 'International'] if 'International' in intl_by_type.columns else 0
            national_pct = intl_by_type.loc[study_type, 'National'] if 'National' in intl_by_type.columns else 0
            report += f"  {study_type} Studies:\n"
            report += f"    International: {intl_pct:.1f}%\n"
            report += f"    National: {national_pct:.1f}%\n"
    
    # Compare applied vs taxonomic international collaboration
    applied_intl = intl_by_type.loc['Applied', 'International'] if 'Applied' in intl_by_type.index and 'International' in intl_by_type.columns else 0
    taxonomic_intl = intl_by_type.loc['Taxonomic', 'International'] if 'Taxonomic' in intl_by_type.index and 'International' in intl_by_type.columns else 0
    intl_diff = applied_intl - taxonomic_intl
    
    report += f"""
KEY FINDINGS
------------
"""
    
    # Test hypothesis: Applied more collaborative than taxonomic (authorship)
    applied_more = applied_author_stats['mean'] > taxonomic_author_stats['mean']
    report += f"""
1. Applied vs Taxonomic Collaboration (Authorship):
   Applied studies: {applied_author_stats['mean']:.2f} authors (mean)
   Taxonomic studies: {taxonomic_author_stats['mean']:.2f} authors (mean)
   Hypothesis {'SUPPORTED' if applied_more else 'NOT SUPPORTED'}: 
   Applied studies have {'more' if applied_more else 'fewer'} authors on average

2. Applied vs Taxonomic International Collaboration:
   Applied studies: {applied_intl:.1f}% international collaboration
   Taxonomic studies: {taxonomic_intl:.1f}% international collaboration
   Difference: {intl_diff:+.1f} percentage points
   Hypothesis {'SUPPORTED' if intl_diff > 5 else 'PARTIALLY SUPPORTED' if intl_diff > 0 else 'NOT SUPPORTED'}: 
   Applied studies show {'higher' if intl_diff > 0 else 'similar' if abs(intl_diff) < 5 else 'lower'} 
   international collaboration rates

3. Temporal Trend (Authorship):
   Authorship has {'increased' if recent_author_stats['mean'] > early_author_stats['mean'] else 'decreased'} 
   from {early_author_stats['mean']:.2f} to {recent_author_stats['mean']:.2f} authors per paper

4. Collaboration Patterns:
   {collab_dist_props.get('3-5 authors', 0) + collab_dist_props.get('6-10 authors', 0) + collab_dist_props.get('10+ authors', 0):.1f}% 
   of papers have 3+ authors (multi-author collaboration)

LIMITATIONS
-----------
"""
    if has_full_author_data:
        report += """
- International collaboration detection uses heuristic approach (country keywords in affiliations) and may miss collaborations if country names are not clearly present in affiliations
- International collaboration analysis is approximate - 79.2% of papers have unknown collaboration status due to missing or unclear affiliation data
- Study type classification based on Research_Theme field (LLM-coded, may have classification errors)
"""
    else:
        report += """
- **CRITICAL**: Publish or Perish export only includes FIRST AUTHOR in Authors field
- AuthorCount field in export is unreliable (shows 1 for all papers)
- Collaboration analysis severely limited - cannot accurately determine multi-author papers
- **RECOMMENDATION**: Run fetch_authors.py to get full author data from OpenAlex API
- International collaboration analysis limited - uses Region_Global as proxy (less reliable)
- Study type classification based on Research_Theme field
"""
    
    # Save report
    with open(output_dir / "rq4_collaboration_report.txt", 'w') as f:
        f.write(report)
    
    # Save detailed data
    yearly_author.to_csv(output_dir / "yearly_author_stats.csv")
    yearly_collab_props.to_csv(output_dir / "yearly_collaboration_proportions.csv")
    collab_by_type.to_csv(output_dir / "collaboration_by_study_type.csv")
    collab_dist_table.to_csv(output_dir / "collaboration_distribution_by_year.csv", index=False)
    intl_by_type.to_csv(output_dir / "international_collaboration_by_study_type.csv")
    yearly_intl_props.to_csv(output_dir / "yearly_international_collaboration.csv")
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"\nAnalysis complete! Files saved to {output_dir}/")
    print(f"  - rq4_collaboration_report.txt")
    print(f"  - collaboration_distribution_by_year.csv (main table)")
    print(f"  - yearly_author_stats.csv")
    print(f"  - yearly_collaboration_proportions.csv")
    print(f"  - collaboration_by_study_type.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ4: Collaboration analysis")
    add_query_arg(parser)
    args = parser.parse_args()
    analyze_collaboration(PipelinePaths(args.query_id))

