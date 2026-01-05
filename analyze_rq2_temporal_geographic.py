"""
RQ2: Temporal and Geographic Growth Analysis

Research Question: How has the geographic distribution of Trichoptera research 
changed over time? Has there been a shift from European and North American 
focus to more global distribution, particularly in South America and Asia?

Note: Due to 200-result cap per year, analysis focuses on PROPORTIONAL trends
rather than absolute publication volumes.
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
INPUT_CSV = "data/processed/trichoptera_scopus_coded.csv"
OUTPUT_DIR = "analysis/rq2_temporal_geographic"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

def analyze_temporal_geographic():
    """Analyze temporal and geographic trends in Trichoptera research"""
    
    # Load data
    print("Loading data...")
    df = pd.read_csv(INPUT_CSV)
    
    # Clean and prepare data
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[df['Year'].between(2010, 2025)]
    
    # Filter out papers with "Not Trichoptera-focused"
    df = df[df['Trichoptera_Relevance'] != 'Not Trichoptera-focused']
    
    print(f"Analyzing {len(df)} papers from 2010-2025")
    
    # Define time periods
    early_period = df[df['Year'].between(2010, 2015)]
    recent_period = df[df['Year'].between(2020, 2025)]
    
    # Define regions for analysis
    regions_of_interest = {
        'South America': ['Neotropical'],
        'Asia': ['Oriental', 'East Palearctic'],
        'Europe': ['Palearctic'],
        'North America': ['Nearctic'],
        'Other': ['Afrotropical', 'Australasian', 'Global']
    }
    
    # Helper function to categorize regions
    def categorize_region(region):
        if pd.isna(region) or region == 'Not Specified':
            return 'Unknown'
        for category, reg_list in regions_of_interest.items():
            if region in reg_list:
                return category
        return 'Other'
    
    df['Region_Category'] = df['Region_Global'].apply(categorize_region)
    early_period['Region_Category'] = early_period['Region_Global'].apply(categorize_region)
    recent_period['Region_Category'] = recent_period['Region_Global'].apply(categorize_region)
    
    # Calculate proportions by time period
    early_counts = early_period['Region_Category'].value_counts()
    recent_counts = recent_period['Region_Category'].value_counts()
    
    early_props = (early_counts / len(early_period) * 100).round(2)
    recent_props = (recent_counts / len(recent_period) * 100).round(2)
    
    # Year-by-year analysis
    yearly_region = df.groupby(['Year', 'Region_Category']).size().unstack(fill_value=0)
    yearly_props = yearly_region.div(yearly_region.sum(axis=1), axis=0) * 100
    
    # Country-level analysis (top countries)
    country_counts = df['Country'].value_counts().head(20)
    country_by_period = pd.DataFrame({
        'Early (2010-2015)': early_period['Country'].value_counts(),
        'Recent (2020-2025)': recent_period['Country'].value_counts()
    }).fillna(0)
    country_by_period['Change'] = country_by_period['Recent (2020-2025)'] - country_by_period['Early (2010-2015)']
    country_by_period = country_by_period.sort_values('Recent (2020-2025)', ascending=False).head(15)
    
    # Generate report
    report = f"""
TEMPORAL AND GEOGRAPHIC GROWTH ANALYSIS (RQ2)
============================================

Research Question: How has the geographic distribution of Trichoptera research 
changed over time? Has there been a shift from European and North American 
focus to more global distribution, particularly in South America and Asia?

Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Dataset: {len(df)} papers (2010-2025)
Note: Analysis uses PROPORTIONAL trends due to 200-result cap per year

TIME PERIOD COMPARISON
-----------------------
Early Period (2010-2015): {len(early_period)} papers
Recent Period (2020-2025): {len(recent_period)} papers

REGIONAL DISTRIBUTION - EARLY PERIOD (2010-2015)
-------------------------------------------------
"""
    
    for region in ['South America', 'Asia', 'Europe', 'North America', 'Other', 'Unknown']:
        if region in early_props.index:
            count = early_counts[region]
            prop = early_props[region]
            report += f"  {region}: {count} papers ({prop}%)\n"
    
    report += f"""
REGIONAL DISTRIBUTION - RECENT PERIOD (2020-2025)
--------------------------------------------------
"""
    
    for region in ['South America', 'Asia', 'Europe', 'North America', 'Other', 'Unknown']:
        if region in recent_props.index:
            count = recent_counts[region]
            prop = recent_props[region]
            report += f"  {region}: {count} papers ({prop}%)\n"
    
    # Calculate changes
    report += f"""
REGIONAL SHIFTS (Early vs Recent)
----------------------------------
"""
    
    for region in ['South America', 'Asia', 'Europe', 'North America']:
        early_pct = early_props.get(region, 0)
        recent_pct = recent_props.get(region, 0)
        change = recent_pct - early_pct
        change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
        report += f"  {region}: {early_pct:.1f}% → {recent_pct:.1f}% ({change_str})\n"
    
    report += f"""
TOP COUNTRIES - COMPARISON
---------------------------
"""
    
    for country in country_by_period.index:
        early = int(country_by_period.loc[country, 'Early (2010-2015)'])
        recent = int(country_by_period.loc[country, 'Recent (2020-2025)'])
        change = int(country_by_period.loc[country, 'Change'])
        change_str = f"+{change}" if change > 0 else str(change)
        report += f"  {country}: {early} → {recent} papers ({change_str})\n"
    
    report += f"""
YEAR-BY-YEAR REGIONAL TRENDS
-----------------------------
"""
    
    for year in sorted(yearly_props.index):
        report += f"\n{year}:\n"
        for region in ['South America', 'Asia', 'Europe', 'North America']:
            if region in yearly_props.columns:
                prop = yearly_props.loc[year, region]
                count = yearly_region.loc[year, region]
                report += f"  {region}: {count} papers ({prop:.1f}%)\n"
    
    report += f"""
KEY FINDINGS
------------
"""
    
    # Test hypothesis: South America and Asia increase
    sa_early = early_props.get('South America', 0)
    sa_recent = recent_props.get('South America', 0)
    asia_early = early_props.get('Asia', 0)
    asia_recent = recent_props.get('Asia', 0)
    europe_early = early_props.get('Europe', 0)
    europe_recent = recent_props.get('Europe', 0)
    na_early = early_props.get('North America', 0)
    na_recent = recent_props.get('North America', 0)
    
    report += f"""
1. South America: {'INCREASED' if sa_recent > sa_early else 'DECREASED'} 
   ({sa_early:.1f}% → {sa_recent:.1f}%)

2. Asia: {'INCREASED' if asia_recent > asia_early else 'DECREASED'}
   ({asia_early:.1f}% → {asia_recent:.1f}%)

3. Europe: {'DECREASED' if europe_recent < europe_early else 'INCREASED'}
   ({europe_early:.1f}% → {europe_recent:.1f}%)

4. North America: {'DECREASED' if na_recent < na_early else 'INCREASED'}
   ({na_early:.1f}% → {na_recent:.1f}%)

HYPOTHESIS TEST
---------------
Hypothesis: Shift from Europe/North America to South America/Asia

Result: {'SUPPORTED' if (sa_recent > sa_early or asia_recent > asia_early) and (europe_recent < europe_early or na_recent < na_early) else 'PARTIALLY SUPPORTED' if (sa_recent > sa_early or asia_recent > asia_early) or (europe_recent < europe_early or na_recent < na_early) else 'NOT CLEARLY SUPPORTED'}

LIMITATIONS
-----------
- Publication volume analysis limited by 200-result cap per year
- Analysis focuses on proportional trends rather than absolute counts
- Regional classification based on Region_Global field (may miss some nuances)
- Country data may be incomplete for some papers

"""
    
    # Save report
    with open(f"{OUTPUT_DIR}/rq2_temporal_geographic_report.txt", 'w') as f:
        f.write(report)
    
    # Save detailed data
    yearly_props.to_csv(f"{OUTPUT_DIR}/yearly_regional_proportions.csv")
    country_by_period.to_csv(f"{OUTPUT_DIR}/country_comparison.csv")
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"\nAnalysis complete! Files saved to {OUTPUT_DIR}/")
    print(f"  - rq2_temporal_geographic_report.txt")
    print(f"  - yearly_regional_proportions.csv")
    print(f"  - country_comparison.csv")


if __name__ == "__main__":
    analyze_temporal_geographic()

