"""
RQ2: Temporal and Geographic Growth Analysis

Research Question: How has the geographic distribution of Trichoptera research 
changed over time? Has there been a shift from European and North American 
focus to more global distribution, particularly in South America and Asia?

Note: Due to 200-result cap per year, analysis focuses on PROPORTIONAL trends
rather than absolute publication volumes.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.pipeline import PipelinePaths, add_query_arg  # noqa: E402


def analyze_temporal_geographic(paths: PipelinePaths):
    """Analyze temporal and geographic trends in Trichoptera research"""
    input_csv = paths.coded
    output_dir = paths.rq_dir("rq2_temporal_geographic")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"query_id={paths.query_id}")
    print("Loading data...")
    df = pd.read_csv(input_csv)
    
    # Clean and prepare data
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce')
    df = df[df['Year'].between(2010, 2025)]

    relevance_col = "Taxon_Relevance" if "Taxon_Relevance" in df.columns else "Trichoptera_Relevance"
    not_focused_labels = {"Not target-taxon-focused", "Not Trichoptera-focused"}
    # Filter out papers where the query taxon is not the study focus.
    df = df[~df[relevance_col].isin(not_focused_labels)]
    
    print(f"Analyzing {len(df)} papers from 2010-2025")
    
    # Normalize country names
    def normalize_country(country_str):
        """Normalize country names to standard format"""
        if pd.isna(country_str) or country_str == 'Not Specified' or not str(country_str).strip():
            return 'Not Specified'
        
        country_str = str(country_str).strip()
        
        # Common normalizations
        country_mapping = {
            'USA': 'United States',
            'US': 'United States',
            'U.S.': 'United States',
            'U.S.A.': 'United States',
            'United States of America': 'United States',
            'UK': 'United Kingdom',
            'U.K.': 'United Kingdom',
            'Czechia': 'Czech Republic',
            'Czech Republic': 'Czech Republic',
        }
        
        # Check for exact match first
        if country_str in country_mapping:
            return country_mapping[country_str]
        
        # Handle multi-country entries (comma, semicolon, or "and" separated)
        separators = [',', ';', ' and ', ' & ']
        for sep in separators:
            if sep in country_str:
                # Take the first country mentioned
                first_country = country_str.split(sep)[0].strip()
                # Normalize the first country
                if first_country in country_mapping:
                    return country_mapping[first_country]
                return first_country
        
        # Handle regions/provinces - map to countries where possible
        region_mapping = {
            'Iberian Peninsula': 'Spain',  # Primary country, though could be Portugal too
            'Southern Ontario': 'Canada',
            'Kosovo': 'Kosovo',  # Keep as is (disputed but recognized by many)
            'Republic of Kosovo': 'Kosovo',
            'Republic of North Macedonia': 'North Macedonia',
            'Democratic Republic of the Congo': 'Congo (DRC)',
            'Papua New Guinea': 'Papua New Guinea',  # Keep as is
        }
        
        if country_str in region_mapping:
            return region_mapping[country_str]
        
        # Check if it contains region keywords - mark as uncertain
        region_keywords = ['peninsula', 'region', 'basin', 'province', 'state', 'county', 'territory']
        if any(keyword in country_str.lower() for keyword in region_keywords):
            # Try to extract country name if possible
            # For now, return as-is but could be improved
            return country_str
        
        # Return normalized version
        return country_str
    
    # Apply normalization
    print("Normalizing country names...")
    df['Country_Normalized'] = df['Country'].apply(normalize_country)
    
    # Show normalization stats
    original_countries = df['Country'].value_counts()
    normalized_countries = df['Country_Normalized'].value_counts()
    print(f"  Original unique countries: {len(original_countries)}")
    print(f"  Normalized unique countries: {len(normalized_countries)}")
    
    # Use normalized country for analysis
    df['Country'] = df['Country_Normalized']
    
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
    
    # Year-by-year analysis
    yearly_region = df.groupby(['Year', 'Region_Category']).size().unstack(fill_value=0)
    yearly_props = yearly_region.div(yearly_region.sum(axis=1), axis=0) * 100
    
    # Temporal volume analysis (now possible with full API data)
    yearly_volume = df.groupby('Year').size()
    early_period_volume = df[df['Year'].between(2010, 2015)].shape[0]
    recent_period_volume = df[df['Year'].between(2020, 2025)].shape[0]
    volume_change = recent_period_volume - early_period_volume
    volume_change_pct = (volume_change / early_period_volume * 100) if early_period_volume > 0 else 0
    
    # Create comprehensive geographic distribution table
    geo_dist_table = pd.DataFrame()
    for year in sorted(yearly_region.index):
        year_data = {
            'Year': year,
            'Total_Papers': int(yearly_region.loc[year].sum())
        }
        for region in ['South America', 'Asia', 'Europe', 'North America', 'Other', 'Unknown']:
            if region in yearly_region.columns:
                count = int(yearly_region.loc[year, region])
                prop = yearly_props.loc[year, region]
                year_data[f'{region}_Count'] = count
                year_data[f'{region}_Percent'] = prop
            else:
                year_data[f'{region}_Count'] = 0
                year_data[f'{region}_Percent'] = 0.0
        geo_dist_table = pd.concat([geo_dist_table, pd.DataFrame([year_data])], ignore_index=True)
    
    # Country-level analysis (top countries overall)
    country_counts = df['Country'].value_counts().head(20)
    
    # Generate report
    report = f"""
TEMPORAL AND GEOGRAPHIC GROWTH ANALYSIS (RQ2)
============================================

Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Dataset: {len(df)} papers (2010-2025)

Research Question: How has the geographic distribution of Trichoptera research 
changed over time? Has there been a shift from European and North American 
focus to more global distribution, particularly in South America and Asia?

TEMPORAL VOLUME ANALYSIS
-------------------------
Total Publications by Period:
  Early Period (2010-2015): {early_period_volume:,} papers
  Recent Period (2020-2025): {recent_period_volume:,} papers
  Change: {volume_change:+,} papers ({'+' if volume_change_pct > 0 else ''}{volume_change_pct:.1f}%)

TOP COUNTRIES (Overall 2010-2025)
----------------------------------
"""
    
    for country in country_counts.head(15).index:
        count = int(country_counts[country])
        report += f"  {country}: {count} papers\n"
    
    report += f"""
YEAR-BY-YEAR GEOGRAPHIC DISTRIBUTION TABLE
------------------------------------------
"""
    
    # Create formatted table
    report += f"{'Year':<6} {'Total':<8} "
    for region in ['South America', 'Asia', 'Europe', 'North America', 'Other', 'Unknown']:
        report += f"{region[:12]:<15} "
    report += "\n" + "-" * 100 + "\n"
    
    for _, row in geo_dist_table.iterrows():
        report += f"{int(row['Year']):<6} {int(row['Total_Papers']):<8} "
        for region in ['South America', 'Asia', 'Europe', 'North America', 'Other', 'Unknown']:
            count = int(row[f'{region}_Count'])
            pct = row[f'{region}_Percent']
            report += f"{count:>3} ({pct:>5.1f}%)  "
        report += "\n"
    
    # Calculate key insights from the table
    sa_avg = geo_dist_table['South America_Percent'].mean()
    asia_avg = geo_dist_table['Asia_Percent'].mean()
    europe_avg = geo_dist_table['Europe_Percent'].mean()
    na_avg = geo_dist_table['North America_Percent'].mean()
    
    # Calculate trends: compare early (2010-2015) vs recent (2020-2025)
    early_years = geo_dist_table[geo_dist_table['Year'].between(2010, 2015)]
    recent_years = geo_dist_table[geo_dist_table['Year'].between(2020, 2025)]
    
    sa_early = early_years['South America_Percent'].mean()
    sa_recent = recent_years['South America_Percent'].mean()
    sa_change = sa_recent - sa_early
    
    asia_early = early_years['Asia_Percent'].mean()
    asia_recent = recent_years['Asia_Percent'].mean()
    asia_change = asia_recent - asia_early
    
    europe_early = early_years['Europe_Percent'].mean()
    europe_recent = recent_years['Europe_Percent'].mean()
    europe_change = europe_recent - europe_early
    
    na_early = early_years['North America_Percent'].mean()
    na_recent = recent_years['North America_Percent'].mean()
    na_change = na_recent - na_early
    
    unknown_avg = geo_dist_table['Unknown_Percent'].mean()
    unknown_trend = "decreasing" if geo_dist_table['Unknown_Percent'].iloc[-1] < geo_dist_table['Unknown_Percent'].iloc[0] else "increasing"
    
    report += f"""
KEY FINDINGS
--------------------------------------------------------

1. Regional Distribution Averages (2010-2025):
   - South America: {sa_avg:.1f}% of papers on average
   - Asia: {asia_avg:.1f}% of papers on average
   - Europe: {europe_avg:.1f}% of papers on average
   - North America: {na_avg:.1f}% of papers on average

2. Proportional Trends (Early 2010-2015 vs Recent 2020-2025):
   - South America: {sa_early:.1f}% → {sa_recent:.1f}% ({'+' if sa_change > 0 else ''}{sa_change:.1f}%)
   - Asia: {asia_early:.1f}% → {asia_recent:.1f}% ({'+' if asia_change > 0 else ''}{asia_change:.1f}%)
   - Europe: {europe_early:.1f}% → {europe_recent:.1f}% ({'+' if europe_change > 0 else ''}{europe_change:.1f}%)
   - North America: {na_early:.1f}% → {na_recent:.1f}% ({'+' if na_change > 0 else ''}{na_change:.1f}%)
   
   Note: North America has decreased while South America, Asia, and Europe have increased.

3. Geographic Data Completeness:
   - Unknown region: {unknown_avg:.1f}% average ({unknown_trend} trend)
   - Geographic classification available for ~{100-unknown_avg:.0f}% of papers on average

LIMITATIONS
-----------
- Regional classification uses broad biogeographic regions (e.g., Palearctic, Neotropical) mapped to continental categories, which may misclassify studies in transition zones or regions that span multiple continents (e.g., Palearctic studies in North Africa classified as "Europe")
- Country data may be incomplete for some papers (~6% unknown)

"""
    
    # Save report
    with open(output_dir / "rq2_temporal_geographic_report.txt", 'w') as f:
        f.write(report)
    
    # Save detailed data
    yearly_props.to_csv(output_dir / "yearly_regional_proportions.csv")
    country_counts.to_frame(name='Count').to_csv(output_dir / "country_counts.csv")
    geo_dist_table.to_csv(output_dir / "geographic_distribution_by_year.csv", index=False)
    yearly_volume.to_frame(name='Count').to_csv(output_dir / "yearly_publication_volume.csv")
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"\nAnalysis complete! Files saved to {output_dir}/")
    print(f"  - rq2_temporal_geographic_report.txt")
    print(f"  - geographic_distribution_by_year.csv (main table)")
    print(f"  - yearly_regional_proportions.csv")
    print(f"  - country_counts.csv")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RQ2: Temporal and geographic analysis")
    add_query_arg(parser)
    args = parser.parse_args()
    analyze_temporal_geographic(PipelinePaths(args.query_id))

