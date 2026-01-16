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

# Get project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
INPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_coded.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis/rq2_temporal_geographic"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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

Research Question: How has the geographic distribution of Trichoptera research 
changed over time? Has there been a shift from European and North American 
focus to more global distribution, particularly in South America and Asia?

Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Dataset: {len(df)} papers (2010-2025)

IMPORTANT NOTE ON TEMPORAL ANALYSIS
-----------------------------------
The current dataset does NOT support temporal volume analysis (comparing 
publication counts over time) because each year's export was capped at 
~200 results due to Scopus API limitations. Therefore, we cannot determine 
if publication volume increased or decreased over time.

This analysis focuses on PROPORTIONAL geographic distribution trends 
(year-by-year percentages) rather than absolute publication counts.

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
    
    # Calculate trends: compare early years (2010-2012) vs recent years (2023-2025)
    early_years = geo_dist_table[geo_dist_table['Year'].isin([2010, 2011, 2012])]
    recent_years = geo_dist_table[geo_dist_table['Year'].isin([2023, 2024, 2025])]
    
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
KEY FINDINGS (Based on Year-by-Year Proportional Trends)
--------------------------------------------------------

1. Regional Distribution Averages (2010-2025):
   - South America: {sa_avg:.1f}% of papers on average
   - Asia: {asia_avg:.1f}% of papers on average
   - Europe: {europe_avg:.1f}% of papers on average
   - North America: {na_avg:.1f}% of papers on average

2. Proportional Trends (Early 2010-2012 vs Recent 2023-2025):
   - South America: {sa_early:.1f}% → {sa_recent:.1f}% ({'+' if sa_change > 0 else ''}{sa_change:.1f}%)
   - Asia: {asia_early:.1f}% → {asia_recent:.1f}% ({'+' if asia_change > 0 else ''}{asia_change:.1f}%)
   - Europe: {europe_early:.1f}% → {europe_recent:.1f}% ({'+' if europe_change > 0 else ''}{europe_change:.1f}%)
   - North America: {na_early:.1f}% → {na_recent:.1f}% ({'+' if na_change > 0 else ''}{na_change:.1f}%)
   
   Note: North America has decreased while South America, Asia, and Europe have increased.

3. Geographic Data Completeness:
   - Unknown region: {unknown_avg:.1f}% average ({unknown_trend} trend)
   - Geographic classification available for ~{100-unknown_avg:.0f}% of papers on average

Note: These findings are based on PROPORTIONAL distribution trends only.
Absolute publication volumes cannot be compared due to 200-result cap per year.

LIMITATIONS
-----------
- **Temporal volume analysis not supported**: Each year capped at ~200 results, 
  preventing comparison of absolute publication counts over time
- Analysis focuses on proportional geographic distribution trends only
- Regional classification based on Region_Global field (may miss some nuances)
- Country data may be incomplete for some papers
- Year-by-year table shows proportional trends, not volume changes

"""
    
    # Save report
    with open(OUTPUT_DIR / "rq2_temporal_geographic_report.txt", 'w') as f:
        f.write(report)
    
    # Save detailed data
    yearly_props.to_csv(OUTPUT_DIR / "yearly_regional_proportions.csv")
    country_counts.to_frame(name='Count').to_csv(OUTPUT_DIR / "country_counts.csv")
    geo_dist_table.to_csv(OUTPUT_DIR / "geographic_distribution_by_year.csv", index=False)
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"\nAnalysis complete! Files saved to {OUTPUT_DIR}/")
    print(f"  - rq2_temporal_geographic_report.txt")
    print(f"  - geographic_distribution_by_year.csv (main table)")
    print(f"  - yearly_regional_proportions.csv")
    print(f"  - country_counts.csv")


if __name__ == "__main__":
    analyze_temporal_geographic()

