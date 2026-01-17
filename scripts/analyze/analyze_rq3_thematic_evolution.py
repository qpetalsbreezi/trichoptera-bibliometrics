"""
RQ3: Thematic Evolution Analysis

Research Question: How have research themes in Trichoptera studies evolved 
over time? What are the dominant themes and how have they shifted?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Get project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
INPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_coded.csv"
OUTPUT_DIR = PROJECT_ROOT / "analysis/rq3_thematic_evolution"

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

def analyze_thematic_evolution():
    """Analyze thematic evolution in Trichoptera research"""
    
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
    mid_period = df[df['Year'].between(2016, 2020)]
    recent_period = df[df['Year'].between(2021, 2025)]
    
    # Theme distribution by period
    def get_theme_distribution(period_df, period_name):
        theme_counts = period_df['Research_Theme'].value_counts()
        theme_props = (theme_counts / len(period_df) * 100).round(2)
        return theme_counts, theme_props
    
    early_counts, early_props = get_theme_distribution(early_period, "Early")
    mid_counts, mid_props = get_theme_distribution(mid_period, "Mid")
    recent_counts, recent_props = get_theme_distribution(recent_period, "Recent")
    
    # Year-by-year theme analysis
    yearly_theme = df.groupby(['Year', 'Research_Theme']).size().unstack(fill_value=0)
    yearly_theme_props = yearly_theme.div(yearly_theme.sum(axis=1), axis=0) * 100
    
    # Create comprehensive theme distribution table (similar to RQ2)
    theme_dist_table = pd.DataFrame()
    all_themes_list = sorted([t for t in yearly_theme.columns if t != 'Not Specified'])
    
    for year in sorted(yearly_theme.index):
        year_data = {
            'Year': year,
            'Total_Papers': int(yearly_theme.loc[year].sum())
        }
        for theme in all_themes_list:
            if theme in yearly_theme.columns:
                count = int(yearly_theme.loc[year, theme])
                prop = yearly_theme_props.loc[year, theme]
                year_data[f'{theme}_Count'] = count
                year_data[f'{theme}_Percent'] = prop
            else:
                year_data[f'{theme}_Count'] = 0
                year_data[f'{theme}_Percent'] = 0.0
        
        # Add "Not Specified" (Unknown) category
        if 'Not Specified' in yearly_theme.columns:
            not_spec_count = int(yearly_theme.loc[year, 'Not Specified'])
            not_spec_prop = yearly_theme_props.loc[year, 'Not Specified']
            year_data['Not Specified_Count'] = not_spec_count
            year_data['Not Specified_Percent'] = not_spec_prop
        else:
            year_data['Not Specified_Count'] = 0
            year_data['Not Specified_Percent'] = 0.0
            
        theme_dist_table = pd.concat([theme_dist_table, pd.DataFrame([year_data])], ignore_index=True)
    
    # Theme trends over time
    theme_trends = {}
    for theme in df['Research_Theme'].dropna().unique():
        if theme != 'Not Specified':
            theme_data = df[df['Research_Theme'] == theme]
            yearly_counts = theme_data.groupby('Year').size()
            theme_trends[theme] = yearly_counts
    
    # Top themes overall
    top_themes = df['Research_Theme'].value_counts().head(10)
    
    # Theme by region (to see if themes vary geographically)
    theme_by_region = df.groupby(['Region_Global', 'Research_Theme']).size().unstack(fill_value=0)
    theme_by_region_props = theme_by_region.div(theme_by_region.sum(axis=1), axis=0) * 100
    
    # Generate report
    report = f"""
THEMATIC EVOLUTION ANALYSIS (RQ3)
==================================

Research Question: How have research themes in Trichoptera studies evolved 
over time? What are the dominant themes and how have they shifted?

Date: {pd.Timestamp.now().strftime('%Y-%m-%d')}
Dataset: {len(df)} papers (2010-2025)

OVERALL THEME DISTRIBUTION
---------------------------
"""
    
    for theme, count in top_themes.items():
        prop = (count / len(df) * 100)
        report += f"  {theme}: {count} papers ({prop:.1f}%)\n"
    
    report += f"""
THEME DISTRIBUTION BY TIME PERIOD
----------------------------------
Early Period (2010-2015): {len(early_period)} papers
"""
    
    for theme in sorted(early_props.index):
        count = early_counts[theme]
        prop = early_props[theme]
        report += f"  {theme}: {count} papers ({prop:.1f}%)\n"
    
    report += f"""
Mid Period (2016-2020): {len(mid_period)} papers
"""
    
    for theme in sorted(mid_props.index):
        count = mid_counts[theme]
        prop = mid_props[theme]
        report += f"  {theme}: {count} papers ({prop:.1f}%)\n"
    
    report += f"""
Recent Period (2021-2025): {len(recent_period)} papers
"""
    
    for theme in sorted(recent_props.index):
        count = recent_counts[theme]
        prop = recent_props[theme]
        report += f"  {theme}: {count} papers ({prop:.1f}%)\n"
    
    # Calculate theme changes
    report += f"""
THEME SHIFTS (Early → Recent)
-----------------------------
"""
    
    all_themes = set(early_props.index) | set(recent_props.index)
    for theme in sorted(all_themes):
        if theme != 'Not Specified':
            early_pct = early_props.get(theme, 0)
            recent_pct = recent_props.get(theme, 0)
            change = recent_pct - early_pct
            change_str = f"+{change:.1f}%" if change > 0 else f"{change:.1f}%"
            trend = "↑" if change > 2 else "↓" if change < -2 else "→"
            report += f"  {theme}: {early_pct:.1f}% → {recent_pct:.1f}% ({change_str}) {trend}\n"
    
    report += f"""
YEAR-BY-YEAR THEME DISTRIBUTION TABLE
--------------------------------------
"""
    
    # Create formatted table (show top themes in columns)
    # Get top themes by average proportion across all years (exclude 'Not Specified')
    theme_avg_props = yearly_theme_props.drop(columns=['Not Specified'], errors='ignore').mean().sort_values(ascending=False)
    top_themes_for_table = theme_avg_props.head(5).index.tolist()  # Top 5 themes + Not Specified = 6 columns
    
    report += f"{'Year':<6} {'Total':<8} "
    for theme in top_themes_for_table:
        # Shorten theme names for table
        theme_short = theme.replace('Biomonitoring/Water Quality', 'Biomonitor').replace('Taxonomy/Systematics', 'Taxonomy').replace('Ecology/Behavior', 'Ecology').replace('Evolution/Phylogeny', 'Evolution').replace('Materials Science (Silk)', 'Silk')
        report += f"{theme_short[:15]:<17} "
    report += f"{'Unknown':<17} "  # Add Unknown column
    report += "\n" + "-" * 120 + "\n"
    
    for _, row in theme_dist_table.iterrows():
        report += f"{int(row['Year']):<6} {int(row['Total_Papers']):<8} "
        for theme in top_themes_for_table:
            count = int(row[f'{theme}_Count'])
            pct = row[f'{theme}_Percent']
            report += f"{count:>3} ({pct:>5.1f}%)  "
        # Add Not Specified (Unknown)
        not_spec_count = int(row['Not Specified_Count'])
        not_spec_pct = row['Not Specified_Percent']
        report += f"{not_spec_count:>3} ({not_spec_pct:>5.1f}%)  "
        report += "\n"
    
    report += "\n"
    
    report += f"""
THEME BY REGION
---------------
"""
    
    for region in theme_by_region_props.index:
        if region != 'Not Specified':
            report += f"\n{region}:\n"
            top_themes_region = theme_by_region_props.loc[region].sort_values(ascending=False).head(5)
            for theme, prop in top_themes_region.items():
                count = theme_by_region.loc[region, theme]
                report += f"  {theme}: {count} papers ({prop:.1f}%)\n"
    
    report += f"""
KEY FINDINGS
------------
"""
    
    # Identify emerging themes (increased significantly)
    emerging = []
    declining = []
    stable = []
    
    for theme in all_themes:
        if theme != 'Not Specified':
            early_pct = early_props.get(theme, 0)
            recent_pct = recent_props.get(theme, 0)
            change = recent_pct - early_pct
            if change > 3:
                emerging.append((theme, change))
            elif change < -3:
                declining.append((theme, change))
            else:
                stable.append((theme, change))
    
    if emerging:
        report += "\nEmerging Themes (increased >3%):\n"
        for theme, change in sorted(emerging, key=lambda x: x[1], reverse=True):
            report += f"  - {theme}: +{change:.1f}%\n"
    
    if declining:
        report += "\nDeclining Themes (decreased >3%):\n"
        for theme, change in sorted(declining, key=lambda x: x[1]):
            report += f"  - {theme}: {change:.1f}%\n"
    
    report += f"""
LIMITATIONS
-----------
- Theme classification based on LLM coding of abstracts
- Some papers may have multiple themes but only one is assigned
- "Not Specified" papers excluded from theme analysis

"""
    
    # Save report
    with open(OUTPUT_DIR / "rq3_thematic_evolution_report.txt", 'w') as f:
        f.write(report)
    
    # Save detailed data
    yearly_theme_props.to_csv(OUTPUT_DIR / "yearly_theme_proportions.csv")
    theme_by_region_props.to_csv(OUTPUT_DIR / "theme_by_region.csv")
    theme_dist_table.to_csv(OUTPUT_DIR / "theme_distribution_by_year.csv", index=False)
    
    # Create theme trends CSV
    theme_trends_df = pd.DataFrame(theme_trends).fillna(0)
    theme_trends_df.to_csv(OUTPUT_DIR / "theme_trends_by_year.csv")
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"\nAnalysis complete! Files saved to {OUTPUT_DIR}/")
    print(f"  - rq3_thematic_evolution_report.txt")
    print(f"  - theme_distribution_by_year.csv (main table)")
    print(f"  - yearly_theme_proportions.csv")
    print(f"  - theme_by_region.csv")
    print(f"  - theme_trends_by_year.csv")


if __name__ == "__main__":
    analyze_thematic_evolution()

