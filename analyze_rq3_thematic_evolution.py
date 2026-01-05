"""
RQ3: Thematic Evolution Analysis

Research Question: How have research themes in Trichoptera studies evolved 
over time? What are the dominant themes and how have they shifted?
"""

import pandas as pd
import numpy as np
from pathlib import Path

# Configuration
INPUT_CSV = "data/processed/trichoptera_scopus_coded.csv"
OUTPUT_DIR = "analysis/rq3_thematic_evolution"

# Create output directory
Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

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
YEAR-BY-YEAR THEME TRENDS
--------------------------
"""
    
    # Show top 5 themes per year
    for year in sorted(yearly_theme_props.index):
        year_themes = yearly_theme_props.loc[year].sort_values(ascending=False).head(5)
        report += f"\n{year}:\n"
        for theme, prop in year_themes.items():
            count = yearly_theme.loc[year, theme]
            report += f"  {theme}: {count} papers ({prop:.1f}%)\n"
    
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
    with open(f"{OUTPUT_DIR}/rq3_thematic_evolution_report.txt", 'w') as f:
        f.write(report)
    
    # Save detailed data
    yearly_theme_props.to_csv(f"{OUTPUT_DIR}/yearly_theme_proportions.csv")
    theme_by_region_props.to_csv(f"{OUTPUT_DIR}/theme_by_region.csv")
    
    # Create theme trends CSV
    theme_trends_df = pd.DataFrame(theme_trends).fillna(0)
    theme_trends_df.to_csv(f"{OUTPUT_DIR}/theme_trends_by_year.csv")
    
    print("\n" + "="*60)
    print(report)
    print("="*60)
    print(f"\nAnalysis complete! Files saved to {OUTPUT_DIR}/")
    print(f"  - rq3_thematic_evolution_report.txt")
    print(f"  - yearly_theme_proportions.csv")
    print(f"  - theme_by_region.csv")
    print(f"  - theme_trends_by_year.csv")


if __name__ == "__main__":
    analyze_thematic_evolution()

