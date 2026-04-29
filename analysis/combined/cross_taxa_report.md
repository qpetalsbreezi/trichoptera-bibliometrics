# Cross-taxon bibliometric summary (side-by-side)

Generated from `data/processed/*/scopus_api_coded.csv` with the same filters as RQ2–RQ4:
- Years: 2010–2025
- Exclude non–taxon-focused papers (`Taxon_Relevance` not in {Not target-taxon-focused, Not Trichoptera-focused})

### Sample size (2010–2025)

| query_id | All coded (2010–2025) | Taxon-focused (2010–2025) |
| --- | --- | --- |
| ephemeroptera | 4062 | 1486 |
| mosquitoes | 51990 | 22664 |
| odonata | 9203 | 4079 |
| plecoptera | 2272 | 1057 |
| trichoptera | 3456 | 2870 |

### Temporal volume (2010–2015 vs 2020–2025)

| query_id | papers_2010_2015 | papers_2020_2025 | pct_change_papers_recent_vs_early |
| --- | --- | --- | --- |
| ephemeroptera | 411 | 791 | 92.46 |
| mosquitoes | 5528 | 11862 | 114.58 |
| odonata | 413 | 2943 | 612.59 |
| plecoptera | 260 | 537 | 106.54 |
| trichoptera | 925 | 1273 | 37.62 |

### Top research themes (overall distribution)

| query_id | theme_top1 | theme_top1_pct | theme_top2 | theme_top2_pct | theme_top3 | theme_top3_pct | theme_not_specified_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ephemeroptera | Ecology/Behavior | 41.79 | Biomonitoring/Water Quality | 25.71 | Taxonomy/Systematics | 8.28 | 20.86 |
| mosquitoes | Ecology/Behavior | 37.95 | Applied Ecology | 25.9 | Physiology | 13.69 | 12.4 |
| odonata | Ecology/Behavior | 30.4 | Taxonomy/Systematics | 20.81 | Biomonitoring/Water Quality | 5.39 | 32.61 |
| plecoptera | Ecology/Behavior | 36.9 | Biomonitoring/Water Quality | 26.3 | Taxonomy/Systematics | 25.45 | 4.35 |
| trichoptera | Ecology/Behavior | 35.44 | Taxonomy/Systematics | 31.46 | Biomonitoring/Water Quality | 20.94 | 3.07 |

### Geography: average continental shares (RQ2-style buckets)

| query_id | geo_avg_south_america_pct | geo_avg_asia_pct | geo_avg_europe_pct | geo_avg_north_america_pct | geo_avg_unknown_pct |
| --- | --- | --- | --- | --- | --- |
| ephemeroptera | 15.86 | 12.78 | 23.9 | 23.27 | 15.61 |
| mosquitoes | 13.31 | 25.36 | 12.23 | 18.26 | 13.97 |
| odonata | 14.5 | 17.4 | 21.9 | 11.39 | 28.51 |
| plecoptera | 14.16 | 22.28 | 27.47 | 23.3 | 4.2 |
| trichoptera | 18.62 | 18.92 | 29.97 | 19.51 | 4.95 |

### Geography: mean early (2010–2012) vs recent (2023–2025) continental % (percentage-point change)

| query_id | geo_delta_pp_south_america_2010_2012_vs_2023_2025 | geo_delta_pp_asia_2010_2012_vs_2023_2025 | geo_delta_pp_europe_2010_2012_vs_2023_2025 | geo_delta_pp_north_america_2010_2012_vs_2023_2025 |
| --- | --- | --- | --- | --- |
| ephemeroptera | -3.73 | 1.34 | 2.56 | -29.13 |
| mosquitoes | -1.98 | 7.1 | 1.31 | -11.62 |
| odonata | -5.33 | 6.36 | -5.27 | -12.98 |
| plecoptera | 3 | 18.47 | 10.8 | -28.01 |
| trichoptera | 2.98 | 3.9 | 6.74 | -14.54 |

### RQ4A: Authorship structure (OpenAlex counts when available)

| query_id | authors_mean | authors_median | authors_mean_early_2010_2015 | authors_mean_recent_2020_2025 | authors_mean_applied | authors_mean_taxonomic |
| --- | --- | --- | --- | --- | --- | --- |
| ephemeroptera | 4.257 | 4 | 3.74 | 4.514 | 4.546 | 3.246 |
| mosquitoes | 6.499 | 5 | 5.661 | 7.023 | 6.746 | 6.37 |
| odonata | 4.99 | 4 | 4.122 | 5.242 | 5.073 | 3.331 |
| plecoptera | 4.008 | 4 | 3.602 | 4.257 | 4.398 | 2.97 |
| trichoptera | 4.051 | 3 | 3.387 | 4.529 | 4.748 | 3.217 |

### RQ4B: International collaboration (affiliation-country heuristic)

| query_id | intl_collab_info_coverage_pct | intl_collab_pct_overall | intl_collab_pct_known_only_overall | intl_collab_pct_applied | intl_collab_pct_taxonomic |
| --- | --- | --- | --- | --- | --- |
| ephemeroptera | 23.85 | 1.49 | 6.27 | 1.49 | 1.64 |
| mosquitoes | 19.58 | 2.62 | 13.39 | 2.28 | 2.05 |
| odonata | 21.35 | 2.98 | 13.96 | 3.64 | 4.01 |
| plecoptera | 24.03 | 2.56 | 10.67 | 1.7 | 2.99 |
| trichoptera | 20.74 | 2.28 | 11 | 1.19 | 3.12 |
