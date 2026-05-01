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
| ephemeroptera | 411 | 791 | 92.5 |
| mosquitoes | 5528 | 11862 | 114.6 |
| odonata | 413 | 2943 | 612.6 |
| plecoptera | 260 | 537 | 106.5 |
| trichoptera | 925 | 1273 | 37.6 |

Each paper has one **primary research theme** label. Ranks #1–#3 omit “Not Specified” when choosing the three most common themes. **Not Specified %** is separate: the share of papers without a more specific theme.

### Research themes by query

| query_id | theme_top1 | theme_top1_pct | theme_top2 | theme_top2_pct | theme_top3 | theme_top3_pct | theme_not_specified_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| ephemeroptera | Ecology/Behavior | 41.8 | Biomonitoring/Water Quality | 25.7 | Taxonomy/Systematics | 8.3 | 20.9 |
| mosquitoes | Ecology/Behavior | 38.0 | Applied Ecology | 25.9 | Physiology | 13.7 | 12.4 |
| odonata | Ecology/Behavior | 30.4 | Taxonomy/Systematics | 20.8 | Biomonitoring/Water Quality | 5.4 | 32.6 |
| plecoptera | Ecology/Behavior | 36.9 | Biomonitoring/Water Quality | 26.3 | Taxonomy/Systematics | 25.4 | 4.4 |
| trichoptera | Ecology/Behavior | 35.4 | Taxonomy/Systematics | 31.5 | Biomonitoring/Water Quality | 20.9 | 3.1 |

### Geography: average continental shares (RQ2-style buckets)

| query_id | geo_avg_south_america_pct | geo_avg_asia_pct | geo_avg_europe_pct | geo_avg_north_america_pct | geo_avg_unknown_pct |
| --- | --- | --- | --- | --- | --- |
| ephemeroptera | 15.9 | 12.8 | 23.9 | 23.3 | 15.6 |
| mosquitoes | 13.3 | 25.4 | 12.2 | 18.3 | 14.0 |
| odonata | 14.5 | 17.4 | 21.9 | 11.4 | 28.5 |
| plecoptera | 14.2 | 22.3 | 27.5 | 23.3 | 4.2 |
| trichoptera | 18.6 | 18.9 | 30.0 | 19.5 | 5.0 |

### Geography: mean early (2010–2012) vs recent (2023–2025) continental % (percentage-point change)

| query_id | geo_delta_pp_south_america_2010_2012_vs_2023_2025 | geo_delta_pp_asia_2010_2012_vs_2023_2025 | geo_delta_pp_europe_2010_2012_vs_2023_2025 | geo_delta_pp_north_america_2010_2012_vs_2023_2025 |
| --- | --- | --- | --- | --- |
| ephemeroptera | -3.7 | 1.3 | 2.6 | -29.1 |
| mosquitoes | -2.0 | 7.1 | 1.3 | -11.6 |
| odonata | -5.3 | 6.4 | -5.3 | -13.0 |
| plecoptera | 3.0 | 18.5 | 10.8 | -28.0 |
| trichoptera | 3.0 | 3.9 | 6.7 | -14.5 |

### RQ4A: Authorship structure (OpenAlex counts when available)

| query_id | authors_mean | authors_median | authors_mean_early_2010_2015 | authors_mean_recent_2020_2025 | authors_mean_applied | authors_mean_taxonomic |
| --- | --- | --- | --- | --- | --- | --- |
| ephemeroptera | 4.2 | 4.0 | 3.7 | 4.5 | 4.5 | 3.2 |
| mosquitoes | 6.5 | 5.0 | 5.6 | 7.0 | 6.7 | 6.4 |
| odonata | 5.0 | 4.0 | 4.1 | 5.2 | 5.1 | 3.3 |
| plecoptera | 4.0 | 4.0 | 3.6 | 4.2 | 4.4 | 2.9 |
| trichoptera | 4.0 | 3.0 | 3.4 | 4.5 | 4.8 | 3.2 |

### RQ4B: International collaboration (affiliation-country heuristic)

| query_id | intl_collab_info_coverage_pct | intl_collab_pct_overall | intl_collab_pct_known_only_overall | intl_collab_pct_applied | intl_collab_pct_taxonomic |
| --- | --- | --- | --- | --- | --- |
| ephemeroptera | 94.4 | 25.6 | 27.2 | 20.1 | 27.9 |
| mosquitoes | 95.2 | 37.5 | 39.4 | 36.5 | 38.1 |
| odonata | 92.2 | 33.1 | 35.9 | 26.9 | 39.1 |
| plecoptera | 96.0 | 25.2 | 26.3 | 21.1 | 23.4 |
| trichoptera | 95.6 | 27.3 | 28.6 | 25.0 | 29.2 |
