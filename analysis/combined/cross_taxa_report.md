# Cross-taxon bibliometric summary (side-by-side)

Generated from `data/processed/*/scopus_api_coded.csv` with the same filters as RQ2–RQ4:
- Years: 2010–2025
- Document types: Article and Review only (`Type` column)
- Exclude non–taxon-focused papers (`Taxon_Relevance` not in {Not target-taxon-focused, Not Trichoptera-focused})

### Sample size (2010–2025)

| query_id | All coded (2010–2025) | Taxon-focused (2010–2025) |
| --- | --- | --- |
| mosquitoes | 46626 | 34717 |
| ephemeroptera | 3776 | 3098 |
| plecoptera | 2181 | 2032 |
| trichoptera | 3304 | 3099 |
| odonata | 7870 | 4878 |

### Temporal volume (2010–2015 vs 2020–2025)

| query_id | papers_2010_2015 | papers_2020_2025 | pct_change_papers_recent_vs_early |
| --- | --- | --- | --- |
| mosquitoes | 8966 | 17226 | 92.1 |
| ephemeroptera | 990 | 1363 | 37.7 |
| plecoptera | 607 | 919 | 51.4 |
| trichoptera | 999 | 1356 | 35.7 |
| odonata | 1404 | 2186 | 55.7 |

Each paper has one **primary research theme** label. Ranks #1–#3 omit “Not Specified” when choosing the three most common themes. **Not Specified %** is separate: the share of papers without a more specific theme.

### Research themes by query

| query_id | theme_top1 | theme_top1_pct | theme_top2 | theme_top2_pct | theme_top3 | theme_top3_pct | theme_not_specified_pct |
| --- | --- | --- | --- | --- | --- | --- | --- |
| mosquitoes | Applied Ecology | 61.7 | Ecology/Behavior | 17.8 | Physiology | 10.9 | 0.0 |
| ephemeroptera | Ecology/Behavior | 31.4 | Biomonitoring/Water Quality | 31.1 | Taxonomy/Systematics | 27.0 | 0.0 |
| plecoptera | Taxonomy/Systematics | 31.7 | Biomonitoring/Water Quality | 29.2 | Ecology/Behavior | 27.9 | 0.0 |
| trichoptera | Ecology/Behavior | 35.0 | Taxonomy/Systematics | 28.0 | Biomonitoring/Water Quality | 27.7 | 0.0 |
| odonata | Ecology/Behavior | 38.2 | Taxonomy/Systematics | 25.7 | Biomonitoring/Water Quality | 10.7 | 0.0 |

### Geography: average continental shares (RQ2-style buckets)

| query_id | geo_avg_south_america_pct | geo_avg_asia_pct | geo_avg_europe_pct | geo_avg_north_america_pct | geo_avg_unknown_pct |
| --- | --- | --- | --- | --- | --- |
| mosquitoes | 15.1 | 26.3 | 14.9 | 17.7 | 4.8 |
| ephemeroptera | 21.5 | 20.3 | 26.0 | 22.0 | 0.9 |
| plecoptera | 14.3 | 29.2 | 27.5 | 20.1 | 1.3 |
| trichoptera | 19.5 | 17.8 | 32.8 | 19.7 | 1.4 |
| odonata | 17.9 | 25.2 | 30.0 | 15.2 | 3.7 |

### Geography: mean early (2010–2015) vs recent (2020–2025) continental % (percentage-point change)

| query_id | geo_delta_pp_south_america_2010_2015_vs_2020_2025 | geo_delta_pp_asia_2010_2015_vs_2020_2025 | geo_delta_pp_europe_2010_2015_vs_2020_2025 | geo_delta_pp_north_america_2010_2015_vs_2020_2025 |
| --- | --- | --- | --- | --- |
| mosquitoes | 0.9 | 9.1 | -0.6 | -8.4 |
| ephemeroptera | -1.7 | 14.0 | 2.6 | -16.5 |
| plecoptera | -0.2 | 21.0 | -5.5 | -16.1 |
| trichoptera | 3.4 | 5.2 | 3.6 | -10.9 |
| odonata | 1.8 | 9.7 | -2.1 | -9.1 |

### RQ4A: Authorship structure (OpenAlex counts when available)

| query_id | authors_mean | authors_median | authors_mean_early_2010_2015 | authors_mean_recent_2020_2025 | authors_mean_applied | authors_mean_taxonomic |
| --- | --- | --- | --- | --- | --- | --- |
| mosquitoes | 6.7 | 6.0 | 5.8 | 7.3 | 6.9 | 5.8 |
| ephemeroptera | 4.2 | 4.0 | 3.6 | 4.6 | 4.7 | 3.2 |
| plecoptera | 4.0 | 4.0 | 3.5 | 4.5 | 4.7 | 2.9 |
| trichoptera | 4.1 | 4.0 | 3.5 | 4.6 | 4.7 | 3.2 |
| odonata | 4.0 | 3.0 | 3.3 | 4.5 | 5.1 | 3.2 |

### RQ4B: International collaboration (affiliation-country heuristic)

| query_id | intl_collab_info_coverage_pct | intl_collab_pct_overall | intl_collab_pct_known_only_overall | intl_collab_pct_applied | intl_collab_pct_taxonomic |
| --- | --- | --- | --- | --- | --- |
| mosquitoes | 96.1 | 40.4 | 42.0 | 42.0 | 39.1 |
| ephemeroptera | 96.1 | 29.3 | 30.5 | 24.4 | 31.8 |
| plecoptera | 96.9 | 25.8 | 26.6 | 22.7 | 24.3 |
| trichoptera | 96.3 | 27.8 | 28.8 | 24.7 | 29.9 |
| odonata | 92.5 | 33.2 | 35.9 | 27.9 | 38.4 |
