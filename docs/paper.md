# Geographic Shifts and Thematic Evolution in Aquatic Insect Research: A Comparative Bibliometric Analysis of Culicidae, EPT Taxa, and Odonata (2010–2025)

## 1. Introduction

Aquatic insects are among the most important organisms in freshwater ecosystems, contributing to nutrient cycling, energy transfer, decomposition, and predator–prey interactions. Because many species are sensitive to environmental disturbance, aquatic insects are also widely used as indicators of water quality and ecosystem health. Consequently, they have been the focus of extensive research across ecology, conservation, biomonitoring, taxonomy, evolution, and environmental management.

The aquatic insect groups examined in this study differ substantially in both their ecological roles and relevance to human society. Ephemeroptera (mayflies), Plecoptera (stoneflies), and Trichoptera (caddisflies), collectively EPT taxa, are widely used in freshwater biomonitoring because many species are sensitive to pollution and habitat degradation. Odonata (dragonflies and damselflies) are important aquatic predators and are frequently studied as indicators of biodiversity and habitat quality. In contrast, Culicidae (mosquitoes) are among the most intensively studied aquatic insects because of their role as vectors of diseases affecting humans and wildlife. These differences in ecological function and societal importance may influence research priorities, publication output, the geographic distribution of studies, and patterns of scientific collaboration.

Despite the large volume of literature on these taxa, few studies have systematically compared publication patterns across groups or tracked how research effort has changed over time. Bibliometric analysis provides a quantitative framework for examining scientific activity through publication trends, research themes, authorship structure, and collaboration networks. Using Scopus records from 2010 to 2025 for each taxon, with a Scopus–Google Scholar comparison for a benchmark year, we analyzed temporal and geographic patterns, thematic composition, and authorship and collaboration across Culicidae, EPT taxa, and Odonata.

We addressed four questions:

1. How do Scopus and Google Scholar overlap in their coverage of literature on each taxon?
2. How has publication output changed over time and across geographic regions?
3. What research themes dominate the literature for each taxon, and how have they shifted over time?
4. What are the patterns of authorship and international collaboration across taxa and among research specialties?

By systematically comparing these groups, we sought to characterize geographic shifts and thematic evolution in aquatic insect research across disciplines, regions, and the 2010–2025 study period.

## 2. Methods

### 2.1 Study design and data collection

We applied one bibliometric workflow to five aquatic insect taxa (Table 1): Culicidae; Ephemeroptera, Plecoptera, and Trichoptera (EPT, three separate searches); and Odonata. Each taxon had its own Scopus and Google Scholar searches, processed files, and coded dataset (records were not merged across taxa).

**Table 1. Scopus search queries by taxon** (terms searched in title, abstract, and keywords)

| Taxon | Search terms |
| --- | --- |
| Culicidae | mosquito, mosquitoes, Culicidae |
| Ephemeroptera | Ephemeroptera, mayfly, mayflies |
| Plecoptera | Plecoptera, stonefly, stoneflies |
| Trichoptera | Trichoptera, caddisfly, caddisflies, caddis fly, caddis flies |
| Odonata | Odonata, dragonfly, dragonflies, damselfly, damselflies |

Publications were retrieved with Elsevier’s Scopus Search API using the queries in Table 1. Calendar years 2010–2025 were fetched sequentially; large years were subdivided into months or quarters to stay within API pagination limits. The API supplied standard bibliographic metadata (title, journal, year, DOI, citations, document type). Full abstracts and complete author lists were often missing and were obtained as described in section 2.2.

For database comparison, taxon-matched Google Scholar results for 2023 were retrieved with Publish or Perish using the same search terms as in Table 1 (1,000-result cap per query). Scopus records for the same year used the workflow above.

### 2.2 Data processing and metadata enrichment

Yearly Scopus exports were combined and deduplicated in two steps: one row per DOI when present, then one row per normalized title (lowercase, trimmed whitespace; first occurrence retained). Each record kept its publication year for temporal analysis and downstream linking.

Where a DOI was available, missing abstracts were retrieved in fixed order: OpenAlex, Semantic Scholar, CrossRef, then PubMed for biomedical items; the first hit was used. Records without abstracts were still classified; missing abstract text was noted for downstream use. Author lists, author counts, and affiliations were retrieved from OpenAlex via DOI lookup for collaboration analyses and as supplemental geographic context.

### 2.3 Automated classification

Each record retained Scopus bibliographic fields plus four LLM-assigned variables:

- **Country:** primary study country (standard name), or blank if unknown.
- **Global biogeographic region:** Oriental, Neotropical, Nearctic, Palearctic, East Palearctic, Afrotropical, Australasian, Global, or Not Specified (formal biogeographic labels, not map coordinates).
- **Primary research theme:** Taxonomy/Systematics, Ecology/Behavior, Biomonitoring/Water Quality, Evolution/Phylogeny, Conservation, Materials Science (Silk), Physiology, Applied Ecology, Other, or Not Specified. Materials Science (Silk) applies mainly to Trichoptera.
- **Taxon relevance:** Primary focus, Secondary mention, Peripheral, or Not target-taxon-focused.

For continental summaries, biogeographic regions were mapped to South America, Asia, Europe, North America, Other, or Unknown (e.g., Nearctic and Neotropical to North and South America, Palearctic and East Palearctic to Europe and Asia). For selected collaboration comparisons (section 2.4.3), themes were also grouped as *applied* (Biomonitoring/Water Quality, Applied Ecology, Conservation, Materials Science [Silk]), *taxonomic* (Taxonomy/Systematics), or *other* (all remaining themes).

Papers were classified with OpenAI GPT-4o-mini (temperature 0) from title, abstract (or a statement that no abstract was available), and title-matched affiliations. Instructions and category definitions were shared across taxa; each run named the target taxon in the prompt (e.g., Culicidae vs Trichoptera). The model was instructed to assign labels only when supported by the title or abstract, to prefer the most specific applicable category, and not to infer missing study details. Geography prioritized explicit study locations in title/abstract, then other text cues, then affiliations; one primary location when several were mentioned. Theme was a single primary category, with prompt cues separating overlapping labels (taxonomy vs phylogeny, ecology vs biomonitoring, silk-focused materials work). Taxon relevance excluded keyword hits not centered on the taxon. The full prompt is available on request.

### 2.4 Bibliometric analysis

The analyses below correspond to the four questions in the Introduction. Analyses of publication volume, geography, themes, and collaboration used 2010–2025 papers not classified as not target-taxon-focused (Primary focus, Secondary mention, and Peripheral retained). Metrics were computed per taxon with identical rules; combined tables and figures compare all five corpora side by side. The database-overlap analysis (section 2.4.1) used all 2023 records from section 2.1, without the taxon-relevance filter.

#### 2.4.1 Database overlap and coverage

For each taxon, 2023 Scopus and Google Scholar records were paired by DOI, or by title similarity of at least 0.85 on normalized titles when DOIs were absent. We reported overlap, database-unique records, and citation summaries. Language was flagged with a simple character-based rule (non-Latin scripts or common diacritics), not full language identification. Journals were labeled regional when the title suggested national/regional scope, otherwise international/general; missing names were set aside.

#### 2.4.2 Temporal, geographic, and thematic patterns

Publication volume and geography used each taxon’s filtered dataset described above. Countries were harmonized; one primary location was kept when multiple appeared. Summaries were by region and year, with country-level concentration across 2010–2025. Continental shifts compared mean within-year shares between early and recent multi-year periods within the study window. Locations reflect LLM-inferred study geography from text, not geocoding of addresses. Theme counts used each paper’s assigned primary research theme, summarized in multi-year bands and by year; ranked cross-taxa theme summaries exclude Not Specified.

#### 2.4.3 Authorship and collaboration

Team size used OpenAlex author counts, binned as single author, two, three to five, six to ten, or more than ten. Research specialties were primary themes; selected tables use the applied, taxonomic, and other theme groups defined in section 2.3. International collaboration labels used, in order: multiple vs single ISO country codes in author metadata; if missing, country keywords in affiliation text; if still missing, assigned global region (Global → International; any other specified region → National; Not Specified or missing → Unknown). Rates and team size were compared across taxa and among specialties, and among applied, taxonomic, and other groups in selected tables.

### 2.5 Software and reproducibility

Analyses were implemented in Python using pandas and provider API clients for Scopus, OpenAlex, and OpenAI. The same workflow was repeated for each taxon with taxon-specific search strings (Table 1) and classification prompts. Query definitions, prompts, and analysis code are available on request.

## 3. Results

### 3.1 Database overlap and coverage (2023)

Scopus returned very different numbers of 2023 records for each taxon (Table 2). For example, Culicidae returned 4,307 records and Plecoptera returned 153. Google Scholar was searched with the same terms, but each query returned at most 1,000 records (Methods, section 2.1). This comparison used all 2023 search hits matched by DOI or similar title (Methods, sections 2.1 and 2.4.1) and did not apply the taxon-focus screen used in sections 3.2–3.4.

**Table 2. Scopus and Google Scholar coverage by taxon (2023)**

| Metric | Culicidae | Ephemeroptera | Plecoptera | Trichoptera | Odonata |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scopus total | 4,307 | 409 | 153 | 261 | 903 |
| Google Scholar total | 1,000 | 997 | 980 | 993 | 1,000 |
| Overlap (both) | 863 | 369 | 136 | 230 | 400 |
| Overlap / Scopus (%) | 20.0 | 90.2 | 88.9 | 88.1 | 44.3 |

Google Scholar returned close to 1,000 records for every taxon (Table 2), reflecting the export limit. The share of Scopus records also found in Google Scholar ranged from 20.0% (Culicidae) to 88–90% (Ephemeroptera, Plecoptera, and Trichoptera) and was 44.3% for Odonata. Between 136 and 863 records appeared in both databases, depending on taxon.

### 3.2 Temporal and geographic variation (2010–2025)

Taxon-focused publication output rose in every taxon between the early and recent multi-year windows (Table 3). For example, Odonata increased from 413 papers in 2010–2015 to 2,943 in 2020–2025 (+612.6%), the largest relative gain, whereas Trichoptera increased from 925 to 1,273 (+37.6%). Results below use taxon-focused Scopus papers from 2010–2025 (Methods, sections 2.3 and 2.4.2). Between 36.6% and 83.0% of coded records were taxon-focused, depending on taxon (Table 3).

**Table 3. Publication volume (2010–2025)**

| Metric | Culicidae | Ephemeroptera | Plecoptera | Trichoptera | Odonata |
| --- | ---: | ---: | ---: | ---: | ---: |
| All coded papers | 51,990 | 4,062 | 2,272 | 3,456 | 9,203 |
| Taxon-focused papers | 22,664 | 1,486 | 1,057 | 2,870 | 4,079 |
| Taxon-focused (%) | 43.6 | 36.6 | 46.5 | 83.0 | 44.3 |
| Early window (2010–2015) | 5,528 | 411 | 260 | 925 | 413 |
| Recent window (2020–2025) | 11,862 | 791 | 537 | 1,273 | 2,943 |
| Percent change (early vs recent) | 114.6 | 92.5 | 106.5 | 37.6 | 612.6 |

Culicidae accounted for the largest absolute volume in both windows (Table 3). Annual taxon-focused counts rose steadily for Culicidae and increased sharply for Odonata after about 2017 (Figure 1).

**Figure 1.** Taxon-focused publication counts by year, 2010–2025, for all five taxa. Counts use the same taxon-focus screen and deduplication rules as Table 3.

Continent labels describe where the study took place (Methods, section 2.3), not author addresses alone. Mean continental shares differed across taxa: Culicidae papers were most Asia-weighted (25.4% on average), whereas Ephemeroptera, Plecoptera, and Trichoptera were most Europe-weighted (24–30%). Odonata had the highest Unknown share (28.5%). North America's share fell in every taxon when comparing 2010–2012 with 2023–2025 (Table 4; Figure 2).

**Table 4. Change in mean continental share (percentage points): 2010–2012 vs 2023–2025**

| Continent | Culicidae | Ephemeroptera | Plecoptera | Trichoptera | Odonata |
| --- | ---: | ---: | ---: | ---: | ---: |
| South America | −2.0 | −3.7 | 3.0 | 3.0 | −5.3 |
| Asia | 7.1 | 1.3 | 18.5 | 3.9 | 6.4 |
| Europe | 1.3 | 2.6 | 10.8 | 6.7 | −5.3 |
| North America | −11.6 | −29.1 | −28.0 | −14.5 | −13.0 |

**Figure 2.** Change in mean continental share (percentage points) from 2010–2012 to 2023–2025. Early and recent values are the mean of within-year continental percentages in each three-year band. Red bars indicate an increase; blue bars indicate a decrease. Values are given in Table 4.

### 3.3 Thematic evolution (2010–2025)

Ecology/Behavior was the most common primary research theme for every taxon (Table 5). For example, it accounted for 41.8% of Ephemeroptera papers and 38.0% of Culicidae papers, but the second- and third-ranked themes diverged: Culicidae papers were often classified as Applied Ecology (25.9%) or Physiology (13.7%), whereas Trichoptera papers were often Taxonomy/Systematics (31.5%) or Biomonitoring/Water Quality (20.9%). Results use taxon-focused papers from 2010–2025 with one primary theme label each (Methods, sections 2.3 and 2.4.2). Ranks #1–#3 exclude papers labeled Not Specified; Not Specified is reported separately.

**Table 5. Ranked primary research themes by taxon (2010–2025)**

| Rank | Culicidae | Ephemeroptera | Plecoptera | Trichoptera | Odonata |
| --- | --- | --- | --- | --- | --- |
| #1 | Ecology/Behavior (38.0%) | Ecology/Behavior (41.8%) | Ecology/Behavior (36.9%) | Ecology/Behavior (35.4%) | Ecology/Behavior (30.4%) |
| #2 | Applied Ecology (25.9%) | Biomonitoring/Water Quality (25.7%) | Biomonitoring/Water Quality (26.3%) | Taxonomy/Systematics (31.5%) | Taxonomy/Systematics (20.8%) |
| #3 | Physiology (13.7%) | Taxonomy/Systematics (8.3%) | Taxonomy/Systematics (25.4%) | Biomonitoring/Water Quality (20.9%) | Biomonitoring/Water Quality (5.4%) |
| Not Specified | 12.4% | 20.9% | 4.4% | 3.1% | 32.6% |

*Ranks #1–#3 exclude papers labeled Not Specified.*

Culicidae stood apart from the EPT taxa and Odonata in its emphasis on applied and physiological work (Table 5). Plecoptera and Trichoptera had high shares of taxonomy and biomonitoring among their top three themes; Ephemeroptera combined ecology, biomonitoring, and a smaller taxonomy share. Not Specified labels were uncommon for Plecoptera and Trichoptera (3–4%) but much more frequent for Odonata (32.6%) and Ephemeroptera (20.9%).

**Figure 3.** Share of papers by ranked primary theme (#1–#3) for each taxon, 2010–2025. Rankings exclude Not Specified; #1, #2, and #3 denote the first, second, and third most frequent themes within each taxon. Not Specified rates are given in Table 5.

### 3.4 Authorship and collaboration (2010–2025)

This subsection reports team size, author-count groups (for example, single author versus three to five authors), year-by-year splits for taxonomic versus non-taxonomic papers, and international collaboration class (section 2.4.3).

Analytical sample: *N* = 2,845 after combining OpenAlex author metadata and excluding papers with no authors recorded (author count zero). That filter makes this sample slightly smaller than §3.2–3.3.

Study-type groups for comparison: *Applied* = Biomonitoring/Water Quality, Applied Ecology, Conservation, Materials Science (Silk); *taxonomic* = Taxonomy/Systematics; *other* = all remaining themes. *Non-taxonomic* tables combine applied and other papers. Team sizes use the OpenAlex author count (Methods, section 2.4.3).

#### Overall authorship

| Measure | Value |
|--------|------:|
| Mean authors per paper | 4.05 |
| Median | 3.0 |
| Standard deviation | 2.95 |
| Range | 1–49 |

#### Authorship by period

| Period | Papers | Mean authors | Median authors |
|--------|-------:|-------------:|---------------:|
| Early (2010–2015) | 920 | 3.39 | 3.0 |
| Recent (2020–2025) | 1,261 | 4.53 | 4.0 |

#### Authorship by study type

| Study type | Papers | Mean authors | Median authors |
|------------|-------:|-------------:|---------------:|
| Applied | 672 | 4.74 | 4.0 |
| Taxonomic | 896 | 3.22 | 3.0 |

#### Collaboration size (all papers)

Author-count groups: 3–5 and 6–10 are inclusive; 10+ means eleven or more authors.

| Category | Papers | % |
|----------|-------:|--:|
| Single author | 246 | 8.7 |
| 2 authors | 590 | 20.7 |
| 3–5 authors | 1,442 | 50.7 |
| 6–10 authors | 498 | 17.5 |
| 10+ authors | 69 | 2.4 |

#### Collaboration size by study type

Percentages are within each study-type group.

##### Applied (*n* = 672)

| Category | % |
|----------|--:|
| Single author | 5.4 |
| 2 authors | 13.7 |
| 3–5 authors | 50.9 |
| 6–10 authors | 26.6 |
| 10+ authors | 3.4 |

##### Taxonomic (*n* = 896)

| Category | % |
|----------|--:|
| Single author | 14.7 |
| 2 authors | 30.6 |
| 3–5 authors | 45.3 |
| 6–10 authors | 7.6 |
| 10+ authors | 1.8 |

##### Other (*n* = 1,277)

| Category | % |
|----------|--:|
| Single author | 6.1 |
| 2 authors | 17.5 |
| 3–5 authors | 54.3 |
| 6–10 authors | 19.7 |
| 10+ authors | 2.3 |

#### Year-by-year collaboration (taxonomic studies only)

*N* = papers with Taxonomy/Systematics as primary theme that year. The Single through 10+ entries are counts, with percentages out of that year’s taxonomic total.

| Year | *N* | Mean | Median | Min | Max | Single | 2 | 3–5 | 6–10 | 10+ |
|-----:|----:|-----:|-------:|----:|----:|-------|---:|----:|-----:|----:|
| 2010 | 53 | 2.21 | 2 | 1 | 7 | 15 (28.3%) | 26 (49.1%) | 10 (18.9%) | 2 (3.8%) | 0 (0.0%) |
| 2011 | 40 | 2.38 | 2 | 1 | 6 | 9 (22.5%) | 16 (40.0%) | 13 (32.5%) | 2 (5.0%) | 0 (0.0%) |
| 2012 | 32 | 2.28 | 2 | 1 | 4 | 6 (18.8%) | 13 (40.6%) | 13 (40.6%) | 0 (0.0%) | 0 (0.0%) |
| 2013 | 52 | 2.92 | 3 | 1 | 8 | 8 (15.4%) | 13 (25.0%) | 27 (51.9%) | 4 (7.7%) | 0 (0.0%) |
| 2014 | 48 | 2.81 | 2 | 1 | 7 | 9 (18.8%) | 15 (31.2%) | 20 (41.7%) | 4 (8.3%) | 0 (0.0%) |
| 2015 | 41 | 3.95 | 2 | 1 | 15 | 3 (7.3%) | 18 (43.9%) | 12 (29.3%) | 6 (14.6%) | 2 (4.9%) |
| 2016 | 58 | 2.93 | 2 | 1 | 15 | 13 (22.4%) | 21 (36.2%) | 19 (32.8%) | 3 (5.2%) | 2 (3.4%) |
| 2017 | 65 | 2.94 | 3 | 1 | 11 | 15 (23.1%) | 17 (26.2%) | 29 (44.6%) | 2 (3.1%) | 2 (3.1%) |
| 2018 | 69 | 2.72 | 3 | 1 | 9 | 7 (10.1%) | 26 (37.7%) | 34 (49.3%) | 2 (2.9%) | 0 (0.0%) |
| 2019 | 54 | 3.31 | 2 | 1 | 16 | 8 (14.8%) | 19 (35.2%) | 22 (40.7%) | 3 (5.6%) | 2 (3.7%) |
| 2020 | 67 | 3.84 | 3 | 1 | 28 | 6 (9.0%) | 17 (25.4%) | 36 (53.7%) | 5 (7.5%) | 3 (4.5%) |
| 2021 | 69 | 3.22 | 2 | 1 | 17 | 12 (17.4%) | 26 (37.7%) | 24 (34.8%) | 5 (7.2%) | 2 (2.9%) |
| 2022 | 60 | 3.47 | 3 | 1 | 10 | 7 (11.7%) | 15 (25.0%) | 31 (51.7%) | 7 (11.7%) | 0 (0.0%) |
| 2023 | 70 | 3.56 | 3 | 1 | 19 | 5 (7.1%) | 18 (25.7%) | 40 (57.1%) | 6 (8.6%) | 1 (1.4%) |
| 2024 | 68 | 3.90 | 4 | 1 | 8 | 6 (8.8%) | 8 (11.8%) | 44 (64.7%) | 10 (14.7%) | 0 (0.0%) |
| 2025 | 50 | 4.38 | 4 | 1 | 22 | 3 (6.0%) | 6 (12.0%) | 32 (64.0%) | 7 (14.0%) | 2 (4.0%) |

#### Year-by-year collaboration (non-taxonomic: applied + other)

| Year | *N* | Mean | Median | Min | Max | Single | 2 | 3–5 | 6–10 | 10+ |
|-----:|----:|-----:|-------:|----:|----:|-------|---:|----:|-----:|----:|
| 2010 | 96 | 3.51 | 3 | 1 | 8 | 6 (6.2%) | 22 (22.9%) | 56 (58.3%) | 12 (12.5%) | 0 (0.0%) |
| 2011 | 109 | 3.58 | 3 | 1 | 10 | 8 (7.3%) | 24 (22.0%) | 65 (59.6%) | 12 (11.0%) | 0 (0.0%) |
| 2012 | 116 | 3.28 | 3 | 1 | 10 | 13 (11.2%) | 27 (23.3%) | 66 (56.9%) | 10 (8.6%) | 0 (0.0%) |
| 2013 | 105 | 3.56 | 3 | 1 | 11 | 7 (6.7%) | 30 (28.6%) | 52 (49.5%) | 15 (14.3%) | 1 (1.0%) |
| 2014 | 103 | 3.72 | 3 | 1 | 17 | 5 (4.9%) | 26 (25.2%) | 59 (57.3%) | 12 (11.7%) | 1 (1.0%) |
| 2015 | 125 | 4.13 | 4 | 1 | 12 | 5 (4.0%) | 30 (24.0%) | 64 (51.2%) | 25 (20.0%) | 1 (0.8%) |
| 2016 | 117 | 4.69 | 4 | 1 | 37 | 5 (4.3%) | 17 (14.5%) | 61 (52.1%) | 31 (26.5%) | 3 (2.6%) |
| 2017 | 100 | 4.62 | 4 | 1 | 15 | 1 (1.0%) | 15 (15.0%) | 57 (57.0%) | 24 (24.0%) | 3 (3.0%) |
| 2018 | 97 | 4.27 | 4 | 1 | 12 | 6 (6.2%) | 14 (14.4%) | 55 (56.7%) | 21 (21.6%) | 1 (1.0%) |
| 2019 | 104 | 5.20 | 4 | 1 | 41 | 4 (3.8%) | 14 (13.5%) | 56 (53.8%) | 25 (24.0%) | 5 (4.8%) |
| 2020 | 143 | 4.39 | 4 | 1 | 12 | 7 (4.9%) | 15 (10.5%) | 86 (60.1%) | 31 (21.7%) | 4 (2.8%) |
| 2021 | 163 | 4.66 | 4 | 1 | 21 | 14 (8.6%) | 22 (13.5%) | 79 (48.5%) | 41 (25.2%) | 7 (4.3%) |
| 2022 | 142 | 4.82 | 4 | 1 | 22 | 10 (7.0%) | 15 (10.6%) | 79 (55.6%) | 30 (21.1%) | 8 (5.6%) |
| 2023 | 140 | 4.71 | 4 | 1 | 18 | 8 (5.7%) | 20 (14.3%) | 68 (48.6%) | 40 (28.6%) | 4 (2.9%) |
| 2024 | 144 | 5.42 | 5 | 1 | 49 | 8 (5.6%) | 12 (8.3%) | 66 (45.8%) | 51 (35.4%) | 7 (4.9%) |
| 2025 | 145 | 5.37 | 5 | 1 | 33 | 7 (4.8%) | 13 (9.0%) | 67 (46.2%) | 50 (34.5%) | 8 (5.5%) |

#### International collaboration

Labels follow section 2.4.3: affiliation-based country cues when OpenAlex affiliation text was used in the analysis, otherwise inference from the assigned global biogeographic region.

##### Overall (*n* = 2,845)

| Label | Papers | % |
|-------|-------:|--:|
| International | 65 | 2.3 |
| National | 526 | 18.5 |
| Unknown | 2,254 | 79.2 |

##### By study type

Percentages sum to 100% within each study-type group.

| Study type | International | National | Unknown |
|------------|----------------:|---------:|--------:|
| Applied | 1.2 | 20.7 | 78.1 |
| Taxonomic | 3.1 | 15.4 | 81.5 |
| Other | 2.3 | 19.5 | 78.2 |

## 4. Discussion

### 4.1 Database overlap and coverage

The RQ1 comparison in section 3.1 is constrained most strongly by the Publish or Perish retrieval cap of 1,000 Google Scholar records per query. Because that ceiling binds for every taxon, Google Scholar list sizes near 1,000 do not support rank-order claims about which taxa have the largest literature online. They only describe the first tranche of hits returned under a fixed export limit.

Pairing a complete 2023 Scopus slice with a truncated Google Scholar sample also limits how overlap percentages should be read. For Ephemeroptera, Plecoptera, and Trichoptera, where Scopus returned fewer than about 1,000 records, 88–90% of the Scopus list also appeared in the Google Scholar export (Table 2). That pattern is consistent with substantial agreement on a bounded set of records, but it does not reveal how many additional Google Scholar hits lie beyond the first 1,000. For Culicidae, Scopus alone returned 4,307 records—more than four times the export limit—so the 20.0% overlap with Scopus mainly reflects incomplete Google Scholar sampling rather than a direct estimate of database disagreement across the full corpus. Odonata (903 Scopus records; 44.3% overlap) falls between these cases, and the same truncation issue applies.

We therefore treat RQ1 as a feasibility check, not a definitive audit of either database. Scopus offers consistent metadata retrieval across 2010–2025 without the export ceiling that affects Google Scholar in this workflow, so it remains the primary source for sections 3.2–3.4. Title- and DOI-based matching also leaves some records unpaired, and the two platforms index different source types and apply different completeness rules; those differences were not resolved here. Stronger statements about database equivalence would require an uncapped or otherwise comparable Google Scholar retrieval strategy, ideally with independent validation of missed records.

### 4.2 Temporal and geographic patterns

Section 3.2 shows that taxon-focused output rose in every group between 2010–2015 and 2020–2025, but the scale of that growth differs sharply by taxon. Culicidae dominates absolute counts, which is expected given broad search terms and the size of the global mosquito and vector-control literature. Odonata had the largest relative increase (+612.6%), partly because its early-window count was small (413 papers); large percent changes on a small base should be read cautiously. Trichoptera showed the smallest relative gain (+37.6%), which may reflect a more stable core literature rather than lack of interest, but we did not test that here.

Only 36.6–83.0% of coded Scopus records were taxon-focused (Table 3), so keyword retrieval still pulls in many off-target papers—especially for Culicidae and Ephemeroptera. Volume trends therefore mix true research growth with how well the search-and-screen pipeline isolates each taxon.

Geographic patterns also differ by taxon (Table 4; Figure 2). Culicidae papers were more Asia-weighted on average, whereas Ephemeroptera, Plecoptera, and Trichoptera were more Europe-weighted—patterns that align broadly with where vector research and classical EPT taxonomy and biomonitoring are concentrated, though we did not map individual countries here. Odonata had the highest Unknown share (28.5% on average), so its continental breakdown is less reliable than for Plecoptera or Trichoptera (4–5% Unknown).

The decline in North America's mean share in every taxon (Table 4) is a consistent descriptive pattern, but it should not be taken as proof of a real shift in where field work occurs. Continent labels come from automated coding of study location in title and abstract (Methods, section 2.3), and broad regions are collapsed into continental bins—for example, Palearctic studies appear under Europe and Oriental studies under Asia. Improved location reporting in recent papers, changes in abstract content, or coding inconsistency could all contribute to apparent regional shifts without any change in where insects are actually studied. Continental change values compare the mean of within-year percentages in 2010–2012 with 2023–2025; they are descriptive summaries, not formal trend tests.

Together, the volume and geography results support a comparative picture—high-output Culicidae literature versus smaller EPT corpora, a late rise in Odonata output, and taxon-specific continental profiles—but causal explanations would require country-level validation, manual auditing of coded locations, and literature outside Scopus.

### 4.3 Thematic evolution

Section 3.3 shows a shared headline—Ecology/Behavior ranks first for every taxon—yet the secondary themes separate Culicidae from the stream-insect groups. Culicidae’s profile (Applied Ecology and Physiology among the top three) fits a literature dominated by vector biology, disease transmission, and control-oriented work rather than freshwater community ecology alone. Ephemeroptera, Plecoptera, and Trichoptera instead combine Ecology/Behavior with Biomonitoring/Water Quality and/or Taxonomy/Systematics, which aligns with the long use of EPT taxa in water-quality assessment and systematic description. Trichoptera and Plecoptera show the strongest taxonomy signals among the top three (31.5% and 25.4%, respectively), whereas Ephemeroptera places more weight on biomonitoring (25.7%) with a smaller taxonomy share (8.3%).

Odonata resembles EPT taxa in ranking ecology and taxonomy highly, but its top three themes account for a smaller fraction of papers overall because Not Specified is much more common (32.6%). Ephemeroptera also has a elevated Not Specified rate (20.9%). For those taxa, the ranked themes in Table 5 describe the papers the model could classify confidently, not the full thematic landscape. Low Not Specified rates for Plecoptera and Trichoptera (3–4%) suggest more stable theme assignment there, but that could reflect clearer abstracts, taxon-specific wording, or classifier behavior rather than inherently simpler research topics.

Several methodological limits apply across taxa. Each paper received only one primary theme, so interdisciplinary studies were forced into a single category and co-occurring topics are lost. Rankings exclude Not Specified when choosing #1–#3, which raises the apparent share of named themes among classifiable papers. Infrequent categories—including Materials Science (Silk) for Trichoptera—can fall outside the top three even when they matter for a subset of the literature. We report overall ranked composition for 2010–2025 rather than formal tests of theme change over time; describing how theme mixes shifted between early and recent periods would require band-by-band comparisons for each taxon.

Despite these limits, the thematic comparison still supports the paper’s broader contrast: Culicidae literature reads as applied and biomedical, EPT taxa as ecology-, biomonitoring-, and taxonomy-oriented, and Odonata as mixed but harder to classify with the current labels. Richer theme analysis would need manual coding audits, multi-label schemes, or region-specific breakdowns beyond the cross-taxa summary in Table 5.

### 4.4 Authorship and collaboration

Mean authors per paper was 4.05 (median 3). Mean team size increased between 2010–2015 and 2020–2025 (3.39 vs. 4.53 authors, +1.14); this is descriptive only and does not use a formal statistical model of trends over time.

Applied-themed papers (§3.4) averaged more authors (4.74) than taxonomic papers (3.22), a gap of 1.53 authors on average, consistent with larger teams in the applied-theme group under this grouping. Another 1,277 other-theme papers contribute to the overall and yearly tables.

In total, 70.6% of papers had three or more authors. Contrasts between author-count groups (for example, more papers with six to ten authors in applied than in taxonomic groups) summarize how collaboration size differs by theme split.

International labels are unreliable for this dataset: only 2.3% of papers were classified as International overall, and 79.2% were Unknown under the rules in section 2.4.3. Taxonomic papers had a slightly higher International share (3.1%) than applied papers (1.2%), a −1.9 point gap relative to a simple guess that applied work would be more international. With so much missing signal, between-type gaps should be read as how often the assignment rules produced a label, not as proof of cross-border collaboration rates.

Study-type comparisons also depend on how well the primary theme codes match the papers. “Applied” here means only papers whose primary coded theme falls in the four applied categories, not every applied study on caddisflies. Excluding papers with no authors recorded yields a slightly smaller authorship sample (2,845) than the geographic and thematic sample (2,870).
