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

Scopus search volume in 2023 differed by more than an order of magnitude across taxa (Table 2). For example, Culicidae returned 4,307 records whereas Plecoptera returned 153. Google Scholar totals were retrieved with the same search terms but were capped at 1,000 records per query (Methods, section 2.1).

Analytical sample: all deduplicated 2023 Scopus records from section 2.1 and taxon-matched Google Scholar results from Publish or Perish (Methods, sections 2.1 and 2.4.1). Records were paired by DOI when present, otherwise by normalized-title similarity of at least 0.85. This comparison did not apply the taxon-relevance filter used in sections 3.2–3.4; it describes database retrieval for the raw search terms in Table 1.

**Table 2. Scopus and Google Scholar coverage by taxon (2023)**

| Metric | Culicidae | Ephemeroptera | Plecoptera | Trichoptera | Odonata |
| --- | ---: | ---: | ---: | ---: | ---: |
| Scopus total | 4,307 | 409 | 153 | 261 | 903 |
| Google Scholar total | 1,000 | 997 | 980 | 993 | 1,000 |
| Overlap (both) | 863 | 369 | 136 | 230 | 400 |
| Overlap / Scopus (%) | 20.0 | 90.2 | 88.9 | 88.1 | 44.3 |

Google Scholar list sizes clustered at or just below 1,000 for every taxon (Table 2). Overlap with Scopus ranged from 20.0% (Culicidae) to 88–90% (Ephemeroptera, Plecoptera, and Trichoptera) and was 44.3% for Odonata. The number of records appearing in both databases was 136–863 depending on taxon.

### 3.2 Temporal and geographic variation (2010–2025)

Analytical sample: *N* = 2,870 papers (2010–2025), excluding records classified as not Trichoptera–focused (Methods, section 2.4). Each publication appears at most once: Scopus-based records were deduplicated first by DOI, then by normalized title, so the counts do not double-count the same article.

Country and global biogeographic region come from the automated coding step (title, abstract, affiliations when used). They describe where the study was inferred to take place, not a separate analysis of author mailing addresses alone.

#### Temporal volume

| Period | Papers |
|--------|-------:|
| 2010–2015 (early) | 925 |
| 2020–2025 (recent) | 1,273 |
| Change (recent minus early) | +348 (+37.6%) |

#### Top countries (overall, 2010–2025)

After harmonizing country names, the most frequent primary country labels were:

| Country | Papers |
|---------|-------:|
| United States | 433 |
| Brazil | 333 |
| Not Specified | 154 |
| China | 135 |
| Japan | 121 |
| Russia | 112 |
| Germany | 92 |
| Canada | 88 |
| Australia | 80 |
| India | 70 |
| United Kingdom | 63 |
| Spain | 62 |
| Argentina | 50 |
| Poland | 42 |
| New Zealand | 40 |

#### Year-by-year geographic distribution (continental categories)

Global biogeographic regions were mapped to continental groups for this table:

- South America ← Neotropical  
- Asia ← Oriental + East Palearctic  
- Europe ← Palearctic  
- North America ← Nearctic  
- Other ← Afrotropical, Australasian, Global  
- Unknown ← missing or Not Specified after classification  

Each cell is the count followed by the percent of that year’s total.

| Year | *N* | South America | Asia | Europe | N. America | Other | Unknown |
|-----:|----:|---------------|------|--------|------------|-------|---------|
| 2010 | 150 | 22 (14.7%) | 27 (18.0%) | 31 (20.7%) | 48 (32.0%) | 19 (12.7%) | 3 (2.0%) |
| 2011 | 150 | 30 (20.0%) | 22 (14.7%) | 37 (24.7%) | 43 (28.7%) | 14 (9.3%) | 4 (2.7%) |
| 2012 | 149 | 22 (14.8%) | 26 (17.4%) | 43 (28.9%) | 36 (24.2%) | 11 (7.4%) | 11 (7.4%) |
| 2013 | 157 | 26 (16.6%) | 27 (17.2%) | 49 (31.2%) | 34 (21.7%) | 16 (10.2%) | 5 (3.2%) |
| 2014 | 152 | 31 (20.4%) | 23 (15.1%) | 42 (27.6%) | 37 (24.3%) | 11 (7.2%) | 8 (5.3%) |
| 2015 | 167 | 34 (20.4%) | 22 (13.2%) | 57 (34.1%) | 35 (21.0%) | 13 (7.8%) | 6 (3.6%) |
| 2016 | 177 | 26 (14.7%) | 34 (19.2%) | 53 (29.9%) | 40 (22.6%) | 12 (6.8%) | 12 (6.8%) |
| 2017 | 167 | 36 (21.6%) | 34 (20.4%) | 52 (31.1%) | 27 (16.2%) | 9 (5.4%) | 9 (5.4%) |
| 2018 | 167 | 36 (21.6%) | 28 (16.8%) | 57 (34.1%) | 27 (16.2%) | 11 (6.6%) | 8 (4.8%) |
| 2019 | 161 | 29 (18.0%) | 29 (18.0%) | 53 (32.9%) | 29 (18.0%) | 14 (8.7%) | 7 (4.3%) |
| 2020 | 210 | 50 (23.8%) | 52 (24.8%) | 52 (24.8%) | 29 (13.8%) | 19 (9.0%) | 8 (3.8%) |
| 2021 | 232 | 33 (14.2%) | 54 (23.3%) | 84 (36.2%) | 31 (13.4%) | 23 (9.9%) | 7 (3.0%) |
| 2022 | 205 | 39 (19.0%) | 47 (22.9%) | 59 (28.8%) | 39 (19.0%) | 10 (4.9%) | 11 (5.4%) |
| 2023 | 211 | 46 (21.8%) | 39 (18.5%) | 67 (31.8%) | 37 (17.5%) | 10 (4.7%) | 12 (5.7%) |
| 2024 | 213 | 41 (19.2%) | 48 (22.5%) | 66 (31.0%) | 23 (10.8%) | 19 (8.9%) | 16 (7.5%) |
| 2025 | 202 | 35 (17.3%) | 42 (20.8%) | 64 (31.7%) | 26 (12.9%) | 18 (8.9%) | 17 (8.4%) |

### 3.3 Thematic evolution (2010–2025)

Analytical sample: same *N* and filters as §3.2. Each paper has one primary research theme from the LLM (Methods, sections 2.3–2.4.2).

The following tables summarize overall theme frequencies, year-by-year shares for the most common themes, and theme composition within each biogeographic region.

#### Overall theme distribution

| Theme | Papers | % of sample |
|-------|-------:|------------:|
| Ecology/Behavior | 1,017 | 35.4 |
| Taxonomy/Systematics | 903 | 31.5 |
| Biomonitoring/Water Quality | 601 | 20.9 |
| Evolution/Phylogeny | 125 | 4.4 |
| Not Specified | 88 | 3.1 |
| Materials Science (Silk) | 50 | 1.7 |
| Physiology | 36 | 1.3 |
| Conservation | 28 | 1.0 |
| Other | 21 | 0.7 |
| Applied Ecology | 1 | 0.0 |

#### Year-by-year theme distribution (selected categories)

The five theme columns are the themes with the highest average share across years (Not Specified is excluded when choosing those five). The Unknown column is Not Specified. Short headers map to full theme names: Ecology = Ecology/Behavior, Taxonomy = Taxonomy/Systematics, Biomonitor = Biomonitoring/Water Quality, Evolution = Evolution/Phylogeny, Silk = Materials Science (Silk). Percentages are the share of that year’s papers.

| Year | *N* | Ecology | Taxonomy | Biomonitor | Evolution | Silk | Unknown |
|-----:|----:|--------:|---------:|-----------:|----------:|-----:|--------:|
| 2010 | 150 | 46 (30.7%) | 53 (35.3%) | 32 (21.3%) | 9 (6.0%) | 5 (3.3%) | 4 (2.7%) |
| 2011 | 150 | 64 (42.7%) | 40 (26.7%) | 34 (22.7%) | 5 (3.3%) | 2 (1.3%) | 3 (2.0%) |
| 2012 | 149 | 60 (40.3%) | 33 (22.1%) | 33 (22.1%) | 6 (4.0%) | 3 (2.0%) | 8 (5.4%) |
| 2013 | 157 | 62 (39.5%) | 52 (33.1%) | 27 (17.2%) | 5 (3.2%) | 4 (2.5%) | 2 (1.3%) |
| 2014 | 152 | 64 (42.1%) | 48 (31.6%) | 21 (13.8%) | 6 (3.9%) | 3 (2.0%) | 4 (2.6%) |
| 2015 | 167 | 73 (43.7%) | 42 (25.1%) | 32 (19.2%) | 8 (4.8%) | 6 (3.6%) | 4 (2.4%) |
| 2016 | 177 | 58 (32.8%) | 59 (33.3%) | 36 (20.3%) | 9 (5.1%) | 6 (3.4%) | 6 (3.4%) |
| 2017 | 167 | 57 (34.1%) | 65 (38.9%) | 25 (15.0%) | 7 (4.2%) | 0 (0.0%) | 5 (3.0%) |
| 2018 | 167 | 50 (29.9%) | 69 (41.3%) | 28 (16.8%) | 6 (3.6%) | 1 (0.6%) | 6 (3.6%) |
| 2019 | 161 | 52 (32.3%) | 55 (34.2%) | 37 (23.0%) | 3 (1.9%) | 4 (2.5%) | 5 (3.1%) |
| 2020 | 210 | 84 (40.0%) | 67 (31.9%) | 42 (20.0%) | 2 (1.0%) | 3 (1.4%) | 6 (2.9%) |
| 2021 | 232 | 81 (34.9%) | 69 (29.7%) | 59 (25.4%) | 10 (4.3%) | 0 (0.0%) | 6 (2.6%) |
| 2022 | 205 | 73 (35.6%) | 60 (29.3%) | 49 (23.9%) | 9 (4.4%) | 3 (1.5%) | 6 (2.9%) |
| 2023 | 211 | 71 (33.6%) | 71 (33.6%) | 40 (19.0%) | 9 (4.3%) | 2 (0.9%) | 9 (4.3%) |
| 2024 | 213 | 56 (26.3%) | 68 (31.9%) | 49 (23.0%) | 19 (8.9%) | 5 (2.3%) | 7 (3.3%) |
| 2025 | 202 | 66 (32.7%) | 52 (25.7%) | 57 (28.2%) | 12 (5.9%) | 3 (1.5%) | 7 (3.5%) |

#### Theme composition by biogeographic region

Percentages are within each region. Each block lists the five most frequent primary themes from the automated summary; other themes may appear in the full counts.

##### Afrotropical

| Theme | Papers | % |
|-------|-------:|--:|
| Biomonitoring/Water Quality | 44 | 48.9 |
| Taxonomy/Systematics | 25 | 27.8 |
| Ecology/Behavior | 18 | 20.0 |
| Conservation | 2 | 2.2 |
| Physiology | 1 | 1.1 |

##### Australasian

| Theme | Papers | % |
|-------|-------:|--:|
| Ecology/Behavior | 71 | 53.8 |
| Taxonomy/Systematics | 38 | 28.8 |
| Biomonitoring/Water Quality | 18 | 13.6 |
| Evolution/Phylogeny | 3 | 2.3 |
| Conservation | 1 | 0.8 |

##### East Palearctic

| Theme | Papers | % |
|-------|-------:|--:|
| Taxonomy/Systematics | 33 | 42.9 |
| Ecology/Behavior | 23 | 29.9 |
| Biomonitoring/Water Quality | 12 | 15.6 |
| Evolution/Phylogeny | 4 | 5.2 |
| Not Specified | 2 | 2.6 |

##### Global (multi-regional syntheses)

| Theme | Papers | % |
|-------|-------:|--:|
| Ecology/Behavior | 5 | 71.4 |
| Biomonitoring/Water Quality | 1 | 14.3 |
| Evolution/Phylogeny | 1 | 14.3 |

##### Nearctic

| Theme | Papers | % |
|-------|-------:|--:|
| Ecology/Behavior | 236 | 43.6 |
| Biomonitoring/Water Quality | 147 | 27.2 |
| Taxonomy/Systematics | 85 | 15.7 |
| Evolution/Phylogeny | 26 | 4.8 |
| Materials Science (Silk) | 24 | 4.4 |

##### Neotropical

| Theme | Papers | % |
|-------|-------:|--:|
| Taxonomy/Systematics | 231 | 43.1 |
| Ecology/Behavior | 195 | 36.4 |
| Biomonitoring/Water Quality | 92 | 17.2 |
| Evolution/Phylogeny | 7 | 1.3 |
| Not Specified | 5 | 0.9 |

##### Oriental

| Theme | Papers | % |
|-------|-------:|--:|
| Taxonomy/Systematics | 215 | 45.1 |
| Ecology/Behavior | 112 | 23.5 |
| Biomonitoring/Water Quality | 91 | 19.1 |
| Evolution/Phylogeny | 34 | 7.1 |
| Materials Science (Silk) | 11 | 2.3 |

##### Palearctic

| Theme | Papers | % |
|-------|-------:|--:|
| Ecology/Behavior | 329 | 38.0 |
| Taxonomy/Systematics | 259 | 29.9 |
| Biomonitoring/Water Quality | 188 | 21.7 |
| Evolution/Phylogeny | 39 | 4.5 |
| Conservation | 16 | 1.8 |

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

Publication volume increased between multi-year bands: 925 papers in 2010–2015 versus 1,273 in 2020–2025 (+348, +37.6%). The most common primary country labels were the United States and Brazil; several other countries contributed large counts (see §3.2).

When each continent’s within-year percentage is averaged across all years, approximate mean shares are South America 18.6%, Asia 18.9%, Europe 30.0%, and North America 19.5%, with the rest in Other and Unknown as tabulated. Comparing the mean of annual percentages in 2010–2012 with 2023–2025, North America’s share in the continental summary table fell from 28.3% to 13.7% (−14.5 percentage points), while South America, Asia, and Europe rose (+3.0, +3.9, and +6.7 points). In that breakdown, North America’s share in the same table declined as the other regions increased. The Unknown category averaged 5.0% per year and tended to rise in later years, so roughly 95% of papers received a continent other than Unknown on average.

These continent labels collapse broad biogeographic regions (for example, Palearctic work can appear under the Europe column). Country can be missing or Not Specified for a minority of papers (on the order of ~6% in the automated report). All regional percentages reflect inferred study geography from the automated coding step, not a separate tabulation by author address alone.

### 4.3 Thematic evolution

With one primary LLM-assigned theme per paper, Ecology/Behavior (35.4%), Taxonomy/Systematics (31.5%), and Biomonitoring/Water Quality (20.9%) accounted for most of the sample; every other theme was under 5%. Regionally, Biomonitoring/Water Quality was most common in the Afrotropical group (48.9%). Ecology/Behavior led in Australasian (53.8%) and Nearctic (43.6%) groups. Taxonomy/Systematics was the single most frequent theme in East Palearctic, Neotropical, and Oriental regions (about 43–45% in each).

Comparing all papers in 2010–2015 with all papers in 2021–2025, the share of Biomonitoring/Water Quality rose by about +4.5 percentage points and Ecology/Behavior fell by about −7.2 points (changes larger than three percentage points were treated as noteworthy in the analysis). Those values are shares of papers within each multi-year band, not averages of the within-year percentages in §3.3.

Single-theme coding forces interdisciplinary work into one category and hides co-occurring topics. Not Specified themes (3.1% overall) also appear as Unknown in the year-by-year table; that table only shows the five themes with the highest average within-year share plus Unknown, so infrequent themes do not receive their own columns even when they appear in the dataset.

### 4.4 Authorship and collaboration

Mean authors per paper was 4.05 (median 3). Mean team size increased between 2010–2015 and 2020–2025 (3.39 vs. 4.53 authors, +1.14); this is descriptive only and does not use a formal statistical model of trends over time.

Applied-themed papers (§3.4) averaged more authors (4.74) than taxonomic papers (3.22), a gap of 1.53 authors on average, consistent with larger teams in the applied-theme group under this grouping. Another 1,277 other-theme papers contribute to the overall and yearly tables.

In total, 70.6% of papers had three or more authors. Contrasts between author-count groups (for example, more papers with six to ten authors in applied than in taxonomic groups) summarize how collaboration size differs by theme split.

International labels are unreliable for this dataset: only 2.3% of papers were classified as International overall, and 79.2% were Unknown under the rules in section 2.4.3. Taxonomic papers had a slightly higher International share (3.1%) than applied papers (1.2%), a −1.9 point gap relative to a simple guess that applied work would be more international. With so much missing signal, between-type gaps should be read as how often the assignment rules produced a label, not as proof of cross-border collaboration rates.

Study-type comparisons also depend on how well the primary theme codes match the papers. “Applied” here means only papers whose primary coded theme falls in the four applied categories, not every applied study on caddisflies. Excluding papers with no authors recorded yields a slightly smaller authorship sample (2,845) than the geographic and thematic sample (2,870).
