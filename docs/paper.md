# 1. Introduction

Trichoptera (caddisflies) are ecologically important aquatic insects. Their larvae are key components of stream food webs and are widely used in water-quality monitoring and conservation. Research on this group spans many areas, including taxonomy, ecology, evolution, water quality, and applied studies. However, few studies synthesize the broader picture: how publications are distributed over time and across regions, which topics dominate, how authors collaborate, and how different databases compare.

This study uses bibliometrics (the statistical analysis of scientific publications) to map the Trichoptera literature. It addresses four questions: (1) How complete and comparable are different databases? (2) How has publication output changed over time and across regions? (3) How have research themes shifted? (4) What are the patterns of authorship and international collaboration, including differences between applied work and taxonomy? The goal is a quantitative, reproducible synthesis of the indexed literature, rather than a narrative review based on a curated subset of studies.

The manuscript is structured as follows: Methods (§2), Results (§3), and Discussion (§4). Results follow the order of the questions above.

## 2. Methods

### 2.1 Overview

The Methods section follows the workflow used to construct and analyze the bibliographic dataset. First, records were collected from the Scopus database, and a smaller dataset was collected from Google Scholar for database comparison. Second, records were cleaned and duplicates were removed. Third, missing metadata such as abstracts, author lists, and affiliations were added from other bibliographic sources. Fourth, each paper was classified into predefined thematic and geographic categories using a large language model. Finally, the resulting dataset was analyzed to address the research questions. The subsections below describe each stage of this process; subsection 2.7 summarizes software, credentials, and reproducibility.

### 2.2 Data collection

#### 2.2.1 Scopus retrieval

Relevant publications were retrieved from the Scopus database using Elsevier’s Scopus Search API. The search was limited to records in which Trichoptera or common English names for caddisflies (e.g., caddisfly, caddisflies) appeared in the title, abstract, or keywords. To obtain complete coverage for each calendar year, the search period was sometimes divided into months or quarters and records were retrieved sequentially. The default metadata supplied by this interface includes core bibliographic information such as title, journal, publication year, DOI, citation count, document type, and pagination fields. However, full abstracts and complete author lists were often missing, so this information was obtained from other sources as described below.

#### 2.2.2 Google Scholar

For database coverage, records from Scopus were compared with Trichoptera-related results from Google Scholar for one publication year (the same calendar year for both sources). Google Scholar results were retrieved using the Publish or Perish software, whereas Scopus records were retrieved through the API described above. The two sources therefore differ in how queries are defined and in how complete the returned lists are. The comparison reports overlap between the two lists, records unique to each database, and basic citation statistics, without assuming equivalent search logic or coverage.

### 2.3 Data cleaning

Yearly Scopus exports were combined into a single dataset so that publication counts by year would not double-count the same article. Duplicate removal proceeded in two steps. First, when a digital object identifier (DOI) was present, we retained a single row per DOI. Second, we removed remaining duplicates by title: titles were normalized to lowercase with leading and trailing whitespace removed, and the first occurrence of each normalized title was kept. Each retained record kept its publication year for later temporal analysis and for linking to downstream metadata and analyses.

### 2.4 Metadata completion

#### 2.4.1 Abstract retrieval

Thematic and geographic coding required abstracts, which were often not included in the Scopus records. Where a DOI was available, abstracts were retrieved from other open bibliographic services in a fixed order: OpenAlex, then Semantic Scholar, then CrossRef, and then PubMed for biomedical items. The first source that returned an abstract was used. We tracked how many abstracts were recovered from each source and how many records still lacked an abstract. Records without abstracts were still classified later but were marked as missing abstract text.


#### 2.4.2 Author and affiliation data

Complete author lists and affiliation information were retrieved from OpenAlex using DOI-based record lookups. For each publication, we retained full author names, total author count, and affiliation context linked to the listed authors. These metadata supported collaboration analyses, including international co-authorship patterns, and provided additional geographic context when study location was not explicit in the title or abstract.

### 2.5 Classification

#### 2.5.1 Schema

Classification added four variables to each record, while bibliographic fields such as title and year continued to come from Scopus. Country was recorded as free text, representing the primary country where the research was conducted, expressed as a standard country name, or left blank when no location could be inferred. Global biogeographic region was assigned as one of the following categories: Oriental, Neotropical, Nearctic, Palearctic, East Palearctic, Afrotropical, Australasian, Global, or Not Specified. Research theme was assigned as one of the following categories: Taxonomy/Systematics, Ecology/Behavior, Biomonitoring/Water Quality, Evolution/Phylogeny, Conservation, Materials Science (Silk), Physiology, Applied Ecology, Other, or Not Specified. Trichoptera relevance was assigned as one of the following categories: Primary focus, Secondary mention, Peripheral, or Not Trichoptera-focused.

#### 2.5.2 LLM classification

Each paper was classified automatically using OpenAI’s GPT-4o-mini model with deterministic settings and the coding schema described above. For each record, the model received the title, abstract (if available), and affiliations when available. Affiliations were matched to the Scopus record by title and appended after the abstract. The same written instructions and category definitions were applied to every paper so that the coding process could be documented and reproduced (see subsection 2.7). Each record was then represented by the original bibliographic fields, the coded variables, and an indicator of whether an abstract was available for that paper.

#### 2.5.3 Prompt instructions

A single standardized instruction set was used for all records to ensure consistent classification across the dataset. For each paper, the model received the title, abstract (or an explicit indication that no abstract was available), and affiliation context when available. The instructions required evidence-based coding, emphasized choosing the most specific category supported by the text, and prohibited inventing missing study details.

Geographic coding followed a structured decision order. The model prioritized explicit study-location information in the title or abstract, then other geographic cues in the study description, and used affiliation context only when direct location information was unavailable. When multiple locations were mentioned, the model selected the primary research location. Global region labels were then assigned from the inferred location, with reserved categories for genuinely global studies and for cases where location could not be determined. Outputs were constrained to a consistent structured format so classifications could be parsed and analyzed reproducibly.

### 2.6 Analysis

#### 2.6.1 Analytical sample (years and relevance filter)

Analyses for research questions 2–4 were restricted to publications from 2010 to 2025 and to papers classified as having at least some substantive Trichoptera focus. Papers classified as not Trichoptera-focused were excluded. The resulting set served as the analytical corpus for geographic, thematic, and collaboration analyses, with the usual limitations of automated relevance classification.

#### 2.6.2 Database overlap, citations, language, and journal type

For the 2023 comparison year, records in Scopus and Google Scholar were paired where possible by matching DOIs, and when DOIs were missing, by high similarity between titles so that minor formatting differences still identified the same article. We reported how many records appeared in both databases, how many were unique to each database, and summarized citation-related measures for comparison. Language of the title and abstract was described using a simple rule based on character patterns, for example non-Latin scripts or common diacritics associated with non-English text. This is a rough indicator only and not full language identification. Journals were classified from their titles as regional when the name suggested a national or regional scope. When no such cue appeared but a journal name was present, the outlet was treated as international or general. Names that were missing or unusable were set aside separately. This rule avoids treating most major journals as “unknown” solely because the title does not name a country.

#### 2.6.3 Temporal and geographic variation

Publication volume and geographic patterns were summarized over time using the filtered, coded dataset. Country labels were harmonized to standard forms, and when multiple countries were present in a single record, one primary study location was retained for consistency. The main temporal summaries were reported at the regional level by year, with country-level summaries used to describe overall geographic concentration across the study period. These results reflect locations inferred during classification rather than an independent geocoding workflow.

#### 2.6.4 Thematic evolution

Theme frequencies were taken from the classifier’s research theme variable. Themes were summarized across three multi-year bands (early, middle, and recent within 2010–2025) and also year by year, so that short-term fluctuations and longer trends could both be observed.

#### 2.6.5 Authorship, collaboration size, and international collaboration

Authorship and collaboration were analyzed using enriched author metadata, with records grouped into predefined collaboration-size categories. For thematic comparisons, papers were collapsed into applied, taxonomic, and other groups. International collaboration was estimated from affiliation text using a conservative heuristic: papers were treated as international when affiliation information indicated contributors from more than one country, national when evidence suggested a single country, and unclear when affiliation evidence was insufficient. Collaboration rates were compared across thematic groups, and team size was summarized over time. Team size and internationality were interpreted as related but distinct dimensions of collaboration.

### 2.7 Software, credentials, and reproducibility

The steps above were implemented in Python using common tools for data tables, web requests, and the OpenAI service. Access credentials for Scopus and OpenAI were stored locally and not distributed with any public code archive. Reproducibility rests on the documented coding scheme, prompts, and analysis rules rather than on sharing full intermediate files, which can be large; such files may be archived outside the code repository when needed.

## 3. Results

### 3.1 Database overlap and coverage (2023)

Records from 2023 were compared between Scopus (API export) and Google Scholar (Publish or Perish). Pairs were matched by DOI when possible, otherwise by high title similarity (Methods, subsection 2.6.2). The tables below report list sizes, overlap, journal and language splits, and citation summaries.

#### Basic statistics

| | Scopus | Google Scholar |
|---|--------|----------------|
| Total papers | 261 | 993 |
| Ratio (Google Scholar / Scopus) | — | 3.80 |

#### Overlap analysis

| | Count |
|---|------:|
| Papers in both databases | 230 |
| Matched by DOI | 88 |
| Matched by title similarity | 142 |
| Unique to Scopus only | 31 (11.9% of Scopus records) |
| Unique to Google Scholar only | 763 (76.8% of Google Scholar records) |

#### Journal type distribution

Journal labels are International, Regional, or Unknown (Methods, subsection 2.6.2).

| Label | Scopus | Google Scholar |
|-------|--------|----------------|
| International | 243 (93.1%) | 794 (80.0%) |
| Regional | 18 (6.9%) | 28 (2.8%) |
| Unknown | — | 171 (17.2%) |

#### Language distribution

| | Scopus | Google Scholar |
|---|--------|----------------|
| English | 244 (93.5%) | 842 (84.8%) |
| Non-English | 17 (6.5%) | 151 (15.2%) |

#### Citation statistics

| | Scopus | Google Scholar |
|---|--------|----------------|
| Mean citations per record | 4.51 | 7.32 |
| Median | 2.00 | 1.00 |
| Minimum | 0 | 0 |
| Maximum | 49 | 598 |
| Sum of citation counts (all records) | 1,177 | 7,272 |

### 3.2 Temporal and geographic variation (2010–2025)

Analytical sample: *N* = 2,870 papers (2010–2025), excluding records classified as not Trichoptera–focused (Methods, subsection 2.6.1). Counts use deduplicated Scopus-based records (one row per DOI, then per normalized title).

Country and `Region_Global` come from the classifier (title, abstract, affiliations when used). They describe inferred study geography, not a separate analysis of author mailing addresses alone.

#### Temporal volume

| Period | Papers |
|--------|-------:|
| 2010–2015 (early) | 925 |
| 2020–2025 (recent) | 1,273 |
| Change (recent minus early) | +348 (+37.6%) |

#### Top countries (overall, 2010–2025)

After harmonizing country strings, the most frequent primary country labels were:

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

Classifier-assigned `Region_Global` was mapped to display continents for this table:

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

Analytical sample: same *N* and filters as §3.2. Each paper has one primary `Research_Theme` from the LLM (Methods, subsections 2.5–2.6.4).

The following tables summarize overall theme frequencies, year-by-year shares for the most common themes, and theme composition within each biogeographic region.

#### Overall theme distribution

| Theme | Papers | % of corpus |
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

Columns are the five themes with highest mean within-year share (excluding Not Specified from the ranking), plus Not Specified as Unknown; percentages are within-year.

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

Percentages are within each region. Each block lists the five most frequent primary themes from the automated summary (other themes may occur in the full cross-tabulation).

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

This subsection reports team size, collaboration bins, year-by-year splits for taxonomic versus non-taxonomic papers, and a keyword-based international label.

Analytical sample: *N* = 2,845 after merging OpenAlex author fields and excluding papers with Author_Count_Actual equal to zero. That filter makes this sample slightly smaller than §3.2–3.3.

Study-type groups for comparison: *Applied* = Biomonitoring/Water Quality, Applied Ecology, Conservation, Materials Science (Silk); *taxonomic* = Taxonomy/Systematics; *other* = all remaining themes. *Non-taxonomic* tables combine applied and other papers. Team sizes use Author_Count_Actual (Methods, subsection 2.6.5).

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

Collaboration bins: 3–5 and 6–10 inclusive; 10+ = 11 or more authors.

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

*N* = papers with Taxonomy/Systematics as primary theme that year. Collaboration columns are count (percent of that year’s taxonomic total).

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

#### International collaboration (affiliation heuristic)

Collaboration_Type was assigned using keyword-based country cues in affiliation text, with Region_Global as a fallback when needed (Methods, subsection 2.6.5).

##### Overall (*n* = 2,845)

| Label | Papers | % |
|-------|-------:|--:|
| International | 65 | 2.3 |
| National | 526 | 18.5 |
| Unknown | 2,254 | 79.2 |

##### By study type

Row percentages sum to 100% within each study-type group.

| Study type | International | National | Unknown |
|------------|----------------:|---------:|--------:|
| Applied | 1.2 | 20.7 | 78.1 |
| Taxonomic | 3.1 | 15.4 | 81.5 |
| Other | 2.3 | 19.5 | 78.2 |

## 4. Discussion

### 4.1 Database overlap and coverage

For 2023, Google Scholar returned many more rows than Scopus (993 vs. 261; ratio 3.8×). The two sources use different query and completeness logic, so these totals should not be read as unbiased estimates of a single underlying “true” population.

Overlap was nonetheless substantial: 230 papers (88.1% of the Scopus list) appeared in both databases, including 88 matched by DOI and 142 by title similarity. Thirty-one records were unique to Scopus (11.9% of Scopus rows) and 763 were unique to Google Scholar (76.8% of Google Scholar rows). Journal-type and language splits show how each export encodes metadata. Citation statistics reflect each database’s own citation field, not a hand-checked citation audit across the open literature.

The Google Scholar export was limited by Publish or Perish’s retrieval cap (1,000 results), so its list size reflects that ceiling as well as the query. Language labels use a simple character-based heuristic and can misclassify some text. Journal categories are rule-based and coarse; they are not a formal measure of journal prestige.

### 4.2 Temporal and geographic patterns

Publication volume increased between multi-year bands: 925 papers in 2010–2015 versus 1,273 in 2020–2025 (+348, +37.6%). The most common primary country labels were the United States and Brazil; several other countries contributed large counts (see §3.2).

When each continent’s within-year percentage is averaged across all years, approximate mean shares are South America 18.6%, Asia 18.9%, Europe 30.0%, and North America 19.5%, with the rest in Other and Unknown as tabulated. Comparing the mean of annual percentages in 2010–2012 with 2023–2025, North America’s display share fell from 28.3% to 13.7% (−14.5 percentage points), while South America, Asia, and Europe rose (+3.0, +3.9, and +6.7 points). In that decomposition, North America’s mapped share declined as the other regions increased. The Unknown category averaged 5.0% per year and tended to rise in later years, so roughly 95% of papers received a non-unknown mapped continent on average.

These continent labels collapse broad biogeographic regions (for example, Palearctic work can appear under the Europe display column). Country can be missing or Not Specified for a minority of papers (on the order of ~6% in the automated report). All regional percentages reflect inferred study geography from the classifier, not a separate tabulation by author address alone.

### 4.3 Thematic evolution

With one primary LLM-assigned theme per paper, Ecology/Behavior (35.4%), Taxonomy/Systematics (31.5%), and Biomonitoring/Water Quality (20.9%) accounted for most of the corpus; every other theme was under 5%. Regionally, Biomonitoring/Water Quality was most common in the Afrotropical subset (48.9%). Ecology/Behavior led in Australasian (53.8%) and Nearctic (43.6%) subsets. Taxonomy/Systematics was the modal theme in East Palearctic, Neotropical, and Oriental regions (about 43–45% in each).

Comparing pooled 2010–2015 with pooled 2021–2025, the share of Biomonitoring/Water Quality rose by about +4.5 percentage points and Ecology/Behavior fell by about −7.2 points (using the project script’s >3 point threshold for flagging change). Those values are shares of papers within each multi-year band, not averages of the within-year percentages in §3.3.

Single-theme coding forces interdisciplinary work into one bucket and hides co-occurring topics. Not Specified themes (3.1% overall) also appear as Unknown in the year-by-year table; that table’s columns follow the script’s rule (top themes by mean within-year share), so infrequent themes do not receive their own columns even when present in the raw data.

### 4.4 Authorship and collaboration

Mean authors per paper was 4.05 (median 3). Mean team size increased between pooled 2010–2015 and 2020–2025 (3.39 vs. 4.53 authors, +1.14); this is descriptive only and not a formal time-series model.

Applied-themed papers (§3.4) averaged more authors (4.74) than taxonomic papers (3.22), a gap of 1.53 authors on average, consistent with larger teams in the applied-theme bucket under this grouping. Another 1,277 other-theme papers contribute to the overall and yearly tables.

In total, 70.6% of papers had three or more authors. Bin-level contrasts (for example, more papers with six to ten authors in applied than in taxonomic groups) summarize how collaboration size differs by theme split.

International labels are noisy: only 2.3% of papers were classified as International overall, and 79.2% were Unknown under the heuristic. Taxonomic papers had a slightly higher International share (3.1%) than applied papers (1.2%), a −1.9 point gap relative to a naive expectation that applied work would be more international. With so much missing signal, between-type gaps should be read as how often the keyword rules fired, not as proof of cross-border collaboration rates.

Study-type comparisons also depend on Research_Theme accuracy. “Applied” here means only papers whose primary coded theme falls in the four applied categories, not every applied study on caddisflies. Dropping zero-author rows yields a slightly smaller authorship sample (2,845) than the geographic and thematic corpus (2,870).
