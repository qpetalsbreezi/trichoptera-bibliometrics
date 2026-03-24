# 1. Introduction

Trichoptera (caddisflies) are ecologically important aquatic insects. Their larvae are key components of stream food webs and are widely used in water-quality monitoring and conservation. Research on this group spans many areas, including taxonomy, ecology, evolution, water quality, and applied studies. However, few studies synthesize the broader picture: how publications are distributed over time and across regions, which topics dominate, how authors collaborate, and how different databases compare.

This study uses bibliometrics, the statistical analysis of scientific publications, to map the Trichoptera literature. The study addresses four questions: (1) How complete and comparable are different databases? (2) How has publication output changed over time and across regions? (3) How have research themes shifted? (4) What are the patterns of authorship and international collaboration, including differences between applied work and taxonomy? The aim is to provide a quantitative, reproducible synthesis of the indexed literature, rather than a narrative literature review based on a curated subset of studies.

The paper is organized in numbered sections. Section 1 is the introduction. Section 2 describes the methods (subsections 2.1–2.7). The Results and Discussion sections will present findings and interpretation in the full manuscript.

## 2. Methods

### 2.1 Overview

The Methods section follows the workflow used to construct and analyze the bibliographic dataset. First, records were collected from the Scopus database, and a smaller dataset was collected from Google Scholar for database comparison. Second, records were cleaned and duplicates were removed. Third, missing metadata such as abstracts, author lists, and affiliations were added from other bibliographic sources. Fourth, each paper was classified into predefined thematic and geographic categories using a large language model. Finally, the resulting dataset was analyzed to address the research questions. The subsections below describe each stage of this process; subsection 2.7 summarizes software, credentials, and reproducibility.

### 2.2 Data collection

#### 2.2.1 Scopus retrieval

Relevant publications were retrieved from the Scopus database using Elsevier’s Scopus Search API. The search was limited to records in which Trichoptera or common English names for caddisflies (e.g., caddisfly, caddisflies) appeared in the title, abstract, or keywords. To obtain complete coverage for each calendar year, the search period was sometimes divided into months or quarters and records were retrieved sequentially. The default metadata supplied by this interface includes core bibliographic information such as title, journal, publication year, DOI, citation count, document type, and pagination fields. However, full abstracts and complete author lists were often missing, so this information was obtained from other sources as described below.

#### 2.2.2 Google Scholar

For the research question on database coverage, records from Scopus were compared with Trichoptera-related results from Google Scholar for one publication year, matched to the same calendar year. Google Scholar results were retrieved using the Publish or Perish software, whereas Scopus records were retrieved through the API described above. The two sources therefore differ in how queries are defined and in how complete the returned lists are. The comparison reports overlap between the two lists, records unique to each database, and basic citation statistics, without assuming equivalent search logic or coverage.

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

#### 2.6.2 RQ1: Database overlap, citations, language, and journal type

For the 2023 comparison year, records in Scopus and Google Scholar were paired where possible by matching DOIs, and when DOIs were missing, by high similarity between titles so that minor formatting differences still identified the same article. We reported how many records appeared in both databases, how many were unique to each database, and summarized citation-related measures for comparison. Language of the title and abstract was described using a simple rule based on character patterns, for example non-Latin scripts or common diacritics associated with non-English text. This is a rough indicator only and not full language identification. Journals were classified from their titles as regional when the name suggested a national or regional scope. When no such cue appeared but a journal name was present, the outlet was treated as international or general. Names that were missing or unusable were set aside separately. This rule avoids treating most major journals as “unknown” solely because the title does not name a country.

#### 2.6.3 RQ2: Temporal and geographic variation

Publication volume and geographic patterns were summarized over time using the filtered, coded dataset. Country labels were harmonized to standard forms, and when multiple countries were present in a single record, one primary study location was retained for consistency. The main temporal summaries were reported at the regional level by year, with country-level summaries used to describe overall geographic concentration across the study period. These results reflect locations inferred during classification rather than an independent geocoding workflow.

#### 2.6.4 RQ3: Thematic evolution

Theme frequencies were taken from the classifier’s research theme variable. Themes were summarized across three multi-year bands (early, middle, and recent within 2010–2025) and also year by year, so that short-term fluctuations and longer trends could both be observed.

#### 2.6.5 RQ4: Authorship, collaboration size, and international collaboration

Authorship and collaboration were analyzed using enriched author metadata, with records grouped into predefined collaboration-size categories. For thematic comparisons, papers were collapsed into applied, taxonomic, and other groups. International collaboration was estimated from affiliation text using a conservative heuristic: papers were treated as international when affiliation information indicated contributors from more than one country, national when evidence suggested a single country, and unclear when affiliation evidence was insufficient. Collaboration rates were compared across thematic groups, and team size was summarized over time. Team size and internationality were interpreted as related but distinct dimensions of collaboration.

### 2.7 Software, credentials, and reproducibility

The steps above were implemented in Python using common tools for data tables, web requests, and the OpenAI service. Access credentials for Scopus and OpenAI were stored locally and not distributed with any public code archive. Reproducibility rests on the documented coding scheme, prompts, and analysis rules rather than on sharing full intermediate files, which can be large; such files may be archived outside the code repository when needed.
