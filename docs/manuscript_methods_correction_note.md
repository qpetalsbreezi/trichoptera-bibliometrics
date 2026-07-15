# Manuscript Methods Correction Note

**Purpose:** Handoff for an agent revising the plain-language Methods section of the aquatic-insect bibliometrics manuscript.  
**Repo:** `/Users/pramod/Desktop/Saanvi/Projects/trichoptera-bibliometrics`  
**Source manuscript draft reviewed:** User-provided plain-language Methods (sections 2.1–2.5), structurally aligned with `docs/paper.md` but rewritten for accessibility.  
**Verification basis:** Python pipeline scripts, `config/queries.json`, `data/taxon_schema.json`, and generated analysis outputs under `analysis/combined/`.

---

## Executive summary

The draft Methods are **mostly accurate** on study design, Scopus retrieval, LLM classification, inclusion filters, time windows, and software stack. Six corrections are **required** because they misstate what the code does. Several **optional clarifications** would improve precision and align the prose with limitations already implicit in the codebase.

**Do not add** a Scopus vs Google Scholar / database-overlap subsection. The author confirmed that database comparison was **intentionally removed** from this manuscript version. Do not reintroduce RQ1 methods or results unless explicitly requested later.

---

## Required corrections (priority order)

### 1. Fix Table 2 — continental aggregation from biogeographic regions

**Problem in draft:** Table 2 says biogeographic regions collapse as “Nearctic and Neotropical → North and South America; Palearctic and East Palearctic → Europe and Asia.” That incorrectly implies **both** Palearctic and East Palearctic map to **both** Europe and Asia.

**What the code does:** Identical logic in `scripts/analyze/analyze_rq2_temporal_geographic.py` (lines ~121–136) and `scripts/analyze/analyze_cross_taxa_summary.py` (`categorize_region`, lines ~111–125):

| Continental bucket (`Region_Category`) | Source biogeographic regions (`Region_Global`) |
| --- | --- |
| **South America** | Neotropical |
| **North America** | Nearctic |
| **Europe** | Palearctic |
| **Asia** | Oriental, East Palearctic |
| **Other** | Afrotropical, Australasian, Global |
| **Unknown** | Not Specified (and unmapped values) |

**Action:** Replace Table 2 “Built from” text with the table above. Add **Other** and **Unknown** as analysis categories (they appear in RQ2 outputs and figures). Clarify that East Palearctic maps to **Asia only**, not Europe.

**Suggested Table 2 replacement (continental row):**

> **Continental scale:** South America, Asia, Europe, North America, Other, Unknown — built from biogeographic regions as follows: Neotropical → South America; Nearctic → North America; Palearctic → Europe; Oriental and East Palearctic → Asia; Afrotropical, Australasian, and Global → Other; Not Specified → Unknown.

---

### 2. Fix title deduplication wording (Section 2.2)

**Problem in draft:** “Among any records that remained, papers sharing the same title were merged into a single entry.”

**What the code does:** `scripts/process/combine_scopus_api_years.py` (lines ~69–85):

1. Drop duplicate **DOIs** (keep first).
2. **Only among rows with missing/empty DOI**, drop duplicate normalized titles (lowercase, trimmed; keep first).

Title deduplication does **not** run on records that already have a DOI.

**Action:** Replace with wording such as:

> After DOI deduplication, records that still lacked a DOI were deduplicated again by normalized title (lowercase, trimmed whitespace; first occurrence retained). Records with a DOI were not second-pass deduplicated by title.

---

### 3. Fix PubMed abstract retrieval (Section 2.2)

**Problem in draft:** “…Crossref, and then PubMed **for medical and biological papers**…”

**What the code does:** `scripts/fetch/fetch_abstracts.py` (lines ~247–277) tries PubMed for **every** record still missing an abstract after OpenAlex, Semantic Scholar, and CrossRef. There is **no** biomedical eligibility filter.

**Note:** `docs/paper.md` also says “for biomedical items”; that line is likewise imprecise relative to the code. Correct both if editing the repo manuscript.

**Action:** Replace with:

> …CrossRef, and then PubMed; the first abstract returned was used.

Optional nuance: PubMed is often empty for non-biomedical records, but the attempt is universal.

---

### 4. Fix international-collaboration fallback (Section 2.4.2)

**Problem in draft:** The three-step hierarchy reads as if all steps infer **author** international collaboration. Step 3 (“fall back on the broad world region: Global → International; any other named region → National”) is **misleading**.

**What the code does:** `scripts/analyze/analyze_rq4_collaboration.py` (`detect_international_collab`, lines ~264–321) and `scripts/analyze/analyze_cross_taxa_summary.py` (`detect_international_collab`, lines ~202–250):

1. **Primary:** Multiple vs single ISO-3166 alpha-2 codes in `Author_Country_Codes` (from OpenAlex).
2. **Secondary:** Keyword search for country names in `Author_Affiliations` (limited keyword list: USA, UK, Germany, France, Brazil, China, Japan, Australia, Canada, Italy, Spain).
3. **Fallback:** Uses LLM-assigned **`Region_Global`** — the **inferred study location**, not author countries:
   - Any named region except `Global` and `Not Specified` → classified **National**
   - `Global` → **International**
   - `Not Specified` or missing → **Unknown**

This fallback conflates **where the research was done** with **whether authors collaborated across countries**. It is a known approximation; RQ4 text reports note high uncertainty when affiliation data are weak.

**Action:** Revise Section 2.4.2 to:

1. Keep steps 1–2 as author-based signals.
2. Reframe step 3 explicitly as a **last-resort proxy** using study-region labels, not author geography.
3. Add a limitation sentence, e.g.:

> When author country codes and affiliation text were insufficient, we used the assigned study biogeographic region as an approximate proxy; this step does not directly measure author nationality and should be interpreted cautiously.

**Optional (recommended for Results/Discussion, not strictly Methods):** Cross-taxa metrics distinguish `intl_collab_pct_overall` (Unknown included in denominator) from `intl_collab_pct_known_only_overall` (International share among papers classified National or International only). See `analysis/combined/cross_taxa_metrics.csv` columns `intl_collab_pct_overall`, `intl_collab_info_coverage_pct`, `intl_collab_pct_known_only_overall`.

---

### 5. Fix Section 2.4 opening — separate per-taxon analyses

**Problem in draft:** “All of them used the same set of papers: those published between 2010 and 2025 that were about **at least one of the five target groups**.”

**What the code does:** Each `query_id` in `config/queries.json` (`mosquitoes`, `ephemeroptera`, `plecoptera`, `trichoptera`, `odonata`) has its **own** Scopus corpus, processed files, coded CSV (`data/processed/<query_id>/scopus_api_coded.csv`), and analysis outputs (`analysis/<query_id>/`). Cross-taxa tables (`scripts/analyze/analyze_cross_taxa_summary.py`, `analysis/combined/`) **compare metrics side by side**; they do **not** merge records into one pooled dataset.

**Action:** Replace with wording such as:

> Analyses of publication volume, geography, themes, and collaboration used taxon-focused Scopus papers from 2010–2025 **for each group separately**, with identical rules applied in every group. Combined tables and figures place the five groups side by side; records were not merged across groups.

---

### 6. Move or broaden the “no statistical tests” disclaimer

**Problem in draft:** The disclaimer appears only at the end of Section 2.4.2 (authorship), but early-vs-recent comparisons for **geography and themes** are in Section 2.4.1 and use the same descriptive window comparison (2010–2015 vs 2020–2025, percentage-point differences).

**What the code does:** No inferential tests in RQ2–RQ4 scripts. `analyze_cross_taxa_summary.py` defines `EARLY_WINDOW = (2010, 2015)`, `RECENT_WINDOW = (2020, 2025)` with 2016–2019 as an implicit buffer for theme-shift deltas only.

**Action:** Place one consolidated disclaimer at the **start of Section 2.4** (or end of 2.4 covering both 2.4.1 and 2.4.2):

> Early-versus-recent comparisons report descriptive differences between fixed multi-year windows (2010–2015 vs 2020–2025). We did not apply formal inferential trend tests.

Remove the duplicate or vague closing sentence in 2.4.2 that mentions “each group’s share of papers” without specifying geographic, thematic, or authorship context.

---

## Optional clarifications (recommended, not strictly errors)

### A. Scopus search field scope (Section 2.1)

Draft does not state where terms were searched. `config/queries.json` uses Elsevier syntax `TITLE-ABS-KEY(...)`, i.e. **title, abstract, and keywords**.

**Suggested addition:** “Search terms were matched in title, abstract, and keywords (Scopus `TITLE-ABS-KEY` syntax).”

### B. LLM geographic extraction priority (Section 2.3)

Draft lists: (1) locations in title/abstract, (2) other geographic clues, (3) author institutions.

**Code prompt** (`scripts/process/llm_code_taxon.py`, lines ~89–94, ~118–127) also includes:

- Study sites named in abstract (rivers, lakes, regions)
- **Species-name geographic indicators** (e.g., *japonica* → Japan, *sinensis* → China) **before** affiliations

**Suggested addition:** Insert species-name inference between text clues and affiliations if describing the prompt faithfully.

### C. Author affiliations supplied to the LLM (Section 2.3)

Draft: “author institutions” always provided.

**Code:** `llm_code_taxon.py` reads `paths.with_abstracts` and optionally merges `Author_Affiliations` from `paths.with_authors` **on Title only** (lines ~227–238). If the authors file is missing, classification is abstract/title-only with a warning.

**Suggested nuance:** “When available, author affiliations from OpenAlex were included; otherwise classification relied on title and abstract only.”

### D. Relevance label canonical name (Section 2.3 / Table 1)

Draft uses “Not group-focused.” Schema and code use **`Not target-taxon-focused`** (`data/taxon_schema.json`, filter sets in all RQ2–RQ4 scripts: `NOT_FOCUS_LABELS = {"Not target-taxon-focused", "Not Trichoptera-focused"}`).

Plain language “Not group-focused” is fine in prose **if** you note it corresponds to the excluded category. Peripheral, Secondary mention, and Primary focus **are retained** in analysis — draft is correct on that.

### E. Scopus standard-view limitations (Section 2.1 or 2.2)

`fetch_scopus_api.py` header comments: standard API view does not include abstracts and often returns **first author only** in metadata. The pipeline compensates via OpenAlex/Semantic Scholar/CrossRef/PubMed and `fetch_authors.py`. One sentence acknowledging initial Scopus metadata gaps would strengthen Methods transparency.

### F. Author-count exclusions “fewer than 2%” (Section 2.4.2)

This claim appears in `docs/paper.md` but is **not recomputed** in generated glossary outputs. RQ4/cross-taxa code excludes papers with `Author_Count_Actual` missing or zero after OpenAlex merge (`author_count > 0` filter).

**Action:** Either verify the percentage from processed data before keeping “2%”, or soften to: “papers without an OpenAlex author count were excluded from author-statistics rows.”

### G. Theme ranking excludes “Not Specified” (Section 2.4.1)

Draft states this correctly. Code: theme rank summaries and cross-taxa top-theme rows exclude `Not Specified`; those papers remain in denominators for other theme counts unless otherwise noted. RQ3 reports note ~3–33% Not Specified depending on taxon (`theme_not_specified_pct` in `analysis/combined/cross_taxa_metrics.csv`).

### H. Materials Science (Silk) theme (Table 1)

Draft correctly notes this is effectively Trichoptera-specific. Prompt instructs: use only when silk properties are studied (`llm_code_taxon.py`). Category exists in schema for all taxa but is rarely applicable outside caddisflies.

---

## Sections that are accurate — do not change unless stylistic edit

| Topic | Verified against |
| --- | --- |
| Five taxa, separate workflows, no cross-taxon record merge | `scripts/lib/pipeline.py`, `config/queries.json` |
| Search term lists (common + scientific names) | `config/queries.json` |
| Scopus Search API, 2010–2025, year-by-year fetch, month/quarter splits for large years | `scripts/fetch/fetch_scopus_api.py` |
| May 2026 retrieval; 2025 undercount caveat | `docs/paper.md` (prose); raw CSV timestamps ~2026-01 |
| Metadata fields from Scopus (title, journal, year, DOI, citations, document type) | `fetch_scopus_api.py` |
| Abstract enrichment order: OpenAlex → Semantic Scholar → CrossRef → PubMed | `fetch_abstracts.py` |
| OpenAlex for author lists, counts, affiliations | `fetch_authors.py` |
| GPT-4o-mini, temperature 0, shared schema, taxon named in prompt | `llm_code_taxon.py` |
| Four LLM fields: Country, biogeographic region, research theme, group relevance | `data/taxon_schema.json` |
| Table 1 allowed values for region, theme, relevance | `data/taxon_schema.json` |
| Theme groups for collaboration: Applied / Taxonomic / Other | `analyze_rq4_collaboration.py`, `analyze_cross_taxa_summary.py` |
| Inclusion: Primary focus, Secondary mention, Peripheral retained; not-group-focused excluded | All RQ2–RQ4 analyze scripts |
| Geographic summaries use LLM-inferred study location, not geocoded addresses | RQ2 + LLM prompt |
| Early window 2010–2015, recent window 2020–2025 for volume, geography, themes | `analyze_cross_taxa_summary.py` |
| Continental shift = mean of yearly continental shares, early vs recent, pp change | RQ2 + cross-taxa |
| Mean/median authors from OpenAlex; applied vs taxonomic comparisons | RQ4 + cross-taxa |
| Python, pandas, API clients for Scopus, OpenAlex, OpenAI | `requirements.txt`, scripts |
| Code/query/prompt availability on request | Consistent with repo practice |

---

## Explicit non-actions

1. **Do not** restore Scopus vs Google Scholar methods (RQ1). Author removed this from the current manuscript version. Existing RQ1 code and `analysis/combined/overall_bibliometric_report.md` RQ1 table remain in repo but are out of scope for this Methods revision.

2. **Do not** change inclusion logic to Primary-focus-only unless the author requests it. Code intentionally keeps Secondary mention and Peripheral.

3. **Do not** cite Holt et al. (2013) as implemented in code unless adding a conceptual citation only. Region names appear in schema/prompt; the pipeline does not programmatically enforce Holt boundaries.

---

## File map for verifying edits

| Manuscript topic | Primary code files |
| --- | --- |
| Query definitions | `config/queries.json` |
| Scopus fetch | `scripts/fetch/fetch_scopus_api.py` |
| Combine + dedupe | `scripts/process/combine_scopus_api_years.py` |
| Abstracts | `scripts/fetch/fetch_abstracts.py` |
| Authors | `scripts/fetch/fetch_authors.py` |
| LLM coding | `scripts/process/llm_code_taxon.py`, `data/taxon_schema.json` |
| RQ2 geography/temporal | `scripts/analyze/analyze_rq2_temporal_geographic.py` |
| RQ3 themes | `scripts/analyze/analyze_rq3_thematic_evolution.py` |
| RQ4 collaboration | `scripts/analyze/analyze_rq4_collaboration.py` |
| Cross-taxa metrics (canonical for paper tables) | `scripts/analyze/analyze_cross_taxa_summary.py`, `analysis/combined/cross_taxa_metrics.csv` |
| Manuscript source in repo | `docs/paper.md` |
| User plain-language draft | Provided in chat (not yet a repo file) |

---

## Suggested edit checklist for correcting agent

- [ ] Rewrite Table 2 continental mapping (Required #1)
- [ ] Fix title dedup sentence in 2.2 (Required #2)
- [ ] Fix PubMed sentence in 2.2 (Required #3)
- [ ] Revise international-collaboration fallback + add limitation in 2.4.2 (Required #4)
- [ ] Fix 2.4 intro “at least one group” phrasing (Required #5)
- [ ] Consolidate no-statistical-tests disclaimer for 2.4.1 + 2.4.2 (Required #6)
- [ ] Optionally add TITLE-ABS-KEY scope, species-name inference, affiliation availability, Scopus metadata limits
- [ ] Verify or soften “<2%” author exclusion claim
- [ ] Sync matching fixes into `docs/paper.md` where the same imprecise wording exists (PubMed, Table 2 if present, dedup, collaboration fallback)
- [ ] Do **not** add Google Scholar / database overlap methods

---

## Copy-ready replacement snippets

### Section 2.2 — deduplication (replace existing paragraph clause)

> The yearly Scopus exports were combined into a single list and deduplicated in two passes. First, each DOI was kept only once. Second, among records that still had no DOI, duplicate titles were removed after normalizing titles to lowercase and trimming whitespace; the first occurrence was retained. Each record kept its publication year for temporal analysis and downstream linking.

### Section 2.2 — abstracts (replace PubMed clause)

> When a DOI was available, missing abstracts were retrieved in fixed order from OpenAlex, Semantic Scholar, CrossRef, and then PubMed; the first abstract returned was used.

### Section 2.4 — opening (replace misleading sentence)

> The analyses below address publication volume, geography, research themes, and collaboration. Unless stated otherwise, they used taxon-focused Scopus papers from 2010–2025 **within each group’s separate dataset** (Primary focus, Secondary mention, and Peripheral retained; Not group-focused excluded). The same rules were applied to every group; combined tables and figures compare the five groups side by side without merging records across groups. Early-versus-recent comparisons are descriptive only (2010–2015 vs 2020–2025); we did not apply formal inferential trend tests.

### Section 2.4.2 — collaboration fallback (replace step 3)

> Third, when author country information was still unavailable, we used the paper’s assigned study biogeographic region as an approximate proxy: studies labeled Global were treated as international, other named regions as national, and studies with no region as unknown. Because this fallback reflects inferred study location rather than author countries, collaboration labels based on it should be interpreted cautiously.

---

*Generated from codebase review, 2026-06-29. Database comparison (RQ1) intentionally omitted per author instruction.*
