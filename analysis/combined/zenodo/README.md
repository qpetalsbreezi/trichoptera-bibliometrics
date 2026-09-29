# Supporting data and code

This deposit accompanies the manuscript *Spatial and Temporal Diversity in the Research Literature on Selected Groups of Aquatic Insects*. Deposit DOI: [10.5281/zenodo.22847390](https://doi.org/10.5281/zenodo.22847390).

## Files

**scopus_records_labeled.csv**: Deduplicated Article/Review records for five insect groups, 2010-2025 (n = 63,757). Identifiers and derived labels only.

**scopus_search_strings.txt**: Five executable Scopus queries (`TITLE-ABS-KEY` + `PUBYEAR`), as run on 12 April 2026.

**retrieval_scripts.zip**: Scopus retrieval and OpenAlex enrichment scripts.

**classification_prompts.txt**: Full prompts for both models, with settings.

**analysis_scripts.zip**: Classification, validation, agreement, and figure scripts.

**validation_agreement_full_table.csv**: Agreement % and Cohen's kappa for GPT-4o-mini, Gemini 3.1 Pro, and a human coder (validation subsample n = 1,376).

## Columns in `scopus_records_labeled.csv`

| Column | Contents |
|---|---|
| `DOI` | Digital Object Identifier (empty if missing). |
| `Scopus_EID` | Scopus Electronic Identifier. |
| `publication_year` | Publication year (2010-2025). |
| `insect_group` | Target group: Culicidae, Ephemeroptera, Plecoptera, Trichoptera, or Odonata. |
| `country` | Primary study country assigned by the LLM workflow (empty if not determined). |
| `biogeographic_region` | Nearctic, Neotropical, Oriental, Palearctic, East Palearctic, Afrotropical, Australasian, Global, or Not Specified. |
| `research_theme` | Taxonomy/Systematics; Ecology/Behavior; Biomonitoring/Water Quality; Evolution/Phylogeny; Conservation; Physiology; Vector Management; Other. (Workflow label Applied Ecology is deposited as Vector Management.) |
| `group_relevance` | Primary focus; Secondary mention; or Not target-taxon-focused. Group-focused analyses use Primary focus or Secondary mention (n = 47,824). |

Records are identifiers plus derived labels only. Elsevier's terms restrict redistributing raw Scopus bibliographic content. Full records can be reconstructed from the DOIs using `scopus_search_strings.txt` and Scopus access.

## Software versions

Python 3.9.6; pandas 2.3.3; requests 2.32.5; tqdm 4.70.0; openai 2.48.0; google-genai 1.47.0; numpy 2.0.2; matplotlib 3.9.4.
