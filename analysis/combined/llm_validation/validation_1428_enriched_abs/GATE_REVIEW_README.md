# Gate conflict review (110 papers)

Eval: Article / Review only (n=1,376), after dropping 52 Conference Papers. Full recode after abstract enrichment (96.7% abstracts).

Review file: `gate_conflicts_for_human_review.csv`

Columns: `id`, `taxon`, `Title`, `DOI`, `Abstract`, `GPT_keep_vs_drop`, `Gemini_keep_vs_drop`, `GPT_taxon_focus`, `Gemini_taxon_focus`, `GPT_research_theme`, `Gemini_research_theme`, `your_decision`

Fill `your_decision`: **KEEP** or **DROP** for that `taxon`.

KEEP = that taxon is part of the study. DROP = not (or name-only).

If Abstract is empty (14 rows), open the clickable `DOI` (`https://doi.org/...`) and decide from the page. Otherwise use Title + Abstract.

`gate_conflicts_keys.csv` is for analysis only (do not send to reviewers).
