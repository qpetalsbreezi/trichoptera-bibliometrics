# Gate conflict review (118 papers)

Eval: Article / Review / Conference Paper only (n=1,428), full recode after abstract enrichment (96.4% abstracts).

Review file: `gate_conflicts_for_human_review.csv`

Columns: `id`, `taxon`, `Title`, `Abstract`, `your_decision`

Fill `your_decision`: **KEEP** or **DROP** for that `taxon`.

KEEP = that taxon is part of the study. DROP = not (or name-only).

If Abstract is a `https://doi.org/...` link (15 rows), open the DOI and decide from the page. Otherwise use Title + Abstract.

`gate_conflicts_keys.csv` is for analysis only (do not send to reviewers).
