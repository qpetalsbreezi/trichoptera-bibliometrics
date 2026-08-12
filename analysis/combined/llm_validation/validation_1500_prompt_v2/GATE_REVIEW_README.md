# Gate conflict review (162 papers)

Eval filter: **Article / Review / Conference Paper** only (72 other document types dropped from the 1,500 sample).

Review file: `gate_conflicts_for_human_review.csv`

Columns: `id`, `taxon`, `Title`, `Abstract`, `your_decision`

Fill `your_decision`: KEEP or DROP for that `taxon`.

KEEP = that taxon is part of the study. DROP = not (or name-only).

If Abstract is blank, decide from Title alone.

`gate_conflicts_keys.csv` is for analysis only (do not send to reviewers).
Excluded types: `eval_excluded_non_article_review_conference.csv`.
