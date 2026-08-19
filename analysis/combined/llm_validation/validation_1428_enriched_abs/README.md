# Enriched-abstract validation (n=1,376)

Article / Review only (52 Conference Papers removed; listed in `eval_excluded_conference_papers.csv`). Abstracts from DOI-waterfall + Europe PMC + Elsevier Article Retrieval + Springer Nature Meta API.

Models: GPT-4o-mini vs Gemini 3.1 Pro (`gemini-3.1-pro-preview`).

Abstracts: **96.7%** (1,331 / 1,376). Gate conflicts for human review: `gate_conflicts_for_human_review.csv` (110 rows; 96 with abstract, 14 with DOI only / empty Abstract). Guide: `GATE_REVIEW_README.md`.

Latest dual-model agreement (Article / Review only):

| Metric | Agreement | κ |
|--------|----------:|--:|
| Keep vs drop | 92.0% | 0.63 |
| Taxon focus label | 82.3% | 0.73 |
| Research theme | 77.8% | 0.74 |
| Focus label and theme both | 66.6% | — |
