# Human review — No-peripheral (n=1,376)

Eval: Article / Review only (n=1,376). Prompt: **No-peripheral** (3 taxon-focus tiers: Primary focus, Secondary mention, Not target-taxon-focused).

Review file: `all_rows_for_human_review.csv`

Columns: `id`, `taxon`, `Title`, `DOI`, `Abstract`, `GPT_keep_vs_drop`, `Gemini_keep_vs_drop`, `GPT_taxon_focus`, `Gemini_taxon_focus`, `GPT_research_theme`, `Gemini_research_theme`, `your_decision`

Fill `your_decision`: **KEEP** or **DROP** for that `taxon`.

KEEP = Primary focus or Secondary mention (taxon is part of the study). DROP = Not target-taxon-focused (absent, incidental, or name-only).

If Abstract is `(no abstract available)` (45 rows), open the clickable `DOI` (`https://doi.org/...`) or use Title alone.

| | n |
|--|--:|
| All rows | 1,376 |
| With abstract | 1,331 |
| Title only | 45 |
| GPT vs Gemini gate conflict | 122 |

`all_rows_keys.csv` is for analysis only (do not send to reviewers).
