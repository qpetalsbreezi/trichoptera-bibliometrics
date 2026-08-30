# Human vs LLM evaluation (n=1,376)

Frozen dual-model sample: `../validation_1376_no_peripheral_20260819/`  
Blind coding sheet: `human_blind_coding_1376.csv`  
Prompt for validators: `PROMPT_WORDING_for_validators.txt`

## Join key

`coding_row_id` in the human file **=** `id` in `all_rows_keys.csv` (1…1376).  
Do **not** join only on DOI or `row_key` — some EIDs/DOIs appear in more than one taxon sample.

## When labels arrive

1. Put the completed sheet here (CSV or Excel), same columns:
   - `coding_row_id`
   - `taxon_focus` — Primary focus | Secondary mention | Not target-taxon-focused
   - `research_theme` — schema themes only
2. Check progress:

```bash
python scripts/process/llm_validation_human_eval.py --status
```

3. Run full evaluation (all 1,376 labeled):

```bash
python scripts/process/llm_validation_human_eval.py
```

4. Or score a partial return:

```bash
python scripts/process/llm_validation_human_eval.py --allow-partial \
  --human analysis/combined/llm_validation/human_blind_coding/YOUR_FILE.csv
```

## Four comparisons

| # | Comparison | Notes |
|---|---|---|
| 1 | GPT-4o-mini vs human | Production model |
| 2 | Gemini Pro vs human | Second model |
| 3 | GPT∩Gemini agree vs human | Rows where both models match on focus **and** theme |
| 4 | Gemini vs GPT | Recomputed on the labeled subset (full-sample numbers already exist) |

Also written: **GPT∩Gemini gate-agree vs human** (KEEP/DROP only; larger subset).

## Output table format (same as email)

| Metric | Agreement | κ |
|---|---:|---:|
| Keep vs drop | … | … |
| Taxon focus label | … | … |
| Research theme | … | … |
| Focus label and theme both | … | — |

Outputs go to `../human_eval/`:

- `agreement_summary.md` — paste-ready tables
- `agreement_metrics.csv`
- `disagreements_vs_human.csv`
- `joined_labeled.csv`
- confusion matrices (focus/theme × GPT/Gemini)

## Notes

- Human labels are the reference for comparisons 1–3.
- Incomplete rows (blank focus or theme) are skipped when `--allow-partial` is set.
- Common aliases are normalized (e.g. “Not group-focused” → Not target-taxon-focused).
