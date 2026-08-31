# Human vs LLM evaluation (n=1,376)

Human labels: `human_labeled_1376_20260830.csv`  
Model labels: `../validation_1376_no_peripheral_20260819/`

Join key: `coding_row_id` = `id` in `all_rows_keys.csv` (not DOI alone).

```bash
# Labeling progress
python scripts/process/llm_validation_human_eval.py --status

# Build human_llm.csv (prints κ tables to terminal)
python scripts/process/llm_validation_human_eval.py \
  --human analysis/combined/llm_validation/human_blind_coding/human_labeled_1376_20260830.csv
```

Output: `../human_eval/human_llm.csv` only.
