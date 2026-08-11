# LLM validation plan (short)

**Goal.** Check labeling consistency without hand-labeling ~2,000 papers.  
**Status.** Complete for the frozen n=1,500 sample (prompt_v2).

## Design

| Item | Choice |
|------|--------|
| Model A | GPT-4o-mini (production coder) |
| Model B | Gemini 3.1 Pro (`gemini-3.1-pro-preview`) |
| Prompt / schema | Shared `llm_code_taxon.build_prompt` + `taxon_schema.json` (Taxon_Relevance has no `Not Specified`) |
| Sample | Frozen `sample_manifest.csv`: 300×5 groups = 1,500, stratified by year band × abstract available |
| Fields compared | `Taxon_Relevance`, `Research_Theme` (+ binary in-set gate) |
| Human review | Optional: adjudicate gate disagreements / theme neighbor fights in `disagreements.csv` |

## Canonical results

| Path | Role |
|------|------|
| `analysis/combined/llm_validation/sample_manifest.csv` | Frozen sample IDs |
| `validation_1500_prompt_v2/` | Final dual-model re-code (mini + 3.1-pro, shared prompt) |

Headline agreement (n=1,500): relevance ~81%, theme ~76%, both ~65%, binary gate ~89%.  
(`before_after_vs_old_prompt.txt` in that folder records the pre–prompt-fix lift; intermediate pilots/baselines were not retained.)

## How to re-run

```bash
# Freeze sample (once; --force only to redraw)
python scripts/process/llm_validation_sample.py

# Dual-model coding (resumable). Example: full frozen sample with current prompt
python scripts/process/llm_validation_pilot100.py \
  --manifest analysis/combined/llm_validation/sample_manifest.csv \
  --out-subdir validation_1500_prompt_v2 \
  --gpt-model gpt-4o-mini \
  --gemini-model gemini-3.1-pro-preview \
  --threads 6

# Compare only (after both CSVs exist)
python scripts/process/llm_validation_pilot100.py \
  --manifest analysis/combined/llm_validation/sample_manifest.csv \
  --out-subdir validation_1500_prompt_v2 \
  --compare-only
```

Needs `OPENAI_API_KEY` and `GEMINI_API_KEY` in `.env`.

## Manuscript use

- Lead with **binary gate** agreement (inclusion filter).
- Report exact relevance / theme with κ; note residual fights are mostly adjacent tiers and Ecology ↔ Applied ↔ Biomonitoring.
- Keep Limitations honest: two LLMs can share errors; this is reproducibility, not human gold-standard accuracy.

## Out of scope (for now)

- Full-corpus re-code with the new prompt (~71k)
- Large manual gold set
