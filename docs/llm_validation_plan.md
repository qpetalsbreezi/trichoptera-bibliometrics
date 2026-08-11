# LLM validation plan (short)

**Goal.** Check labeling consistency without hand-labeling 2,000 papers.  
**Status.** Planned — not yet run.

## Design

| Item | Choice |
|------|--------|
| Model A | GPT-4o-mini (existing coded files) |
| Model B | Gemini 3.5 Flash (`gemini-3.5-flash`) |
| Prompt / schema | Same as production (`llm_code_taxon.py`, `taxon_schema.json`) |
| Sample | 300 articles per group (1,500 total), stratified by year band (2010–2015 / 2016–2019 / 2020–2025) and abstract available vs not when possible |
| Fields compared | `Taxon_Relevance`, `Research_Theme` |
| Human review | Spreadsheet of **A vs B disagreements** only (plus a small agree-set spot-check); coauthors adjudicate |

## Steps

1. Draw stratified random sample IDs from each group’s coded file; freeze the sample list.
2. Re-code the sample with Gemini using the same prompt and allowed values.
3. Join A and B labels; compute agreement % and Cohen’s κ per field (overall and by group).
4. Export disagreement rows to a review spreadsheet (title, abstract snippet, A label, B label, blank human column).
5. After adjudication, summarize agreement, disagreement patterns, and with/without-abstract splits.
6. Add a short Validation subsection + table to the manuscript; keep Limitations honest (two LLMs can share errors).

## Deliverables

- `analysis/combined/llm_validation/` — sample list, Gemini outputs, agreement metrics, disagreement CSV  
- Methods note + small results table in the manuscript  

## Out of scope (for now)

- Full 2,000-article manual gold set  
- Two-stage exclude-then-label redesign  
- Re-running all ~71k records on Gemini (sample first; expand only if needed)
