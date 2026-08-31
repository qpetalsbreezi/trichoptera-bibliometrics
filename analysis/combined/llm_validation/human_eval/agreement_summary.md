# Human vs LLM agreement

Human file: `analysis/combined/llm_validation/human_blind_coding/human_labeled_1376_20260830.csv`
Model run: `validation_1376_no_peripheral_20260819`
Labeled rows scored: **1376** / 1376

## GPT-4o-mini vs human

| Metric | Agreement | κ |
|---|---:|---:|
| Keep vs drop | 91.9% | 0.73 |
| Taxon focus label | 85.0% | 0.76 |
| Research theme | 74.6% | 0.69 |
| Focus label and theme both | 68.0% | — |

## Gemini Pro vs human

| Metric | Agreement | κ |
|---|---:|---:|
| Keep vs drop | 90.8% | 0.71 |
| Taxon focus label | 87.6% | 0.81 |
| Research theme | 73.0% | 0.68 |
| Focus label and theme both | 68.0% | — |

## GPT∩Gemini agree vs human

| Metric | Agreement | κ |
|---|---:|---:|
| Keep vs drop | 96.7% | 0.88 |
| Taxon focus label | 94.0% | 0.90 |
| Research theme | 86.2% | 0.83 |
| Focus label and theme both | 83.0% | — |

## GPT∩Gemini gate-agree vs human

| Metric | Agreement | κ |
|---|---:|---:|
| Keep vs drop | 95.4% | 0.84 |
| Taxon focus label | 89.2% | 0.82 |
| Research theme | 76.2% | 0.71 |
| Focus label and theme both | 70.7% | — |

## Gemini Pro vs GPT-4o-mini (labeled subset)

| Metric | Agreement | κ |
|---|---:|---:|
| Keep vs drop | 91.1% | 0.72 |
| Taxon focus label | 86.9% | 0.79 |
| Research theme | 76.2% | 0.71 |
| Focus label and theme both | 69.6% | — |
