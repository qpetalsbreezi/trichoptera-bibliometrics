# Glossary: rows in the multi-taxon bibliometric tables

Unless stated otherwise, the main tables use publication years **2010–2025** and only studies **focused on the target taxon** (studies marked as not focused on that taxon are excluded). **RQ1** is different: it reports a Scopus vs Google Scholar overlap comparison and is not drawn from that same analytic sample.

---

## Shared scope (RQ2–RQ4)

**Taxon-focused** counts include only studies that pass the relevance screen. **All coded papers** is the larger count before that screen (same year window).

---

## RQ1 — Database coverage

| Row title | Definition |
| --- | --- |
| **Benchmark year** | Publication year used for the side-by-side Scopus and Google Scholar comparison for that taxon. |
| **Scopus total** | Hits from the Scopus search at the benchmark year. |
| **Google Scholar total** | Hits from the Google Scholar search at the benchmark year. |
| **Overlap (both)** | How many records showed up in both lists. |
| **Overlap / Scopus (%)** | Overlap divided by Scopus total, times 100. (Skipped if Scopus total is zero.) |
| **GS/Scopus ratio** | Google Scholar count ÷ Scopus count for the same comparison. |

---

## RQ2 — Publication volume (2010–2025)

| Row title | Definition |
| --- | --- |
| **All coded papers (2010–2025)** | Papers in the dataset for those years before removing not-target-taxon work. |
| **Taxon-focused papers (2010–2025)** | Same window, relevance filter applied. |
| **Taxon-focused (2010–2015)** | Taxon-focused count for 2010 through 2015. |
| **Taxon-focused (2020–2025)** | Taxon-focused count for 2020 through 2025. |
| **Percent change 2010–15 vs 2020–25 (taxon-focused)** | \(\frac{N_{2020\text{–}25} - N_{2010\text{–}15}}{N_{2010\text{–}15}} \times 100\). If the early-period count is zero, the value is 0. |

### RQ2 — Year-by-year *N* (taxon-focused)

| Row title | Definition |
| --- | --- |
| **Year** | Publication year. |
| *(one column per taxon)* | Taxon-focused papers in that year. |

---

## RQ2 — Mean continental % (mean of yearly %, 2010–2025)

Each paper has a region-of-study label, rolled up to continents (South America, Asia, Europe, North America, plus unknown). Within a year we compute what fraction of taxon-focused papers fall in each bucket. The five “Mean … %” rows average those yearly percentages over 2010–2025—one row per bucket.

| Row title | Definition |
| --- | --- |
| **Mean South America %** through **Mean Unknown %** | For each label, the mean across years of that year’s percentage in that geographic bucket (see the paragraph above this table). |

---

## RQ2 — Continental % change (pp): mean 2010–2012 vs mean 2023–2025

We average the yearly continental percentages for 2010–2012 (“early”) and separately for 2023–2025 (“recent”), then subtract: recent − early, in percentage points.

| Row title | Definition |
| --- | --- |
| **Delta South America (pp)** | Change for South America. |
| **Delta Asia (pp)** | Change for Asia. |
| **Delta Europe (pp)** | Change for Europe. |
| **Delta North America (pp)** | Change for North America. |

---

## RQ3 — Research themes

Percentages use the taxon-focused sample. The three ranked theme rows ignore “Not Specified” when choosing ranks, then report each label’s share of all papers. “Not Specified %” is reported separately as its share of the full sample.

| Row title | Definition |
| --- | --- |
| **Top theme #1** | Most frequent theme after dropping “Not Specified” from the ranking. |
| **Top theme #1 %** | % of papers with that theme. |
| **Top theme #2** | Second place, same rule. |
| **Top theme #2 %** | % with theme #2. |
| **Top theme #3** | Third place. |
| **Top theme #3 %** | % with theme #3. |
| **Not Specified %** | % of papers assigned the “Not Specified” theme label. |

---

## RQ3 — Theme share change (percentage points): 2010–2015 vs 2021–2025

Percentages use taxon-focused papers with one primary theme label. For each theme row, **early** is the share in 2010–2015 and **recent** is the share in 2021–2025; cell values are recent − early in percentage points.

| Row title | Definition |
| --- | --- |
| **Ecology/Behavior** through **Not Specified** | Change in the share of papers assigned that primary theme between the early and recent bands. |

---

## RQ4A — Authorship structure

Author counts use each paper’s recorded author number when available; papers without a count are omitted from these rows.

| Row title | Definition |
| --- | --- |
| **Mean authors** | Mean authors per paper (taxon-focused, 2010–2025). |
| **Median authors** | Median for the same set. |
| **Mean authors early (2010-2015)** | Mean for 2010–2015 only. |
| **Mean authors recent (2020-2025)** | Mean for 2020–2025 only. |
| **Mean authors (applied)** | Mean for applied themes: Biomonitoring/Water Quality, Applied Ecology, Conservation, Materials Science (Silk). |
| **Mean authors (taxonomic)** | Mean where theme is Taxonomy/Systematics. |

---

## RQ4B — International collaboration %

**International:** affiliations span more than one country (country-level metadata when available; otherwise inference from affiliation text or broad region). **National:** one country identifiable. **Unknown:** collaboration type could not be assigned.

| Row title | Definition |
| --- | --- |
| **Papers with known affiliation-country signal (%)** | % of papers classified as International or National (not Unknown). |
| **Intl collaboration % (overall)** | International as a % of all taxon-focused papers (Unknown stays in the base). |
| **Intl collaboration % (known affiliations only)** | International ÷ (International + National) × 100; 0 if that denominator is empty. |
| **Intl collaboration % (applied)** | % International within the applied-theme papers (same theme list as RQ4A). |
| **Intl collaboration % (taxonomic)** | % International within Taxonomy/Systematics. |
