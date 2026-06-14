"""
Cross-taxon summary report (side-by-side).

This intentionally mirrors the filtering logic used by the existing RQ2–RQ4
scripts:
- restrict to 2010–2025
- exclude papers where the query taxon is not the study focus
  (Taxon_Relevance / legacy Trichoptera_Relevance label "Not target-taxon-focused"
   or legacy "Not Trichoptera-focused")

Outputs:
- analysis/combined/cross_taxa_report.md
- analysis/combined/cross_taxa_metrics.csv
- analysis/combined/yearly_publication_volume_by_query.csv (long format for plotting)
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.format_metrics import fmt_integerish, fmt_ratio_or_pct, round_one_decimal  # noqa: E402
from lib.pipeline import PipelinePaths, load_queries_config, paper_query_order  # noqa: E402


NOT_FOCUS_LABELS = {"Not target-taxon-focused", "Not Trichoptera-focused"}


def _norm_str(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def yearly_publication_volume_long(query_ids: list[str]) -> pd.DataFrame:
    """
    One row per (query_id, year) for 2010–2025 inclusive.
    ``n_all_coded``: year filter only; ``n_taxon_focused``: after relevance exclusion
    (same rules as ``filter_analysis_frame``).
    """
    query_ids = paper_query_order(load_queries_config(), query_ids)
    return pd.concat([_yearly_volume_single_query(q) for q in query_ids], ignore_index=True)


def _yearly_volume_single_query(query_id: str) -> pd.DataFrame:
    paths = PipelinePaths(query_id)
    df_in = pd.read_csv(paths.coded, low_memory=False)
    d = filter_year_window_only(df_in)
    if d.empty:
        return pd.DataFrame(
            {
                "query_id": query_id,
                "year": list(range(2010, 2026)),
                "n_all_coded": [0] * 16,
                "n_taxon_focused": [0] * 16,
            }
        )
    d = d.copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    d = d[d["Year"].notna()]
    d["Year"] = d["Year"].astype(int)
    rel_col = "Taxon_Relevance" if "Taxon_Relevance" in d.columns else "Trichoptera_Relevance"
    if rel_col not in d.columns:
        raise SystemExit(f"Missing relevance column for query_id={query_id!r}.")

    all_c = d.groupby("Year").size()
    foc = d[~d[rel_col].isin(NOT_FOCUS_LABELS)].groupby("Year").size()
    rows = []
    for y in range(2010, 2026):
        rows.append(
            {
                "query_id": query_id,
                "year": y,
                "n_all_coded": int(all_c.loc[y]) if y in all_c.index else 0,
                "n_taxon_focused": int(foc.loc[y]) if y in foc.index else 0,
            }
        )
    return pd.DataFrame(rows)


def filter_year_window_only(df: pd.DataFrame) -> pd.DataFrame:
    """2010–2025 only; no relevance filter (denominator for 'raw' vs taxon-focused)."""
    d = df.copy()
    d["Year"] = pd.to_numeric(d["Year"], errors="coerce")
    return d[d["Year"].between(2010, 2025)]


def filter_analysis_frame(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Year"] = pd.to_numeric(df["Year"], errors="coerce")
    df = df[df["Year"].between(2010, 2025)]

    rel_col = "Taxon_Relevance" if "Taxon_Relevance" in df.columns else "Trichoptera_Relevance"
    if rel_col not in df.columns:
        raise SystemExit("Missing relevance column (expected Taxon_Relevance or Trichoptera_Relevance).")

    df = df[~df[rel_col].isin(NOT_FOCUS_LABELS)]
    return df


def categorize_region(region) -> str:
    """Same continental buckets as analyze_rq2_temporal_geographic.py."""
    if pd.isna(region) or region == "Not Specified":
        return "Unknown"

    regions_of_interest = {
        "South America": ["Neotropical"],
        "Asia": ["Oriental", "East Palearctic"],
        "Europe": ["Palearctic"],
        "North America": ["Nearctic"],
        "Other": ["Afrotropical", "Australasian", "Global"],
    }
    for category, reg_list in regions_of_interest.items():
        if region in reg_list:
            return category
    return "Other"


def yearly_continental_props(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Region_Category"] = d["Region_Global"].apply(categorize_region)
    yearly = d.groupby(["Year", "Region_Category"]).size().unstack(fill_value=0)
    props = yearly.div(yearly.sum(axis=1), axis=0) * 100
    return props


def mean_category_share(props: pd.DataFrame, category: str) -> float:
    if category not in props.columns:
        return 0.0
    return float(props[category].mean())


def compare_early_recent(props: pd.DataFrame, early_years: list[int], recent_years: list[int]) -> dict[str, float]:
    early = props[props.index.isin(early_years)]
    recent = props[props.index.isin(recent_years)]
    out: dict[str, float] = {}
    for cat in ["South America", "Asia", "Europe", "North America", "Other", "Unknown"]:
        e = float(early[cat].mean()) if cat in early.columns else 0.0
        r = float(recent[cat].mean()) if cat in recent.columns else 0.0
        out[f"{cat}_early_mean_pct"] = e
        out[f"{cat}_recent_mean_pct"] = r
        out[f"{cat}_delta_pp"] = r - e
    return out


def merge_authors_like_rq4(df: pd.DataFrame, paths: PipelinePaths) -> tuple[pd.DataFrame, bool]:
    authors_csv = paths.with_authors
    if not authors_csv.exists():
        return df, False

    authors_df = pd.read_csv(authors_csv)
    merge_cols = ["DOI"] if "DOI" in df.columns and "DOI" in authors_df.columns else ["Title", "Year"]
    author_cols = ["Author_Count_Actual", "Author_Affiliations", "All_Authors"]
    if "Author_Country_Codes" in authors_df.columns:
        author_cols.append("Author_Country_Codes")
    merged = df.merge(
        authors_df[merge_cols + author_cols],
        on=merge_cols,
        how="left",
        suffixes=("", "_author"),
    )
    return merged, True


def compute_author_count(df: pd.DataFrame, has_full_author_data: bool) -> pd.Series:
    if has_full_author_data and "Author_Count_Actual" in df.columns:
        return df["Author_Count_Actual"].fillna(0).astype(int)
    # Fallback heuristic (same spirit as RQ4), but should rarely trigger if with_authors exists.
    def count_authors(authors_str):
        if pd.isna(authors_str) or not authors_str:
            return 1
        authors_str = str(authors_str).strip()
        if not authors_str or authors_str.lower() == "nan":
            return 1
        if "," in authors_str or ";" in authors_str or " and " in authors_str.lower() or " & " in authors_str:
            count = 1
            if "," in authors_str:
                count = max(count, authors_str.count(",") + 1)
            if ";" in authors_str:
                count = max(count, authors_str.count(";") + 1)
            if " and " in authors_str.lower():
                count = max(count, authors_str.lower().count(" and ") + 1)
            if " & " in authors_str:
                count = max(count, authors_str.count(" & ") + 1)
            return count
        return 1

    if "Authors" not in df.columns:
        return pd.Series([1] * len(df))
    return df["Authors"].apply(count_authors)


def detect_international_collab(row, has_affiliations: bool) -> str:
    """Same heuristic as analyze_rq4_collaboration.py (abbreviated)."""
    codes_raw = str(row.get("Author_Country_Codes", "") or "").strip()
    if codes_raw and codes_raw.lower() != "nan":
        tokens = [t.strip().upper() for t in codes_raw.replace("|", ";").replace(",", ";").split(";")]
        codes = {t for t in tokens if len(t) == 2 and t.isalpha()}
        if len(codes) > 1:
            return "International"
        if len(codes) == 1:
            return "National"
        return "Unknown"

    if has_affiliations:
        affiliations = str(row.get("Author_Affiliations", ""))
        if pd.isna(affiliations) or not affiliations or affiliations == "nan":
            return "Unknown"

        country_keywords = {
            "USA": ["United States", "USA", "US", "America"],
            "UK": ["United Kingdom", "UK", "England", "Scotland", "Wales"],
            "Germany": ["Germany", "Deutschland"],
            "France": ["France", "Français"],
            "Brazil": ["Brazil", "Brasil"],
            "China": ["China", "Chinese"],
            "Japan": ["Japan", "Japanese"],
            "Australia": ["Australia", "Australian"],
            "Canada": ["Canada", "Canadian"],
            "Italy": ["Italy", "Italian"],
            "Spain": ["Spain", "Spanish", "España"],
        }

        countries_found = set()
        affiliations_lower = affiliations.lower()
        for country, keywords in country_keywords.items():
            if any(kw.lower() in affiliations_lower for kw in keywords):
                countries_found.add(country)

        if len(countries_found) > 1:
            return "International"
        if len(countries_found) == 1:
            return "National"
        return "Unknown"

    region = str(row.get("Region_Global", ""))
    if region and region not in ("Not Specified", "Global"):
        return "National"
    if region == "Global":
        return "International"
    return "Unknown"


def study_type_from_theme(theme: str) -> str:
    applied_themes = [
        "Biomonitoring/Water Quality",
        "Applied Ecology",
        "Conservation",
        "Materials Science (Silk)",
    ]
    if theme in applied_themes:
        return "Applied"
    if theme == "Taxonomy/Systematics":
        return "Taxonomic"
    return "Other"


RQ3_THEME_SHIFT_THEMES = [
    "Ecology/Behavior",
    "Taxonomy/Systematics",
    "Biomonitoring/Water Quality",
    "Applied Ecology",
    "Not Specified",
]

THEME_SHIFT_EARLY_YEARS = (2010, 2015)
THEME_SHIFT_RECENT_YEARS = (2021, 2025)


def theme_share_pct(df: pd.DataFrame, theme: str, year_lo: int, year_hi: int) -> float:
    sub = df[df["Year"].between(year_lo, year_hi)]
    if len(sub) == 0:
        return 0.0
    themes = sub["Research_Theme"].fillna("").astype(str).str.strip()
    return float((themes == theme).mean() * 100.0)


def theme_shift_long(query_ids: list[str]) -> pd.DataFrame:
    cfg = load_queries_config()
    rows: list[dict] = []
    for q in paper_query_order(cfg, query_ids):
        paths = PipelinePaths(q)
        df = filter_analysis_frame(pd.read_csv(paths.coded, low_memory=False))
        for theme in RQ3_THEME_SHIFT_THEMES:
            early = theme_share_pct(df, theme, *THEME_SHIFT_EARLY_YEARS)
            recent = theme_share_pct(df, theme, *THEME_SHIFT_RECENT_YEARS)
            rows.append(
                {
                    "theme": theme,
                    "query_id": q,
                    "early_pct": round_one_decimal(early),
                    "recent_pct": round_one_decimal(recent),
                    "delta_pp": round_one_decimal(recent - early),
                }
            )
    return pd.DataFrame(rows)


def theme_shift_delta_wide_table(df_long: pd.DataFrame, query_order: list[str]) -> str:
    """Markdown table: themes as rows, query_id columns, delta_pp values."""
    pivot = df_long.pivot(index="theme", columns="query_id", values="delta_pp")
    pivot = pivot.reindex(RQ3_THEME_SHIFT_THEMES)
    pivot = pivot.reindex(columns=query_order)
    lines = [
        "| Theme | " + " | ".join(query_order) + " |",
        "| --- | " + " | ".join(["---:"] * len(query_order)) + " |",
    ]
    for theme in pivot.index:
        cells = []
        for q in query_order:
            val = pivot.loc[theme, q]
            if pd.isna(val):
                cells.append("—")
            else:
                v = float(val)
                cells.append(f"{v:+.1f}" if v != 0 else "0.0")
        lines.append(f"| {theme} | " + " | ".join(cells) + " |")
    return "\n".join(lines)

@dataclass
class TaxonMetrics:
    query_id: str
    n_papers_raw_year: int
    n_papers: int

    early_2010_2015: int
    recent_2020_2025: int
    pct_change_recent_vs_early: float

    theme_top1: str
    theme_top1_pct: float
    theme_top2: str
    theme_top2_pct: float
    theme_top3: str
    theme_top3_pct: float
    theme_not_specified_pct: float

    geo_sa_avg_pct: float
    geo_asia_avg_pct: float
    geo_europe_avg_pct: float
    geo_na_avg_pct: float
    geo_unknown_avg_pct: float

    geo_sa_delta_2010_2012_vs_2023_2025_pp: float
    geo_asia_delta_pp: float
    geo_europe_delta_pp: float
    geo_na_delta_pp: float

    authors_mean: float
    authors_median: float
    authors_early_mean: float
    authors_recent_mean: float

    applied_mean_authors: float
    taxonomic_mean_authors: float

    intl_pct_overall: float
    intl_info_coverage_pct: float
    intl_pct_known_only_overall: float
    intl_pct_applied: float
    intl_pct_taxonomic: float


def compute_metrics_for_query(query_id: str) -> TaxonMetrics:
    paths = PipelinePaths(query_id)
    df_in = pd.read_csv(paths.coded, low_memory=False)
    n_raw = int(len(filter_year_window_only(df_in)))
    df = filter_analysis_frame(df_in)

    n = len(df)
    if n == 0:
        raise SystemExit(f"No rows after filtering for query_id={query_id!r}.")

    early = int(df[df["Year"].between(2010, 2015)].shape[0])
    recent = int(df[df["Year"].between(2020, 2025)].shape[0])
    pct_change = ((recent - early) / early * 100.0) if early > 0 else 0.0

    themes = df["Research_Theme"].fillna("").astype(str).str.strip()
    vc_all = themes.value_counts(normalize=True) * 100.0
    ns_pct = float(vc_all.get("Not Specified", 0.0)) if "Not Specified" in vc_all.index else 0.0

    # Rank "real" themes excluding Not Specified so the top-3 row is interpretable.
    vc_rank = vc_all.drop(labels=["Not Specified"], errors="ignore")
    top = vc_rank.head(3)
    top_names = list(top.index)
    top_pcts = [float(top.loc[name]) for name in top_names] if len(top) else []
    while len(top_names) < 3:
        top_names.append("")
        top_pcts.append(0.0)

    props = yearly_continental_props(df)
    sa_avg = mean_category_share(props, "South America")
    asia_avg = mean_category_share(props, "Asia")
    eu_avg = mean_category_share(props, "Europe")
    na_avg = mean_category_share(props, "North America")
    unk_avg = mean_category_share(props, "Unknown")

    deltas = compare_early_recent(props, [2010, 2011, 2012], [2023, 2024, 2025])

    merged, has_authors = merge_authors_like_rq4(df, paths)
    author_count = compute_author_count(merged, has_authors and "Author_Count_Actual" in merged.columns)
    merged = merged[author_count > 0].copy()
    merged["AuthorCount"] = author_count[author_count > 0]

    has_affiliations = has_authors and "Author_Affiliations" in merged.columns
    merged["Collaboration_Type"] = merged.apply(lambda r: detect_international_collab(r, has_affiliations), axis=1)

    authors_mean = float(merged["AuthorCount"].mean())
    authors_median = float(merged["AuthorCount"].median())

    early_df = merged[merged["Year"].between(2010, 2015)]
    recent_df = merged[merged["Year"].between(2020, 2025)]
    authors_early_mean = float(early_df["AuthorCount"].mean()) if len(early_df) else 0.0
    authors_recent_mean = float(recent_df["AuthorCount"].mean()) if len(recent_df) else 0.0

    merged["Study_Type"] = merged["Research_Theme"].apply(study_type_from_theme)
    applied = merged[merged["Study_Type"] == "Applied"]
    tax = merged[merged["Study_Type"] == "Taxonomic"]

    def intl_pct(sub: pd.DataFrame) -> float:
        if len(sub) == 0:
            return 0.0
        return float((sub["Collaboration_Type"] == "International").mean() * 100.0)

    def intl_pct_known_only(sub: pd.DataFrame) -> float:
        if len(sub) == 0:
            return 0.0
        known = sub[sub["Collaboration_Type"].isin(["International", "National"])]
        if len(known) == 0:
            return 0.0
        return float((known["Collaboration_Type"] == "International").mean() * 100.0)

    intl_overall = intl_pct(merged)
    known_mask = merged["Collaboration_Type"].isin(["International", "National"])
    intl_info_coverage = float(known_mask.mean() * 100.0) if len(merged) else 0.0
    intl_known_overall = intl_pct_known_only(merged)
    intl_applied = intl_pct(applied)
    intl_tax = intl_pct(tax)

    return TaxonMetrics(
        query_id=query_id,
        n_papers_raw_year=n_raw,
        n_papers=n,
        early_2010_2015=early,
        recent_2020_2025=recent,
        pct_change_recent_vs_early=pct_change,
        theme_top1=top_names[0],
        theme_top1_pct=top_pcts[0],
        theme_top2=top_names[1],
        theme_top2_pct=top_pcts[1],
        theme_top3=top_names[2],
        theme_top3_pct=top_pcts[2],
        theme_not_specified_pct=ns_pct,
        geo_sa_avg_pct=sa_avg,
        geo_asia_avg_pct=asia_avg,
        geo_europe_avg_pct=eu_avg,
        geo_na_avg_pct=na_avg,
        geo_unknown_avg_pct=unk_avg,
        geo_sa_delta_2010_2012_vs_2023_2025_pp=float(deltas["South America_delta_pp"]),
        geo_asia_delta_pp=float(deltas["Asia_delta_pp"]),
        geo_europe_delta_pp=float(deltas["Europe_delta_pp"]),
        geo_na_delta_pp=float(deltas["North America_delta_pp"]),
        authors_mean=authors_mean,
        authors_median=authors_median,
        authors_early_mean=authors_early_mean,
        authors_recent_mean=authors_recent_mean,
        applied_mean_authors=float(applied["AuthorCount"].mean()) if len(applied) else 0.0,
        taxonomic_mean_authors=float(tax["AuthorCount"].mean()) if len(tax) else 0.0,
        intl_pct_overall=intl_overall,
        intl_info_coverage_pct=intl_info_coverage,
        intl_pct_known_only_overall=intl_known_overall,
        intl_pct_applied=intl_applied,
        intl_pct_taxonomic=intl_tax,
    )


def metrics_to_row(m: TaxonMetrics) -> dict:
    return {
        "query_id": m.query_id,
        "n_papers_2010_2025_raw": m.n_papers_raw_year,
        "n_papers_2010_2025_focused": m.n_papers,
        "papers_2010_2015": m.early_2010_2015,
        "papers_2020_2025": m.recent_2020_2025,
        "pct_change_papers_recent_vs_early": round_one_decimal(m.pct_change_recent_vs_early),
        "theme_top1": m.theme_top1,
        "theme_top1_pct": round_one_decimal(m.theme_top1_pct),
        "theme_top2": m.theme_top2,
        "theme_top2_pct": round_one_decimal(m.theme_top2_pct),
        "theme_top3": m.theme_top3,
        "theme_top3_pct": round_one_decimal(m.theme_top3_pct),
        "theme_not_specified_pct": round_one_decimal(m.theme_not_specified_pct),
        "geo_avg_south_america_pct": round_one_decimal(m.geo_sa_avg_pct),
        "geo_avg_asia_pct": round_one_decimal(m.geo_asia_avg_pct),
        "geo_avg_europe_pct": round_one_decimal(m.geo_europe_avg_pct),
        "geo_avg_north_america_pct": round_one_decimal(m.geo_na_avg_pct),
        "geo_avg_unknown_pct": round_one_decimal(m.geo_unknown_avg_pct),
        "geo_delta_pp_south_america_2010_2012_vs_2023_2025": round_one_decimal(
            m.geo_sa_delta_2010_2012_vs_2023_2025_pp
        ),
        "geo_delta_pp_asia_2010_2012_vs_2023_2025": round_one_decimal(m.geo_asia_delta_pp),
        "geo_delta_pp_europe_2010_2012_vs_2023_2025": round_one_decimal(m.geo_europe_delta_pp),
        "geo_delta_pp_north_america_2010_2012_vs_2023_2025": round_one_decimal(m.geo_na_delta_pp),
        "authors_mean": round_one_decimal(m.authors_mean),
        "authors_median": round_one_decimal(float(m.authors_median)),
        "authors_mean_early_2010_2015": round_one_decimal(m.authors_early_mean),
        "authors_mean_recent_2020_2025": round_one_decimal(m.authors_recent_mean),
        "authors_mean_applied": round_one_decimal(m.applied_mean_authors),
        "authors_mean_taxonomic": round_one_decimal(m.taxonomic_mean_authors),
        "intl_collab_pct_overall": round_one_decimal(m.intl_pct_overall),
        "intl_collab_info_coverage_pct": round_one_decimal(m.intl_info_coverage_pct),
        "intl_collab_pct_known_only_overall": round_one_decimal(m.intl_pct_known_only_overall),
        "intl_collab_pct_applied": round_one_decimal(m.intl_pct_applied),
        "intl_collab_pct_taxonomic": round_one_decimal(m.intl_pct_taxonomic),
    }


def render_markdown(rows: list[dict]) -> str:
    cfg = load_queries_config()
    df = pd.DataFrame(rows)
    order = paper_query_order(cfg, df["query_id"].tolist())
    df["query_id"] = pd.Categorical(df["query_id"], categories=order, ordered=True)
    df = df.sort_values("query_id")

    count_columns = frozenset(
        {
            "n_papers_2010_2025_raw",
            "n_papers_2010_2025_focused",
            "papers_2010_2015",
            "papers_2020_2025",
            # After rename in the sample-size table:
            "All coded (2010–2025)",
            "Taxon-focused (2010–2025)",
        }
    )

    def tbl(cols: list[str], title: str) -> str:
        def _fmt_cell(column: str, v) -> str:
            if column != "query_id" and pd.isna(v):
                return ""
            if column == "query_id":
                return str(v)
            if column in count_columns:
                return fmt_integerish(v)
            return fmt_ratio_or_pct(v)

        sub = df[["query_id"] + cols].copy()
        if cols == ["n_papers_2010_2025_raw", "n_papers_2010_2025_focused"]:
            sub = sub.rename(
                columns={
                    "n_papers_2010_2025_raw": "All coded (2010–2025)",
                    "n_papers_2010_2025_focused": "Taxon-focused (2010–2025)",
                }
            )
        lines = [f"### {title}", ""]
        lines.append("| " + " | ".join(sub.columns) + " |")
        lines.append("| " + " | ".join(["---"] * len(sub.columns)) + " |")
        for _, r in sub.iterrows():
            lines.append("| " + " | ".join(_fmt_cell(c, r[c]) for c in sub.columns) + " |")
        lines.append("")
        return "\n".join(lines)

    out: list[str] = []
    out.append("# Cross-taxon bibliometric summary (side-by-side)")
    out.append("")
    out.append("Generated from `data/processed/*/scopus_api_coded.csv` with the same filters as RQ2–RQ4:")
    out.append("- Years: 2010–2025")
    out.append("- Exclude non–taxon-focused papers (`Taxon_Relevance` not in {Not target-taxon-focused, Not Trichoptera-focused})")
    out.append("")
    out.append(
        tbl(
            ["n_papers_2010_2025_raw", "n_papers_2010_2025_focused"],
            "Sample size (2010–2025)",
        )
    )
    out.append(tbl(
        ["papers_2010_2015", "papers_2020_2025", "pct_change_papers_recent_vs_early"],
        "Temporal volume (2010–2015 vs 2020–2025)",
    ))
    out.append(
        "Each paper has one **primary research theme** label. Ranks #1–#3 omit “Not Specified” when choosing the three most common themes. "
        "**Not Specified %** is separate: the share of papers without a more specific theme."
    )
    out.append("")
    out.append(tbl(
        [
            "theme_top1",
            "theme_top1_pct",
            "theme_top2",
            "theme_top2_pct",
            "theme_top3",
            "theme_top3_pct",
            "theme_not_specified_pct",
        ],
        "Research themes by query",
    ))
    out.append(tbl(
        [
            "geo_avg_south_america_pct",
            "geo_avg_asia_pct",
            "geo_avg_europe_pct",
            "geo_avg_north_america_pct",
            "geo_avg_unknown_pct",
        ],
        "Geography: average continental shares (RQ2-style buckets)",
    ))
    out.append(tbl(
        [
            "geo_delta_pp_south_america_2010_2012_vs_2023_2025",
            "geo_delta_pp_asia_2010_2012_vs_2023_2025",
            "geo_delta_pp_europe_2010_2012_vs_2023_2025",
            "geo_delta_pp_north_america_2010_2012_vs_2023_2025",
        ],
        "Geography: mean early (2010–2012) vs recent (2023–2025) continental % (percentage-point change)",
    ))
    out.append(tbl(
        [
            "authors_mean",
            "authors_median",
            "authors_mean_early_2010_2015",
            "authors_mean_recent_2020_2025",
            "authors_mean_applied",
            "authors_mean_taxonomic",
        ],
        "RQ4A: Authorship structure (OpenAlex counts when available)",
    ))
    out.append(tbl(
        [
            "intl_collab_info_coverage_pct",
            "intl_collab_pct_overall",
            "intl_collab_pct_known_only_overall",
            "intl_collab_pct_applied",
            "intl_collab_pct_taxonomic",
        ],
        "RQ4B: International collaboration (affiliation-country heuristic)",
    ))
    return "\n".join(out)


def main():
    parser = argparse.ArgumentParser(description="Generate a cross-taxon side-by-side summary report.")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Query ids to include (default: all keys in config/queries.json).",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/combined",
        help="Output directory (default: analysis/combined).",
    )
    args = parser.parse_args()

    cfg = load_queries_config()
    queries = paper_query_order(cfg, args.queries or list((cfg.get("queries") or {}).keys()))
    if not queries:
        raise SystemExit("No queries found.")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for q in queries:
        rows.append(metrics_to_row(compute_metrics_for_query(q)))

    df = pd.DataFrame(rows)
    df["query_id"] = pd.Categorical(df["query_id"], categories=queries, ordered=True)
    df = df.sort_values("query_id")
    csv_path = out_dir / "cross_taxa_metrics.csv"
    md_path = out_dir / "cross_taxa_report.md"
    df.to_csv(csv_path, index=False)
    md_path.write_text(render_markdown(rows), encoding="utf-8")

    yvol = yearly_publication_volume_long(queries)
    yvol_path = out_dir / "yearly_publication_volume_by_query.csv"
    yvol.to_csv(yvol_path, index=False)

    theme_shift = theme_shift_long(queries)
    theme_shift_path = out_dir / "theme_shift_by_query.csv"
    theme_shift.to_csv(theme_shift_path, index=False)

    meta = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "queries": queries,
        "outputs": {
            "markdown": str(md_path),
            "csv": str(csv_path),
            "yearly_publication_volume_csv": str(yvol_path),
            "theme_shift_by_query_csv": str(theme_shift_path),
        },
    }
    (out_dir / "cross_taxa_report_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {md_path}")
    print(f"Wrote: {csv_path}")
    print(f"Wrote: {yvol_path}")
    print(f"Wrote: {theme_shift_path}")


if __name__ == "__main__":
    main()
