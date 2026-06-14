"""
Multi-taxon overall report: **tables only** (no methods, limitations, or narrative).

Combines:
- RQ1: parsed from each `analysis/<id>/rq1_coverage/coverage_report.txt`
- RQ2–RQ4: same metrics as `analyze_cross_taxa_summary.py`

Outputs:
- analysis/combined/overall_bibliometric_report.md (appends bibliometric_tables_glossary.md when present)
- analysis/combined/overall_bibliometric_report_meta.json
- analysis/combined/yearly_publication_volume_by_query.csv (long format; same as cross_taxa script)
- analysis/combined/theme_shift_by_query.csv (long format; early/recent/delta by theme and query)
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.format_metrics import fmt_integerish, fmt_ratio_or_pct, round_one_decimal  # noqa: E402
from lib.pipeline import PROJECT_ROOT, load_queries_config, paper_query_order  # noqa: E402

import analyze_cross_taxa_summary as xtax  # noqa: E402

# Row labels (after metric_label_map) whose cells are counts/years, not % or ratios.
_RQ1_INT_METRIC_ROWS = frozenset(
    {
        "Benchmark year",
        "Scopus total",
        "Google Scholar total",
        "Overlap (both)",
    }
)
_RQ2_VOLUME_INT_METRIC_ROWS = frozenset(
    {
        "All coded papers (2010–2025)",
        "Taxon-focused papers (2010–2025)",
        "Taxon-focused (2010–2015)",
        "Taxon-focused (2020–2025)",
    }
)


def _md_table(
    df: pd.DataFrame,
    columns: list[str] | None = None,
    *,
    count_columns: frozenset[str] | None = None,
    int_metric_rows: frozenset[str] | None = None,
) -> str:
    count_columns = count_columns or frozenset()
    int_metric_rows = int_metric_rows or frozenset()
    cols = list(columns or df.columns)
    sub = df[cols].copy()
    has_metric = "metric" in cols
    lines = [
        "| " + " | ".join(str(c) for c in sub.columns) + " |",
        "| " + " | ".join(["---"] * len(sub.columns)) + " |",
    ]
    for _, row in sub.iterrows():
        mname = str(row["metric"]) if has_metric else ""
        cells: list[str] = []
        for c in cols:
            v = row[c]
            if c == "metric":
                cells.append(str(v))
            elif pd.isna(v):
                cells.append("")
            elif c in count_columns:
                cells.append(fmt_integerish(v))
            elif has_metric and mname in int_metric_rows:
                cells.append(fmt_integerish(v))
            else:
                cells.append(fmt_ratio_or_pct(v))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def _transposed_metrics_table(
    df: pd.DataFrame,
    value_cols: list[str],
    metric_label_map: dict[str, str] | None = None,
    *,
    int_metric_rows: frozenset[str] | None = None,
) -> str:
    """
    Render a comparison-first table:
      - rows = metrics
      - columns = query_id only
    """
    metric_label_map = metric_label_map or {}
    q_order = [str(q) for q in df["query_id"]]
    sub = df[["query_id"] + value_cols].copy()
    sub["query_id"] = sub["query_id"].astype(str)
    sub = sub.set_index("query_id")
    tdf = sub.T
    tdf.index = [metric_label_map.get(idx, idx) for idx in tdf.index]
    tdf.index.name = "metric"
    tdf = tdf.reset_index()
    col_order = ["metric"] + q_order
    tdf = tdf[[c for c in col_order if c in tdf.columns]]
    return _md_table(tdf, col_order, int_metric_rows=int_metric_rows)


def _yearly_wide_md(df_long: pd.DataFrame, value_col: str, title: str, query_order: list[str]) -> str:
    """Years as rows, query_id as columns — easy copy into Excel / chart tools."""
    w = df_long.pivot(index="year", columns="query_id", values=value_col).reset_index()
    w.columns.name = None
    w = w.rename(columns={"year": "Year"})
    for q in query_order:
        if q not in w.columns:
            w[q] = 0
    col_order = ["Year"] + query_order
    return f"### {title}\n\n" + _md_table(w[col_order], col_order, count_columns=frozenset(col_order)) + "\n"


def parse_rq1_coverage(text: str) -> dict[str, str | int | float | None]:
    """Extract headline RQ1 stats from a coverage_report.txt body."""

    def one_int(pattern: str) -> int | None:
        m = re.search(pattern, text, re.MULTILINE)
        return int(m.group(1)) if m else None

    def one_float(pattern: str) -> float | None:
        m = re.search(pattern, text, re.MULTILINE)
        return float(m.group(1)) if m else None

    year = one_int(r"Publication Year:\s*(\d+)")
    scopus = one_int(r"Total Papers:\s*\n\s*-\s*Scopus:\s*(\d+)")
    gs = one_int(r"Total Papers:\s*\n\s*-\s*Scopus:\s*\d+\s*\n\s*-\s*Google Scholar:\s*(\d+)")
    ratio = one_float(r"Ratio \(GS/Scopus\):\s*([\d.]+)")
    both = one_int(r"Papers in both databases:\s*(\d+)")
    return {
        "rq1_benchmark_year": year,
        "rq1_scopus_total": scopus,
        "rq1_gs_total": gs,
        "rq1_gs_scopus_ratio": ratio,
        "rq1_overlap_both": both,
    }


def load_rq1_row(query_id: str) -> dict[str, str | int | float | None]:
    path = PROJECT_ROOT / "analysis" / query_id / "rq1_coverage" / "coverage_report.txt"
    base: dict[str, str | int | float | None] = {
        "query_id": query_id,
        "rq1_benchmark_year": None,
        "rq1_scopus_total": None,
        "rq1_gs_total": None,
        "rq1_gs_scopus_ratio": None,
        "rq1_overlap_both": None,
    }
    if not path.exists():
        return base
    parsed = parse_rq1_coverage(path.read_text(encoding="utf-8", errors="replace"))
    base.update(parsed)
    return base


def _sort_by_paper_order(df: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    out = df.copy()
    out["query_id"] = pd.Categorical(out["query_id"], categories=order, ordered=True)
    return out.sort_values("query_id")


def build_report(
    metrics_rows: list[dict],
    rq1_rows: list[dict],
    yearly_long: pd.DataFrame | None = None,
    query_order: list[str] | None = None,
    theme_shift_long: pd.DataFrame | None = None,
) -> str:
    cfg = load_queries_config()
    qord = query_order or paper_query_order(
        cfg, [r["query_id"] for r in metrics_rows if "query_id" in r]
    )
    df = _sort_by_paper_order(pd.DataFrame(metrics_rows), qord)
    df_rq1 = _sort_by_paper_order(pd.DataFrame(rq1_rows), qord)

    parts: list[str] = []
    parts.append("# Overall bibliometric data (multi-taxon)")
    parts.append("")
    parts.append(f"*Generated {pd.Timestamp.now().strftime('%Y-%m-%d')}.*")
    parts.append("")
    parts.append("Definitions of table rows are in the **Glossary** at the end of this file.")
    parts.append("")

    parts.append("## RQ1 — Database coverage")
    parts.append("")
    display = df_rq1.copy()
    s = pd.to_numeric(display["rq1_scopus_total"], errors="coerce")
    o = pd.to_numeric(display["rq1_overlap_both"], errors="coerce")
    overlap_pct = (o / s * 100.0).where(s > 0)
    display["rq1_overlap_pct_of_scopus"] = overlap_pct.map(
        lambda x: round_one_decimal(float(x)) if pd.notna(x) else x
    )
    ratio_num = pd.to_numeric(display["rq1_gs_scopus_ratio"], errors="coerce")
    display["rq1_gs_scopus_ratio"] = ratio_num.map(
        lambda x: round_one_decimal(float(x)) if pd.notna(x) else x
    )
    rq1_value_cols = [
        "rq1_benchmark_year",
        "rq1_scopus_total",
        "rq1_gs_total",
        "rq1_overlap_both",
        "rq1_overlap_pct_of_scopus",
        "rq1_gs_scopus_ratio",
    ]
    parts.append(
        _transposed_metrics_table(
            display,
            [c for c in rq1_value_cols if c in display.columns],
            metric_label_map={
                "rq1_benchmark_year": "Benchmark year",
                "rq1_scopus_total": "Scopus total",
                "rq1_gs_total": "Google Scholar total",
                "rq1_overlap_both": "Overlap (both)",
                "rq1_overlap_pct_of_scopus": "Overlap / Scopus (%)",
                "rq1_gs_scopus_ratio": "GS/Scopus ratio",
            },
            int_metric_rows=_RQ1_INT_METRIC_ROWS,
        )
    )
    parts.append("")

    parts.append("## RQ2 — Publication volume (2010–2025)")
    parts.append("")
    vol_cols = [
        "query_id",
        "n_papers_2010_2025_raw",
        "n_papers_2010_2025_focused",
        "papers_2010_2015",
        "papers_2020_2025",
        "pct_change_papers_recent_vs_early",
    ]
    vdf = df[vol_cols].copy()
    parts.append(
        _transposed_metrics_table(
            vdf,
            [
                "n_papers_2010_2025_raw",
                "n_papers_2010_2025_focused",
                "papers_2010_2015",
                "papers_2020_2025",
                "pct_change_papers_recent_vs_early",
            ],
            metric_label_map={
                "n_papers_2010_2025_raw": "All coded papers (2010–2025)",
                "n_papers_2010_2025_focused": "Taxon-focused papers (2010–2025)",
                "papers_2010_2015": "Taxon-focused (2010–2015)",
                "papers_2020_2025": "Taxon-focused (2020–2025)",
                "pct_change_papers_recent_vs_early": "Percent change 2010–15 vs 2020–25 (taxon-focused)",
            },
            int_metric_rows=_RQ2_VOLUME_INT_METRIC_ROWS,
        )
    )
    parts.append("")
    if yearly_long is not None and len(yearly_long) > 0:
        qord = query_order or qord
        parts.append(
            _yearly_wide_md(
                yearly_long,
                "n_taxon_focused",
                "RQ2 — Year-by-year N (taxon-focused)",
                qord,
            )
        )

    parts.append("## RQ2 — Mean continental % (mean of yearly %, 2010–2025)")
    parts.append("")
    geo_cols = [
        "query_id",
        "geo_avg_south_america_pct",
        "geo_avg_asia_pct",
        "geo_avg_europe_pct",
        "geo_avg_north_america_pct",
        "geo_avg_unknown_pct",
    ]
    gdf = df[geo_cols].copy()
    parts.append(
        _transposed_metrics_table(
            gdf,
            [
                "geo_avg_south_america_pct",
                "geo_avg_asia_pct",
                "geo_avg_europe_pct",
                "geo_avg_north_america_pct",
                "geo_avg_unknown_pct",
            ],
            metric_label_map={
                "geo_avg_south_america_pct": "Mean South America %",
                "geo_avg_asia_pct": "Mean Asia %",
                "geo_avg_europe_pct": "Mean Europe %",
                "geo_avg_north_america_pct": "Mean North America %",
                "geo_avg_unknown_pct": "Mean Unknown %",
            },
        )
    )
    parts.append("")

    parts.append(
        f"## RQ2 — Continental % change (pp): mean {xtax.EARLY_WINDOW_LABEL} vs mean {xtax.RECENT_WINDOW_LABEL}"
    )
    parts.append("")
    dcols = [
        "query_id",
        "geo_delta_pp_south_america_2010_2015_vs_2020_2025",
        "geo_delta_pp_asia_2010_2015_vs_2020_2025",
        "geo_delta_pp_europe_2010_2015_vs_2020_2025",
        "geo_delta_pp_north_america_2010_2015_vs_2020_2025",
    ]
    ddf = df[dcols].copy()
    parts.append(
        _transposed_metrics_table(
            ddf,
            [
                "geo_delta_pp_south_america_2010_2015_vs_2020_2025",
                "geo_delta_pp_asia_2010_2015_vs_2020_2025",
                "geo_delta_pp_europe_2010_2015_vs_2020_2025",
                "geo_delta_pp_north_america_2010_2015_vs_2020_2025",
            ],
            metric_label_map={
                "geo_delta_pp_south_america_2010_2015_vs_2020_2025": "Delta South America (pp)",
                "geo_delta_pp_asia_2010_2015_vs_2020_2025": "Delta Asia (pp)",
                "geo_delta_pp_europe_2010_2015_vs_2020_2025": "Delta Europe (pp)",
                "geo_delta_pp_north_america_2010_2015_vs_2020_2025": "Delta North America (pp)",
            },
        )
    )
    parts.append("")

    parts.append("## RQ3 — Research themes")
    parts.append("")
    parts.append(
        "Each paper carries one **primary research theme** label. The three ranked rows (#1–#3) list the most common themes after excluding “Not Specified” from the ranking. "
        "**Not Specified %** is reported on its own: the proportion of papers left in that category."
    )
    parts.append("")
    th_cols = [
        "query_id",
        "theme_top1",
        "theme_top1_pct",
        "theme_top2",
        "theme_top2_pct",
        "theme_top3",
        "theme_top3_pct",
        "theme_not_specified_pct",
    ]
    tdf = df[th_cols].copy()
    parts.append(
        _transposed_metrics_table(
            tdf,
            [
                "theme_top1",
                "theme_top1_pct",
                "theme_top2",
                "theme_top2_pct",
                "theme_top3",
                "theme_top3_pct",
                "theme_not_specified_pct",
            ],
            metric_label_map={
                "theme_top1": "Top theme #1",
                "theme_top1_pct": "Top theme #1 %",
                "theme_top2": "Top theme #2",
                "theme_top2_pct": "Top theme #2 %",
                "theme_top3": "Top theme #3",
                "theme_top3_pct": "Top theme #3 %",
                "theme_not_specified_pct": "Not Specified %",
            },
        )
    )
    parts.append("")

    if theme_shift_long is not None and len(theme_shift_long) > 0:
        qord = query_order or qord
        parts.append(
            f"## RQ3 — Theme share change (percentage points): {xtax.EARLY_WINDOW_LABEL} vs {xtax.RECENT_WINDOW_LABEL}"
        )
        parts.append("")
        parts.append(
            f"Change in the share of papers assigned each primary theme between the early band ({xtax.EARLY_WINDOW_LABEL}) "
            f"and recent band ({xtax.RECENT_WINDOW_LABEL}). Values are recent minus early percentage points on taxon-focused papers."
        )
        parts.append("")
        parts.append(xtax.theme_shift_delta_wide_table(theme_shift_long, qord))
        parts.append("")

    parts.append("## RQ4A — Authorship structure")
    parts.append("")
    a_cols = [
        "query_id",
        "authors_mean",
        "authors_median",
        "authors_mean_early_2010_2015",
        "authors_mean_recent_2020_2025",
        "authors_mean_applied",
        "authors_mean_taxonomic",
    ]
    adf = df[a_cols].copy()
    parts.append(
        _transposed_metrics_table(
            adf,
            [
                "authors_mean",
                "authors_median",
                "authors_mean_early_2010_2015",
                "authors_mean_recent_2020_2025",
                "authors_mean_applied",
                "authors_mean_taxonomic",
            ],
            metric_label_map={
                "authors_mean": "Mean authors",
                "authors_median": "Median authors",
                "authors_mean_early_2010_2015": "Mean authors early (2010–2015)",
                "authors_mean_recent_2020_2025": "Mean authors recent (2020–2025)",
                "authors_mean_applied": "Mean authors (applied)",
                "authors_mean_taxonomic": "Mean authors (taxonomic)",
            },
        )
    )
    parts.append("")

    parts.append("## RQ4B — International collaboration %")
    parts.append("")
    c_cols = [
        "query_id",
        "intl_collab_info_coverage_pct",
        "intl_collab_pct_overall",
        "intl_collab_pct_known_only_overall",
        "intl_collab_pct_applied",
        "intl_collab_pct_taxonomic",
    ]
    cdf = df[c_cols].copy()
    parts.append(
        _transposed_metrics_table(
            cdf,
            [
                "intl_collab_info_coverage_pct",
                "intl_collab_pct_overall",
                "intl_collab_pct_known_only_overall",
                "intl_collab_pct_applied",
                "intl_collab_pct_taxonomic",
            ],
            metric_label_map={
                "intl_collab_info_coverage_pct": "Papers with known affiliation-country signal (%)",
                "intl_collab_pct_overall": "Intl collaboration % (overall)",
                "intl_collab_pct_known_only_overall": "Intl collaboration % (known affiliations only)",
                "intl_collab_pct_applied": "Intl collaboration % (applied)",
                "intl_collab_pct_taxonomic": "Intl collaboration % (taxonomic)",
            },
        )
    )
    parts.append("")

    return "\n".join(parts)


def _increase_markdown_heading_depth(text: str, add: int = 1) -> str:
    """Nest a snippet under the report H1 by adding ``add`` '#' to each heading line."""
    out: list[str] = []
    for line in text.splitlines():
        m = re.match(r"^(#+)(\s.*)$", line)
        if m:
            hashes = m.group(1)
            rest = m.group(2)
            out.append("#" * (len(hashes) + add) + rest)
        else:
            out.append(line)
    return "\n".join(out)


def _append_glossary_if_present(md: str, glossary_path: Path) -> str:
    if not glossary_path.is_file():
        return md
    body = glossary_path.read_text(encoding="utf-8").strip()
    if not body:
        return md
    nested = _increase_markdown_heading_depth(body, add=1)
    return md.rstrip() + "\n\n---\n\n" + nested + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate single overall bibliometric report (all taxa).")
    parser.add_argument(
        "--queries",
        nargs="*",
        default=None,
        help="Query ids (default: all in config/queries.json).",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/combined",
        help="Output directory relative to project root.",
    )
    args = parser.parse_args()

    cfg = load_queries_config()
    queries = paper_query_order(cfg, args.queries or list((cfg.get("queries") or {}).keys()))
    if not queries:
        raise SystemExit("No queries in config.")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = [xtax.metrics_to_row(xtax.compute_metrics_for_query(q)) for q in queries]
    rq1_rows = [load_rq1_row(q) for q in queries]

    yearly_long = xtax.yearly_publication_volume_long(queries)
    yearly_path = out_dir / "yearly_publication_volume_by_query.csv"
    yearly_long.to_csv(yearly_path, index=False)

    theme_shift = xtax.theme_shift_long(queries)
    theme_shift_path = out_dir / "theme_shift_by_query.csv"
    theme_shift.to_csv(theme_shift_path, index=False)

    md = build_report(metrics_rows, rq1_rows, yearly_long, queries, theme_shift)
    glossary_path = out_dir / "bibliometric_tables_glossary.md"
    md = _append_glossary_if_present(md, glossary_path)
    md_path = out_dir / "overall_bibliometric_report.md"
    md_path.write_text(md, encoding="utf-8")

    meta = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "queries": queries,
        "outputs": {
            "markdown": str(md_path.relative_to(PROJECT_ROOT)),
            "yearly_publication_volume_csv": str(yearly_path.relative_to(PROJECT_ROOT)),
            "theme_shift_by_query_csv": str(theme_shift_path.relative_to(PROJECT_ROOT)),
            "glossary_source": str(glossary_path.relative_to(PROJECT_ROOT))
            if glossary_path.is_file()
            else None,
        },
    }
    (out_dir / "overall_bibliometric_report_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {yearly_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {theme_shift_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
