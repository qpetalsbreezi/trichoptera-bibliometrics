"""
Multi-taxon overall report: **tables only** (no methods, limitations, or narrative).

Combines:
- RQ1: parsed from each `analysis/<id>/rq1_coverage/coverage_report.txt`
- RQ2–RQ4: same metrics as `analyze_cross_taxa_summary.py`

Outputs:
- analysis/combined/overall_bibliometric_report.md
- analysis/combined/overall_bibliometric_report_meta.json
- analysis/combined/yearly_publication_volume_by_query.csv (long format; same as cross_taxa script)
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

from lib.pipeline import PROJECT_ROOT, load_queries_config  # noqa: E402

import analyze_cross_taxa_summary as xtax  # noqa: E402


def _md_table(df: pd.DataFrame, columns: list[str] | None = None) -> str:
    def _fmt(v) -> str:
        if pd.isna(v):
            return ""
        if isinstance(v, bool):
            return str(v)
        if isinstance(v, int):
            return str(v)
        if isinstance(v, float):
            # Render integer-like floats cleanly, otherwise trim trailing zeros.
            if abs(v - round(v)) < 1e-9:
                return str(int(round(v)))
            s = f"{v:.3f}".rstrip("0").rstrip(".")
            return s
        return str(v)

    cols = list(columns or df.columns)
    sub = df[cols].copy()
    lines = [
        "| " + " | ".join(str(c) for c in sub.columns) + " |",
        "| " + " | ".join(["---"] * len(sub.columns)) + " |",
    ]
    for _, row in sub.iterrows():
        lines.append("| " + " | ".join(_fmt(row[c]) for c in sub.columns) + " |")
    return "\n".join(lines) + "\n"


def _transposed_metrics_table(
    df: pd.DataFrame,
    value_cols: list[str],
    metric_label_map: dict[str, str] | None = None,
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
    return _md_table(tdf, col_order)


def _yearly_wide_md(df_long: pd.DataFrame, value_col: str, title: str, query_order: list[str]) -> str:
    """Years as rows, query_id as columns — easy copy into Excel / chart tools."""
    w = df_long.pivot(index="year", columns="query_id", values=value_col).reset_index()
    w.columns.name = None
    w = w.rename(columns={"year": "Year"})
    for q in query_order:
        if q not in w.columns:
            w[q] = 0
    col_order = ["Year"] + query_order
    return f"### {title}\n\n" + _md_table(w[col_order], col_order) + "\n"


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


def build_report(
    metrics_rows: list[dict],
    rq1_rows: list[dict],
    yearly_long: pd.DataFrame | None = None,
    query_order: list[str] | None = None,
) -> str:
    df = pd.DataFrame(metrics_rows).sort_values("query_id")
    df_rq1 = pd.DataFrame(rq1_rows).sort_values("query_id")

    parts: list[str] = []
    parts.append("# Overall bibliometric data (multi-taxon)")
    parts.append("")
    parts.append(f"*Generated {pd.Timestamp.now().strftime('%Y-%m-%d')}.*")
    parts.append("")

    parts.append("## RQ1 — Database coverage")
    parts.append("")
    display = df_rq1.copy()
    s = pd.to_numeric(display["rq1_scopus_total"], errors="coerce")
    o = pd.to_numeric(display["rq1_overlap_both"], errors="coerce")
    display["rq1_overlap_pct_of_scopus"] = (o / s * 100.0).where(s > 0).round(2)
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
                "pct_change_papers_recent_vs_early": "Pct change 2010–15 vs 2020–25 (taxon-focused)",
            },
        )
    )
    parts.append("")
    if yearly_long is not None and len(yearly_long) > 0:
        qord = query_order or sorted(yearly_long["query_id"].unique().tolist())
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

    parts.append("## RQ2 — Continental % change (pp): mean 2010–2012 vs mean 2023–2025")
    parts.append("")
    dcols = [
        "query_id",
        "geo_delta_pp_south_america_2010_2012_vs_2023_2025",
        "geo_delta_pp_asia_2010_2012_vs_2023_2025",
        "geo_delta_pp_europe_2010_2012_vs_2023_2025",
        "geo_delta_pp_north_america_2010_2012_vs_2023_2025",
    ]
    ddf = df[dcols].copy()
    parts.append(
        _transposed_metrics_table(
            ddf,
            [
                "geo_delta_pp_south_america_2010_2012_vs_2023_2025",
                "geo_delta_pp_asia_2010_2012_vs_2023_2025",
                "geo_delta_pp_europe_2010_2012_vs_2023_2025",
                "geo_delta_pp_north_america_2010_2012_vs_2023_2025",
            ],
            metric_label_map={
                "geo_delta_pp_south_america_2010_2012_vs_2023_2025": "Delta South America (pp)",
                "geo_delta_pp_asia_2010_2012_vs_2023_2025": "Delta Asia (pp)",
                "geo_delta_pp_europe_2010_2012_vs_2023_2025": "Delta Europe (pp)",
                "geo_delta_pp_north_america_2010_2012_vs_2023_2025": "Delta North America (pp)",
            },
        )
    )
    parts.append("")

    parts.append("## RQ3 — `Research_Theme` (top 3 ranks exclude Not Specified; includes Not Specified %)")
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
                "authors_mean_early_2010_2015": "Mean authors early (2010-2015)",
                "authors_mean_recent_2020_2025": "Mean authors recent (2020-2025)",
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
    queries = sorted(args.queries or list((cfg.get("queries") or {}).keys()))
    if not queries:
        raise SystemExit("No queries in config.")

    out_dir = PROJECT_ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    metrics_rows = [xtax.metrics_to_row(xtax.compute_metrics_for_query(q)) for q in queries]
    rq1_rows = [load_rq1_row(q) for q in queries]

    yearly_long = xtax.yearly_publication_volume_long(queries)
    yearly_path = out_dir / "yearly_publication_volume_by_query.csv"
    yearly_long.to_csv(yearly_path, index=False)

    md = build_report(metrics_rows, rq1_rows, yearly_long, queries)
    md_path = out_dir / "overall_bibliometric_report.md"
    md_path.write_text(md, encoding="utf-8")

    meta = {
        "generated_at": pd.Timestamp.now().isoformat(),
        "queries": queries,
        "outputs": {
            "markdown": str(md_path.relative_to(PROJECT_ROOT)),
            "yearly_publication_volume_csv": str(yearly_path.relative_to(PROJECT_ROOT)),
        },
    }
    (out_dir / "overall_bibliometric_report_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")

    print(f"Wrote: {md_path.relative_to(PROJECT_ROOT)}")
    print(f"Wrote: {yearly_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
