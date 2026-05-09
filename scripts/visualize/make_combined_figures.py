"""
Generate multi-taxon bibliometric figures for journal use (PDF + PNG).

Reads analysis/combined CSVs plus config/queries.json (taxon order and display labels).

RQ1 metrics come from each taxon's ``analysis/<id>/rq1_coverage/coverage_report.txt``
(same as overall_bibliometric_report).

Yearly continental line charts recompute shares from each taxon's
``data/processed/<id>/scopus_api_coded.csv`` using the same filters as
``analyze_cross_taxa_summary`` (so manifest records SHA-256 of those files).

Writes:
  - analysis/combined/figures/*.pdf and *.png
  - analysis/combined/figures/figures_manifest.json (inputs, hashes, optional git SHA)
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_ANALYZE_DIR = _SCRIPTS_DIR / "analyze"
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))
if str(_ANALYZE_DIR) not in sys.path:
    sys.path.append(str(_ANALYZE_DIR))

import analyze_cross_taxa_summary as xtax  # noqa: E402
from analyze_overall_bibliometric_report import load_rq1_row  # noqa: E402
from lib.pipeline import PROJECT_ROOT, PipelinePaths, load_queries_config  # noqa: E402

# Okabe–Ito palette (colorblind-friendly), one color per taxon in sorted query_id order.
_TAXON_COLORS: list[tuple[float, float, float]] = [
    (0.90, 0.62, 0.00),  # orange
    (0.35, 0.70, 0.90),  # sky blue
    (0.00, 0.62, 0.45),  # bluish green
    (0.80, 0.75, 0.00),  # yellow
    (0.00, 0.45, 0.70),  # blue
    (0.80, 0.40, 0.00),  # vermillion
    (0.55, 0.35, 0.64),  # purple
]


def _setup_rc() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _query_order(cfg: dict) -> list[str]:
    return sorted((cfg.get("queries") or {}).keys())


def _taxon_color_map(query_ids: list[str]) -> dict[str, tuple]:
    return {q: _TAXON_COLORS[i % len(_TAXON_COLORS)] for i, q in enumerate(query_ids)}


def _short_label(cfg: dict, qid: str) -> str:
    q = (cfg.get("queries") or {}).get(qid) or {}
    lab = str(q.get("label") or qid)
    if "(" in lab:
        lab = lab.split("(")[0].strip()
    return lab[:22] + ("…" if len(lab) > 22 else "")


def _sha256(path: Path) -> str | None:
    if not path.is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_head() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            stderr=subprocess.DEVNULL,
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return None


def _save(fig: plt.Figure, out_dir: Path, stem: str) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    fig.savefig(pdf, bbox_inches="tight")
    fig.savefig(png, bbox_inches="tight")
    plt.close(fig)
    return {"pdf": str(pdf.relative_to(PROJECT_ROOT)), "png": str(png.relative_to(PROJECT_ROOT))}


def fig_rq2_temporal_facets(
    yearly: pd.DataFrame,
    query_ids: list[str],
    colors: dict[str, tuple],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    n = len(query_ids)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), sharex=True, sharey=False)
    if n == 1:
        axes = np.array([axes])
    years = np.arange(2010, 2026)
    for ax, q in zip(axes, query_ids):
        sub = yearly[yearly["query_id"] == q].sort_values("year")
        y = sub.set_index("year")["n_taxon_focused"].reindex(years, fill_value=0).values
        ax.fill_between(years, y, alpha=0.25, color=colors[q])
        ax.plot(years, y, color=colors[q], linewidth=1.6)
        ax.set_title(labels[q])
        ax.set_xlim(2009.5, 2025.5)
        ax.set_xticks([2010, 2015, 2020, 2025])
    axes[0].set_ylabel("Taxon-focused N")
    fig.supxlabel("Publication year", y=0.02)
    fig.suptitle("Taxon-focused publication volume by year (2010–2025)", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_temporal_taxon_focused_facets")


def fig_rq2_temporal_log_overlay(
    yearly: pd.DataFrame,
    query_ids: list[str],
    colors: dict[str, tuple],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for q in query_ids:
        sub = yearly[yearly["query_id"] == q].sort_values("year")
        y = sub["n_taxon_focused"].to_numpy(dtype=float)
        x = sub["year"].to_numpy()
        ax.plot(x, np.log10(y + 1), label=labels[q], color=colors[q], linewidth=1.8)
    ax.set_xlabel("Publication year")
    ax.set_ylabel(r"$\log_{10}$(N + 1), taxon-focused")
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("Cross-taxon publication volume (log scale, comparable shape)")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_temporal_taxon_focused_log_overlay")


def fig_rq1_database_coverage(
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    rows = []
    for q in query_ids:
        r = load_rq1_row(q)
        rows.append(
            {
                "query_id": q,
                "scopus": float(r.get("rq1_scopus_total") or 0),
                "gs": float(r.get("rq1_gs_total") or 0),
                "overlap": float(r.get("rq1_overlap_both") or 0),
            }
        )
    df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    x = np.arange(len(query_ids))
    w = 0.25
    ax.bar(x - w, df["scopus"], width=w, label="Scopus total", color="#333333", edgecolor="none")
    ax.bar(x, df["overlap"], width=w, label="Overlap (both)", color="#888888", edgecolor="none")
    ax.bar(x + w, df["gs"], width=w, label="Google Scholar total", color="#BBBBBB", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=25, ha="right")
    ax.set_ylabel("Publication count (RQ1 benchmark year)")
    ax.set_title("Database coverage comparison (RQ1)")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq1_database_counts")


def fig_rq2_geo_mean_stacked(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    cats = [
        ("geo_avg_south_america_pct", "South America"),
        ("geo_avg_asia_pct", "Asia"),
        ("geo_avg_europe_pct", "Europe"),
        ("geo_avg_north_america_pct", "North America"),
        ("geo_avg_unknown_pct", "Unknown"),
    ]
    mat = np.array([[float(m.loc[q, col]) for col, _ in cats] for q in query_ids])
    fig, ax = plt.subplots(figsize=(7, 4))
    left = np.zeros(len(query_ids))
    stack_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#999999"]
    for i, (col, name) in enumerate(cats):
        ax.barh(
            [labels[q] for q in query_ids],
            mat[:, i],
            left=left,
            label=name,
            color=stack_colors[i % len(stack_colors)],
        )
        left = left + mat[:, i]
    ax.set_xlabel("Mean continental share (% of papers, 2010–2025)")
    ax.set_title("Geographic composition by taxon (mean yearly %)")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", borderaxespad=0.0, fontsize=7)
    ax.set_xlim(0, 100)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_geo_mean_continental_stacked")


def fig_rq2_geo_delta_heatmap(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    cols = [
        ("geo_delta_pp_south_america_2010_2012_vs_2023_2025", "S. America"),
        ("geo_delta_pp_asia_2010_2012_vs_2023_2025", "Asia"),
        ("geo_delta_pp_europe_2010_2012_vs_2023_2025", "Europe"),
        ("geo_delta_pp_north_america_2010_2012_vs_2023_2025", "N. America"),
    ]
    m = metrics.set_index("query_id").loc[query_ids]
    mat = np.array([[float(m.loc[q, c]) for c, _ in cols] for q in query_ids])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    vmax = max(np.abs(mat).max(), 1e-6)
    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lbl for _, lbl in cols], rotation=20, ha="right")
    ax.set_yticks(range(len(query_ids)))
    ax.set_yticklabels([labels[q] for q in query_ids])
    ax.set_title("Δ continental % (mean 2023–2025 minus mean 2010–2012, pp)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Percentage points")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_geo_delta_heatmap")


def fig_rq3_theme_top_shares(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(query_ids))
    w = 0.2
    parts = [
        ("theme_top1_pct", "Top theme #1 %"),
        ("theme_top2_pct", "Top theme #2 %"),
        ("theme_top3_pct", "Top theme #3 %"),
        ("theme_not_specified_pct", "Not specified %"),
    ]
    for i, (col, leg) in enumerate(parts):
        ax.bar(x + (i - 1.5) * w, [float(m.loc[q, col]) for q in query_ids], width=w, label=leg)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Share of taxon-focused papers (%)")
    ax.set_title("Research theme concentration (RQ3)")
    ax.legend(loc="upper right", ncol=2, fontsize=7)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_theme_ranked_shares")


def fig_rq4_authorship_collaboration(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, tuple],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.8))
    x = np.arange(len(query_ids))
    w = 0.35
    early = [float(m.loc[q, "authors_mean_early_2010_2015"]) for q in query_ids]
    recent = [float(m.loc[q, "authors_mean_recent_2020_2025"]) for q in query_ids]
    ax1.bar(x - w / 2, early, width=w, label="2010–2015", color="#6BAED6", edgecolor="none")
    ax1.bar(x + w / 2, recent, width=w, label="2020–2025", color="#08519C", edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax1.set_ylabel("Mean author count")
    ax1.set_title("Authorship (RQ4A)")
    ax1.legend()

    intl = [float(m.loc[q, "intl_collab_pct_known_only_overall"]) for q in query_ids]
    ax2.bar(x, intl, color=[colors[q] for q in query_ids], edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax2.set_ylabel("International collaboration %")
    ax2.set_title("Collaboration among papers with known affiliation signal (RQ4B)")
    ax2.set_ylim(0, max(intl) * 1.15 if intl else 1)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq4_authors_and_intl_collab")


# --- RQ1 (additional) ---


def fig_rq1_overlap_pct_and_ratio(
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    overlap_pct: list[float] = []
    ratios: list[float] = []
    for q in query_ids:
        r = load_rq1_row(q)
        s = float(r.get("rq1_scopus_total") or 0)
        o = float(r.get("rq1_overlap_both") or 0)
        overlap_pct.append(100.0 * o / s if s > 0 else 0.0)
        ratios.append(float(r.get("rq1_gs_scopus_ratio") or 0.0))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9, 3.6))
    x = np.arange(len(query_ids))
    ax1.bar(x, overlap_pct, color="#4DAF4A", edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[q] for q in query_ids], rotation=22, ha="right")
    ax1.set_ylabel("Overlap / Scopus (%)")
    ax1.set_title("RQ1 — Records appearing in both databases")
    ax1.set_ylim(0, min(105, max(overlap_pct) * 1.2 if overlap_pct else 100))

    ax2.bar(x, ratios, color="#984EA3", edgecolor="none")
    ax2.set_xticks(x)
    ax2.set_xticklabels([labels[q] for q in query_ids], rotation=22, ha="right")
    ax2.set_ylabel("Google Scholar / Scopus ratio")
    ax2.set_title("RQ1 — Relative database yield")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq1_overlap_pct_and_gs_scopus_ratio")


# --- RQ2 volume (additional) ---


def fig_rq2_sample_size_raw_vs_focused(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(query_ids))
    w = 0.35
    raw = [float(m.loc[q, "n_papers_2010_2025_raw"]) for q in query_ids]
    foc = [float(m.loc[q, "n_papers_2010_2025_focused"]) for q in query_ids]
    ax.bar(x - w / 2, raw, width=w, label="All coded (2010–2025)", color="#9ECAE1", edgecolor="none")
    ax.bar(x + w / 2, foc, width=w, label="Taxon-focused (2010–2025)", color="#3182BD", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=22, ha="right")
    ax.set_ylabel("Publication count")
    ax.set_title("RQ2 — Analytic sample vs full coded window")
    ax.legend(loc="upper right")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_sample_size_all_coded_vs_taxon_focused")


def fig_rq2_early_vs_recent_window_counts(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    x = np.arange(len(query_ids))
    w = 0.35
    early = [float(m.loc[q, "papers_2010_2015"]) for q in query_ids]
    recent = [float(m.loc[q, "papers_2020_2025"]) for q in query_ids]
    ax.bar(x - w / 2, early, width=w, label="Taxon-focused 2010–2015", color="#FDAE6B", edgecolor="none")
    ax.bar(x + w / 2, recent, width=w, label="Taxon-focused 2020–2025", color="#D95F0E", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=22, ha="right")
    ax.set_ylabel("Publication count")
    ax.set_title("RQ2 — Early vs recent six-year windows (taxon-focused)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_taxon_focused_early_vs_recent_counts")


def fig_rq2_pct_change_recent_vs_early(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, tuple],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    vals = [float(m.loc[q, "pct_change_papers_recent_vs_early"]) for q in query_ids]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    y_pos = np.arange(len(query_ids))
    ax.barh(y_pos, vals, color=[colors[q] for q in query_ids], edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[q] for q in query_ids])
    ax.set_xlabel("Percent change (2020–2025 vs 2010–2015, taxon-focused N)")
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax.set_title("RQ2 — Growth contrast across taxa")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_pct_change_taxon_focused_recent_vs_early")


def fig_rq2_temporal_all_coded_facets(
    yearly: pd.DataFrame,
    query_ids: list[str],
    colors: dict[str, tuple],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    n = len(query_ids)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), sharex=True, sharey=False)
    if n == 1:
        axes = np.array([axes])
    years = np.arange(2010, 2026)
    for ax, q in zip(axes, query_ids):
        sub = yearly[yearly["query_id"] == q].sort_values("year")
        y = sub.set_index("year")["n_all_coded"].reindex(years, fill_value=0).values
        ax.fill_between(years, y, alpha=0.25, color=colors[q])
        ax.plot(years, y, color=colors[q], linewidth=1.6)
        ax.set_title(labels[q])
        ax.set_xlim(2009.5, 2025.5)
        ax.set_xticks([2010, 2015, 2020, 2025])
    axes[0].set_ylabel("All coded N")
    fig.supxlabel("Publication year", y=0.02)
    fig.suptitle("RQ2 — All coded papers by year (2010–2025, before relevance filter)", y=1.02)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_temporal_all_coded_facets")


def fig_rq2_temporal_all_coded_log_overlay(
    yearly: pd.DataFrame,
    query_ids: list[str],
    colors: dict[str, tuple],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    fig, ax = plt.subplots(figsize=(6.5, 4))
    for q in query_ids:
        sub = yearly[yearly["query_id"] == q].sort_values("year")
        y = sub["n_all_coded"].to_numpy(dtype=float)
        x = sub["year"].to_numpy()
        ax.plot(x, np.log10(y + 1), label=labels[q], color=colors[q], linewidth=1.8)
    ax.set_xlabel("Publication year")
    ax.set_ylabel(r"$\log_{10}$(N + 1), all coded")
    ax.legend(loc="upper left", frameon=True)
    ax.set_title("RQ2 — All coded volume (log scale, cross-taxon)")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_temporal_all_coded_log_overlay")


# --- RQ2 geography (additional) ---


_GEO_REGION_LINE_COLORS: dict[str, tuple] = {
    "South America": (0.90, 0.62, 0.00),
    "Asia": (0.35, 0.70, 0.90),
    "Europe": (0.00, 0.62, 0.45),
    "North America": (0.80, 0.40, 0.00),
    "Unknown": (0.55, 0.55, 0.55),
    "Other": (0.55, 0.35, 0.64),
}


def _yearly_continental_props_by_query(query_ids: list[str]) -> dict[str, pd.DataFrame]:
    """Year × region % table per taxon (same filters as cross_taxa_summary)."""
    out: dict[str, pd.DataFrame] = {}
    years = list(range(2010, 2026))
    for q in query_ids:
        paths = PipelinePaths(q)
        if not paths.coded.is_file():
            out[q] = pd.DataFrame()
            continue
        df_in = pd.read_csv(paths.coded, low_memory=False)
        d = xtax.filter_analysis_frame(df_in)
        if d.empty:
            out[q] = pd.DataFrame()
            continue
        props = xtax.yearly_continental_props(d)
        props = props.reindex(years).fillna(0)
        out[q] = props
    return out


def fig_rq2_geo_temporal_lines_by_taxon(
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    props_by_q = _yearly_continental_props_by_query(query_ids)
    regions = ["South America", "Asia", "Europe", "North America", "Unknown", "Other"]
    n = len(query_ids)
    fig, axes = plt.subplots(1, n, figsize=(2.5 * n, 3.2), sharex=True, sharey=True)
    if n == 1:
        axes = np.array([axes])
    legend_ax = None
    for ax, q in zip(axes, query_ids):
        props = props_by_q.get(q, pd.DataFrame())
        if props.empty:
            ax.set_title(labels[q] + "\n(no data)")
            continue
        if legend_ax is None:
            legend_ax = ax
        for reg in regions:
            if reg not in props.columns:
                continue
            ax.plot(
                props.index,
                props[reg].values,
                label=reg if ax is legend_ax else "_nolegend_",
                color=_GEO_REGION_LINE_COLORS.get(reg, (0.2, 0.2, 0.2)),
                linewidth=1.2,
            )
        ax.set_title(labels[q])
        ax.set_xlim(2009.5, 2025.5)
        ax.set_xticks([2010, 2015, 2020, 2025])
        ax.set_ylim(0, 100)
    axes[0].set_ylabel("Share of taxon-focused papers (%)")
    fig.supxlabel("Publication year", y=0.02)
    handles, leg_labels = (legend_ax or axes[0]).get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    fig.suptitle("RQ2 — Continental composition over time (yearly %)", y=1.12)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_geo_continental_share_over_time_facets")


def fig_rq2_geo_mean_grouped_vertical(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    cats = [
        ("geo_avg_south_america_pct", "S. Am."),
        ("geo_avg_asia_pct", "Asia"),
        ("geo_avg_europe_pct", "Europe"),
        ("geo_avg_north_america_pct", "N. Am."),
        ("geo_avg_unknown_pct", "Unk."),
    ]
    fig, ax = plt.subplots(figsize=(10, 4))
    x = np.arange(len(query_ids))
    n_c = len(cats)
    w = 0.15
    for i, (col, name) in enumerate(cats):
        offset = (i - (n_c - 1) / 2) * w
        heights = [float(m.loc[q, col]) for q in query_ids]
        ax.bar(x + offset, heights, width=w * 0.95, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Mean yearly % (2010–2025)")
    ax.set_title("RQ2 — Mean continental share by taxon (grouped)")
    ax.legend(ncol=5, loc="upper right", fontsize=7)
    ax.set_ylim(0, max(40, max(float(m.loc[q, c]) for q in query_ids for c, _ in cats) * 1.1))
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_geo_mean_continental_grouped_bars")


def fig_rq2_geo_delta_grouped_bars(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    cols = [
        ("geo_delta_pp_south_america_2010_2012_vs_2023_2025", "S. Am."),
        ("geo_delta_pp_asia_2010_2012_vs_2023_2025", "Asia"),
        ("geo_delta_pp_europe_2010_2012_vs_2023_2025", "Europe"),
        ("geo_delta_pp_north_america_2010_2012_vs_2023_2025", "N. Am."),
    ]
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(9.5, 4))
    x = np.arange(len(query_ids))
    n_c = len(cols)
    w = 0.18
    for i, (col, name) in enumerate(cols):
        offset = (i - (n_c - 1) / 2) * w
        heights = [float(m.loc[q, col]) for q in query_ids]
        ax.bar(x + offset, heights, width=w * 0.9, label=name)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Δ percentage points")
    ax.set_title("RQ2 — Continental shift (mean 2023–2025 − mean 2010–2012)")
    ax.legend(ncol=4, loc="upper right", fontsize=7)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_geo_delta_continental_grouped_bars")


# --- RQ3 (additional) ---


def fig_rq3_top1_theme_labeled_bars(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y_pos = np.arange(len(query_ids))
    pcts = [float(m.loc[q, "theme_top1_pct"]) for q in query_ids]
    names = [str(m.loc[q, "theme_top1"])[:28] for q in query_ids]
    ax.barh(y_pos, pcts, color="#66C2A5", edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[q] for q in query_ids])
    ax.set_xlabel("Share of taxon-focused papers (%)")
    ax.set_title("RQ3 — Most common primary theme (#1 rank) with label")
    for i, (pct, nm) in enumerate(zip(pcts, names, strict=True)):
        ax.text(pct + 0.8, i, nm, va="center", fontsize=7, clip_on=False)
    ax.set_xlim(0, max(pcts) * 1.35 if pcts else 50)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_top_theme_labeled_horizontal")


def fig_rq3_theme_top3_heatmap(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    mat = np.array(
        [
            [
                float(m.loc[q, "theme_top1_pct"]),
                float(m.loc[q, "theme_top2_pct"]),
                float(m.loc[q, "theme_top3_pct"]),
            ]
            for q in query_ids
        ]
    )
    fig, ax = plt.subplots(figsize=(4.5, 4.8))
    im = ax.imshow(mat, aspect="auto", cmap="YlGnBu", vmin=0, vmax=max(mat.max(), 1))
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["#1 %", "#2 %", "#3 %"])
    ax.set_yticks(range(len(query_ids)))
    ax.set_yticklabels([labels[q] for q in query_ids])
    ax.set_title("RQ3 — Ranked theme shares (heatmap)")
    for i in range(len(query_ids)):
        for j in range(3):
            v = mat[i, j]
            ax.text(
                j,
                i,
                f"{v:.1f}",
                ha="center",
                va="center",
                color="white" if v > 25 else "black",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of papers")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_ranked_theme_pct_heatmap")


def fig_rq3_not_specified_only(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, tuple],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    vals = [float(m.loc[q, "theme_not_specified_pct"]) for q in query_ids]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    x = np.arange(len(query_ids))
    ax.bar(x, vals, color=[colors[q] for q in query_ids], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Not specified (%)")
    ax.set_title("RQ3 — Unspecified primary theme by taxon")
    ax.set_ylim(0, max(vals) * 1.15 if vals else 1)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_not_specified_pct_by_taxon")


# --- RQ4 (additional) ---


def fig_rq4_mean_vs_median_authors(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(query_ids))
    w = 0.35
    mean_v = [float(m.loc[q, "authors_mean"]) for q in query_ids]
    med_v = [float(m.loc[q, "authors_median"]) for q in query_ids]
    ax.bar(x - w / 2, mean_v, width=w, label="Mean", color="#8DA0CB", edgecolor="none")
    ax.bar(x + w / 2, med_v, width=w, label="Median", color="#FC8D62", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Author count")
    ax.set_title("RQ4A — Mean vs median authors (taxon-focused, 2010–2025)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq4_mean_vs_median_authors")


def fig_rq4_authors_applied_vs_taxonomic(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(8, 3.6))
    x = np.arange(len(query_ids))
    w = 0.35
    ap = [float(m.loc[q, "authors_mean_applied"]) for q in query_ids]
    tx = [float(m.loc[q, "authors_mean_taxonomic"]) for q in query_ids]
    ax.bar(x - w / 2, ap, width=w, label="Applied themes", color="#A6D854", edgecolor="none")
    ax.bar(x + w / 2, tx, width=w, label="Taxonomic theme", color="#FFD92F", edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Mean author count")
    ax.set_title("RQ4A — Team size by study type (theme-based split)")
    ax.legend()
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq4_mean_authors_applied_vs_taxonomic")


def fig_rq4_intl_collab_definitions_comparison(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    keys = [
        ("intl_collab_pct_overall", "Overall"),
        ("intl_collab_pct_known_only_overall", "Known affil."),
        ("intl_collab_pct_applied", "Applied"),
        ("intl_collab_pct_taxonomic", "Taxonomic"),
    ]
    fig, ax = plt.subplots(figsize=(11, 4.2))
    x = np.arange(len(query_ids))
    n_k = len(keys)
    w = 0.17
    for i, (col, name) in enumerate(keys):
        offset = (i - (n_k - 1) / 2) * w
        vals = [float(m.loc[q, col]) for q in query_ids]
        ax.bar(x + offset, vals, width=w * 0.92, label=name)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=22, ha="right")
    ax.set_ylabel("International collaboration %")
    ax.set_title("RQ4B — Collaboration rate under alternative denominators")
    ax.legend(ncol=4, loc="upper right", fontsize=7)
    hi = max(float(m.loc[q, col]) for q in query_ids for col, _ in keys)
    ax.set_ylim(0, min(100, hi * 1.12 + 2))
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq4_intl_collab_overall_known_applied_taxonomic")


def fig_rq4_affiliation_signal_coverage(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, tuple],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    vals = [float(m.loc[q, "intl_collab_info_coverage_pct"]) for q in query_ids]
    fig, ax = plt.subplots(figsize=(7.5, 3.5))
    x = np.arange(len(query_ids))
    ax.bar(x, vals, color=[colors[q] for q in query_ids], edgecolor="none")
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Papers with known affiliation signal (%)")
    ax.set_title("RQ4B — Coverage of affiliation-country heuristic")
    ax.set_ylim(0, 100)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq4_affiliation_country_signal_coverage")


def fig_rq4_scatter_mean_authors_vs_intl_known(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, tuple],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for q in query_ids:
        ax.scatter(
            float(m.loc[q, "authors_mean"]),
            float(m.loc[q, "intl_collab_pct_known_only_overall"]),
            s=120,
            color=colors[q],
            edgecolors="black",
            linewidths=0.4,
            zorder=3,
        )
        ax.annotate(labels[q], (float(m.loc[q, "authors_mean"]), float(m.loc[q, "intl_collab_pct_known_only_overall"])), fontsize=7, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel("Mean author count (2010–2025)")
    ax.set_ylabel("Intl collaboration % (known affiliations)")
    ax.set_title("RQ4 — Team size vs collaboration (by taxon)")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq4_scatter_mean_authors_vs_intl_collab_known")


def _coded_csv_manifest(query_ids: list[str]) -> dict[str, str | None]:
    out: dict[str, str | None] = {}
    for q in query_ids:
        p = PipelinePaths(q).coded
        out[q] = _sha256(p) if p.is_file() else None
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Build multi-taxon bibliometric figures from combined CSVs.")
    parser.add_argument(
        "--combined-dir",
        default="analysis/combined",
        help="Directory with yearly_publication_volume_by_query.csv and cross_taxa_metrics.csv.",
    )
    parser.add_argument(
        "--out-dir",
        default="analysis/combined/figures",
        help="Figure output directory (relative to project root).",
    )
    args = parser.parse_args()

    combined = PROJECT_ROOT / args.combined_dir
    out_dir = PROJECT_ROOT / args.out_dir
    yearly_path = combined / "yearly_publication_volume_by_query.csv"
    metrics_path = combined / "cross_taxa_metrics.csv"
    if not yearly_path.is_file():
        raise SystemExit(f"Missing {yearly_path}. Run analyze_cross_taxa_summary.py first.")
    if not metrics_path.is_file():
        raise SystemExit(f"Missing {metrics_path}. Run analyze_cross_taxa_summary.py first.")

    _setup_rc()
    cfg = load_queries_config()
    query_ids = _query_order(cfg)
    colors = _taxon_color_map(query_ids)
    labels = {q: _short_label(cfg, q) for q in query_ids}

    yearly = pd.read_csv(yearly_path)
    metrics = pd.read_csv(metrics_path)

    outputs: list[dict[str, str]] = []
    # RQ2 temporal (core)
    outputs.append(fig_rq2_temporal_facets(yearly, query_ids, colors, labels, out_dir))
    outputs.append(fig_rq2_temporal_log_overlay(yearly, query_ids, colors, labels, out_dir))
    outputs.append(fig_rq2_temporal_all_coded_facets(yearly, query_ids, colors, labels, out_dir))
    outputs.append(fig_rq2_temporal_all_coded_log_overlay(yearly, query_ids, colors, labels, out_dir))
    # RQ2 volume / sample
    outputs.append(fig_rq2_sample_size_raw_vs_focused(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq2_early_vs_recent_window_counts(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq2_pct_change_recent_vs_early(metrics, query_ids, labels, colors, out_dir))
    # RQ1
    outputs.append(fig_rq1_database_coverage(query_ids, labels, out_dir))
    outputs.append(fig_rq1_overlap_pct_and_ratio(query_ids, labels, out_dir))
    # RQ2 geography
    outputs.append(fig_rq2_geo_mean_stacked(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq2_geo_mean_grouped_vertical(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq2_geo_delta_heatmap(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq2_geo_delta_grouped_bars(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq2_geo_temporal_lines_by_taxon(query_ids, labels, out_dir))
    # RQ3
    outputs.append(fig_rq3_theme_top_shares(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq3_top1_theme_labeled_bars(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq3_theme_top3_heatmap(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq3_not_specified_only(metrics, query_ids, labels, colors, out_dir))
    # RQ4
    outputs.append(fig_rq4_authorship_collaboration(metrics, query_ids, labels, colors, out_dir))
    outputs.append(fig_rq4_mean_vs_median_authors(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq4_authors_applied_vs_taxonomic(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq4_intl_collab_definitions_comparison(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq4_affiliation_signal_coverage(metrics, query_ids, labels, colors, out_dir))
    outputs.append(fig_rq4_scatter_mean_authors_vs_intl_known(metrics, query_ids, labels, colors, out_dir))

    manifest = {
        "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
        "git_commit": _git_head(),
        "query_ids": query_ids,
        "inputs": {
            "yearly_publication_volume_by_query.csv": {
                "path": str(yearly_path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(yearly_path),
                "n_rows": int(len(yearly)),
            },
            "cross_taxa_metrics.csv": {
                "path": str(metrics_path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(metrics_path),
                "n_rows": int(len(metrics)),
            },
            "coded_scopus_api_csv_sha256_by_query": _coded_csv_manifest(query_ids),
        },
        "figures": outputs,
    }
    manifest_path = out_dir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(outputs)} figures under {out_dir.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
