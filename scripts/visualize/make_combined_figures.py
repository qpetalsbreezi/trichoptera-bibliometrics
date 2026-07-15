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
import matplotlib.colors as mcolors  # noqa: E402
import matplotlib.ticker as mticker  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
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
from lib.pipeline import PROJECT_ROOT, PipelinePaths, load_queries_config, paper_query_order, paper_taxon_label  # noqa: E402

# ---------------------------------------------------------------------------
# Color design (journal / color-vision deficiency)
# - Taxa: Paul Tol "bright" subset (no yellow; strong pairwise separation).
# - Geography: Tol "muted"–style hues orthogonal to taxa blues/cyans.
# - RQ1 bars: semantic (indexed corpus vs intersection vs web-scale).
# - Comparisons (two-bar): single-hue or single-family ramps, not rainbow.
# ---------------------------------------------------------------------------
# Stable per-taxon colors (independent of bar/panel order).
_TAXON_HEX_BY_QUERY: dict[str, str] = {
    "mosquitoes": "#4477AA",
    "ephemeroptera": "#EE6677",
    "plecoptera": "#228833",
    "trichoptera": "#AA3377",
    "odonata": "#66CCEE",
}

# Same keys as Region_Category in analyze_cross_taxa_summary.
_REGION_HEX: dict[str, str] = {
    "South America": "#117733",
    "Asia": "#CC6677",
    "Europe": "#332288",
    "North America": "#6699CC",
    "Unknown": "#BBBBBB",
    "Other": "#AA4499",
}

# RQ1 — three databases (meaning-driven; hexes distinct from North America and Europe region colors).
_RQ1_SCOPUS_HEX = "#0D3D5C"
_RQ1_OVERLAP_HEX = "#5B9BD4"
_RQ1_GS_HEX = "#EE7733"

# RQ1 — secondary single-metric panels (hexes distinct from taxon green and purple in _TAXON_HEX).
_RQ1_OVERLAP_PCT_HEX = "#00796B"
_RQ1_GS_RATIO_HEX = "#88419D"

# Two-bar “before vs after” families (distinct roles; avoid reusing geo reds).
_PAIR_AUTHORS_TIME = ("#B8C9E0", "#1C3D5A")  # early, recent
_PAIR_WINDOW_COUNTS = ("#E6C8A8", "#7A3E1D")  # early window, recent window
_PAIR_SAMPLE_SCOPE = ("#D4DCE8", "#3D5A80")  # all coded, taxon-focused
_PAIR_MEAN_MEDIAN = ("#8DA0CB", "#FC8D62")  # mean, median (Tableau-style pairing)
_PAIR_APPLIED_TAX = ("#66A61E", "#E6AB02")  # applied, taxonomic (brown-yellow)

# RQ3 — ranked shares + NS (sequential purple ramp; first swatch ≠ Europe indigo used in geo charts).
_THEME_RANK_HEX = ["#5C4E75", "#8A7CA8", "#BEB3D4", "#AEAEAE"]

# RQ3 — theme categories for shift chart (Table 6); distinct from geo and taxon palettes.
_THEME_CATEGORY_HEX: dict[str, str] = {
    "Ecology/Behavior": "#228833",
    "Taxonomy/Systematics": "#AA3377",
    "Biomonitoring/Water Quality": "#4477AA",
    "Applied Ecology": "#EE7733",
    "Not Specified": "#BBBBBB",
}
_THEME_CATEGORY_SHORT: dict[str, str] = {
    "Ecology/Behavior": "Ecology",
    "Taxonomy/Systematics": "Taxonomy",
    "Biomonitoring/Water Quality": "Biomonitoring",
    "Applied Ecology": "Vector Mgmt.",
    "Not Specified": "Not spec.",
}
_THEME_PANEL_TITLE: dict[str, str] = {
    "Ecology/Behavior": "Ecology/Behavior",
    "Taxonomy/Systematics": "Taxonomy",
    "Biomonitoring/Water Quality": "Biomonitoring",
    "Applied Ecology": "Vector Management",
    "Not Specified": "Not Specified",
}

# RQ4B — four collaboration definitions (hexes distinct from taxon green and purple in _TAXON_HEX).
_INTL_DEFINITION_HEX = [
    "#555555",  # overall
    "#2A6F97",  # known affiliations (steel teal)
    "#D95F0E",  # applied subset
    "#8067AC",  # taxonomic subset
]

# Diverging Δpp heatmap (ColorBrewer-style, balanced around white).
_DIVERGING_PP = mcolors.LinearSegmentedColormap.from_list(
    "delta_pp",
    ["#2166AC", "#92C5DE", "#F7F7F7", "#F4A582", "#B2182B"],
)
_DELTA_DECREASE_HEX = "#2166AC"
_DELTA_INCREASE_HEX = "#B2182B"
_GEO_SHIFT_POS_HEX = "#228833"
_GEO_SHIFT_NEG_HEX = "#CC0000"

# Paper figures (Figures 1–4): shared typography
_PAPER_TITLE_SIZE = 10
_PAPER_TITLE_PAD = 6
_PAPER_SUPTITLE_Y = 1.02


def _paper_suptitle(fig: plt.Figure, text: str, *, y: float = _PAPER_SUPTITLE_Y) -> None:
    fig.suptitle(text, y=y, fontsize=_PAPER_TITLE_SIZE)


def _paper_panel_title(ax, text: str) -> None:
    ax.set_title(text, fontsize=_PAPER_TITLE_SIZE, pad=_PAPER_TITLE_PAD)


def _paper_figure_title(ax, text: str) -> None:
    ax.set_title(text, fontsize=_PAPER_TITLE_SIZE, pad=_PAPER_TITLE_PAD + 2)


def _setup_rc() -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "font.size": 9,
            "figure.titlesize": _PAPER_TITLE_SIZE,
            "axes.titlesize": _PAPER_TITLE_SIZE,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.grid": True,
            "grid.alpha": 0.25,
        }
    )


def _query_order(cfg: dict) -> list[str]:
    return paper_query_order(cfg)


def _taxon_color_map(query_ids: list[str]) -> dict[str, str]:
    fallback = ["#4477AA", "#EE6677", "#228833", "#AA3377", "#66CCEE"]
    out: dict[str, str] = {}
    for i, q in enumerate(query_ids):
        out[q] = _TAXON_HEX_BY_QUERY.get(q, fallback[i % len(fallback)])
    return out


def _short_label(cfg: dict, qid: str) -> str:
    return paper_taxon_label(qid, cfg)


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


def _save(
    fig: plt.Figure,
    out_dir: Path,
    stem: str,
    *,
    pad_inches: float | None = None,
    bbox_inches: str | None = "tight",
) -> dict[str, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"{stem}.pdf"
    png = out_dir / f"{stem}.png"
    save_kw: dict[str, object] = {}
    if bbox_inches is not None:
        save_kw["bbox_inches"] = bbox_inches
    if pad_inches is not None:
        save_kw["pad_inches"] = pad_inches
    fig.savefig(pdf, **save_kw)
    fig.savefig(png, **save_kw)
    plt.close(fig)
    return {"pdf": str(pdf.relative_to(PROJECT_ROOT)), "png": str(png.relative_to(PROJECT_ROOT))}


def _style_taxon_xaxis(
    ax,
    query_ids: list[str],
    labels: dict[str, str],
    *,
    fontsize: int = 8,
    horizontal: bool = True,
) -> None:
    """Center taxon names under grouped bars; avoid ha='right' drift on rotated labels."""
    x = np.arange(len(query_ids))
    ax.set_xticks(x)
    if horizontal:
        ax.set_xticklabels([labels[q] for q in query_ids], ha="center", fontsize=fontsize)
        ax.tick_params(axis="x", pad=6)
    else:
        ax.set_xticklabels(
            [labels[q] for q in query_ids],
            rotation=45,
            ha="right",
            rotation_mode="anchor",
            fontsize=fontsize,
        )
        ax.tick_params(axis="x", pad=2)


def fig_rq2_temporal_facets(
    yearly: pd.DataFrame,
    query_ids: list[str],
    colors: dict[str, str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    n = len(query_ids)
    fig, axes = plt.subplots(1, n, figsize=(2.4 * n, 2.8), sharex=True, sharey=False)
    if n == 1:
        axes = np.array([axes])
    years = np.arange(2010, 2026)
    for ax, q in zip(axes, query_ids):
        ax.axvspan(2010, 2015, color="#888888", alpha=0.07, zorder=0)
        ax.axvspan(2020, 2025, color="#888888", alpha=0.12, zorder=0)
        sub = yearly[yearly["query_id"] == q].sort_values("year")
        y = sub.set_index("year")["n_taxon_focused"].reindex(years, fill_value=0).values
        ax.fill_between(years, y, alpha=0.25, color=colors[q], zorder=2)
        ax.plot(years, y, color=colors[q], linewidth=1.6, zorder=3)
        _paper_panel_title(ax, labels[q])
        ax.set_xlim(2009.5, 2025.5)
        ax.set_xticks([2010, 2015, 2020, 2025])
        ax.set_xticklabels([2010, 2015, 2020, 2025], fontsize=7)
    axes[0].set_ylabel("Group-focused N")
    fig.supxlabel("Publication year", y=0.04, fontsize=9)
    _paper_suptitle(fig, "Group-focused publication volume by year")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_temporal_taxon_focused_facets")


def fig_rq2_temporal_log_overlay(
    yearly: pd.DataFrame,
    query_ids: list[str],
    colors: dict[str, str],
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
    ax.bar(x - w, df["scopus"], width=w, label="Scopus total", color=_RQ1_SCOPUS_HEX, edgecolor="none")
    ax.bar(x, df["overlap"], width=w, label="Overlap (both)", color=_RQ1_OVERLAP_HEX, edgecolor="none")
    ax.bar(x + w, df["gs"], width=w, label="Google Scholar total", color=_RQ1_GS_HEX, edgecolor="none")
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
    for i, (col, name) in enumerate(cats):
        reg_key = ["South America", "Asia", "Europe", "North America", "Unknown"][i]
        ax.barh(
            [labels[q] for q in query_ids],
            mat[:, i],
            left=left,
            label=name,
            color=_REGION_HEX[reg_key],
            edgecolor="white",
            linewidth=0.4,
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
        ("geo_delta_pp_south_america_2010_2015_vs_2020_2025", "S. America"),
        ("geo_delta_pp_asia_2010_2015_vs_2020_2025", "Asia"),
        ("geo_delta_pp_europe_2010_2015_vs_2020_2025", "Europe"),
        ("geo_delta_pp_north_america_2010_2015_vs_2020_2025", "N. America"),
    ]
    m = metrics.set_index("query_id").loc[query_ids]
    mat = np.array([[float(m.loc[q, c]) for c, _ in cols] for q in query_ids])
    fig, ax = plt.subplots(figsize=(5.5, 4))
    vmax = max(np.abs(mat).max(), 1e-6)
    im = ax.imshow(mat, aspect="auto", cmap=_DIVERGING_PP, vmin=-vmax, vmax=vmax)
    ax.set_xticks(range(len(cols)))
    ax.set_xticklabels([lbl for _, lbl in cols], rotation=20, ha="right")
    ax.set_yticks(range(len(query_ids)))
    ax.set_yticklabels([labels[q] for q in query_ids])
    ax.set_title(f"Δ continental % (mean {xtax.RECENT_WINDOW_LABEL} minus mean {xtax.EARLY_WINDOW_LABEL}, pp)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Percentage points")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_geo_delta_heatmap")


def _short_theme_label(name: str) -> str:
    mapping = {
        "Ecology/Behavior": "Ecology",
        "Taxonomy/Systematics": "Taxonomy",
        "Biomonitoring/Water Quality": "Biomonitoring",
        "Applied Ecology": "Applied Ecol.",
        "Physiology": "Physiology",
        "Conservation": "Conservation",
    }
    return mapping.get(name, name[:18])


def fig_rq3_theme_top_shares(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(9.5, 4.6))
    x = np.arange(len(query_ids))
    w = 0.22
    parts = [
        ("theme_top1_pct", "theme_top1", "Rank #1"),
        ("theme_top2_pct", "theme_top2", "Rank #2"),
        ("theme_top3_pct", "theme_top3", "Rank #3"),
    ]
    ymax = 0.0
    for i, (col_pct, col_name, leg) in enumerate(parts):
        offsets = x + (i - 1) * w
        heights = [float(m.loc[q, col_pct]) for q in query_ids]
        ymax = max(ymax, max(heights) if heights else 0.0)
        ax.bar(
            offsets,
            heights,
            width=w,
            label=leg,
            color=_THEME_RANK_HEX[i],
            edgecolor="white",
            linewidth=0.35,
        )
        for j, q in enumerate(query_ids):
            nm = _short_theme_label(str(m.loc[q, col_name]))
            ax.text(
                offsets[j],
                heights[j] + 0.6,
                nm,
                ha="center",
                va="bottom",
                fontsize=6,
                rotation=55,
                rotation_mode="anchor",
            )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Share of taxon-focused papers (%)")
    ax.set_title("Ranked primary themes by taxon (2010–2025; labels name each rank)")
    ax.set_ylim(0, ymax * 1.28 if ymax else 50)
    ax.legend(loc="upper right", fontsize=7, title="Rank (excl. Not Specified)")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_theme_ranked_shares")


def fig_rq3_theme_shift_delta_grouped_bars(
    theme_shift: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    themes = list(xtax.RQ3_THEME_SHIFT_THEMES)
    pivot = theme_shift.pivot(index="theme", columns="query_id", values="delta_pp")
    pivot = pivot.reindex(themes).reindex(columns=query_ids)
    fig, ax = plt.subplots(figsize=(11, 4.5))
    x = np.arange(len(query_ids))
    n_t = len(themes)
    w = 0.14
    abs_max = 1.0
    for i, theme in enumerate(themes):
        offset = (i - (n_t - 1) / 2) * w
        heights = [float(pivot.loc[theme, q]) for q in query_ids]
        abs_max = max(abs_max, max(abs(h) for h in heights))
        ax.bar(
            x + offset,
            heights,
            width=w * 0.92,
            label=_THEME_CATEGORY_SHORT.get(theme, theme[:12]),
            color=_THEME_CATEGORY_HEX.get(theme, "#888888"),
            edgecolor="white",
            linewidth=0.35,
        )
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Δ percentage points")
    ax.set_title(f"Theme share shift ({xtax.RECENT_WINDOW_LABEL} minus {xtax.EARLY_WINDOW_LABEL})")
    ylim = abs_max * 1.12
    ax.set_ylim(-ylim, ylim)
    ax.legend(ncol=5, loc="upper right", fontsize=7)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_theme_shift_delta_grouped_bars")


def fig_rq3_theme_shift_delta_facets(
    theme_shift: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    themes = list(xtax.RQ3_THEME_SHIFT_THEMES)
    pivot = theme_shift.pivot(index="theme", columns="query_id", values="delta_pp")
    pivot = pivot.reindex(themes).reindex(columns=query_ids)
    n = len(themes)
    fig, axes = plt.subplots(1, n, figsize=(2.55 * n, 3.6), sharex=True)
    if n == 1:
        axes = np.array([axes])
    x = np.arange(len(query_ids))
    for ax, theme in zip(axes, themes):
        heights = [float(pivot.loc[theme, q]) for q in query_ids]
        ax.bar(
            x,
            heights,
            width=0.62,
            color=[colors[q] for q in query_ids],
            edgecolor="white",
            linewidth=0.45,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        _paper_panel_title(ax, _THEME_PANEL_TITLE.get(theme, theme))
        _style_taxon_xaxis(ax, query_ids, labels, fontsize=7, horizontal=False)
        ymax = max((abs(h) for h in heights), default=1.0)
        ax.set_ylim(-ymax * 1.18, ymax * 1.18)
    axes[0].set_ylabel("Δ percentage points")
    taxon_handles = [
        Patch(facecolor=colors[q], edgecolor="white", linewidth=0.45, label=labels[q])
        for q in query_ids
    ]
    fig.legend(
        handles=taxon_handles,
        loc="upper center",
        ncol=len(query_ids),
        bbox_to_anchor=(0.5, 1.08),
        fontsize=7,
        frameon=False,
    )
    _paper_suptitle(
        fig,
        f"Theme share shift ({xtax.RECENT_WINDOW_LABEL} minus {xtax.EARLY_WINDOW_LABEL}; separate y-axis per panel)",
    )
    fig.tight_layout()
    fig.subplots_adjust(top=0.78)
    return _save(fig, out_dir, "fig_rq3_theme_shift_delta_facets")


def _yearly_mean_authors_long(query_ids: list[str]) -> pd.DataFrame:
    """Yearly mean author count per taxon (same filters as cross_taxa_summary / Table 7)."""
    rows: list[dict] = []
    for q in query_ids:
        paths = PipelinePaths(q)
        df_in = pd.read_csv(paths.coded, low_memory=False)
        df = xtax.filter_analysis_frame(df_in)
        merged, has_authors = xtax.merge_authors_like_rq4(df, paths)
        use_actual = has_authors and "Author_Count_Actual" in merged.columns
        author_count = xtax.compute_author_count(merged, use_actual)
        merged = merged[author_count > 0].copy()
        merged["AuthorCount"] = author_count[author_count > 0]
        for year, sub in merged.groupby("Year"):
            yi = int(year)
            if 2010 <= yi <= 2025:
                rows.append(
                    {
                        "query_id": q,
                        "year": yi,
                        "mean_authors": float(sub["AuthorCount"].mean()),
                    }
                )
    return pd.DataFrame(rows)


def _plot_mean_authors_by_year(
    ax,
    yearly_authors: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
) -> None:
    """Left panel of fig_rq4_authors_and_intl_collab: yearly mean OpenAlex author counts."""
    years = np.arange(2010, 2026)
    ax.axvspan(2010, 2015, color="#888888", alpha=0.07, zorder=0)
    ax.axvspan(2020, 2025, color="#888888", alpha=0.12, zorder=0)
    for q in query_ids:
        sub = yearly_authors[yearly_authors["query_id"] == q].sort_values("year")
        y = sub.set_index("year")["mean_authors"].reindex(years)
        ax.plot(
            years,
            y,
            label=labels[q],
            color=colors[q],
            linewidth=1.8,
            marker="o",
            markersize=3.5,
            zorder=3,
        )
    ax.set_xlim(2009.5, 2025.5)
    ax.set_xticks([2010, 2015, 2020, 2025])
    ax.set_ylabel("Mean author count")
    _paper_panel_title(ax, "Mean authors per paper (2010–2025)")
    ax.legend(fontsize=7, loc="upper left", frameon=True)


def fig_rq4_mean_authors_by_year(
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    """Standalone left panel: mean authors per paper by year (Figure 4 left)."""
    yearly_authors = _yearly_mean_authors_long(query_ids)
    fig, ax = plt.subplots(1, 1, figsize=(6.2, 4.1))
    _plot_mean_authors_by_year(ax, yearly_authors, query_ids, labels, colors)
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq4_mean_authors_by_year")


def fig_rq4_authorship_collaboration(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    yearly_authors = _yearly_mean_authors_long(query_ids)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.1))
    _plot_mean_authors_by_year(ax1, yearly_authors, query_ids, labels, colors)

    x = np.arange(len(query_ids))
    intl = [float(m.loc[q, "intl_collab_pct_overall"]) for q in query_ids]
    intl_w = 0.52
    _, c_recent = _PAIR_AUTHORS_TIME
    ax2.bar(
        x,
        intl,
        width=intl_w,
        color=c_recent,
        edgecolor="white",
        linewidth=0.6,
    )
    _style_taxon_xaxis(ax2, query_ids, labels)
    ax2.set_ylabel("International collaboration %")
    _paper_panel_title(ax2, "International collaboration (2010–2025, all papers)")
    ax2.set_ylim(0, max(intl) * 1.15 if intl else 1)
    _paper_suptitle(fig, "Authorship and international collaboration by taxon")
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
    ax1.bar(x, overlap_pct, color=_RQ1_OVERLAP_PCT_HEX, edgecolor="none")
    ax1.set_xticks(x)
    ax1.set_xticklabels([labels[q] for q in query_ids], rotation=22, ha="right")
    ax1.set_ylabel("Overlap / Scopus (%)")
    ax1.set_title("RQ1 — Records appearing in both databases")
    ax1.set_ylim(0, min(105, max(overlap_pct) * 1.2 if overlap_pct else 100))

    ax2.bar(x, ratios, color=_RQ1_GS_RATIO_HEX, edgecolor="none")
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
    c_raw, c_foc = _PAIR_SAMPLE_SCOPE
    ax.bar(x - w / 2, raw, width=w, label="All coded (2010–2025)", color=c_raw, edgecolor="none")
    ax.bar(x + w / 2, foc, width=w, label="Taxon-focused (2010–2025)", color=c_foc, edgecolor="none")
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
    c_early, c_recent = _PAIR_WINDOW_COUNTS
    ax.bar(x - w / 2, early, width=w, label="Taxon-focused 2010–2015", color=c_early, edgecolor="none")
    ax.bar(x + w / 2, recent, width=w, label="Taxon-focused 2020–2025", color=c_recent, edgecolor="none")
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
    colors: dict[str, str],
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
    colors: dict[str, str],
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
    colors: dict[str, str],
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
                color=_REGION_HEX.get(reg, "#222222"),
                linewidth=1.35,
            )
        ax.set_title(labels[q])
        ax.set_xlim(2009.5, 2025.5)
        ax.set_xticks([2010, 2015, 2020, 2025])
        # Cap at 50% so typical continental shares use more vertical space (full 0–100 looks squeezed).
        ax.set_ylim(0, 50)
    axes[0].set_ylabel("Share of taxon-focused papers (%)")
    fig.supxlabel("Publication year", y=0.02)
    handles, leg_labels = (legend_ax or axes[0]).get_legend_handles_labels()
    if handles:
        fig.legend(handles, leg_labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 1.08), fontsize=7)
    fig.suptitle("RQ2 — Continental composition over time (yearly %; y-axis 0–50%)", y=1.12)
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
    reg_keys = ["South America", "Asia", "Europe", "North America", "Unknown"]
    for i, (col, name) in enumerate(cats):
        offset = (i - (n_c - 1) / 2) * w
        heights = [float(m.loc[q, col]) for q in query_ids]
        ax.bar(
            x + offset,
            heights,
            width=w * 0.95,
            label=name,
            color=_REGION_HEX[reg_keys[i]],
            edgecolor="white",
            linewidth=0.35,
        )
    ax.set_xticks(x)
    ax.set_xticklabels([labels[q] for q in query_ids], rotation=20, ha="right")
    ax.set_ylabel("Mean yearly % (2010–2025)")
    ax.set_title("RQ2 — Mean continental share by taxon (grouped)")
    ax.legend(ncol=5, loc="upper right", fontsize=7)
    ax.set_ylim(0, max(40, max(float(m.loc[q, c]) for q in query_ids for c, _ in cats) * 1.1))
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq2_geo_mean_continental_grouped_bars")


def _symmetric_pp_ylim(ax, values: list[float], *, step: int = 10) -> float:
    """Symmetric Δpp axis so zero sits in the vertical center of the plot."""
    abs_max = max((abs(v) for v in values), default=1.0)
    ylim = step * int(np.ceil(abs_max / step))
    ylim = max(ylim, step)
    ax.set_ylim(-ylim, ylim)
    ax.yaxis.set_major_locator(mticker.MultipleLocator(step))
    return ylim


def fig_rq2_geo_delta_grouped_bars(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    cols = [
        ("geo_delta_pp_south_america_2010_2015_vs_2020_2025", "South America"),
        ("geo_delta_pp_asia_2010_2015_vs_2020_2025", "Asia"),
        ("geo_delta_pp_europe_2010_2015_vs_2020_2025", "Europe"),
        ("geo_delta_pp_north_america_2010_2015_vs_2020_2025", "North America"),
    ]
    m = metrics.set_index("query_id").loc[query_ids]
    all_vals = [float(m.loc[q, col]) for q in query_ids for col, _ in cols]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    n_cont = len(cols)
    n_tax = len(query_ids)
    x = np.arange(n_cont)
    w = 0.14
    for i, q in enumerate(query_ids):
        offset = (i - (n_tax - 1) / 2) * w
        heights = [float(m.loc[q, col]) for col, _ in cols]
        ax.bar(
            x + offset,
            heights,
            width=w * 0.92,
            label=labels[q],
            color=colors[q],
            edgecolor="white",
            linewidth=0.35,
            zorder=3,
        )
    ax.axhline(0, color="black", linewidth=0.9, zorder=2)
    _symmetric_pp_ylim(ax, all_vals, step=10)
    ax.set_xlim(-0.55, n_cont - 0.45)
    ax.set_xticks(x)
    ax.set_xticklabels([name for _, name in cols], ha="center", fontsize=8)
    ax.tick_params(axis="x", pad=6)
    ax.set_ylabel("Δ percentage points")
    _paper_figure_title(ax, f"Continental shift ({xtax.EARLY_WINDOW_LABEL} vs {xtax.RECENT_WINDOW_LABEL} mean share)")
    ax.legend(
        ncol=len(query_ids),
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        fontsize=7,
        frameon=False,
    )
    fig.subplots_adjust(bottom=0.22, top=0.90)
    _symmetric_pp_ylim(ax, all_vals, step=10)
    return _save(fig, out_dir, "fig_rq2_geo_delta_continental_grouped_bars")


def _annotate_pp_barh(ax, val: float, xlim: float) -> None:
    """Place Δpp label just beyond the bar tip, clear of the zero line."""
    label = f"{val:+.1f}"
    pad = max(1.0, xlim * 0.025)
    if val >= 0:
        x_pos = max(val + pad, pad)
        ha = "left"
    else:
        x_pos = min(val - pad, -pad)
        ha = "right"
    ax.text(
        x_pos,
        0,
        label,
        ha=ha,
        va="center",
        fontsize=7,
        color="#1a1a1a",
        clip_on=False,
    )


def fig_rq2_geo_delta_compositional_matrix(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    """Continent × taxon matrix of horizontal Δpp bars (green = increase, red = decrease)."""
    cols = [
        ("geo_delta_pp_south_america_2010_2015_vs_2020_2025", "S. America"),
        ("geo_delta_pp_asia_2010_2015_vs_2020_2025", "Asia"),
        ("geo_delta_pp_europe_2010_2015_vs_2020_2025", "Europe"),
        ("geo_delta_pp_north_america_2010_2015_vs_2020_2025", "N. America"),
    ]
    m = metrics.set_index("query_id").loc[query_ids]
    all_vals = [float(m.loc[q, col]) for q in query_ids for col, _ in cols]
    xlim = max(10, 10 * int(np.ceil(max(abs(v) for v in all_vals) / 10)))

    n_rows = len(cols)
    n_cols = len(query_ids)
    title_text = f"Continental compositional shift ({xtax.EARLY_WINDOW_LABEL} vs {xtax.RECENT_WINDOW_LABEL} mean share)"

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(2.05 * n_cols, 0.65 * n_rows + 0.95),
        sharex=True,
        gridspec_kw={"hspace": 0.08, "wspace": 0.10},
    )
    if n_rows == 1:
        axes = np.array([axes])
    if n_cols == 1:
        axes = axes.reshape(-1, 1)

    grid_ticks = list(range(-xlim, xlim + 1, 10))
    for r, (col_key, region) in enumerate(cols):
        for c, q in enumerate(query_ids):
            ax = axes[r, c]
            val = float(m.loc[q, col_key])
            color = _GEO_SHIFT_POS_HEX if val >= 0 else _GEO_SHIFT_NEG_HEX
            for tick in grid_ticks:
                if tick != 0:
                    ax.axvline(tick, color="#E6E6E6", linewidth=0.55, zorder=1)
            ax.barh(0, val, height=0.58, color=color, edgecolor="none", zorder=3)
            ax.axvline(0, color="#444444", linewidth=0.65, zorder=2)
            ax.set_xlim(-xlim, xlim)
            if r == n_rows - 1:
                ax.set_ylim(-0.78, 0.48)
            else:
                ax.set_ylim(-0.65, 0.65)
            ax.set_yticks([])
            ax.grid(False)
            _annotate_pp_barh(ax, val, xlim)
            for spine in ("top", "right", "left"):
                ax.spines[spine].set_visible(False)
            ax.spines["bottom"].set_visible(r == n_rows - 1)
            ax.spines["bottom"].set_linewidth(0.55)
            if r < n_rows - 1:
                ax.tick_params(axis="x", bottom=False, labelbottom=False)
            else:
                ax.xaxis.set_major_locator(mticker.MultipleLocator(10))
                ax.tick_params(axis="x", labelsize=6.5, length=2, pad=5)
                if c == n_cols // 2:
                    ax.set_xlabel(
                        "Δ percentage points (compositional shift)",
                        fontsize=9,
                        labelpad=7,
                    )

        ax_left = axes[r, 0]
        ax_left.set_ylabel(
            region,
            rotation=0,
            ha="right",
            va="center",
            fontsize=8,
            labelpad=8,
        )
        ax_left.yaxis.set_label_coords(-0.20, 0.5)

    # Reserve headroom above the data grid, then place headers and title in figure coords.
    fig.subplots_adjust(left=0.11, right=0.99, top=0.50, bottom=0.17, hspace=0.08, wspace=0.10)
    fig.text(
        0.5,
        0.865,
        title_text,
        ha="center",
        va="center",
        fontsize=_PAPER_TITLE_SIZE,
        transform=fig.transFigure,
    )
    for c, q in enumerate(query_ids):
        pos = axes[0, c].get_position()
        fig.text(
            pos.x0 + pos.width / 2,
            0.72,
            labels[q],
            ha="center",
            va="center",
            fontsize=_PAPER_TITLE_SIZE,
            transform=fig.transFigure,
        )

    return _save(
        fig,
        out_dir,
        "fig_rq2_geo_delta_compositional_matrix",
        bbox_inches=None,
    )


# --- RQ3 (additional) ---


def fig_rq3_top1_theme_labeled_bars(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
    out_dir: Path,
) -> dict[str, str]:
    m = metrics.set_index("query_id").loc[query_ids]
    fig, ax = plt.subplots(figsize=(8.5, 4.2))
    y_pos = np.arange(len(query_ids))
    pcts = [float(m.loc[q, "theme_top1_pct"]) for q in query_ids]
    names = [str(m.loc[q, "theme_top1"])[:28] for q in query_ids]
    ax.barh(y_pos, pcts, color=[colors[q] for q in query_ids], edgecolor="none")
    ax.set_yticks(y_pos)
    ax.set_yticklabels([labels[q] for q in query_ids])
    ax.set_xlabel("Share of taxon-focused papers (%)")
    ax.set_title("RQ3 — Most common primary theme (#1 rank) with label")
    for i, (pct, nm) in enumerate(zip(pcts, names)):
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
    im = ax.imshow(mat, aspect="auto", cmap="BuPu", vmin=0, vmax=max(mat.max(), 1))
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
                color="white" if v > 22 else "#1a1a1a",
                fontsize=8,
            )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="% of papers")
    fig.tight_layout()
    return _save(fig, out_dir, "fig_rq3_ranked_theme_pct_heatmap")


def fig_rq3_not_specified_only(
    metrics: pd.DataFrame,
    query_ids: list[str],
    labels: dict[str, str],
    colors: dict[str, str],
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
    c_mean, c_med = _PAIR_MEAN_MEDIAN
    ax.bar(x - w / 2, mean_v, width=w, label="Mean", color=c_mean, edgecolor="none")
    ax.bar(x + w / 2, med_v, width=w, label="Median", color=c_med, edgecolor="none")
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
    c_ap, c_tx = _PAIR_APPLIED_TAX
    ax.bar(x - w / 2, ap, width=w, label="Applied themes", color=c_ap, edgecolor="none")
    ax.bar(x + w / 2, tx, width=w, label="Taxonomic theme", color=c_tx, edgecolor="none")
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
        ax.bar(
            x + offset,
            vals,
            width=w * 0.92,
            label=name,
            color=_INTL_DEFINITION_HEX[i],
            edgecolor="white",
            linewidth=0.35,
        )
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
    colors: dict[str, str],
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
    colors: dict[str, str],
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
            edgecolors="#2C2C2C",
            linewidths=0.5,
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
    parser.add_argument(
        "--only",
        default="",
        help="Comma-separated figure stems to build (e.g. fig_rq4_mean_authors_by_year). Default: all.",
    )
    args = parser.parse_args()

    combined = PROJECT_ROOT / args.combined_dir
    out_dir = PROJECT_ROOT / args.out_dir
    yearly_path = combined / "yearly_publication_volume_by_query.csv"
    metrics_path = combined / "cross_taxa_metrics.csv"
    theme_shift_path = combined / "theme_shift_by_query.csv"
    if not yearly_path.is_file():
        raise SystemExit(f"Missing {yearly_path}. Run analyze_cross_taxa_summary.py first.")
    if not metrics_path.is_file():
        raise SystemExit(f"Missing {metrics_path}. Run analyze_cross_taxa_summary.py first.")
    if not theme_shift_path.is_file():
        raise SystemExit(f"Missing {theme_shift_path}. Run analyze_cross_taxa_summary.py first.")

    _setup_rc()
    cfg = load_queries_config()
    query_ids = _query_order(cfg)
    colors = _taxon_color_map(query_ids)
    labels = {q: _short_label(cfg, q) for q in query_ids}

    yearly = pd.read_csv(yearly_path)
    metrics = pd.read_csv(metrics_path)
    theme_shift = pd.read_csv(theme_shift_path)

    only = {s.strip() for s in args.only.split(",") if s.strip()}
    if only:
        builders: dict[str, object] = {
            "fig_rq2_temporal_taxon_focused_facets": lambda: fig_rq2_temporal_facets(
                yearly, query_ids, colors, labels, out_dir
            ),
            "fig_rq3_theme_shift_delta_facets": lambda: fig_rq3_theme_shift_delta_facets(
                theme_shift, query_ids, labels, colors, out_dir
            ),
            "fig_rq3_theme_shift_delta_grouped_bars": lambda: fig_rq3_theme_shift_delta_grouped_bars(
                theme_shift, query_ids, labels, out_dir
            ),
            "fig_rq4_mean_authors_by_year": lambda: fig_rq4_mean_authors_by_year(
                query_ids, labels, colors, out_dir
            ),
            "fig_rq4_authors_and_intl_collab": lambda: fig_rq4_authorship_collaboration(
                metrics, query_ids, labels, colors, out_dir
            ),
        }
        unknown = only - set(builders)
        if unknown:
            raise SystemExit(f"Unknown --only figure(s): {', '.join(sorted(unknown))}")
        outputs = [builders[stem]() for stem in sorted(only)]
        manifest = {
            "generated_at": pd.Timestamp.now(tz="UTC").isoformat(),
            "git_commit": _git_head(),
            "query_ids": query_ids,
            "only": sorted(only),
            "figures": outputs,
        }
        manifest_path = out_dir / "figures_manifest.json"
        out_dir.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Wrote {len(outputs)} figure(s) to {out_dir.relative_to(PROJECT_ROOT)}/")
        for o in outputs:
            print(f"  {o['png']}")
        return

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
    outputs.append(fig_rq2_geo_delta_grouped_bars(metrics, query_ids, labels, colors, out_dir))
    outputs.append(fig_rq2_geo_delta_compositional_matrix(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq2_geo_temporal_lines_by_taxon(query_ids, labels, out_dir))
    # RQ3
    outputs.append(fig_rq3_theme_top_shares(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq3_theme_shift_delta_grouped_bars(theme_shift, query_ids, labels, out_dir))
    outputs.append(fig_rq3_theme_shift_delta_facets(theme_shift, query_ids, labels, colors, out_dir))
    outputs.append(fig_rq3_top1_theme_labeled_bars(metrics, query_ids, labels, colors, out_dir))
    outputs.append(fig_rq3_theme_top3_heatmap(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq3_not_specified_only(metrics, query_ids, labels, colors, out_dir))
    # RQ4
    outputs.append(fig_rq4_authorship_collaboration(metrics, query_ids, labels, colors, out_dir))
    outputs.append(fig_rq4_mean_authors_by_year(query_ids, labels, colors, out_dir))
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
            "theme_shift_by_query.csv": {
                "path": str(theme_shift_path.relative_to(PROJECT_ROOT)),
                "sha256": _sha256(theme_shift_path),
                "n_rows": int(len(theme_shift)),
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
