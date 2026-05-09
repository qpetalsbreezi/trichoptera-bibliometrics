"""
Generate multi-taxon bibliometric figures for journal use (PDF + PNG).

Reads only committed-style artifacts under analysis/combined/:
  - yearly_publication_volume_by_query.csv
  - cross_taxa_metrics.csv
  - config/queries.json (taxon order and display labels)

RQ1 metrics are read from per-taxon coverage_report.txt (same as overall_bibliometric_report).

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

from analyze_overall_bibliometric_report import load_rq1_row  # noqa: E402
from lib.pipeline import PROJECT_ROOT, load_queries_config  # noqa: E402

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
    outputs.append(fig_rq2_temporal_facets(yearly, query_ids, colors, labels, out_dir))
    outputs.append(fig_rq2_temporal_log_overlay(yearly, query_ids, colors, labels, out_dir))
    outputs.append(fig_rq1_database_coverage(query_ids, labels, out_dir))
    outputs.append(fig_rq2_geo_mean_stacked(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq2_geo_delta_heatmap(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq3_theme_top_shares(metrics, query_ids, labels, out_dir))
    outputs.append(fig_rq4_authorship_collaboration(metrics, query_ids, labels, colors, out_dir))

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
        },
        "figures": outputs,
    }
    manifest_path = out_dir / "figures_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Wrote {len(outputs)} figures under {out_dir.relative_to(PROJECT_ROOT)}")
    print(f"Wrote {manifest_path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
