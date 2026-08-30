"""Build country×country international collaboration matrices (top 6).

Uses group-focused Article/Review papers (2010–2025) with OpenAlex
Author_Country_Codes and ≥2 distinct author countries.
"""
from __future__ import annotations

import itertools
import sys
from collections import Counter, defaultdict
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from analyze.analyze_cross_taxa_summary import (  # noqa: E402
    EARLY_WINDOW,
    RECENT_WINDOW,
    filter_analysis_frame,
    merge_authors_like_rq4,
)
from lib.pipeline import (  # noqa: E402
    PROJECT_ROOT,
    PipelinePaths,
    load_queries_config,
    normalize_research_theme,
    paper_query_order,
    paper_taxon_label,
)

OUT = PROJECT_ROOT / "analysis" / "combined" / "country_collaboration_matrices"
TOP_N = 6


def parse_codes(s) -> list[str]:
    if pd.isna(s) or not str(s).strip() or str(s).lower() == "nan":
        return []
    toks = [t.strip().upper() for t in str(s).replace("|", ";").replace(",", ";").split(";")]
    return sorted({t for t in toks if len(t) == 2 and t.isalpha()})


def fmt_pct_change(early: int, recent: int) -> str:
    if early == 0 and recent == 0:
        return "n/a"
    if early == 0:
        return "n/a"
    return f"{(recent - early) / early * 100:.1f}%"


def build_for_query(query_id: str, label: str) -> dict:
    paths = PipelinePaths(query_id)
    df = filter_analysis_frame(pd.read_csv(paths.coded, low_memory=False))
    n_focused = len(df)
    merged, has = merge_authors_like_rq4(df, paths)
    if not has or "Author_Country_Codes" not in merged.columns:
        raise SystemExit(f"{query_id}: missing Author_Country_Codes")

    merged = merged.copy()
    merged["codes"] = merged["Author_Country_Codes"].map(parse_codes)
    merged["n_countries"] = merged["codes"].map(len)
    intl = merged[merged["n_countries"] >= 2].copy()
    n_intl = len(intl)

    country_involvement = Counter()
    for codes in intl["codes"]:
        for c in codes:
            country_involvement[c] += 1
    top = [c for c, _ in country_involvement.most_common(TOP_N)]

    def window_mask(years: tuple[int, int]):
        return intl["Year"].between(*years)

    pair_total: Counter = Counter()
    pair_early: Counter = Counter()
    pair_recent: Counter = Counter()
    diag_total: Counter = Counter()
    diag_early: Counter = Counter()
    diag_recent: Counter = Counter()

    for _, row in intl.iterrows():
        codes = [c for c in row["codes"] if c in top]
        if not codes:
            continue
        y = int(row["Year"])
        in_early = EARLY_WINDOW[0] <= y <= EARLY_WINDOW[1]
        in_recent = RECENT_WINDOW[0] <= y <= RECENT_WINDOW[1]
        for c in set(codes):
            diag_total[c] += 1
            if in_early:
                diag_early[c] += 1
            if in_recent:
                diag_recent[c] += 1
        for a, b in itertools.combinations(sorted(set(codes)), 2):
            pair_total[(a, b)] += 1
            if in_early:
                pair_early[(a, b)] += 1
            if in_recent:
                pair_recent[(a, b)] += 1

    # Display matrix with pct change
    rows = []
    def cell_label(n: int, e: int, r: int) -> str:
        pct = fmt_pct_change(e, r)
        if pct == "n/a":
            return f"{n} (n/a)"
        if pct == "0.0%":
            return f"{n} (0%)"
        if pct.startswith("-"):
            return f"{n} ({pct})"
        return f"{n} (+{pct})"

    for a in top:
        row = {}
        for b in top:
            if a == b:
                row[b] = str(diag_total[a])
            else:
                key = tuple(sorted((a, b)))
                row[b] = cell_label(pair_total[key], pair_early[key], pair_recent[key])
        rows.append(row)

    mat = pd.DataFrame(rows, index=top, columns=top)

    # Counts-only matrix
    counts_rows = []
    for a in top:
        crow = {}
        for b in top:
            if a == b:
                crow[b] = diag_total[a]
            else:
                crow[b] = pair_total[tuple(sorted((a, b)))]
        counts_rows.append(crow)
    counts = pd.DataFrame(counts_rows, index=top, columns=top)

    # Long pair detail
    pair_rows = []
    for a, b in itertools.combinations(top, 2):
        key = tuple(sorted((a, b)))
        n = pair_total[key]
        e = pair_early[key]
        r = pair_recent[key]
        pct = fmt_pct_change(e, r)
        pair_rows.append(
            {
                "group": label,
                "country_a": key[0],
                "country_b": key[1],
                "n_total": n,
                "n_early_2010_2015": e,
                "n_recent_2020_2025": r,
                "pct_change": None if pct == "n/a" else float(pct.replace("%", "")),
                "cell": cell_label(n, e, r),
            }
        )
    pairs_long = pd.DataFrame(pair_rows)

    return {
        "label": label,
        "query_id": query_id,
        "n_focused": n_focused,
        "n_intl": n_intl,
        "top": top,
        "mat": mat,
        "counts": counts,
        "pairs_long": pairs_long,
        "country_involvement": country_involvement,
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    cfg = load_queries_config()
    queries = paper_query_order(cfg, list((cfg.get("queries") or {}).keys()))

    html_parts = [
        "<html><head><meta charset='utf-8'><title>Country collaboration matrices (top 6)</title>",
        "<style>body{font-family:Georgia,serif;margin:2rem} table{border-collapse:collapse;margin:1rem 0}"
        "th,td{border:1px solid #333;padding:4px 8px;font-size:12px} th{background:#eee}</style></head><body>",
        "<h1>Country×country international collaboration (top 6)</h1>",
        "<p>Group-focused Article/Review, 2010–2025. Cells: total N (+ early→recent % change). "
        "Diagonal: international articles involving that country.</p>",
    ]

    stacked_counts = []
    all_pairs = []
    summary_lines = []

    for q in queries:
        label = paper_taxon_label(q, cfg)
        print(f"Building {label} ({q})...")
        res = build_for_query(q, label)

        mat_path = OUT / f"{q}_top6_counts_with_pct_change.csv"
        counts_path = OUT / f"{q}_top6_counts.csv"
        pairs_path = OUT / f"{q}_top6_pairs_long.csv"
        detail_path = OUT / f"{q}_top6_pair_detail.csv"

        res["mat"].to_csv(mat_path)
        res["counts"].to_csv(counts_path)
        res["pairs_long"].to_csv(pairs_path, index=False)
        res["pairs_long"].to_csv(detail_path, index=False)

        # pct of intl papers involving each top country
        inv = pd.DataFrame(
            {
                "country": res["top"],
                "n_intl_papers_involving": [res["country_involvement"][c] for c in res["top"]],
                "pct_of_intl": [
                    round(100 * res["country_involvement"][c] / res["n_intl"], 1) if res["n_intl"] else 0.0
                    for c in res["top"]
                ],
            }
        )
        inv.to_csv(OUT / f"{q}_top6_pct_of_intl_papers.csv", index=False)

        c = res["counts"].copy()
        c.insert(0, "group", label)
        stacked_counts.append(c.reset_index().rename(columns={"index": "country"}))
        all_pairs.append(res["pairs_long"])

        summary_lines.append(
            f"{label} — {res['n_intl']:,} international articles of {res['n_focused']:,} group-focused"
        )
        print(f"  {summary_lines[-1]}")
        print(f"  top6: {', '.join(res['top'])}")

        html_parts.append(f"<h2>{label}</h2>")
        html_parts.append(f"<p>{summary_lines[-1]}</p>")
        html_parts.append(res["mat"].to_html())

    if stacked_counts:
        pd.concat(stacked_counts, ignore_index=True).to_csv(
            OUT / "all_groups_top6_counts_stacked.csv", index=False
        )
    if all_pairs:
        pd.concat(all_pairs, ignore_index=True).to_csv(
            OUT / "all_groups_top6_pair_detail.csv", index=False
        )

    html_parts.append("</body></html>")
    (OUT / "country_collaboration_matrices_top6.html").write_text("\n".join(html_parts), encoding="utf-8")

    caption = (
        "Table 8. Country×country international collaboration (top 6 countries), by aquatic insect group "
        "(2010–2025). Off-diagonal cells show the total number of group-focused articles whose OpenAlex "
        "author-country list includes both countries, followed in parentheses by the percent change from "
        "early (2010–2015) to recent (2020–2025), calculated as (recent − early) / early × 100. Diagonal "
        "cells show the number of international articles involving that country. The matrix is symmetric.\n\n"
        + "\n".join(summary_lines)
        + "\n"
    )
    (OUT / "suggested_table_caption.txt").write_text(caption, encoding="utf-8")
    print(f"\nWrote outputs under {OUT}")


if __name__ == "__main__":
    main()
