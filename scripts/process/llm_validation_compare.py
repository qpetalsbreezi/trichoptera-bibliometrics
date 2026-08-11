#!/usr/bin/env python3
"""Compare GPT (A) vs Gemini (B) labels; export metrics and disagreement review sheet."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.llm_validation import (  # noqa: E402
    SPOTCHECK_N,
    abstract_snippet,
    agreement_metrics_path,
    classify_disagreement_type,
    cohens_kappa,
    disagreement_review_path,
    gemini_coded_path,
    is_gemini_all_ns_raw,
    sample_manifest_path,
    spotcheck_path,
    validation_dir,
)
from lib.pipeline import normalize_research_theme  # noqa: E402


def _clean(series: pd.Series, field: str) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    if field == "Research_Theme":
        return s.map(normalize_research_theme)
    return s


def join_ab() -> pd.DataFrame:
    manifest = pd.read_csv(sample_manifest_path())
    gemini = pd.read_csv(gemini_coded_path())
    b_cols = [
        "query_id",
        "row_key",
        "Taxon_Relevance_B",
        "Research_Theme_B",
        "Country_B",
        "Region_Global_B",
        "model_B",
        "raw_json_B",
    ]
    missing = [c for c in b_cols if c not in gemini.columns]
    if missing:
        raise SystemExit(f"Gemini coded file missing columns: {missing}")
    merged = manifest.merge(gemini[b_cols], on=["query_id", "row_key"], how="inner")
    if len(merged) == 0:
        raise SystemExit("No overlapping rows between sample manifest and Gemini outputs.")
    merged["Taxon_Relevance_A"] = _clean(merged["Taxon_Relevance_A"], "Taxon_Relevance")
    merged["Taxon_Relevance_B"] = _clean(merged["Taxon_Relevance_B"], "Taxon_Relevance")
    merged["Research_Theme_A"] = _clean(merged["Research_Theme_A"], "Research_Theme")
    merged["Research_Theme_B"] = _clean(merged["Research_Theme_B"], "Research_Theme")
    merged["agree_relevance"] = merged["Taxon_Relevance_A"] == merged["Taxon_Relevance_B"]
    merged["agree_theme"] = merged["Research_Theme_A"] == merged["Research_Theme_B"]
    merged["agree_both"] = merged["agree_relevance"] & merged["agree_theme"]
    if "raw_json_B" in merged.columns:
        merged["gemini_all_ns"] = merged["raw_json_B"].map(is_gemini_all_ns_raw)
    else:
        merged["gemini_all_ns"] = False
    return merged


def metrics_for(df: pd.DataFrame, scope: str, query_id: str = "") -> list[dict]:
    rows = []
    n = len(df)
    for field, a_col, b_col, agree_col in (
        ("Taxon_Relevance", "Taxon_Relevance_A", "Taxon_Relevance_B", "agree_relevance"),
        ("Research_Theme", "Research_Theme_A", "Research_Theme_B", "agree_theme"),
    ):
        agree_n = int(df[agree_col].sum())
        rows.append(
            {
                "scope": scope,
                "query_id": query_id,
                "field": field,
                "n": n,
                "agreement_n": agree_n,
                "agreement_pct": round(100.0 * agree_n / n, 2) if n else float("nan"),
                "cohens_kappa": round(
                    cohens_kappa(df[a_col].tolist(), df[b_col].tolist()), 4
                ),
            }
        )
    both_n = int(df["agree_both"].sum())
    rows.append(
        {
            "scope": scope,
            "query_id": query_id,
            "field": "both_fields",
            "n": n,
            "agreement_n": both_n,
            "agreement_pct": round(100.0 * both_n / n, 2) if n else float("nan"),
            "cohens_kappa": float("nan"),
        }
    )
    return rows


def build_metrics(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(metrics_for(merged, "overall"))
    abs_series = merged["abstract_available"].map(
        lambda v: str(v).strip().lower() in ("true", "1", "yes")
        if not isinstance(v, bool)
        else v
    )
    for abs_flag, label in ((True, "with_abstract"), (False, "without_abstract")):
        sub = merged[abs_series == abs_flag]
        if len(sub):
            rows.extend(metrics_for(sub, label))
    no_all_ns = merged[~merged["gemini_all_ns"]]
    if len(no_all_ns):
        rows.extend(metrics_for(no_all_ns, "excluding_gemini_all_ns"))
    for qid, g in merged.groupby("query_id"):
        rows.extend(metrics_for(g, "by_group", query_id=str(qid)))
    return pd.DataFrame(rows)


def build_disagreements(merged: pd.DataFrame) -> pd.DataFrame:
    dis = merged[~merged["agree_both"]].copy()
    dis["abstract_snippet"] = dis["Abstract"].map(lambda x: abstract_snippet(x, 500))
    dis["disagreement_type"] = dis.apply(classify_disagreement_type, axis=1)
    out = pd.DataFrame(
        {
            "query_id": dis["query_id"],
            "taxon_label": dis.get("taxon_label", ""),
            "row_key": dis["row_key"],
            "Year": dis["Year"],
            "year_band": dis["year_band"],
            "abstract_available": dis["abstract_available"],
            "Title": dis["Title"],
            "DOI": dis.get("DOI", ""),
            "abstract_snippet": dis["abstract_snippet"],
            "disagreement_type": dis["disagreement_type"],
            "Taxon_Relevance_A": dis["Taxon_Relevance_A"],
            "Taxon_Relevance_B": dis["Taxon_Relevance_B"],
            "agree_relevance": dis["agree_relevance"],
            "Research_Theme_A": dis["Research_Theme_A"],
            "Research_Theme_B": dis["Research_Theme_B"],
            "agree_theme": dis["agree_theme"],
            "Human_Taxon_Relevance": "",
            "Human_Research_Theme": "",
            "Reviewer_notes": "",
        }
    )
    return out.sort_values(["query_id", "Year", "Title"]).reset_index(drop=True)


def build_spotcheck(merged: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    agree = merged[merged["agree_both"]].copy()
    if agree.empty:
        return agree
    parts = []
    for qid, g in agree.groupby("query_id"):
        take = min(max(1, n // max(1, agree["query_id"].nunique())), len(g))
        parts.append(g.sample(n=take, random_state=seed))
    spot = pd.concat(parts, ignore_index=True)
    spot["abstract_snippet"] = spot["Abstract"].map(lambda x: abstract_snippet(x, 500))
    return pd.DataFrame(
        {
            "query_id": spot["query_id"],
            "taxon_label": spot.get("taxon_label", ""),
            "row_key": spot["row_key"],
            "Year": spot["Year"],
            "Title": spot["Title"],
            "DOI": spot.get("DOI", ""),
            "abstract_snippet": spot["abstract_snippet"],
            "Taxon_Relevance_A": spot["Taxon_Relevance_A"],
            "Taxon_Relevance_B": spot["Taxon_Relevance_B"],
            "Research_Theme_A": spot["Research_Theme_A"],
            "Research_Theme_B": spot["Research_Theme_B"],
            "Human_OK": "",
            "Reviewer_notes": "",
        }
    )


def main():
    parser = argparse.ArgumentParser(description="Compare GPT vs Gemini validation labels")
    parser.add_argument("--spotcheck-n", type=int, default=SPOTCHECK_N)
    parser.add_argument("--seed", type=int, default=20260810)
    args = parser.parse_args()

    if not sample_manifest_path().exists():
        raise SystemExit(f"Missing {sample_manifest_path()}")
    if not gemini_coded_path().exists():
        raise SystemExit(
            f"Missing {gemini_coded_path()}\n"
            "Run: python scripts/process/llm_validation_gemini.py"
        )

    validation_dir().mkdir(parents=True, exist_ok=True)
    merged = join_ab()
    metrics = build_metrics(merged)
    disagreements = build_disagreements(merged)
    spot = build_spotcheck(merged, n=args.spotcheck_n, seed=args.seed)

    metrics.to_csv(agreement_metrics_path(), index=False)
    disagreements.to_csv(disagreement_review_path(), index=False)
    spot.to_csv(spotcheck_path(), index=False)

    print(f"Joined rows: {len(merged):,}")
    print(f"Disagreements (either field): {len(disagreements):,}")
    if len(disagreements):
        print("\nDisagreement types:")
        print(disagreements["disagreement_type"].value_counts().to_string())
    print(f"Spot-check agrees: {len(spot):,}")
    print("\nOverall:")
    overall = metrics[metrics["scope"] == "overall"]
    print(overall[["field", "n", "agreement_pct", "cohens_kappa"]].to_string(index=False))
    print(f"\n✓ {agreement_metrics_path()}")
    print(f"✓ {disagreement_review_path()}")
    print(f"✓ {spotcheck_path()}")


if __name__ == "__main__":
    main()
