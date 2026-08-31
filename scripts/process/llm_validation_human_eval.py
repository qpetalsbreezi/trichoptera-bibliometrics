#!/usr/bin/env python3
"""Score human labels against GPT-4o-mini and Gemini on the frozen n=1376 sample.

Four comparisons (as planned for Prof. Resh):
  1. GPT-4o-mini vs human
  2. Gemini Pro vs human
  3. GPT∩Gemini agree subset vs human
  4. GPT vs Gemini (recomputed on the labeled subset; already done for full sample)

Usage:
  # Status check (how many rows labeled so far)
  python scripts/process/llm_validation_human_eval.py --status

  # Full eval once taxon_focus + research_theme are filled
  python scripts/process/llm_validation_human_eval.py \\
      --human analysis/combined/llm_validation/human_blind_coding/human_blind_coding_1376.csv

  # Partial: score only completed rows
  python scripts/process/llm_validation_human_eval.py --allow-partial
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.llm_validation import (  # noqa: E402
    RELEVANCE_OFF_TARGET,
    cohens_kappa,
    normalize_taxon_relevance,
    validation_dir,
)
from lib.pipeline import normalize_research_theme, project_root  # noqa: E402

DEFAULT_HUMAN = (
    validation_dir()
    / "human_blind_coding"
    / "human_blind_coding_1376.csv"
)
DEFAULT_RUN = (
    validation_dir() / "validation_1376_no_peripheral_20260819"
)
DEFAULT_OUT = validation_dir() / "human_eval"

FOCUS_KEEP = {"Primary focus", "Secondary mention"}
FOCUS_ALLOWED = FOCUS_KEEP | {RELEVANCE_OFF_TARGET}

# Common human typos / shorthand → schema labels
FOCUS_ALIASES = {
    "primary": "Primary focus",
    "primary focus": "Primary focus",
    "secondary": "Secondary mention",
    "secondary mention": "Secondary mention",
    "not target": RELEVANCE_OFF_TARGET,
    "not target-taxon-focused": RELEVANCE_OFF_TARGET,
    "not-target-taxon-focused": RELEVANCE_OFF_TARGET,
    "not target taxon focused": RELEVANCE_OFF_TARGET,
    "not group-focused": RELEVANCE_OFF_TARGET,
    "not group focused": RELEVANCE_OFF_TARGET,
    "drop": RELEVANCE_OFF_TARGET,
    "out": RELEVANCE_OFF_TARGET,
    "keep": "Primary focus",  # ambiguous; prefer explicit labels
    "peripheral": RELEVANCE_OFF_TARGET,
    "not specified": RELEVANCE_OFF_TARGET,
}

THEME_ALLOWED = {
    "Taxonomy/Systematics",
    "Ecology/Behavior",
    "Biomonitoring/Water Quality",
    "Evolution/Phylogeny",
    "Conservation",
    "Physiology",
    "Applied Ecology",
    "Other",
    "Not Specified",
}


def _norm_text(v) -> str:
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return ""
    return str(v).strip()


def normalize_human_focus(value) -> str:
    s = _norm_text(value)
    if not s:
        return ""
    if s in FOCUS_ALLOWED:
        return s
    aliased = FOCUS_ALIASES.get(s.lower())
    if aliased:
        return aliased
    return normalize_taxon_relevance(s)


def normalize_human_theme(value) -> str:
    s = normalize_research_theme(_norm_text(value))
    if not s:
        return ""
    # Case-insensitive match to allowed themes
    for allowed in THEME_ALLOWED:
        if s.lower() == allowed.lower():
            return allowed
    return s


def is_keep(focus: str) -> bool:
    return focus in FOCUS_KEEP


def keep_or_drop(focus: str) -> str:
    return "keep" if is_keep(focus) else "drop"


def load_human(path: Path) -> pd.DataFrame:
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)
    # Accept alternate column names from returned sheets
    rename = {}
    cols_lower = {c.lower().strip(): c for c in df.columns}
    for want, aliases in {
        "coding_row_id": ["coding_row_id", "id", "row_id", "coding id"],
        "taxon_focus": ["taxon_focus", "taxon_relevance", "focus", "group relevance"],
        "research_theme": ["research_theme", "theme", "primary theme"],
        "target_taxon": ["target_taxon", "taxon", "group"],
        "article_title": ["article_title", "title"],
        "article_doi": ["article_doi", "doi"],
    }.items():
        if want in df.columns:
            continue
        for a in aliases:
            if a in cols_lower:
                rename[cols_lower[a]] = want
                break
    if rename:
        df = df.rename(columns=rename)
    required = {"coding_row_id", "taxon_focus", "research_theme"}
    missing = required - set(df.columns)
    if missing:
        raise SystemExit(f"Human file missing columns: {sorted(missing)}")
    df["coding_row_id"] = pd.to_numeric(df["coding_row_id"], errors="coerce").astype("Int64")
    df["focus_Human"] = df["taxon_focus"].map(normalize_human_focus)
    df["theme_Human"] = df["research_theme"].map(normalize_human_theme)
    df["labeled"] = (df["focus_Human"] != "") & (df["theme_Human"] != "")
    return df


def load_model_run(run_dir: Path) -> pd.DataFrame:
    """Load frozen sample labels.

    Prefer ``all_rows_keys.csv`` (unique ``id`` = coding_row_id). Do not merge
    GPT/Gemini on ``row_key`` alone — some EIDs appear in more than one taxon
    sample and would duplicate rows.
    """
    keys = pd.read_csv(run_dir / "all_rows_keys.csv")
    required = {
        "id",
        "Taxon_Relevance_A",
        "Taxon_Relevance_B",
        "Research_Theme_A",
        "Research_Theme_B",
    }
    missing = required - set(keys.columns)
    if missing:
        raise SystemExit(f"{run_dir / 'all_rows_keys.csv'} missing columns: {sorted(missing)}")

    merged = keys.copy()
    merged["coding_row_id"] = merged["id"].astype(int)
    merged["focus_gpt"] = merged["Taxon_Relevance_A"].map(normalize_taxon_relevance)
    merged["theme_gpt"] = merged["Research_Theme_A"].map(normalize_research_theme)
    merged["focus_gemini"] = merged["Taxon_Relevance_B"].map(normalize_taxon_relevance)
    merged["theme_gemini"] = merged["Research_Theme_B"].map(normalize_research_theme)

    # Attach title from GPT file via (query_id, row_key) — unique within sample
    gpt_path = run_dir / "gpt_coded.csv"
    if gpt_path.exists():
        gpt = pd.read_csv(gpt_path, usecols=["query_id", "row_key", "Title"])
        gpt = gpt.drop_duplicates(subset=["query_id", "row_key"], keep="first")
        merged = merged.merge(gpt, on=["query_id", "row_key"], how="left")

    if "models_agree_taxon" in merged.columns:
        merged["models_agree_focus"] = merged["models_agree_taxon"].map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
            if not isinstance(v, bool)
            else v
        )
    else:
        merged["models_agree_focus"] = merged["focus_gpt"] == merged["focus_gemini"]

    theme_agree_src = merged["models_agree_theme"] if "models_agree_theme" in merged.columns else None
    if theme_agree_src is not None:
        merged["models_agree_theme"] = theme_agree_src.map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
            if not isinstance(v, bool)
            else v
        )
    else:
        merged["models_agree_theme"] = merged["theme_gpt"] == merged["theme_gemini"]

    merged["models_agree_both"] = merged["models_agree_focus"] & merged["models_agree_theme"]

    if "models_agree_gate" in merged.columns:
        gate_src = merged["models_agree_gate"]
        merged["models_agree_gate"] = gate_src.map(
            lambda v: str(v).strip().lower() in ("true", "1", "yes")
            if not isinstance(v, bool)
            else v
        )
    else:
        merged["models_agree_gate"] = merged["focus_gpt"].map(is_keep) == merged["focus_gemini"].map(
            is_keep
        )
    return merged


def pair_metrics(
    df: pd.DataFrame,
    left_focus: str,
    left_theme: str,
    right_focus: str,
    right_theme: str,
    comparison: str,
) -> list[dict]:
    """Agreement metrics with the same four rows as the Gemini vs OpenAI table."""
    n = len(df)
    if n == 0:
        return []

    def row(field: str, agree_mask, left_vals, right_vals, kappa: bool = True) -> dict:
        agree_n = int(agree_mask.sum())
        return {
            "comparison": comparison,
            "metric": field,
            "n": n,
            "agreement_n": agree_n,
            "agreement_pct": round(100.0 * agree_n / n, 2),
            "cohens_kappa": (
                round(cohens_kappa(left_vals, right_vals), 4) if kappa else float("nan")
            ),
        }

    gate_l = df[left_focus].map(is_keep).map({True: "keep", False: "drop"})
    gate_r = df[right_focus].map(is_keep).map({True: "keep", False: "drop"})
    agree_gate = gate_l == gate_r
    agree_focus = df[left_focus] == df[right_focus]
    agree_theme = df[left_theme] == df[right_theme]
    agree_both = agree_focus & agree_theme

    return [
        row(
            "Keep vs drop",
            agree_gate,
            gate_l.tolist(),
            gate_r.tolist(),
        ),
        row(
            "Taxon focus label",
            agree_focus,
            df[left_focus].tolist(),
            df[right_focus].tolist(),
        ),
        row(
            "Research theme",
            agree_theme,
            df[left_theme].tolist(),
            df[right_theme].tolist(),
        ),
        row(
            "Focus label and theme both",
            agree_both,
            [],
            [],
            kappa=False,
        ),
    ]


def export_human_llm(df: pd.DataFrame, out_path: Path) -> pd.DataFrame:
    """Single audit table: human review fields + H/GPT/Gemini labels grouped by field."""
    title = df["article_title"] if "article_title" in df.columns else df.get("Title", "")
    if isinstance(title, pd.Series):
        if "Title" in df.columns:
            title = title.fillna(df["Title"])
    else:
        title = df["Title"]

    doi = df["DOI"] if "DOI" in df.columns else df.get("article_doi", "")
    if isinstance(doi, pd.Series) and "article_doi" in df.columns:
        doi = doi.fillna(df["article_doi"])

    out = pd.DataFrame(
        {
            "coding_row_id": df["coding_row_id"],
            "target_taxon": df["target_taxon"],
            "article_title": title,
            "article_abstract": df.get("article_abstract", ""),
            "doi": doi,
            "year": df["Year"] if "Year" in df.columns else df.get("year", ""),
            "focus_Human": df["focus_Human"],
            "focus_gpt": df["focus_gpt"],
            "focus_gemini": df["focus_gemini"],
            "keep_or_drop_Human": df["focus_Human"].map(keep_or_drop),
            "keep_or_drop_gpt": df["focus_gpt"].map(keep_or_drop),
            "keep_or_drop_gemini": df["focus_gemini"].map(keep_or_drop),
            "theme_Human": df["theme_Human"],
            "theme_gpt": df["theme_gpt"],
            "theme_gemini": df["theme_gemini"],
        }
    )
    out = out.sort_values("coding_row_id").reset_index(drop=True)
    out.to_csv(out_path, index=False)
    return out


def print_table(metrics: pd.DataFrame, comparison: str) -> None:
    sub = metrics[metrics["comparison"] == comparison]
    if sub.empty:
        return
    print(f"\n### {comparison} (n={int(sub['n'].iloc[0])})")
    print(f"{'Metric':<32} {'Agreement':>10} {'κ':>8}")
    for _, r in sub.iterrows():
        k = "—" if pd.isna(r["cohens_kappa"]) else f"{r['cohens_kappa']:.2f}"
        print(f"{r['metric']:<32} {r['agreement_pct']:>9.1f}% {k:>8}")


def status_report(human: pd.DataFrame, models: pd.DataFrame) -> None:
    n = len(human)
    labeled = int(human["labeled"].sum())
    print(f"Human file rows: {n}")
    print(f"Labeled (focus+theme): {labeled} / {n} ({100.0 * labeled / n:.1f}%)")
    if "coding_row_id" in human.columns and "coding_row_id" in models.columns:
        missing_ids = set(models["coding_row_id"]) - set(human["coding_row_id"].dropna().astype(int))
        extra_ids = set(human["coding_row_id"].dropna().astype(int)) - set(models["coding_row_id"])
        print(f"IDs in model sample missing from human file: {len(missing_ids)}")
        print(f"Extra IDs in human file: {len(extra_ids)}")
    bad_focus = human.loc[human["focus_Human"] != "", "focus_Human"]
    bad_focus = bad_focus[~bad_focus.isin(FOCUS_ALLOWED)]
    if len(bad_focus):
        print(f"\nUnexpected focus labels ({len(bad_focus)}):")
        print(bad_focus.value_counts().head(10).to_string())
    bad_theme = human.loc[human["theme_Human"] != "", "theme_Human"]
    bad_theme = bad_theme[~bad_theme.isin(THEME_ALLOWED)]
    if len(bad_theme):
        print(f"\nUnexpected theme labels ({len(bad_theme)}):")
        print(bad_theme.value_counts().head(10).to_string())
    if labeled:
        by = human[human["labeled"]].groupby(
            human.get("target_taxon", pd.Series(["?"] * len(human)))
            if "target_taxon" in human.columns
            else pd.Series(["all"] * len(human)),
            dropna=False,
        ).size()
        print("\nLabeled by taxon:")
        print(by.to_string())


def run_eval(
    human_path: Path,
    run_dir: Path,
    out_dir: Path,
    allow_partial: bool,
    status_only: bool,
) -> None:
    human = load_human(human_path)
    models = load_model_run(run_dir)

    if status_only:
        status_report(human, models)
        return

    status_report(human, models)

    joined = models.merge(
        human[
            [
                c
                for c in [
                    "coding_row_id",
                    "focus_Human",
                    "theme_Human",
                    "labeled",
                    "target_taxon",
                    "article_title",
                    "article_abstract",
                    "article_doi",
                ]
                if c in human.columns
            ]
        ],
        on="coding_row_id",
        how="inner",
    )
    labeled = joined[joined["labeled"]].copy()
    n_lab = len(labeled)
    if n_lab == 0:
        raise SystemExit(
            "No completed human labels found (both taxon_focus and research_theme required)."
        )
    if n_lab < len(joined) and not allow_partial:
        raise SystemExit(
            f"Only {n_lab}/{len(joined)} rows labeled. "
            "Finish labeling, or re-run with --allow-partial."
        )

    # Flag invalid labels that slipped through
    invalid_focus = ~labeled["focus_Human"].isin(FOCUS_ALLOWED)
    if invalid_focus.any():
        print(
            f"WARNING: {invalid_focus.sum()} rows have focus labels outside schema; "
            "kept as-is for scoring."
        )

    metrics_rows: list[dict] = []
    # 1) GPT vs human
    metrics_rows.extend(
        pair_metrics(labeled, "focus_gpt", "theme_gpt", "focus_Human", "theme_Human", "GPT-4o-mini vs human")
    )
    # 2) Gemini vs human
    metrics_rows.extend(
        pair_metrics(labeled, "focus_gemini", "theme_gemini", "focus_Human", "theme_Human", "Gemini Pro vs human")
    )
    # 3) Models agree vs human
    agree_both = labeled[labeled["models_agree_both"]].copy()
    metrics_rows.extend(
        pair_metrics(
            agree_both,
            "focus_gpt",
            "theme_gpt",
            "focus_Human",
            "theme_Human",
            "GPT∩Gemini agree vs human",
        )
    )
    # Also report gate-agree subset (broader)
    agree_gate = labeled[labeled["models_agree_gate"]].copy()
    metrics_rows.extend(
        pair_metrics(
            agree_gate,
            "focus_gpt",
            "theme_gpt",
            "focus_Human",
            "theme_Human",
            "GPT∩Gemini gate-agree vs human",
        )
    )
    # 4) GPT vs Gemini on labeled subset
    metrics_rows.extend(
        pair_metrics(
            labeled,
            "focus_gpt",
            "theme_gpt",
            "focus_gemini",
            "theme_gemini",
            "Gemini Pro vs GPT-4o-mini (labeled subset)",
        )
    )

    metrics = pd.DataFrame(metrics_rows)
    out_dir.mkdir(parents=True, exist_ok=True)

    human_llm = export_human_llm(labeled, out_dir / "human_llm.csv")

    for comp in metrics["comparison"].unique():
        print_table(metrics, comp)

    print(f"\nWrote {out_dir / 'human_llm.csv'} ({len(human_llm)} rows)")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--human", type=Path, default=DEFAULT_HUMAN, help="Human-coded CSV or Excel")
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN, help="GPT/Gemini validation folder")
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT, help="Output folder")
    p.add_argument(
        "--allow-partial",
        action="store_true",
        help="Score completed rows even if the full sample is not labeled yet",
    )
    p.add_argument("--status", action="store_true", help="Only report labeling progress")
    args = p.parse_args()

    if not args.human.exists():
        raise SystemExit(f"Human file not found: {args.human}")
    if not args.run_dir.exists():
        raise SystemExit(f"Validation run dir not found: {args.run_dir}")

    run_eval(
        human_path=args.human,
        run_dir=args.run_dir,
        out_dir=args.out_dir,
        allow_partial=args.allow_partial,
        status_only=args.status,
    )


if __name__ == "__main__":
    main()
