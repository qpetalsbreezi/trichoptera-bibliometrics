#!/usr/bin/env python3
"""Re-code Culicidae validation rows with current prompt (GPT + Gemini); compare to human."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.llm_validation import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    RELEVANCE_FIELD,
    abstract_text_ok,
    cohens_kappa,
    load_schema_for_llm,
    normalize_research_theme,
    normalize_taxon_relevance,
    validation_dir,
)
from lib.pipeline import get_query_config, load_dotenv  # noqa: E402
from process.llm_code_taxon import classify  # noqa: E402
from process.llm_validation_gemini import classify_gemini, get_gemini_client  # noqa: E402

DEFAULT_RUN = validation_dir() / "validation_1376_no_peripheral_20260819"
DEFAULT_HUMAN = (
    validation_dir() / "human_blind_coding" / "human_labeled_1376_20260830.csv"
)
DEFAULT_OUT = validation_dir() / "culicidae_prompt_test"

FOCUS_KEEP = {"Primary focus", "Secondary mention"}


def keep_or_drop(focus: str) -> str:
    return "keep" if focus in FOCUS_KEEP else "drop"


def fmt_agree_kappa(left: pd.Series, right: pd.Series, kappa: bool = True) -> str:
    agree = (left == right).mean()
    if kappa:
        return f"{100 * agree:.1f}% (κ {cohens_kappa(left, right):.2f})"
    return f"{100 * agree:.1f}%"


def agreement_tables(df: pd.DataFrame, version: str) -> dict:
    """Four tables for one prompt version (old or new columns)."""
    sfx = "_old" if version == "old" else "_new"
    fg, tg = f"focus_gpt{sfx}", f"theme_gpt{sfx}"
    fgm, tgm = f"focus_gemini{sfx}", f"theme_gemini{sfx}"
    agree_both = (df[fg] == df[fgm]) & (df[tg] == df[tgm])
    sub = df[agree_both]
    n_sub = len(sub)

    def row(metric: str) -> dict:
        if metric == "gate":
            gpt = df[fg].map(keep_or_drop)
            gem = df[fgm].map(keep_or_drop)
            hum = df.focus_Human.map(keep_or_drop)
            sg = sub[fg].map(keep_or_drop)
            return {
                "gpt_vs_human": fmt_agree_kappa(gpt, hum),
                "gemini_vs_human": fmt_agree_kappa(gem, hum),
                "both_vs_human": fmt_agree_kappa(sg, sub.focus_Human.map(keep_or_drop)),
                "gpt_vs_gemini": fmt_agree_kappa(gpt, gem),
            }
        if metric == "focus":
            return {
                "gpt_vs_human": fmt_agree_kappa(df[fg], df.focus_Human),
                "gemini_vs_human": fmt_agree_kappa(df[fgm], df.focus_Human),
                "both_vs_human": fmt_agree_kappa(sub[fg], sub.focus_Human),
                "gpt_vs_gemini": fmt_agree_kappa(df[fg], df[fgm]),
            }
        if metric == "theme":
            return {
                "gpt_vs_human": fmt_agree_kappa(df[tg], df.theme_Human),
                "gemini_vs_human": fmt_agree_kappa(df[tgm], df.theme_Human),
                "both_vs_human": fmt_agree_kappa(sub[tg], sub.theme_Human),
                "gpt_vs_gemini": fmt_agree_kappa(df[tg], df[tgm]),
            }
        both_gpt = (df[fg] == df.focus_Human) & (df[tg] == df.theme_Human)
        both_gem = (df[fgm] == df.focus_Human) & (df[tgm] == df.theme_Human)
        both_sub = (sub[fg] == sub.focus_Human) & (sub[tg] == sub.theme_Human)
        both_gg = (df[fg] == df[fgm]) & (df[tg] == df[tgm])
        return {
            "gpt_vs_human": f"{100 * both_gpt.mean():.1f}%",
            "gemini_vs_human": f"{100 * both_gem.mean():.1f}%",
            "both_vs_human": f"{100 * both_sub.mean():.1f}%",
            "gpt_vs_gemini": f"{100 * both_gg.mean():.1f}%",
        }

    return {
        "subset_n_agree_both": n_sub,
        "table1_gate": row("gate"),
        "table2_focus": row("focus"),
        "table3_theme": row("theme"),
        "table4_both": row("both"),
    }


def print_tables(tables: dict, label: str) -> None:
    print(f"\n{'='*60}\n{label}  |  GPT∩Gemini agree n={tables['subset_n_agree_both']}\n{'='*60}")
    for i, key in enumerate(
        ["table1_gate", "table2_focus", "table3_theme", "table4_both"], start=1
    ):
        titles = ["Keep vs drop", "Taxon focus", "Research theme", "Focus+theme both"]
        t = tables[key]
        print(f"\nTable {i}. {titles[i-1]} — Culicidae only")
        print(f"  GPT vs human:        {t['gpt_vs_human']}")
        print(f"  Gemini vs human:     {t['gemini_vs_human']}")
        print(f"  GPT∩Gemini vs human: {t['both_vs_human']}")
        print(f"  GPT vs Gemini:       {t['gpt_vs_gemini']}")


def code_rows(
    cul: pd.DataFrame,
    llm_schema: dict,
    llm_schema_text: str,
    llm_cfg: dict,
    *,
    gpt_client: OpenAI | None,
    gpt_model: str,
    gemini_client,
    gemini_model: str,
    run_gpt: bool,
    run_gemini: bool,
) -> tuple[list[str], list[str], list[str], list[str]]:
    focus_gpt: list[str] = []
    theme_gpt: list[str] = []
    focus_gem: list[str] = []
    theme_gem: list[str] = []

    for _, row in tqdm(cul.iterrows(), total=len(cul), desc="Re-coding Culicidae"):
        abstract = row.get("article_abstract", "")
        abstract_ok = abstract_text_ok(abstract)
        title = row.get("Title", "")
        abs_text = abstract if abstract_ok else ""

        if run_gpt:
            out = classify(
                gpt_client,
                gpt_model,
                title,
                abs_text,
                None,
                llm_schema,
                llm_schema_text,
                llm_cfg,
            )
            focus_gpt.append(normalize_taxon_relevance(out.get(RELEVANCE_FIELD, "")))
            theme_gpt.append(normalize_research_theme(out.get("Research_Theme", "")) or "")
        if run_gemini:
            out = classify_gemini(
                gemini_client,
                gemini_model,
                title,
                abs_text,
                None,
                llm_schema,
                llm_schema_text,
                llm_cfg,
            )
            focus_gem.append(normalize_taxon_relevance(out.get(RELEVANCE_FIELD, "")))
            theme_gem.append(normalize_research_theme(out.get("Research_Theme", "")) or "")

    return focus_gpt, theme_gpt, focus_gem, theme_gem


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--gpt-model", default="gpt-4o-mini")
    p.add_argument("--gemini-model", default="gemini-3.1-pro-preview")
    p.add_argument("--reuse-gpt-from", type=Path, default=None,
                     help="CSV with focus_gpt_new/theme_gpt_new when --gemini-only")
    p.add_argument("--limit", type=int, default=0, help="0 = all Culicidae rows")
    p.add_argument("--high-error-only", action="store_true")
    p.add_argument("--gpt-only", action="store_true")
    p.add_argument("--gemini-only", action="store_true")
    args = p.parse_args()

    run_gpt = not args.gemini_only
    run_gemini = not args.gpt_only

    load_dotenv()
    gpt_client = None
    if run_gpt:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise SystemExit("OPENAI_API_KEY not set in .env")
        gpt_client = OpenAI(api_key=api_key)

    gemini_client = None
    if run_gemini:
        gkey = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not gkey:
            raise SystemExit("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in .env")
        gemini_client = get_gemini_client(gkey)

    keys = pd.read_csv(args.run_dir / "all_rows_keys.csv")
    gpt = pd.read_csv(args.run_dir / "gpt_coded.csv")
    human = pd.read_csv(args.human)

    cul = keys[keys.query_id == "mosquitoes"].merge(
        gpt[["query_id", "row_key", "Title"]],
        on=["query_id", "row_key"],
        how="left",
    )
    cul = cul.merge(
        human[
            [
                "coding_row_id",
                "taxon_focus",
                "research_theme",
                "article_abstract",
            ]
        ],
        left_on="id",
        right_on="coding_row_id",
        how="inner",
    )
    cul["focus_Human"] = cul["taxon_focus"]
    cul["theme_Human"] = cul["research_theme"]
    cul["focus_gpt_old"] = cul["Taxon_Relevance_A"].map(normalize_taxon_relevance)
    cul["theme_gpt_old"] = cul["Research_Theme_A"].map(normalize_research_theme)
    cul["focus_gemini_old"] = cul["Taxon_Relevance_B"].map(normalize_taxon_relevance)
    cul["theme_gemini_old"] = cul["Research_Theme_B"].map(normalize_research_theme)

    if args.high_error_only:
        cul = cul[
            (cul.focus_Human.map(keep_or_drop) == "drop")
            & (cul.focus_gpt_old.map(keep_or_drop) == "keep")
        ].copy()
    if args.limit:
        cul = cul.head(args.limit).copy()

    llm_schema, llm_schema_text = load_schema_for_llm()
    llm_cfg = get_query_config("mosquitoes")["llm"]

    focus_gpt, theme_gpt, focus_gem, theme_gem = code_rows(
        cul,
        llm_schema,
        llm_schema_text,
        llm_cfg,
        gpt_client=gpt_client,
        gpt_model=args.gpt_model,
        gemini_client=gemini_client,
        gemini_model=args.gemini_model,
        run_gpt=run_gpt,
        run_gemini=run_gemini,
    )

    cul = cul.copy()
    if run_gpt:
        cul["focus_gpt_new"] = focus_gpt
        cul["theme_gpt_new"] = theme_gpt
    elif args.reuse_gpt_from:
        prior = pd.read_csv(args.reuse_gpt_from)
        cul = cul.merge(
            prior[["id", "focus_gpt_new", "theme_gpt_new"]],
            on="id",
            how="left",
        )
    if run_gemini:
        cul["focus_gemini_new"] = focus_gem
        cul["theme_gemini_new"] = theme_gem

    args.out_dir.mkdir(parents=True, exist_ok=True)
    out_cols = [
        "id",
        "DOI",
        "Title",
        "focus_Human",
        "theme_Human",
        "focus_gpt_old",
        "focus_gpt_new",
        "focus_gemini_old",
        "focus_gemini_new",
        "theme_gpt_old",
        "theme_gpt_new",
        "theme_gemini_old",
        "theme_gemini_new",
    ]
    cul[out_cols].sort_values("id").to_csv(args.out_dir / "culicidae_recoded.csv", index=False)

    summary = {
        "n": len(cul),
        "gpt_model": args.gpt_model if run_gpt else None,
        "gemini_model": args.gemini_model if run_gemini else None,
        "old": agreement_tables(cul, "old"),
        "new": agreement_tables(cul, "new"),
    }
    (args.out_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print_tables(summary["old"], "OLD PROMPT (checked-in validation run)")
    print_tables(summary["new"], "NEW PROMPT (current build_prompt)")
    print(f"\nWrote {args.out_dir / 'culicidae_recoded.csv'}")


if __name__ == "__main__":
    main()
