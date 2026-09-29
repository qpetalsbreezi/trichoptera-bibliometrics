#!/usr/bin/env python3
"""Test Culicidae prompt variants against human labels; compare to v2 full tweak."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Callable

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.llm_validation import (  # noqa: E402
    RELEVANCE_FIELD,
    abstract_text_ok,
    cohens_kappa,
    load_schema_for_llm,
    normalize_taxon_relevance,
    validation_dir,
)
from lib.pipeline import get_query_config, load_dotenv  # noqa: E402
from process import llm_code_taxon  # noqa: E402

DEFAULT_RUN = validation_dir() / "validation_1376_no_peripheral_20260819"
DEFAULT_HUMAN = (
    validation_dir() / "human_blind_coding" / "human_labeled_1376_20260830.csv"
)
DEFAULT_V2 = validation_dir() / "culicidae_prompt_test" / "culicidae_recoded.csv"
DEFAULT_OUT = validation_dir() / "culicidae_prompt_variants"

FOCUS_KEEP = {"Primary focus", "Secondary mention"}

TITLE_ANCHOR_V2 = (
    '- Exception (mosquitoes / Culicidae): a genus name (*Aedes*, *Anopheles*, *Culex*) or "mosquito" '
    "in the title does NOT require Primary focus if the paper mainly studies the pathogen, disease, "
    'treatment, or repellent chemistry — use "Not target-taxon-focused" or "Secondary mention" as appropriate.'
)

TITLE_ANCHOR_V3 = (
    '- Exception (mosquitoes / Culicidae): if the title names a mosquito genus or "mosquito" but the paper '
    "mainly studies pathogen/disease/treatment outcomes, do NOT default to Primary focus; use Secondary mention "
    "when mosquitoes appear in methods, otherwise Not target-taxon-focused."
)

VECTOR_V2 = """VECTOR / DISEASE (especially mosquitoes / Culicidae):
- Mosquito biology, surveillance, control, vector competence, or vector ecology → "Primary focus" or "Secondary mention" as appropriate.
- Papers mainly about mosquito-borne pathogens, human disease, or clinical/epidemiological outcomes → "Not target-taxon-focused", even if mosquitoes are named as vectors.
- Use "Primary focus" only when mosquito biology, ecology, physiology, or control on the insect is the main subject; if the main subject is a pathogen, disease, Wolbachia/symbiont in mosquitoes, or a multi-vector surveillance program, use "Secondary mention" at most.
- Do not use "Not target-taxon-focused" when mosquitoes are a deliberate part of methods or sampled assemblage (e.g. trap catches, competence assays, vector surveillance) — use "Secondary mention" instead.
- Plant essential oils, repellent chemistry, or GIS disease-receptivity studies without mosquito biology → "Not target-taxon-focused"."""

VECTOR_V3_BALANCED = """VECTOR / DISEASE (especially mosquitoes / Culicidae):
- Mosquito biology, surveillance, control, vector competence, or vector ecology → "Primary focus" or "Secondary mention" as appropriate.
- Papers mainly about mosquito-borne pathogens, human disease, or clinical/epidemiological outcomes → "Not target-taxon-focused" when mosquitoes are named only as background vectors and were not trapped, surveyed, infected, or assayed in methods.
- Use "Primary focus" only when mosquito biology, ecology, physiology, behavior, or insect-directed control is the main subject.
- If the main subject is a pathogen, disease, Wolbachia/symbiont, or human clinical outcome—but mosquitoes were trapped, surveyed, infected, or assayed in methods → "Secondary mention" (not Primary, not Not target).
- Rodent/bird/human-only serology, GIS territory receptivity or disease-risk mapping, LLIN or antimalarial drug trials, and plant repellent chemistry without mosquito biology → "Not target-taxon-focused".
- GM mosquito trials, mosquito movement/behavior studies, mosquito-cell assays, and ethics of mosquito releases → Primary or Secondary if mosquitoes are the experimental system, even when disease is in the title."""

VECTOR_V3_MINIMAL = """VECTOR / DISEASE (especially mosquitoes / Culicidae):
- Mosquito biology, surveillance, control, vector competence, or vector ecology → "Primary focus" or "Secondary mention" as appropriate.
- Papers mainly about mosquito-borne pathogens, human disease, or clinical/epidemiological outcomes → "Not target-taxon-focused", even if mosquitoes are named as vectors.
- Use "Primary focus" only when mosquito biology, ecology, physiology, or control on the insect is the main subject; if the main subject is a pathogen, disease, Wolbachia/symbiont in mosquitoes, or a multi-vector surveillance program, use "Secondary mention" at most.
- If mosquitoes were trapped, surveyed, infected, or assayed in methods, use at least "Secondary mention" even when the headline topic is disease or pathogen (do not use Not target-taxon-focused).
- Plant essential oils, repellent chemistry, or GIS disease-receptivity studies without mosquito biology → "Not target-taxon-focused"."""

# v2 + one minimal bullet each (surgical tests)
FALSE_KEEP_BULLET = (
    '- Rodent/bird/human-only serology, GIS territory-receptivity or disease-risk mapping, '
    'LLIN or antimalarial drug trials, and spillover-risk assessments without mosquito trapping '
    '→ "Not target-taxon-focused".'
)
PREFER_SEC_BULLET = (
    "- When torn between Secondary mention and Not target-taxon-focused on a disease paper, "
    "choose Secondary mention if mosquitoes were trapped, surveyed, infected, or assayed."
)
CAP_PRI_BULLET = (
    "- Do not use Primary focus solely because a mosquito genus appears in the title; "
    "disease-first papers remain Secondary mention at most unless insect biology is the main subject."
)

VECTOR_V2_1_FALSE_KEEP = VECTOR_V2 + "\n" + FALSE_KEEP_BULLET
VECTOR_V2_1_PREFER_SEC = VECTOR_V2 + "\n" + PREFER_SEC_BULLET
VECTOR_V2_1_CAP_PRI = VECTOR_V2 + "\n" + CAP_PRI_BULLET
VECTOR_V2_1_COMBO = VECTOR_V2 + "\n" + FALSE_KEEP_BULLET + "\n" + PREFER_SEC_BULLET


def patch_vector(base: str, vector_block: str) -> str:
    return base.replace(VECTOR_V2, vector_block)


def patch_title(base: str, title_anchor: str) -> str:
    return base.replace(TITLE_ANCHOR_V2, title_anchor)


def patch_prompt(base: str, title_anchor: str | None = None, vector_block: str | None = None) -> str:
    out = base
    if title_anchor:
        out = patch_title(out, title_anchor)
    if vector_block:
        out = patch_vector(out, vector_block)
    return out


def keep_or_drop(focus: str) -> str:
    return "keep" if focus in FOCUS_KEEP else "drop"


def make_classify(prompt_patcher: Callable[[str], str]):
    def classify_variant(
        client: OpenAI,
        model: str,
        title,
        abstract,
        author_affiliations,
        llm_schema: dict,
        llm_schema_text: str,
        llm_cfg: dict,
    ):
        affiliation_text = ""
        if pd.notna(author_affiliations) and str(author_affiliations).strip():
            affiliation_text = f"\n\nAuthor affiliations:\n{str(author_affiliations)}"

        prompt = prompt_patcher(
            llm_code_taxon.build_prompt(
                title, abstract, affiliation_text, llm_schema_text, llm_cfg
            )
        )
        attempt = 0
        while True:
            try:
                response = client.chat.completions.create(
                    model=model,
                    temperature=llm_code_taxon.TEMPERATURE,
                    messages=[
                        {"role": "system", "content": "You are a careful bibliometric classifier."},
                        {"role": "user", "content": prompt},
                    ],
                )
                raw = response.choices[0].message.content.strip()
                parsed = llm_code_taxon.safe_json_loads(raw)
                if parsed is not None:
                    return llm_code_taxon.normalize_classified_output(parsed, llm_schema)
                return llm_code_taxon.placeholder_llm_output(llm_schema)
            except Exception as e:
                if llm_code_taxon.is_daily_rate_limit_error(e):
                    wait = llm_code_taxon.parse_retry_after_seconds(e)
                    if wait is not None and wait <= 120:
                        time.sleep(wait + 1.0)
                        continue
                    raise
                if attempt < 11:
                    time.sleep((2**attempt) * 0.5)
                    attempt += 1
                    continue
                return llm_code_taxon.placeholder_llm_output(llm_schema)

    return classify_variant


VARIANTS: dict[str, Callable[[str], str]] = {
    "v3_balanced": lambda p: patch_prompt(p, TITLE_ANCHOR_V3, VECTOR_V3_BALANCED),
    "v3_minimal": lambda p: patch_prompt(p, vector_block=VECTOR_V3_MINIMAL),
    "v2_1_false_keep": lambda p: patch_prompt(p, vector_block=VECTOR_V2_1_FALSE_KEEP),
    "v2_1_prefer_sec": lambda p: patch_prompt(p, vector_block=VECTOR_V2_1_PREFER_SEC),
    "v2_1_cap_pri": lambda p: patch_prompt(p, vector_block=VECTOR_V2_1_CAP_PRI),
    "v2_1_combo": lambda p: patch_prompt(p, vector_block=VECTOR_V2_1_COMBO),
    "v2_1_anchor": lambda p: patch_prompt(p, title_anchor=TITLE_ANCHOR_V3),
}


def metrics(df: pd.DataFrame, pred_col: str) -> dict:
    n = len(df)
    focus_agree = (df["focus_Human"] == df[pred_col]).mean()
    gate_pred = df[pred_col].map(keep_or_drop)
    gate_agree = (df["gate_Human"] == gate_pred).mean()
    return {
        "n": n,
        "focus_agree_pct": round(100 * focus_agree, 1),
        "focus_kappa": round(cohens_kappa(df["focus_Human"], df[pred_col]), 3),
        "gate_agree_pct": round(100 * gate_agree, 1),
        "gate_kappa": round(cohens_kappa(df["gate_Human"], gate_pred), 3),
        "human_drop_gpt_keep": int(((df["gate_Human"] == "drop") & (gate_pred == "keep")).sum()),
        "human_keep_gpt_drop": int(((df["gate_Human"] == "keep") & (gate_pred == "drop")).sum()),
        "sec_to_not": int(
            ((df["focus_Human"] == "Secondary mention") & (df[pred_col] == "Not target-taxon-focused")).sum()
        ),
        "sec_to_pri": int(
            ((df["focus_Human"] == "Secondary mention") & (df[pred_col] == "Primary focus")).sum()
        ),
    }


def load_culicidae(run_dir: Path, human_path: Path, v2_path: Path | None) -> pd.DataFrame:
    keys = pd.read_csv(run_dir / "all_rows_keys.csv")
    gpt = pd.read_csv(run_dir / "gpt_coded.csv")
    human = pd.read_csv(human_path)
    human["focus_Human"] = human["taxon_focus"]
    human["gate_Human"] = human["taxon_focus"].map(keep_or_drop)

    cul = keys[keys.query_id == "mosquitoes"].merge(
        gpt[["query_id", "row_key", "Title"]],
        on=["query_id", "row_key"],
        how="left",
    )
    cul = cul.merge(
        human[["coding_row_id", "focus_Human", "gate_Human", "article_abstract"]],
        left_on="id",
        right_on="coding_row_id",
        how="inner",
    )
    cul["focus_old"] = cul["Taxon_Relevance_A"]
    cul["gate_old"] = cul["focus_old"].map(keep_or_drop)
    if v2_path and v2_path.exists():
        v2 = pd.read_csv(v2_path)[["id", "focus_new"]].rename(columns={"focus_new": "focus_v2"})
        cul = cul.merge(v2, on="id", how="left")
    return cul


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    p.add_argument("--human", type=Path, default=DEFAULT_HUMAN)
    p.add_argument("--v2-results", type=Path, default=DEFAULT_V2)
    p.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument(
        "--variants",
        nargs="+",
        default=[
            "v2_1_false_keep",
            "v2_1_prefer_sec",
            "v2_1_cap_pri",
            "v2_1_combo",
            "v2_1_anchor",
        ],
        choices=sorted(VARIANTS),
    )
    p.add_argument(
        "--subset",
        choices=["all", "disagree_v2", "errors_v2"],
        default="all",
        help="all=269; disagree_v2=human!=v2; errors_v2=false keep/drop under v2",
    )
    args = p.parse_args()

    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set in .env")

    cul = load_culicidae(args.run_dir, args.human, args.v2_results)
    if "focus_v2" in cul.columns:
        cul["gate_v2"] = cul["focus_v2"].map(keep_or_drop)
        if args.subset == "disagree_v2":
            cul = cul[cul.focus_Human != cul.focus_v2].copy()
        elif args.subset == "errors_v2":
            cul = cul[
                ((cul.gate_Human == "drop") & (cul.gate_v2 == "keep"))
                | ((cul.gate_Human == "keep") & (cul.gate_v2 == "drop"))
            ].copy()

    llm_schema, llm_schema_text = load_schema_for_llm()
    llm_cfg = get_query_config("mosquitoes")["llm"]
    client = OpenAI(api_key=api_key)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    summaries: dict[str, dict] = {}

    if "focus_v2" in cul.columns and args.subset == "all":
        summaries["v2_full"] = metrics(cul.rename(columns={"focus_v2": "focus_pred"}), "focus_pred")

    for name in args.variants:
        classify_fn = make_classify(VARIANTS[name])
        preds: list[str] = []
        for _, row in tqdm(cul.iterrows(), total=len(cul), desc=name):
            abstract = row.get("article_abstract", "")
            abstract_ok = abstract_text_ok(abstract)
            out = classify_fn(
                client,
                args.model,
                row.get("Title", ""),
                abstract if abstract_ok else "",
                None,
                llm_schema,
                llm_schema_text,
                llm_cfg,
            )
            preds.append(normalize_taxon_relevance(out.get(RELEVANCE_FIELD, "")))

        cul[f"focus_{name}"] = preds
        summaries[name] = metrics(cul, f"focus_{name}")

    out_csv = cul[
        ["id", "DOI", "Title", "focus_Human", "focus_old"]
        + ([c for c in cul.columns if c.startswith("focus_v")])
    ].sort_values("id")
    out_csv.to_csv(args.out_dir / f"results_{args.subset}.csv", index=False)
    (args.out_dir / f"summary_{args.subset}.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )

    print(f"Subset: {args.subset} (n={len(cul)})")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
