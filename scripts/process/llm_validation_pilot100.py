#!/usr/bin/env python3
"""100-paper GPT-4o-mini vs Gemini pilot with updated Taxon_Relevance schema.

Draws a stratified subsample from the frozen sample_manifest (20/group),
re-codes BOTH models with the current shared prompt/schema, and writes
agreement outputs under analysis/combined/llm_validation/pilot100/.

Does not overwrite the full 1500 sample_manifest or production coded files.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from threading import Lock

import pandas as pd
from openai import OpenAI
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_PROCESS_DIR = Path(__file__).resolve().parent
for p in (_SCRIPTS_DIR, _PROCESS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.llm_validation import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    DEFAULT_SEED,
    RELEVANCE_FIELD,
    RELEVANCE_OFF_TARGET,
    abstract_snippet,
    abstract_text_ok,
    classify_disagreement_type,
    cohens_kappa,
    load_schema_for_llm,
    normalize_taxon_relevance,
    sample_manifest_path,
    validation_dir,
    validation_query_ids,
)
from lib.pipeline import get_query_config, load_dotenv, normalize_research_theme  # noqa: E402
from llm_code_taxon import (  # noqa: E402
    DEFAULT_MODEL,
    TEMPERATURE,
    DailyRateLimitExceeded,
    build_prompt,
    classify,
    placeholder_llm_output,
)
from llm_validation_gemini import classify_gemini, get_gemini_client, normalize_llm_output  # noqa: E402

PILOT_SEED = 20260811
PILOT_N_PER_GROUP = 20
NUM_THREADS = 6
SAVE_INTERVAL = 20
save_lock = Lock()


# Set by main() so GPT/Gemini writers share an output folder.
_PILOT_OUT_SUBDIR = "pilot100"
_PILOT_MANIFEST_NAME = "pilot100_manifest.csv"


def pilot_dir() -> Path:
    return validation_dir() / _PILOT_OUT_SUBDIR


def pilot_manifest_path() -> Path:
    # Keep each pilot's frozen subsample inside its out-subdir.
    return pilot_dir() / _PILOT_MANIFEST_NAME


def load_coding_manifest(manifest: Path | None, n_per_group: int, seed: int, force: bool) -> pd.DataFrame:
    """Load a frozen manifest, or draw a subsample from sample_manifest.csv."""
    if manifest is not None:
        if not manifest.exists():
            raise SystemExit(f"Missing manifest: {manifest}")
        df = pd.read_csv(manifest)
        for col in (
            "Taxon_Relevance_A",
            "Research_Theme_A",
            "Country_A",
            "Region_Global_A",
            "model_A",
        ):
            if col in df.columns:
                df = df.drop(columns=[col])
        print(f"Using manifest: {manifest} ({len(df)} rows)")
        return df
    return draw_pilot_manifest(n_per_group, seed, force)


def draw_pilot_manifest(n_per_group: int, seed: int, force: bool) -> pd.DataFrame:
    out = pilot_manifest_path()
    if out.exists() and not force:
        df = pd.read_csv(out)
        print(f"Using existing pilot manifest: {out} ({len(df)} rows)")
        return df

    src = sample_manifest_path()
    if not src.exists():
        raise SystemExit(f"Missing frozen sample: {src}")
    full = pd.read_csv(src)
    parts = []
    for i, qid in enumerate(validation_query_ids()):
        g = full[full["query_id"] == qid]
        if g.empty:
            print(f"Warning: no rows for {qid}")
            continue
        take = min(n_per_group, len(g))
        parts.append(g.sample(n=take, random_state=seed + i))
        print(f"{qid:14s}  pilot {take} / {len(g)}")
    pilot = pd.concat(parts, ignore_index=True)
    # Clear stale Model A labels; pilot will re-code both models with new prompt
    for col in (
        "Taxon_Relevance_A",
        "Research_Theme_A",
        "Country_A",
        "Region_Global_A",
        "model_A",
    ):
        if col in pilot.columns:
            pilot = pilot.drop(columns=[col])
    validation_dir().mkdir(parents=True, exist_ok=True)
    pilot_dir().mkdir(parents=True, exist_ok=True)
    pilot.to_csv(out, index=False)
    print(f"✓ Wrote {len(pilot)} rows -> {out}")
    return pilot


def _clean(series: pd.Series, field: str) -> pd.Series:
    s = series.fillna("").astype(str).str.strip()
    if field == "Research_Theme":
        return s.map(normalize_research_theme)
    if field == "Taxon_Relevance":
        return s.map(normalize_taxon_relevance)
    return s


def run_gpt_coding(
    df: pd.DataFrame,
    llm_schema: dict,
    llm_schema_text: str,
    model: str,
    num_threads: int,
    save_interval: int,
) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise SystemExit(
            "OPENAI_API_KEY not set. Add it to .env (uncomment/set OPENAI_API_KEY=...)."
        )
    client = OpenAI(api_key=api_key)
    out_path = pilot_dir() / "gpt_coded.csv"
    pilot_dir().mkdir(parents=True, exist_ok=True)

    coded: dict[str, dict] = {}
    if out_path.exists():
        existing = pd.read_csv(out_path)
        for _, row in existing.iterrows():
            coded[f"{row['query_id']}::{row['row_key']}"] = row.to_dict()
        print(f"GPT resume: {len(coded):,} rows in {out_path}")

    pending = [
        row
        for _, row in df.iterrows()
        if f"{row['query_id']}::{row['row_key']}" not in coded
    ]
    print(f"Model A (GPT): {model}  to code: {len(pending):,} / {len(df):,}")

    def process_row(row: pd.Series) -> tuple[str, dict]:
        qid = row["query_id"]
        llm_cfg = get_query_config(qid)["llm"]
        abstract = row.get("Abstract", "")
        abstract_ok = abstract_text_ok(abstract)
        llm_output = classify(
            client,
            model,
            row.get("Title", ""),
            abstract if abstract_ok else "",
            row.get("Author_Affiliations", None),
            llm_schema,
            llm_schema_text,
            llm_cfg,
        )
        out = {
            "query_id": qid,
            "taxon_label": row.get("taxon_label", ""),
            "row_key": row["row_key"],
            "EID": row.get("EID", ""),
            "DOI": row.get("DOI", ""),
            "Title": row.get("Title", ""),
            "Year": row.get("Year", ""),
            "year_band": row.get("year_band", ""),
            "abstract_available": abstract_ok,
            "Taxon_Relevance_A": normalize_taxon_relevance(
                llm_output.get(RELEVANCE_FIELD, RELEVANCE_OFF_TARGET)
            ),
            "Research_Theme_A": (
                normalize_research_theme(llm_output.get("Research_Theme", "Not Specified"))
                or "Not Specified"
            ),
            "Country_A": llm_output.get("Country", ""),
            "Region_Global_A": llm_output.get("Region_Global", "Not Specified"),
            "model_A": model,
            "raw_json_A": json.dumps(llm_output, ensure_ascii=False),
        }
        # Keep pilot theme aligned with shared post-process (NS → Other when off-target).
        if (
            out["Research_Theme_A"] in ("", "Not Specified")
            and out["Taxon_Relevance_A"] == RELEVANCE_OFF_TARGET
        ):
            out["Research_Theme_A"] = "Other"
        return f"{qid}::{row['row_key']}", out

    def flush():
        with save_lock:
            pd.DataFrame(list(coded.values())).to_csv(out_path, index=False)

    if pending:
        completed = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_row, row): row for row in pending}
            with tqdm(total=len(pending), desc="GPT coding") as pbar:
                for fut in as_completed(futures):
                    try:
                        key, out = fut.result()
                        coded[key] = out
                    except DailyRateLimitExceeded as e:
                        flush()
                        raise SystemExit(
                            f"GPT daily/rolling rate limit; progress saved. Re-run to resume.\n{e}"
                        ) from e
                    except Exception as e:
                        row = futures[fut]
                        key = f"{row['query_id']}::{row['row_key']}"
                        msg = str(e)
                        # Do not persist rate-limit failures as fake labels.
                        if "rate_limit" in msg.lower() or "429" in msg:
                            print(f"\nGPT rate-limit on {key} (left uncoded for resume): {e}")
                            continue
                        print(f"\nGPT error on {key}: {e}")
                        ph = placeholder_llm_output(llm_schema)
                        coded[key] = {
                            "query_id": row["query_id"],
                            "taxon_label": row.get("taxon_label", ""),
                            "row_key": row["row_key"],
                            "EID": row.get("EID", ""),
                            "DOI": row.get("DOI", ""),
                            "Title": row.get("Title", ""),
                            "Year": row.get("Year", ""),
                            "year_band": row.get("year_band", ""),
                            "abstract_available": abstract_text_ok(row.get("Abstract", "")),
                            "Taxon_Relevance_A": RELEVANCE_OFF_TARGET,
                            "Research_Theme_A": ph.get("Research_Theme", "Not Specified"),
                            "Country_A": ph.get("Country", ""),
                            "Region_Global_A": ph.get("Region_Global", "Not Specified"),
                            "model_A": model,
                            "raw_json_A": "{}",
                        }
                    completed += 1
                    pbar.update(1)
                    if completed % save_interval == 0:
                        flush()
        flush()
    print(f"✓ GPT coded -> {out_path} ({len(coded)} rows)")
    return pd.DataFrame(list(coded.values()))


def run_gemini_coding(
    df: pd.DataFrame,
    llm_schema: dict,
    llm_schema_text: str,
    model: str,
    num_threads: int,
    save_interval: int,
) -> pd.DataFrame:
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit("GEMINI_API_KEY (or GOOGLE_API_KEY) not set in .env")
    client = get_gemini_client(api_key)
    out_path = pilot_dir() / "gemini_coded.csv"
    pilot_dir().mkdir(parents=True, exist_ok=True)

    coded: dict[str, dict] = {}
    if out_path.exists():
        existing = pd.read_csv(out_path)
        for _, row in existing.iterrows():
            coded[f"{row['query_id']}::{row['row_key']}"] = row.to_dict()
        print(f"Gemini resume: {len(coded):,} rows in {out_path}")

    pending = [
        row
        for _, row in df.iterrows()
        if f"{row['query_id']}::{row['row_key']}" not in coded
    ]
    print(f"Model B (Gemini): {model}  to code: {len(pending):,} / {len(df):,}")

    def process_row(row: pd.Series) -> tuple[str, dict]:
        qid = row["query_id"]
        llm_cfg = get_query_config(qid)["llm"]
        abstract = row.get("Abstract", "")
        abstract_ok = abstract_text_ok(abstract)
        llm_output = classify_gemini(
            client,
            model,
            row.get("Title", ""),
            abstract if abstract_ok else "",
            row.get("Author_Affiliations", None),
            llm_schema,
            llm_schema_text,
            llm_cfg,
        )
        llm_output = normalize_llm_output(llm_output, llm_schema)
        out = {
            "query_id": qid,
            "taxon_label": row.get("taxon_label", ""),
            "row_key": row["row_key"],
            "EID": row.get("EID", ""),
            "DOI": row.get("DOI", ""),
            "Title": row.get("Title", ""),
            "Year": row.get("Year", ""),
            "year_band": row.get("year_band", ""),
            "abstract_available": abstract_ok,
            "Taxon_Relevance_B": normalize_taxon_relevance(
                llm_output.get(RELEVANCE_FIELD, RELEVANCE_OFF_TARGET)
            ),
            "Research_Theme_B": (
                normalize_research_theme(llm_output.get("Research_Theme", "Not Specified"))
                or "Not Specified"
            ),
            "Country_B": llm_output.get("Country", ""),
            "Region_Global_B": llm_output.get("Region_Global", "Not Specified"),
            "model_B": model,
            "raw_json_B": json.dumps(llm_output, ensure_ascii=False),
        }
        if (
            out["Research_Theme_B"] in ("", "Not Specified")
            and out["Taxon_Relevance_B"] == RELEVANCE_OFF_TARGET
        ):
            out["Research_Theme_B"] = "Other"
        return f"{qid}::{row['row_key']}", out

    def flush():
        with save_lock:
            pd.DataFrame(list(coded.values())).to_csv(out_path, index=False)

    if pending:
        completed = 0
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            futures = {executor.submit(process_row, row): row for row in pending}
            with tqdm(total=len(pending), desc="Gemini coding") as pbar:
                for fut in as_completed(futures):
                    try:
                        key, out = fut.result()
                        coded[key] = out
                    except Exception as e:
                        row = futures[fut]
                        key = f"{row['query_id']}::{row['row_key']}"
                        print(f"\nGemini error on {key}: {e}")
                        coded[key] = {
                            "query_id": row["query_id"],
                            "taxon_label": row.get("taxon_label", ""),
                            "row_key": row["row_key"],
                            "EID": row.get("EID", ""),
                            "DOI": row.get("DOI", ""),
                            "Title": row.get("Title", ""),
                            "Year": row.get("Year", ""),
                            "year_band": row.get("year_band", ""),
                            "abstract_available": abstract_text_ok(row.get("Abstract", "")),
                            "Taxon_Relevance_B": RELEVANCE_OFF_TARGET,
                            "Research_Theme_B": "Not Specified",
                            "Country_B": "",
                            "Region_Global_B": "Not Specified",
                            "model_B": model,
                            "raw_json_B": "{}",
                        }
                    completed += 1
                    pbar.update(1)
                    if completed % save_interval == 0:
                        flush()
        flush()
    print(f"✓ Gemini coded -> {out_path} ({len(coded)} rows)")
    return pd.DataFrame(list(coded.values()))


def compare_and_write(manifest: pd.DataFrame, gpt: pd.DataFrame, gemini: pd.DataFrame) -> None:
    a_cols = [
        "query_id",
        "row_key",
        "Taxon_Relevance_A",
        "Research_Theme_A",
        "Country_A",
        "Region_Global_A",
        "model_A",
        "raw_json_A",
    ]
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
    keep_m = [
        c
        for c in (
            "query_id",
            "taxon_label",
            "row_key",
            "Year",
            "year_band",
            "abstract_available",
            "Title",
            "DOI",
            "Abstract",
        )
        if c in manifest.columns
    ]
    merged = manifest[keep_m].merge(gpt[a_cols], on=["query_id", "row_key"], how="inner")
    merged = merged.merge(gemini[b_cols], on=["query_id", "row_key"], how="inner")
    if len(merged) == 0:
        raise SystemExit("No overlapping rows between GPT and Gemini pilot outputs.")

    merged["Taxon_Relevance_A"] = _clean(merged["Taxon_Relevance_A"], "Taxon_Relevance")
    merged["Taxon_Relevance_B"] = _clean(merged["Taxon_Relevance_B"], "Taxon_Relevance")
    merged["Research_Theme_A"] = _clean(merged["Research_Theme_A"], "Research_Theme")
    merged["Research_Theme_B"] = _clean(merged["Research_Theme_B"], "Research_Theme")
    merged["agree_relevance"] = merged["Taxon_Relevance_A"] == merged["Taxon_Relevance_B"]
    merged["agree_theme"] = merged["Research_Theme_A"] == merged["Research_Theme_B"]
    merged["agree_both"] = merged["agree_relevance"] & merged["agree_theme"]

    in_set = {"Primary focus", "Secondary mention"}
    merged["in_set_A"] = merged["Taxon_Relevance_A"].isin(in_set)
    merged["in_set_B"] = merged["Taxon_Relevance_B"].isin(in_set)
    merged["agree_gate"] = merged["in_set_A"] == merged["in_set_B"]

    n = len(merged)
    metrics_rows = []
    for field, a_col, b_col, agree_col in (
        ("Taxon_Relevance", "Taxon_Relevance_A", "Taxon_Relevance_B", "agree_relevance"),
        ("Research_Theme", "Research_Theme_A", "Research_Theme_B", "agree_theme"),
    ):
        agree_n = int(merged[agree_col].sum())
        metrics_rows.append(
            {
                "scope": "overall",
                "field": field,
                "n": n,
                "agreement_n": agree_n,
                "agreement_pct": round(100.0 * agree_n / n, 2),
                "cohens_kappa": round(cohens_kappa(merged[a_col].tolist(), merged[b_col].tolist()), 4),
            }
        )
    both_n = int(merged["agree_both"].sum())
    metrics_rows.append(
        {
            "scope": "overall",
            "field": "both_fields",
            "n": n,
            "agreement_n": both_n,
            "agreement_pct": round(100.0 * both_n / n, 2),
            "cohens_kappa": float("nan"),
        }
    )
    gate_n = int(merged["agree_gate"].sum())
    metrics_rows.append(
        {
            "scope": "overall",
            "field": "binary_gate",
            "n": n,
            "agreement_n": gate_n,
            "agreement_pct": round(100.0 * gate_n / n, 2),
            "cohens_kappa": round(
                cohens_kappa(
                    merged["in_set_A"].map({True: "in", False: "out"}).tolist(),
                    merged["in_set_B"].map({True: "in", False: "out"}).tolist(),
                ),
                4,
            ),
        }
    )
    metrics_rows.append(
        {
            "scope": "in_set_rate",
            "field": "Taxon_Relevance_A_in_set",
            "n": n,
            "agreement_n": int(merged["in_set_A"].sum()),
            "agreement_pct": round(100.0 * merged["in_set_A"].mean(), 2),
            "cohens_kappa": float("nan"),
        }
    )
    metrics_rows.append(
        {
            "scope": "in_set_rate",
            "field": "Taxon_Relevance_B_in_set",
            "n": n,
            "agreement_n": int(merged["in_set_B"].sum()),
            "agreement_pct": round(100.0 * merged["in_set_B"].mean(), 2),
            "cohens_kappa": float("nan"),
        }
    )

    ns_a = int((merged["Taxon_Relevance_A"].str.lower() == "not specified").sum())
    ns_b = int((merged["Taxon_Relevance_B"].str.lower() == "not specified").sum())
    metrics_rows.append(
        {
            "scope": "ns_count",
            "field": "Taxon_Relevance_A_Not_Specified",
            "n": n,
            "agreement_n": ns_a,
            "agreement_pct": round(100.0 * ns_a / n, 2),
            "cohens_kappa": float("nan"),
        }
    )
    metrics_rows.append(
        {
            "scope": "ns_count",
            "field": "Taxon_Relevance_B_Not_Specified",
            "n": n,
            "agreement_n": ns_b,
            "agreement_pct": round(100.0 * ns_b / n, 2),
            "cohens_kappa": float("nan"),
        }
    )

    # Top confusion pairs for disagreements
    dis = merged[~merged["agree_both"]].copy()
    dis["abstract_snippet"] = dis["Abstract"].map(lambda x: abstract_snippet(x, 500))
    dis["disagreement_type"] = dis.apply(classify_disagreement_type, axis=1)
    disagreements = pd.DataFrame(
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
        }
    ).sort_values(["query_id", "Year", "Title"]).reset_index(drop=True)

    metrics = pd.DataFrame(metrics_rows)
    out_dir = pilot_dir()
    out_dir.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out_dir / "agreement_metrics.csv", index=False)
    disagreements.to_csv(out_dir / "disagreements.csv", index=False)

    print(f"\nJoined rows: {n}")
    print(f"Taxon_Relevance NS counts: GPT={ns_a}, Gemini={ns_b}")
    print("\nOverall agreement:")
    print(metrics[metrics["scope"] == "overall"][["field", "n", "agreement_pct", "cohens_kappa"]].to_string(index=False))

    rel_dis = merged[~merged["agree_relevance"]]
    if len(rel_dis):
        pairs = (
            rel_dis.groupby(["Taxon_Relevance_A", "Taxon_Relevance_B"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .head(8)
        )
        print("\nTop Taxon_Relevance confusion pairs:")
        print(pairs.to_string(index=False))
    theme_dis = merged[~merged["agree_theme"]]
    if len(theme_dis):
        pairs = (
            theme_dis.groupby(["Research_Theme_A", "Research_Theme_B"])
            .size()
            .reset_index(name="n")
            .sort_values("n", ascending=False)
            .head(8)
        )
        print("\nTop Research_Theme confusion pairs:")
        print(pairs.to_string(index=False))

    print(f"\n✓ {out_dir / 'agreement_metrics.csv'}")
    print(f"✓ {out_dir / 'disagreements.csv'}")


def main():
    global _PILOT_OUT_SUBDIR
    parser = argparse.ArgumentParser(description="100-paper GPT vs Gemini pilot (new Taxon_Relevance schema)")
    parser.add_argument("--n-per-group", type=int, default=PILOT_N_PER_GROUP)
    parser.add_argument("--seed", type=int, default=PILOT_SEED)
    parser.add_argument("--threads", type=int, default=NUM_THREADS)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--gpt-model", default=DEFAULT_MODEL)
    parser.add_argument("--gemini-model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument(
        "--out-subdir",
        default="pilot100",
        help="Output folder under analysis/combined/llm_validation/ (default: pilot100)",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=None,
        help="Use this CSV as the coding sample (e.g. sample_manifest.csv for full 1500). "
        "Skips drawing pilot100_manifest.csv.",
    )
    parser.add_argument("--force-sample", action="store_true", help="Redraw pilot100_manifest.csv")
    parser.add_argument("--skip-gpt", action="store_true")
    parser.add_argument("--skip-gemini", action="store_true")
    parser.add_argument("--compare-only", action="store_true")
    args = parser.parse_args()

    _PILOT_OUT_SUBDIR = args.out_subdir
    _ = DEFAULT_SEED

    manifest = load_coding_manifest(
        args.manifest, args.n_per_group, args.seed, force=args.force_sample
    )
    llm_schema, llm_schema_text = load_schema_for_llm()
    print("Taxon_Relevance allowed:", llm_schema.get(RELEVANCE_FIELD))
    print(f"Output dir: {pilot_dir()}")
    print(f"Models: GPT={args.gpt_model}  Gemini={args.gemini_model}")

    gpt_path = pilot_dir() / "gpt_coded.csv"
    gem_path = pilot_dir() / "gemini_coded.csv"

    if not args.compare_only:
        if not args.skip_gpt:
            run_gpt_coding(
                manifest,
                llm_schema,
                llm_schema_text,
                model=args.gpt_model,
                num_threads=args.threads,
                save_interval=args.save_interval,
            )
        if not args.skip_gemini:
            run_gemini_coding(
                manifest,
                llm_schema,
                llm_schema_text,
                model=args.gemini_model,
                num_threads=args.threads,
                save_interval=args.save_interval,
            )

    if not gpt_path.exists() or not gem_path.exists():
        if args.compare_only or (not args.skip_gpt and not args.skip_gemini):
            raise SystemExit(
                f"Need both coded files to compare.\n  GPT: {gpt_path.exists()} {gpt_path}\n"
                f"  Gemini: {gem_path.exists()} {gem_path}"
            )
        print("Skipping compare: waiting for both model outputs (parallel/partial run).")
        return

    gpt = pd.read_csv(gpt_path)
    gemini = pd.read_csv(gem_path)
    n_need = len(manifest)
    if len(gpt) < n_need or len(gemini) < n_need:
        msg = (
            f"GPT {len(gpt):,}/{n_need:,}, Gemini {len(gemini):,}/{n_need:,}"
        )
        if args.compare_only:
            print(f"Warning: comparing incomplete coded files ({msg})")
        else:
            # Avoid overwriting metrics when one parallel worker finishes early.
            print(f"Skipping compare until both models finish ({msg})")
            return
    compare_and_write(manifest, gpt, gemini)


if __name__ == "__main__":
    main()
