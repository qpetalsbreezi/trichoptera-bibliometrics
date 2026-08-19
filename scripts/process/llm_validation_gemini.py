#!/usr/bin/env python3
"""Re-code the frozen validation sample with Gemini (Model B)."""

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
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
_PROCESS_DIR = Path(__file__).resolve().parent
for p in (_SCRIPTS_DIR, _PROCESS_DIR):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from lib.llm_validation import (  # noqa: E402
    DEFAULT_GEMINI_MODEL,
    RELEVANCE_FIELD,
    RELEVANCE_OFF_TARGET,
    abstract_text_ok,
    gemini_coded_path,
    is_all_not_specified,
    load_schema_for_llm,
    normalize_taxon_relevance,
    sample_manifest_path,
    taxon_relevance_is_ns,
    validation_dir,
)
from lib.pipeline import get_query_config, load_dotenv  # noqa: E402
from llm_code_taxon import (  # noqa: E402
    build_prompt,
    normalize_classified_output,
    placeholder_llm_output,
    safe_json_loads,
)

TEMPERATURE = 0
NUM_THREADS = 8
SAVE_INTERVAL = 25
save_lock = Lock()

ALL_NS_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Never use \"Not Specified\" for Taxon_Relevance. "
    "Choose a concrete tier "
    "(Primary focus, Secondary mention, Peripheral, or Not target-taxon-focused). "
    "If the taxon appears in the title, do not use Not target-taxon-focused. "
    "Use \"Not target-taxon-focused\" only when the taxon is absent/irrelevant, "
    "or when the abstract is unavailable AND the title does not name the taxon. "
    "For Research_Theme, prefer a concrete theme; use Not Specified only if "
    "title and abstract are both empty/unavailable; if relevance is "
    "Not target-taxon-focused and no theme fits, use Other."
)


def get_gemini_client(api_key: str):
    try:
        from google import genai
    except ImportError as e:
        raise SystemExit(
            "Missing dependency google-genai. Install with: pip install google-genai"
        ) from e
    # 120s request timeout so a stalled Pro call cannot freeze the thread pool.
    return genai.Client(api_key=api_key, http_options={"timeout": 120_000})


def classify_gemini(
    client,
    model: str,
    title,
    abstract,
    author_affiliations,
    llm_schema: dict,
    llm_schema_text: str,
    llm_cfg: dict,
    max_retries: int = 3,
):
    from google.genai import types

    affiliation_text = ""
    if pd.notna(author_affiliations) and str(author_affiliations).strip():
        affiliation_text = f"\n\nAuthor affiliations:\n{str(author_affiliations)}"

    base_prompt = build_prompt(
        title,
        abstract,
        affiliation_text,
        llm_schema_text,
        llm_cfg,
    )
    retried_all_ns = False

    def _generate(prompt: str):
        return client.models.generate_content(
            model=model,
            contents=prompt,
            config=types.GenerateContentConfig(
                temperature=TEMPERATURE,
                response_mime_type="application/json",
                system_instruction="You are a careful bibliometric classifier.",
            ),
        )

    def _parse_response(raw: str):
        parsed = safe_json_loads(raw)
        if parsed is not None:
            return parsed
        if raw.startswith("```"):
            raw2 = raw.strip("`")
            if raw2.startswith("json"):
                raw2 = raw2[4:]
            return safe_json_loads(raw2.strip())
        return None

    prompt = base_prompt
    for attempt in range(max_retries):
        try:
            response = _generate(prompt)
            raw = (response.text or "").strip()
            parsed = _parse_response(raw)
            if parsed is not None:
                needs_retry = (
                    not retried_all_ns
                    and (
                        is_all_not_specified(parsed, llm_schema)
                        or taxon_relevance_is_ns(parsed)
                    )
                )
                if needs_retry:
                    retried_all_ns = True
                    prompt = base_prompt + ALL_NS_RETRY_SUFFIX
                    continue
                # Final map: Taxon_Relevance Not Specified -> Not target-taxon-focused
                if taxon_relevance_is_ns(parsed):
                    parsed[RELEVANCE_FIELD] = RELEVANCE_OFF_TARGET
                return parsed
            return placeholder_llm_output(llm_schema)
        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep((2**attempt) * 0.5)
                continue
            print(f"Warning: Failed to classify after {max_retries} attempts: {e}")
            return placeholder_llm_output(llm_schema)

    return placeholder_llm_output(llm_schema)


def normalize_llm_output(llm_output: dict, llm_schema: dict) -> dict:
    """Shared GPT/Gemini normalization (NS relevance → off-target; NS theme → Other when off-target)."""
    out = normalize_classified_output(llm_output, llm_schema)
    if RELEVANCE_FIELD in out:
        out[RELEVANCE_FIELD] = normalize_taxon_relevance(out[RELEVANCE_FIELD])
    # Re-apply theme rule after relevance remapping (NS relevance becomes off-target).
    if "Research_Theme" in out:
        theme = "" if out["Research_Theme"] is None else str(out["Research_Theme"]).strip()
        rel = "" if out.get(RELEVANCE_FIELD) is None else str(out.get(RELEVANCE_FIELD, "")).strip()
        if (not theme or theme.lower() in ("not specified", "nan", "none")) and (
            rel == RELEVANCE_OFF_TARGET
        ):
            out["Research_Theme"] = "Other"
    return out


def run_gemini_coding(
    model: str,
    num_threads: int,
    save_interval: int,
    test_size: int | None,
    query_ids: list[str] | None,
):
    load_dotenv()
    api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        raise SystemExit(
            "GEMINI_API_KEY (or GOOGLE_API_KEY) not set. Add it to .env or the environment."
        )
    model = os.getenv("GEMINI_MODEL", model)

    manifest_path = sample_manifest_path()
    if not manifest_path.exists():
        raise SystemExit(
            f"Missing sample manifest: {manifest_path}\n"
            "Run: python scripts/process/llm_validation_sample.py"
        )

    df = pd.read_csv(manifest_path)
    if query_ids:
        df = df[df["query_id"].isin(query_ids)].copy()
    if test_size is not None:
        df = df.head(test_size).copy()

    llm_schema, llm_schema_text = load_schema_for_llm()
    client = get_gemini_client(api_key)

    out_path = gemini_coded_path()
    validation_dir().mkdir(parents=True, exist_ok=True)

    coded: dict[str, dict] = {}
    if out_path.exists():
        existing = pd.read_csv(out_path)
        for _, row in existing.iterrows():
            key = f"{row['query_id']}::{row['row_key']}"
            coded[key] = row.to_dict()
        print(f"Resuming: {len(coded):,} already coded rows in {out_path}")

    pending = []
    for _, row in df.iterrows():
        key = f"{row['query_id']}::{row['row_key']}"
        if key not in coded:
            pending.append(row)

    print(f"Model B: {model}")
    print(f"To code: {len(pending):,}  (of {len(df):,} in scope)")

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
            "Taxon_Relevance_B": llm_output.get(RELEVANCE_FIELD, RELEVANCE_OFF_TARGET),
            "Research_Theme_B": llm_output.get("Research_Theme", "Not Specified"),
            "Country_B": llm_output.get("Country", ""),
            "Region_Global_B": llm_output.get("Region_Global", "Not Specified"),
            "model_B": model,
            "raw_json_B": json.dumps(llm_output, ensure_ascii=False),
        }
        return f"{qid}::{row['row_key']}", out

    def flush():
        with save_lock:
            rows = list(coded.values())
            pd.DataFrame(rows).to_csv(out_path, index=False)

    if not pending:
        print("Nothing to do.")
        return

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
                    print(f"\nError on {key}: {e}")
                    coded[key] = {
                        "query_id": row["query_id"],
                        "taxon_label": row.get("taxon_label", ""),
                        "row_key": row["row_key"],
                        "EID": row.get("EID", ""),
                        "DOI": row.get("DOI", ""),
                        "Title": row.get("Title", ""),
                        "Year": row.get("Year", ""),
                        "year_band": row.get("year_band", ""),
                        "abstract_available": False,
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
    print(f"\n✓ Saved {len(coded):,} Gemini-coded rows to {out_path}")


def main():
    parser = argparse.ArgumentParser(description="Gemini re-coding for LLM validation sample")
    parser.add_argument("--model", default=DEFAULT_GEMINI_MODEL)
    parser.add_argument("--threads", type=int, default=NUM_THREADS)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument("--test-size", type=int, default=None, help="Code only first N sample rows")
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        help="Limit to one or more query_ids (repeatable)",
    )
    args = parser.parse_args()
    run_gemini_coding(
        model=args.model,
        num_threads=args.threads,
        save_interval=args.save_interval,
        test_size=args.test_size,
        query_ids=args.query_ids,
    )


if __name__ == "__main__":
    main()
