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
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.pipeline import (  # noqa: E402
    PipelinePaths,
    add_query_arg,
    get_query_config,
    load_dotenv,
)

DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
NUM_THREADS = 8
SAVE_INTERVAL = 50
LEGACY_RELEVANCE_FIELD = "Trichoptera_Relevance"
RELEVANCE_FIELD = "Taxon_Relevance"

progress_lock = Lock()
save_lock = Lock()


class DailyRateLimitExceeded(Exception):
    """Raised when the API daily request cap is reached."""


def is_daily_rate_limit_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    return (
        "rate_limit_exceeded" in msg
        and ("requests per day" in msg or "rpd" in msg or "used 10000" in msg)
    )


def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        return None


def build_prompt(
    title: str,
    abstract: str,
    affiliation_text: str,
    llm_schema_text: str,
    llm_cfg: dict,
) -> str:
    study = llm_cfg["study_descriptor"]
    taxon = llm_cfg["taxon_name"]
    rel_field = RELEVANCE_FIELD

    return f"""
You are coding academic papers for a bibliometric study on {study}.

Your task is to assign values ONLY for the fields listed below, using the predefined schema.

Schema (allowed values):
{llm_schema_text}

Paper title:
{title}

Paper abstract:
{abstract}{affiliation_text}

CORE RULES (follow strictly):
- Do NOT assume {taxon} are the main focus unless clearly stated.
- Prefer the MOST SPECIFIC allowed value supported by the text.
- Use "Other" ONLY if no allowed value applies.
- For Country: If information is missing, leave empty (do NOT use "Not Specified").
- For other fields: If information is missing, use "Not Specified".
- Do NOT invent taxa, locations, methods, or conclusions.
- Output VALID JSON only. No explanations.

GEOGRAPHIC EXTRACTION PRIORITY:
1. Explicit country/state/city names -> extract country
2. Species names with geographic indicators (japonica->Japan, sinensis->China, etc.) -> infer country
3. Author affiliations or institutional names -> infer country
4. Rivers/lakes/regions -> infer country if context is clear
5. If none of the above, leave Country empty (not "Not Specified")

FOCUS DISCIPLINE RULE:
Before assigning fields, determine whether {taxon} are:
- the PRIMARY study organism
- studied alongside other taxa
- mentioned incidentally
- not a {taxon}-focused paper

Only assign {taxon}-specific taxonomy or themes if supported.

FIELD-SPECIFIC GUIDANCE:

Research_Theme:
- Use "Taxonomy/Systematics" for species descriptions or classifications.
- Use "Evolution/Phylogeny" for phylogenetic analyses.
- Use "Biomonitoring/Water Quality" ONLY if {taxon} are used as indicators.
- Use "Ecology/Behavior" for life history, traits, distributions, interactions.
- Use "Materials Science (Silk)" ONLY if silk properties are studied.
- Use "Applied Ecology" for applied research that doesn't fit other categories.
- Use "Conservation" for conservation-focused studies.
- Use "Physiology" for physiological studies.
- Use "Other" only if none of the above apply.

Country:
- Extract the PRIMARY country where the research was conducted (field site, study location, or primary geographic focus).
- PRIORITY ORDER (use first available):
  1. Explicit country/state/city in abstract or title (highest priority)
  2. Study location mentioned in abstract (rivers, lakes, regions)
  3. Species names with geographic indicators (japonica -> Japan, sinensis -> China, etc.)
  4. Author affiliations (if provided) - infer country from institution names
- Use standard country names: "United States" (not "USA"), "United Kingdom" (not "UK"), "South Korea" (not "Korea").
- If multiple countries mentioned, choose the PRIMARY study location (where fieldwork/data collection occurred).
- If country cannot be determined from title/abstract, leave empty (do NOT use "Not Specified").

Region_Global:
- Map the Country to its biogeographic region (Nearctic, Neotropical, Oriental, Palearctic, East Palearctic, Afrotropical, Australasian).
- If Country is empty but region can be inferred from geographic context, assign region.
- Use "Global" ONLY for meta-analyses, global syntheses, or multi-continent studies.
- Use "Not Specified" ONLY if neither country nor region can be determined.

{rel_field} (JSON field name is fixed in the schema; interpret it as focus on {taxon} for this run):
- "Primary focus": {taxon} are the main study organism.
- "Secondary mention": {taxon} are studied alongside other taxa.
- "Peripheral": {taxon} mentioned but not central to the study.
- "Not target-taxon-focused": Paper does not primarily focus on {taxon}.

OUTPUT FORMAT:
- One JSON object
- One value per field
"""


def classify(
    client: OpenAI,
    model: str,
    title,
    abstract,
    author_affiliations,
    llm_schema: dict,
    llm_schema_text: str,
    llm_cfg: dict,
    max_retries=3,
):
    affiliation_text = ""
    if pd.notna(author_affiliations) and str(author_affiliations).strip():
        affiliation_text = f"\n\nAuthor affiliations:\n{str(author_affiliations)}"

    prompt = build_prompt(
        title,
        abstract,
        affiliation_text,
        llm_schema_text,
        llm_cfg,
    )

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "You are a careful bibliometric classifier."},
                    {"role": "user", "content": prompt},
                ],
            )

            raw = response.choices[0].message.content.strip()
            parsed = safe_json_loads(raw)

            if parsed is not None:
                return parsed

            return {col: "Not Specified" for col in llm_schema.keys()}

        except Exception as e:
            if is_daily_rate_limit_error(e):
                raise DailyRateLimitExceeded(str(e)) from e
            if attempt < max_retries - 1:
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue
            print(f"Warning: Failed to classify after {max_retries} attempts: {e}")
            return {col: "Not Specified" for col in llm_schema.keys()}

    return {col: "Not Specified" for col in llm_schema.keys()}


def run_llm_coding(
    paths: PipelinePaths,
    test_mode: bool,
    test_size: int,
    num_threads: int,
    save_interval: int,
):
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set. Please set it in .env file or environment variable.")

    qconf = get_query_config(paths.query_id)
    llm_cfg = qconf["llm"]

    client = OpenAI(api_key=api_key)

    input_csv = paths.with_abstracts
    authors_csv = paths.with_authors
    schema_file = paths.schema
    output_csv = paths.coded

    df = pd.read_csv(input_csv)

    if authors_csv.exists():
        df_authors = pd.read_csv(authors_csv)
        df = df.merge(
            df_authors[["Title", "Author_Affiliations"]],
            on="Title",
            how="left",
            suffixes=("", "_auth"),
        )
        print(f"Loaded author affiliations for {df['Author_Affiliations'].notna().sum():,} papers")
    else:
        df["Author_Affiliations"] = None
        print("Warning: Author affiliations file not found. Using abstract-only extraction.")

    with open(schema_file, encoding="utf-8") as f:
        schema = json.load(f)

    relevance_allowed_values = None
    schema_columns = schema.get("columns", {})
    if RELEVANCE_FIELD in schema_columns and "allowed_values" in schema_columns[RELEVANCE_FIELD]:
        relevance_allowed_values = schema_columns[RELEVANCE_FIELD]["allowed_values"]
    elif LEGACY_RELEVANCE_FIELD in schema_columns and "allowed_values" in schema_columns[LEGACY_RELEVANCE_FIELD]:
        relevance_allowed_values = schema_columns[LEGACY_RELEVANCE_FIELD]["allowed_values"]
    else:
        relevance_allowed_values = [
            "Primary focus",
            "Secondary mention",
            "Peripheral",
            "Not target-taxon-focused",
            "Not Specified",
        ]

    llm_coded_fields = ["Country", "Region_Global", "Research_Theme", RELEVANCE_FIELD]
    llm_schema = {}
    for col, spec in schema_columns.items():
        if col in llm_coded_fields:
            if "allowed_values" in spec:
                llm_schema[col] = spec["allowed_values"]
            else:
                llm_schema[col] = "short free-text"
    if RELEVANCE_FIELD not in llm_schema:
        llm_schema[RELEVANCE_FIELD] = relevance_allowed_values

    llm_schema_text = json.dumps(llm_schema, indent=2)

    def norm(v):
        if pd.isna(v):
            return ""
        return str(v).strip()

    def make_row_key(row) -> str:
        eid = norm(row.get("EID", ""))
        if eid:
            return f"eid::{eid}"
        scopus_id = norm(row.get("ScopusID", ""))
        if scopus_id:
            return f"scopus::{scopus_id}"
        doi = norm(row.get("DOI", "")).lower()
        if doi:
            return f"doi::{doi}"
        title = norm(row.get("Title", "")).lower()
        year = norm(row.get("Year", ""))
        journal = norm(row.get("Source", "")).lower()
        return f"title-year-source::{title}::{year}::{journal}"

    def relevance_value(row) -> str:
        return norm(row.get(RELEVANCE_FIELD, row.get(LEGACY_RELEVANCE_FIELD, "")))

    def normalize_row_relevance_columns(row_dict: dict) -> dict:
        row_out = dict(row_dict)
        rel_val = norm(row_out.get(RELEVANCE_FIELD, row_out.get(LEGACY_RELEVANCE_FIELD, "")))
        if rel_val:
            row_out[RELEVANCE_FIELD] = rel_val
        row_out.pop(LEGACY_RELEVANCE_FIELD, None)
        return row_out

    def row_is_llm_placeholder(row) -> bool:
        """True if this row was never successfully classified (checkpoint filler or API fallback)."""
        t = norm(row.get("Research_Theme", "")).lower()
        r = relevance_value(row).lower()
        return t in ("", "not specified", "nan") and r in ("", "not specified", "nan")

    def process_paper(row_data):
        row_key, row = row_data
        title = row.get("Title", "")
        abstract = row.get("Abstract", "")
        abstract_available = isinstance(abstract, str) and abstract.strip() != ""
        author_affiliations = row.get("Author_Affiliations", None)

        llm_output = classify(
            client,
            model,
            title,
            abstract if abstract_available else "",
            author_affiliations=author_affiliations,
            llm_schema=llm_schema,
            llm_schema_text=llm_schema_text,
            llm_cfg=llm_cfg,
        )

        new_row = row.to_dict()
        new_row.update(llm_output)
        if RELEVANCE_FIELD not in new_row:
            new_row[RELEVANCE_FIELD] = norm(new_row.get(LEGACY_RELEVANCE_FIELD, ""))
        new_row.pop(LEGACY_RELEVANCE_FIELD, None)
        new_row["abstract_available"] = abstract_available
        return row_key, new_row

    print(f"\nquery_id={paths.query_id}")
    print(f"Using model: {model}")
    if test_mode:
        print(f"TEST MODE: Processing only {test_size} records with {num_threads} threads")
        df = df.head(test_size)
        rows_to_process = [(make_row_key(row), row) for _, row in df.iterrows()]
        coded_dict = {}

        rate_limit_hit = False
        rate_limit_msg = None
        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_title = {
                executor.submit(process_paper, row_data): row_data[0] for row_data in rows_to_process
            }

            with tqdm(total=len(rows_to_process), desc="Coding papers") as pbar:
                for future in as_completed(future_to_title):
                    try:
                        row_key, new_row = future.result()
                        coded_dict[row_key] = new_row
                        pbar.update(1)
                    except DailyRateLimitExceeded as e:
                        rate_limit_hit = True
                        rate_limit_msg = str(e)
                        print("\nDaily API request limit reached. Stopping test run early.")
                        executor.shutdown(wait=False, cancel_futures=True)
                        break
                    except Exception as e:
                        row_key = future_to_title[future]
                        print(f"\nError processing paper {row_key!r}: {e}")
                        row = df[[make_row_key(r) == row_key for _, r in df.iterrows()]].iloc[0]
                        error_row = row.to_dict()
                        error_row.update({col: "Not Specified" for col in llm_schema.keys()})
                        error_row["abstract_available"] = pd.notna(row.get("Abstract"))
                        coded_dict[row_key] = error_row
                        pbar.update(1)

        coded_test = [coded_dict[make_row_key(row)] for _, row in df.iterrows() if make_row_key(row) in coded_dict]
        test_output = paths.processed / "scopus_api_coded_TEST.csv"
        test_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(coded_test).to_csv(test_output, index=False)
        if rate_limit_hit:
            print(f"\n⚠ Stopped early due to daily rate limit: {rate_limit_msg}")
        print(f"\n✓ Test complete! Saved {len(coded_test)} records to: {test_output}")
        return

    print(f"\nStarting LLM coding with {num_threads} threads...")
    df_all = df.copy()
    print(f"Total papers: {len(df_all):,}")
    print(f"Progress will be saved every {save_interval} papers\n")

    def coded_rows_in_input_order():
        """One output row per df_all row, in input order. Missing keys use LLM placeholders."""
        rows_out = []
        for _, r in df_all.iterrows():
            k = make_row_key(r)
            if k in coded_dict:
                rows_out.append(coded_dict[k])
            else:
                d = r.to_dict()
                for col in llm_schema:
                    d[col] = "Not Specified"
                d.pop(LEGACY_RELEVANCE_FIELD, None)
                ab = r.get("Abstract", "")
                d["abstract_available"] = isinstance(ab, str) and ab.strip() != ""
                rows_out.append(d)
        return rows_out

    coded_dict = {}
    if output_csv.exists():
        try:
            df_existing = pd.read_csv(output_csv)
            if "Country" in df_existing.columns or "Research_Theme" in df_existing.columns:
                already_coded_keys = set()
                for _, row in df_existing.iterrows():
                    if not row_is_llm_placeholder(row):
                        already_coded_keys.add(make_row_key(row))
                n_placeholder = int(df_existing.apply(row_is_llm_placeholder, axis=1).sum())
                print(f"Found existing coded file with {len(df_existing)} papers")
                print(
                    f"Resuming: {len(already_coded_keys):,} rows look fully coded; "
                    f"{n_placeholder:,} placeholder rows will be re-processed if still in the input set."
                )
                df = df[~df.apply(make_row_key, axis=1).isin(already_coded_keys)]
                coded_dict = {
                    make_row_key(row): normalize_row_relevance_columns(row.to_dict())
                    for _, row in df_existing.iterrows()
                    if not row_is_llm_placeholder(row)
                }
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")
            print("Starting fresh...")

    rows_to_process = [(make_row_key(row), row) for _, row in df.iterrows()]

    rate_limit_hit = False
    rate_limit_msg = None
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_title = {
            executor.submit(process_paper, row_data): row_data[0] for row_data in rows_to_process
        }

        with tqdm(total=len(rows_to_process), desc="Coding papers") as pbar:
            completed_count = 0
            for future in as_completed(future_to_title):
                try:
                    row_key, new_row = future.result()
                    coded_dict[row_key] = new_row
                    completed_count += 1
                    pbar.update(1)

                    if completed_count % save_interval == 0:
                        with save_lock:
                            coded_list = coded_rows_in_input_order()
                            temp_df = pd.DataFrame(coded_list)
                            output_csv.parent.mkdir(parents=True, exist_ok=True)
                            temp_df.to_csv(output_csv, index=False)
                            pbar.set_postfix({"Saved": f"{completed_count}/{len(rows_to_process)}"})
                except DailyRateLimitExceeded as e:
                    rate_limit_hit = True
                    rate_limit_msg = str(e)
                    print("\nDaily API request limit reached. Stopping run and saving progress.")
                    executor.shutdown(wait=False, cancel_futures=True)
                    break

                except Exception as e:
                    row_key = future_to_title[future]
                    print(f"\nError processing paper {row_key!r}: {e}")
                    row = df_all[[make_row_key(r) == row_key for _, r in df_all.iterrows()]].iloc[0]
                    error_row = row.to_dict()
                    error_row.update({col: "Not Specified" for col in llm_schema.keys()})
                    error_row["abstract_available"] = pd.notna(row.get("Abstract"))
                    coded_dict[row_key] = error_row
                    pbar.update(1)

    coded_list = coded_rows_in_input_order()
    final_df = pd.DataFrame(coded_list)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

    if rate_limit_hit:
        print(f"\n⚠ Run halted due to daily rate limit: {rate_limit_msg}")
        print("Resume this command after your request quota resets; completed rows are preserved.")
        return

    print(f"\n✓ Complete! Saved {len(coded_list):,} records to: {output_csv}")

    if "Country" in final_df.columns:
        country_filled = final_df["Country"].notna() & (final_df["Country"] != "") & (
            final_df["Country"] != "Not Specified"
        )
        print(f"  Papers with Country: {country_filled.sum():,} ({country_filled.sum()/len(final_df)*100:.1f}%)")
    if "Region_Global" in final_df.columns:
        region_filled = final_df["Region_Global"] != "Not Specified"
        print(f"  Papers with Region: {region_filled.sum():,} ({region_filled.sum()/len(final_df)*100:.1f}%)")


def main():
    parser = argparse.ArgumentParser(description="LLM coding for bibliometric pipeline (shared schema)")
    add_query_arg(parser)
    parser.add_argument("--test", action="store_true", help="Process only a small sample")
    parser.add_argument("--test-size", type=int, default=20)
    parser.add_argument("--threads", type=int, default=NUM_THREADS)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    args = parser.parse_args()
    paths = PipelinePaths(args.query_id)
    run_llm_coding(
        paths,
        test_mode=args.test,
        test_size=args.test_size,
        num_threads=args.threads,
        save_interval=args.save_interval,
    )


if __name__ == "__main__":
    main()
