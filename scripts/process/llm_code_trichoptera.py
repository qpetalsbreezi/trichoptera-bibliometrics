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

MODEL = "gpt-4o-mini"
TEMPERATURE = 0
NUM_THREADS = 8
SAVE_INTERVAL = 50

progress_lock = Lock()
save_lock = Lock()


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
    rel_field = llm_cfg["relevance_field"]

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
1. Explicit country/state/city names → extract country
2. Species names with geographic indicators (japonica→Japan, sinensis→China, etc.) → infer country
3. Author affiliations or institutional names → infer country
4. Rivers/lakes/regions → infer country if context is clear
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
  3. Species names with geographic indicators (japonica → Japan, sinensis → China, etc.)
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
- "Not Trichoptera-focused": Paper does not primarily focus on {taxon} (same enum label for every query; compare runs by query_id, not by this string alone).

OUTPUT FORMAT:
- One JSON object
- One value per field
"""


def classify(
    client: OpenAI,
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
                model=MODEL,
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

    llm_coded_fields = ["Country", "Region_Global", "Research_Theme", "Trichoptera_Relevance"]
    llm_schema = {}
    for col, spec in schema["columns"].items():
        if col in llm_coded_fields:
            if "allowed_values" in spec:
                llm_schema[col] = spec["allowed_values"]
            else:
                llm_schema[col] = "short free-text"

    llm_schema_text = json.dumps(llm_schema, indent=2)

    def process_paper(row_data):
        idx, row = row_data
        title = row.get("Title", "")
        abstract = row.get("Abstract", "")
        abstract_available = isinstance(abstract, str) and abstract.strip() != ""
        author_affiliations = row.get("Author_Affiliations", None)

        llm_output = classify(
            client,
            title,
            abstract if abstract_available else "",
            author_affiliations=author_affiliations,
            llm_schema=llm_schema,
            llm_schema_text=llm_schema_text,
            llm_cfg=llm_cfg,
        )

        new_row = row.to_dict()
        new_row.update(llm_output)
        new_row["abstract_available"] = abstract_available
        return idx, new_row

    print(f"\nquery_id={paths.query_id}")
    if test_mode:
        print(f"TEST MODE: Processing only {test_size} records with {num_threads} threads")
        df = df.head(test_size)
        rows_to_process = [(idx, row) for idx, row in df.iterrows()]
        coded_dict = {}

        with ThreadPoolExecutor(max_workers=num_threads) as executor:
            future_to_idx = {
                executor.submit(process_paper, row_data): row_data[0] for row_data in rows_to_process
            }

            with tqdm(total=len(rows_to_process), desc="Coding papers") as pbar:
                for future in as_completed(future_to_idx):
                    try:
                        idx, new_row = future.result()
                        coded_dict[idx] = new_row
                        pbar.update(1)
                    except Exception as e:
                        idx = future_to_idx[future]
                        print(f"\nError processing paper {idx}: {e}")
                        row = df.loc[idx]
                        error_row = row.to_dict()
                        error_row.update({col: "Not Specified" for col in llm_schema.keys()})
                        error_row["abstract_available"] = pd.notna(row.get("Abstract"))
                        coded_dict[idx] = error_row
                        pbar.update(1)

        coded_test = [coded_dict[i] for i in sorted(coded_dict.keys())]
        test_output = paths.processed / "scopus_api_coded_TEST.csv"
        test_output.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(coded_test).to_csv(test_output, index=False)
        print(f"\n✓ Test complete! Saved {len(coded_test)} records to: {test_output}")
        return

    print(f"\nStarting LLM coding with {num_threads} threads...")
    print(f"Total papers: {len(df):,}")
    print(f"Progress will be saved every {save_interval} papers\n")

    coded_dict = {}
    start_index = 0
    if output_csv.exists():
        try:
            df_existing = pd.read_csv(output_csv)
            if "Country" in df_existing.columns or "Research_Theme" in df_existing.columns:
                coded_titles = set(df_existing["Title"].astype(str))
                print(f"Found existing coded file with {len(df_existing)} papers")
                print("Resuming from existing progress...")
                df = df[~df["Title"].astype(str).isin(coded_titles)]
                coded_dict = {i: row.to_dict() for i, row in df_existing.iterrows()}
                start_index = len(df_existing)
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")
            print("Starting fresh...")

    rows_to_process = [(idx, row) for idx, row in df.iterrows()]

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_idx = {
            executor.submit(process_paper, row_data): row_data[0] for row_data in rows_to_process
        }

        with tqdm(total=len(rows_to_process), initial=start_index, desc="Coding papers") as pbar:
            completed_count = 0
            for future in as_completed(future_to_idx):
                try:
                    idx, new_row = future.result()
                    coded_dict[idx] = new_row
                    completed_count += 1
                    pbar.update(1)

                    if completed_count % save_interval == 0:
                        with save_lock:
                            coded_list = [coded_dict[i] for i in sorted(coded_dict.keys())]
                            temp_df = pd.DataFrame(coded_list)
                            output_csv.parent.mkdir(parents=True, exist_ok=True)
                            temp_df.to_csv(output_csv, index=False)
                            pbar.set_postfix({"Saved": f"{completed_count}/{len(rows_to_process)}"})

                except Exception as e:
                    idx = future_to_idx[future]
                    print(f"\nError processing paper {idx}: {e}")
                    row = df.loc[idx]
                    error_row = row.to_dict()
                    error_row.update({col: "Not Specified" for col in llm_schema.keys()})
                    error_row["abstract_available"] = pd.notna(row.get("Abstract"))
                    coded_dict[idx] = error_row
                    pbar.update(1)

    coded_list = [coded_dict[i] for i in sorted(coded_dict.keys())]
    final_df = pd.DataFrame(coded_list)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    final_df.to_csv(output_csv, index=False)

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
