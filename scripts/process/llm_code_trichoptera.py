import pandas as pd
import json
import time
from tqdm import tqdm
from openai import OpenAI
import os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading
from threading import Lock

# Get project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Load environment variables from .env file if it exists
env_file = PROJECT_ROOT / ".env"
if env_file.exists():
    with open(env_file, "r") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                os.environ[key.strip()] = value.strip()

api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise ValueError("OPENAI_API_KEY not set. Please set it in .env file or environment variable.")

client = OpenAI(api_key=api_key)


INPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_with_abstracts.csv"
AUTHORS_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_with_authors.csv"
SCHEMA_FILE = PROJECT_ROOT / "data/trichoptera_schema.json"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_coded.csv"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0

# Multi-threading configuration
NUM_THREADS = 8  # Adjust based on API rate limits (gpt-4o-mini allows higher throughput)
SAVE_INTERVAL = 50  # Save progress every N papers

# -------------------------
# LOAD FILES
# -------------------------
df = pd.read_csv(INPUT_CSV)

# Load author affiliations data for geographic inference
if AUTHORS_CSV.exists():
    df_authors = pd.read_csv(AUTHORS_CSV)
    # Merge author affiliations into main dataframe
    # Use Title as key (should be unique)
    df = df.merge(
        df_authors[['Title', 'Author_Affiliations']],
        on='Title',
        how='left',
        suffixes=('', '_auth')
    )
    print(f"Loaded author affiliations for {df['Author_Affiliations'].notna().sum():,} papers")
else:
    df['Author_Affiliations'] = None
    print("Warning: Author affiliations file not found. Using abstract-only extraction.")

with open(SCHEMA_FILE) as f:
    schema = json.load(f)

# Build LLM coding schema - only include fields that need LLM coding
# Exclude metadata fields (Title, Authors, Year, Journal, DOI) which come from Scopus
llm_coded_fields = ["Country", "Region_Global", "Research_Theme", "Trichoptera_Relevance"]

llm_schema = {}
for col, spec in schema["columns"].items():
    if col in llm_coded_fields:
        if "allowed_values" in spec:
            llm_schema[col] = spec["allowed_values"]
        else:
            # Free-text fields (like Country)
            llm_schema[col] = "short free-text"

llm_schema_text = json.dumps(llm_schema, indent=2)

def safe_json_loads(text):
    try:
        return json.loads(text)
    except Exception:
        return None

# Thread-safe locks for progress tracking and saving
progress_lock = Lock()
save_lock = Lock()

# -------------------------
# CLASSIFIER
# -------------------------
def classify(title, abstract, author_affiliations=None, max_retries=3):
    # Prepare affiliation text
    affiliation_text = ""
    if pd.notna(author_affiliations) and str(author_affiliations).strip():
        affiliation_text = f"\n\nAuthor affiliations:\n{str(author_affiliations)}"
    
    prompt = f"""
You are coding academic papers for a bibliometric study on Trichoptera (caddisflies).

Your task is to assign values ONLY for the fields listed below, using the predefined schema.

Schema (allowed values):
{llm_schema_text}

Paper title:
{title}

Paper abstract:
{abstract}{affiliation_text}

CORE RULES (follow strictly):
- Do NOT assume Trichoptera are the main focus unless clearly stated.
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
Before assigning fields, determine whether Trichoptera are:
- the PRIMARY study organism
- studied alongside other taxa
- mentioned incidentally
- not a Trichoptera-focused paper

Only assign Trichoptera-specific taxonomy or themes if supported.

FIELD-SPECIFIC GUIDANCE:

Research_Theme:
- Use "Taxonomy/Systematics" for species descriptions or classifications.
- Use "Evolution/Phylogeny" for phylogenetic analyses.
- Use "Biomonitoring/Water Quality" ONLY if Trichoptera are used as indicators.
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
     * "Tohoku University" → Japan
     * "University of Latvia" → Latvia
     * "Hôpital du Sacré-Cœur de Montréal" → Canada
     * "Parks Victoria" → Australia
     * "University of Manitoba" → Canada
     * Look for country names in affiliation text
     * Infer from well-known institutions if country not explicit
- Look for explicit mentions of: country names, states/provinces (infer country), cities (infer country), rivers/lakes (infer country), geographic regions.
- Study sites: "X River", "X Lake", "X National Park" → infer country
- Common patterns to extract:
  * "North Carolina" → "United States"
  * "California" → "United States"
  * "Queensland" → "Australia"
  * "Amazon" → "Brazil" (if context suggests Brazil)
  * "Yangtze River" → "China"
  * "japonica" in species name → "Japan" (if study focuses on that species)
- Use standard country names: "United States" (not "USA"), "United Kingdom" (not "UK"), "South Korea" (not "Korea").
- If multiple countries mentioned, choose the PRIMARY study location (where fieldwork/data collection occurred).
- If country cannot be determined from title/abstract, leave empty (do NOT use "Not Specified").
- Be AGGRESSIVE: if ANY geographic indicator exists (explicit or implicit), extract it.
- EXAMPLES:
  * Paper on "Goera japonica" with no location → "Japan" (species name indicates Japanese origin)
  * Paper mentions "North Carolina" → "United States"
  * Paper on "Amazon basin" → "Brazil" (most common country for Amazon basin studies)
  * Paper studying multiple species including "japonica", "kisoensis" → "Japan" (if these are the primary study species)
- IMPORTANT: If the paper focuses on species with geographic names (e.g., "japonica", "sinensis", "americana") and no other location is mentioned, infer the country from the species name.

Region_Global:
- Map the Country to its biogeographic region:
  * United States, Canada, Mexico → "Nearctic"
  * Brazil, Argentina, Chile, Colombia, Peru, Ecuador, Venezuela, etc. → "Neotropical"
  * China, Japan, India, Thailand, Vietnam, Indonesia, Malaysia, Philippines, etc. → "Oriental"
  * UK, Germany, France, Spain, Italy, Russia (European), etc. → "Palearctic"
  * Russia (Asian), Mongolia, Kazakhstan (Asian) → "East Palearctic"
  * South Africa, Kenya, Tanzania, Nigeria, etc. → "Afrotropical"
  * Australia, New Zealand, Papua New Guinea → "Australasian"
- If Country is empty but region can be inferred from geographic context, assign region.
- Use "Global" ONLY for meta-analyses, global syntheses, or multi-continent studies.
- Use "Not Specified" ONLY if neither country nor region can be determined.

Trichoptera_Relevance:
- "Primary focus": Trichoptera are the main study organism.
- "Secondary mention": Trichoptera are studied alongside other taxa.
- "Peripheral": Trichoptera mentioned but not central to the study.
- "Not Trichoptera-focused": Paper does not focus on Trichoptera.

OUTPUT FORMAT:
- One JSON object
- One value per field
"""

    # Retry logic for rate limits
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "You are a careful bibliometric classifier."},
                    {"role": "user", "content": prompt}
                ]
            )

            raw = response.choices[0].message.content.strip()
            parsed = safe_json_loads(raw)

            if parsed is not None:
                return parsed

            # Safe fallback (single-pass philosophy)
            return {col: "Not Specified" for col in llm_schema.keys()}
            
        except Exception as e:
            if attempt < max_retries - 1:
                # Exponential backoff for rate limits
                wait_time = (2 ** attempt) * 0.5
                time.sleep(wait_time)
                continue
            else:
                # Final attempt failed, return fallback
                print(f"Warning: Failed to classify after {max_retries} attempts: {e}")
                return {col: "Not Specified" for col in llm_schema.keys()}
    
    # Should not reach here, but just in case
    return {col: "Not Specified" for col in llm_schema.keys()}


# -------------------------
# PROCESS SINGLE PAPER (for threading)
# -------------------------
def process_paper(row_data):
    """Process a single paper - thread-safe"""
    idx, row = row_data
    
    title = row.get("Title", "")
    abstract = row.get("Abstract", "")
    abstract_available = isinstance(abstract, str) and abstract.strip() != ""
    author_affiliations = row.get("Author_Affiliations", None)

    llm_output = classify(
        title,
        abstract if abstract_available else "",
        author_affiliations=author_affiliations
    )

    new_row = row.to_dict()
    new_row.update(llm_output)
    new_row["abstract_available"] = abstract_available

    return idx, new_row

# -------------------------
# RUN (PILOT FIRST)
# -------------------------

# Test with 5 records first
TEST_MODE = False
TEST_SIZE = 20  # Test with 20 papers to verify multi-threading

if TEST_MODE:
    # Test mode with multi-threading
    print(f"TEST MODE: Processing only {TEST_SIZE} records with {NUM_THREADS} threads")
    df = df.head(TEST_SIZE)
    rows_to_process = [(idx, row) for idx, row in df.iterrows()]
    coded_dict = {}
    
    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        future_to_idx = {executor.submit(process_paper, row_data): row_data[0] 
                         for row_data in rows_to_process}
        
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
    
    # Save test output
    coded_test = [coded_dict[i] for i in sorted(coded_dict.keys())]
    test_output = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_coded_TEST.csv"
    pd.DataFrame(coded_test).to_csv(test_output, index=False)
    print(f"\n✓ Test complete! Saved {len(coded_test)} records to: {test_output}")
    print(f"\nSample output:")
    for i, row in enumerate(coded_test[:3], 1):
        print(f"\n  Paper {i}:")
        print(f"    Title: {row.get('Title', '')[:60]}...")
        print(f"    Country: {row.get('Country', 'N/A')}")
        print(f"    Region_Global: {row.get('Region_Global', 'N/A')}")
        print(f"    Research_Theme: {row.get('Research_Theme', 'N/A')}")
        print(f"    Trichoptera_Relevance: {row.get('Trichoptera_Relevance', 'N/A')}")
else:
    # Full run with multi-threading
    print(f"\nStarting LLM coding with {NUM_THREADS} threads...")
    print(f"Total papers: {len(df):,}")
    print(f"Progress will be saved every {SAVE_INTERVAL} papers\n")
    
    # Check if we can resume from existing file
    coded_dict = {}
    start_index = 0
    if OUTPUT_CSV.exists():
        try:
            df_existing = pd.read_csv(OUTPUT_CSV)
            # Check which papers are already coded (have Country or Research_Theme)
            if 'Country' in df_existing.columns or 'Research_Theme' in df_existing.columns:
                coded_titles = set(df_existing['Title'].astype(str))
                print(f"Found existing coded file with {len(df_existing)} papers")
                print(f"Resuming from existing progress...")
                # Only process papers not already coded
                df = df[~df['Title'].astype(str).isin(coded_titles)]
                coded_dict = {i: row.to_dict() for i, row in df_existing.iterrows()}
                start_index = len(df_existing)
        except Exception as e:
            print(f"Warning: Could not read existing file: {e}")
            print("Starting fresh...")
    
    # Prepare data for threading
    rows_to_process = [(idx, row) for idx, row in df.iterrows()]
    
    # Process with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=NUM_THREADS) as executor:
        # Submit all tasks
        future_to_idx = {executor.submit(process_paper, row_data): row_data[0] 
                         for row_data in rows_to_process}
        
        # Process completed tasks with progress bar
        with tqdm(total=len(rows_to_process), initial=start_index, desc="Coding papers") as pbar:
            completed_count = 0
            for future in as_completed(future_to_idx):
                try:
                    idx, new_row = future.result()
                    coded_dict[idx] = new_row
                    completed_count += 1
                    pbar.update(1)
                    
                    # Save progress periodically (thread-safe)
                    if completed_count % SAVE_INTERVAL == 0:
                        with save_lock:
                            # Convert dict to DataFrame and save
                            coded_list = [coded_dict[i] for i in sorted(coded_dict.keys())]
                            temp_df = pd.DataFrame(coded_list)
                            temp_df.to_csv(OUTPUT_CSV, index=False)
                            pbar.set_postfix({"Saved": f"{completed_count}/{len(rows_to_process)}"})
                            
                except Exception as e:
                    idx = future_to_idx[future]
                    print(f"\nError processing paper {idx}: {e}")
                    # Add error row with fallback values
                    row = df.loc[idx]
                    error_row = row.to_dict()
                    error_row.update({col: "Not Specified" for col in llm_schema.keys()})
                    error_row["abstract_available"] = pd.notna(row.get("Abstract"))
                    coded_dict[idx] = error_row
                    pbar.update(1)
    
    # Final save
    coded_list = [coded_dict[i] for i in sorted(coded_dict.keys())]
    final_df = pd.DataFrame(coded_list)
    final_df.to_csv(OUTPUT_CSV, index=False)
    
    print(f"\n✓ Complete! Saved {len(coded_list):,} records to: {OUTPUT_CSV}")
    
    # Summary statistics
    if 'Country' in final_df.columns:
        country_filled = final_df['Country'].notna() & (final_df['Country'] != '') & (final_df['Country'] != 'Not Specified')
        print(f"  Papers with Country: {country_filled.sum():,} ({country_filled.sum()/len(final_df)*100:.1f}%)")
    if 'Region_Global' in final_df.columns:
        region_filled = final_df['Region_Global'] != 'Not Specified'
        print(f"  Papers with Region: {region_filled.sum():,} ({region_filled.sum()/len(final_df)*100:.1f}%)")
