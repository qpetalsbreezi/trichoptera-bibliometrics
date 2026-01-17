import pandas as pd
import json
import time
from tqdm import tqdm
from openai import OpenAI
import os
from pathlib import Path

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
SCHEMA_FILE = PROJECT_ROOT / "data/trichoptera_schema.json"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_coded.csv"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0

# -------------------------
# LOAD FILES
# -------------------------
df = pd.read_csv(INPUT_CSV)

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

# -------------------------
# CLASSIFIER
# -------------------------
def classify(title, abstract):
    prompt = f"""
You are coding academic papers for a bibliometric study on Trichoptera (caddisflies).

Your task is to assign values ONLY for the fields listed below, using the predefined schema.

Schema (allowed values):
{llm_schema_text}

Paper title:
{title}

Paper abstract:
{abstract}

CORE RULES (follow strictly):
- Do NOT assume Trichoptera are the main focus unless clearly stated.
- Prefer the MOST SPECIFIC allowed value supported by the text.
- Use "Other" ONLY if no allowed value applies.
- If information is missing, use "Not Specified".
- Do NOT invent taxa, locations, methods, or conclusions.
- Output VALID JSON only. No explanations.

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

Region_Global:
- Use "Global" ONLY for meta-analyses or global syntheses.
- Otherwise choose the most specific region stated (Oriental, Neotropical, Nearctic, Palearctic, East Palearctic, Afrotropical, Australasian).
- Use "Not Specified" if region cannot be determined.

Country:
- Extract the primary country where research was conducted.
- Can be inferred from study location, author affiliations, or geographic context.
- Use standard country names (e.g., "United States" not "USA").
- Leave empty if country cannot be determined.

Trichoptera_Relevance:
- "Primary focus": Trichoptera are the main study organism.
- "Secondary mention": Trichoptera are studied alongside other taxa.
- "Peripheral": Trichoptera mentioned but not central to the study.
- "Not Trichoptera-focused": Paper does not focus on Trichoptera.

OUTPUT FORMAT:
- One JSON object
- One value per field
"""

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


# -------------------------
# RUN (PILOT FIRST)
# -------------------------
coded = []

# Test with 5 records first
TEST_MODE = False
TEST_SIZE = 5

if TEST_MODE:
    print(f"TEST MODE: Processing only {TEST_SIZE} records")
    subset = df.head(TEST_SIZE)
    for _, row in tqdm(subset.iterrows(), total=len(subset)):
        title = row.get("Title", "")
        abstract = row.get("Abstract", "")
        abstract_available = isinstance(abstract, str) and abstract.strip() != ""

        llm_output = classify(title, abstract if abstract_available else "")

        new_row = row.to_dict()
        new_row.update(llm_output)
        new_row["abstract_available"] = abstract_available

        coded.append(new_row)
        time.sleep(0.5)
    
    # Save test output
    test_output = PROJECT_ROOT / "data/processed/trichoptera_scopus_api_coded_TEST.csv"
    pd.DataFrame(coded).to_csv(test_output, index=False)
    print(f"\n✓ Test complete! Saved {len(coded)} records to: {test_output}")
    print(f"\nSample output:")
    for i, row in enumerate(coded[:3], 1):
        print(f"\n  Paper {i}:")
        print(f"    Title: {row.get('Title', '')[:60]}...")
        print(f"    Country: {row.get('Country', 'N/A')}")
        print(f"    Region_Global: {row.get('Region_Global', 'N/A')}")
        print(f"    Research_Theme: {row.get('Research_Theme', 'N/A')}")
        print(f"    Trichoptera_Relevance: {row.get('Trichoptera_Relevance', 'N/A')}")
else:
    # Full run
    for _, row in tqdm(df.iterrows(), total=len(df)):
        title = row.get("Title", "")
        abstract = row.get("Abstract", "")
        abstract_available = isinstance(abstract, str) and abstract.strip() != ""

        llm_output = classify(title, abstract if abstract_available else "")

        new_row = row.to_dict()
        new_row.update(llm_output)
        new_row["abstract_available"] = abstract_available

        coded.append(new_row)
        time.sleep(0.5)
    
    pd.DataFrame(coded).to_csv(OUTPUT_CSV, index=False)
    print(f"\n✓ Complete! Saved {len(coded)} records to: {OUTPUT_CSV}")
