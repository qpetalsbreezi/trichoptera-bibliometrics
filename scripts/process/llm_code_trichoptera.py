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


INPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_with_abstracts.csv"
SCHEMA_FILE = PROJECT_ROOT / "data/trichoptera_schema.json"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_coded.csv"

MODEL = "gpt-4o-mini"
TEMPERATURE = 0

# -------------------------
# LOAD FILES
# -------------------------
df = pd.read_csv(INPUT_CSV)

with open(SCHEMA_FILE) as f:
    schema = json.load(f)

# Build LLM coding schema
llm_schema = {}
for col, spec in schema["columns"].items():
    if "allowed_values" in spec:
        llm_schema[col] = spec["allowed_values"]
    elif col in [
        "Taxonomic_Focus",
        "Species_Studied",
        "Country",
        "Notes",
        "Relevance_to_Hypotheses"
    ]:
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

Method_Type:
- Use "Review" ONLY if the paper synthesizes literature without new data or analyses.
- Otherwise prefer Field Study, Modeling, Genetic/Molecular Study, or Other.

Family_Studied:
- Assign a family ONLY if the paper focuses on that family.
- If multiple families, higher taxa, or mixed orders are involved, use "Multiple".

FFG:
- Assign ONLY if feeding or functional role is explicitly discussed.

Habitat_Type:
- Prefer Lotic / Stream / River ONLY if freshwater flow is explicitly implied.

Region_Global:
- Use "Global" ONLY for meta-analyses or global syntheses.
- Otherwise choose the most specific region stated.

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

# subset = df.head(10)
# for _, row in tqdm(subset.iterrows(), total=len(subset)):
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
print("Saved:", OUTPUT_CSV)
