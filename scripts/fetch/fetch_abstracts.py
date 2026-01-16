import pandas as pd
import requests
from tqdm import tqdm
import time
import os
from pathlib import Path

# Get project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
INPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_raw_2010_2025.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_with_abstracts.csv"
SAVE_INTERVAL = 50  # Save every N papers

# Load your Scopus CSV
if OUTPUT_CSV.exists():
    print(f"Resuming from existing file: {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV)
else:
    print(f"Starting fresh from: {INPUT_CSV}")
    df = pd.read_csv(INPUT_CSV)

# Ensure Abstract column exists
if "Abstract" not in df.columns:
    df["Abstract"] = ""

# ---------- OpenAlex ----------
def get_abstract_openalex(doi, max_retries=3):
    if pd.isna(doi) or not doi:
        return None

    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                data = r.json()
                inverted = data.get("abstract_inverted_index")
                if not inverted:
                    return None
                words = {}
                for word, positions in inverted.items():
                    for pos in positions:
                        words[pos] = word
                return " ".join(words[i] for i in sorted(words))
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            return None
    return None


# ---------- Semantic Scholar ----------
def get_abstract_semantic(doi, max_retries=3):
    if pd.isna(doi) or not doi:
        return None

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "abstract"}
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json().get("abstract")
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            return None
    return None


# ---------- Main enrichment loop ----------
processed = 0
total = len(df)

for idx, row in tqdm(df.iterrows(), total=total, desc="Fetching abstracts"):
    if isinstance(row.get("Abstract"), str) and row["Abstract"].strip():
        continue  # already filled

    doi = row.get("DOI")

    abstract = get_abstract_openalex(doi)

    if not abstract:
        abstract = get_abstract_semantic(doi)

    if abstract:
        df.at[idx, "Abstract"] = abstract

    processed += 1
    
    # Save periodically
    if processed % SAVE_INTERVAL == 0:
        df.to_csv(OUTPUT_CSV, index=False)
        tqdm.write(f"Progress saved: {processed}/{total} papers processed")

    time.sleep(0.2)  # be polite to APIs

# Final save
print(f"\nSaving final results to {OUTPUT_CSV}...")
df.to_csv(OUTPUT_CSV, index=False)
print(f"✓ Complete! Processed {total} papers")
