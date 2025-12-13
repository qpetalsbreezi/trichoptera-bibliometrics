import pandas as pd
import requests
from tqdm import tqdm
import time

# Load your Scopus CSV
df = pd.read_csv("data/trichoptera_scopus_raw.csv")

# Ensure Abstract column exists
if "Abstract" not in df.columns:
    df["Abstract"] = ""

# ---------- OpenAlex ----------
def get_abstract_openalex(doi):
    if pd.isna(doi):
        return None

    url = f"https://api.openalex.org/works/https://doi.org/{doi}"
    r = requests.get(url, timeout=10)

    if r.status_code != 200:
        return None

    data = r.json()
    inverted = data.get("abstract_inverted_index")

    if not inverted:
        return None

    words = {}
    for word, positions in inverted.items():
        for pos in positions:
            words[pos] = word

    return " ".join(words[i] for i in sorted(words))


# ---------- Semantic Scholar ----------
def get_abstract_semantic(doi):
    if pd.isna(doi):
        return None

    url = f"https://api.semanticscholar.org/graph/v1/paper/DOI:{doi}"
    params = {"fields": "abstract"}
    r = requests.get(url, params=params, timeout=10)

    if r.status_code != 200:
        return None

    return r.json().get("abstract")


# ---------- Main enrichment loop ----------
for idx, row in tqdm(df.iterrows(), total=len(df)):
    if isinstance(row.get("Abstract"), str) and row["Abstract"].strip():
        continue  # already filled

    doi = row.get("DOI")

    abstract = get_abstract_openalex(doi)

    if not abstract:
        abstract = get_abstract_semantic(doi)

    if abstract:
        df.at[idx, "Abstract"] = abstract

    time.sleep(0.2)  # be polite to APIs

# Save enriched file
df.to_csv("data/trichoptera_scopus_with_abstracts.csv", index=False)
