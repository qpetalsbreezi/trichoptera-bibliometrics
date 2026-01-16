import pandas as pd
import requests
from tqdm import tqdm
import time
import os
from pathlib import Path
import json

# Get project root directory (two levels up from this script)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# Configuration
INPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_raw_2010_2025.csv"
OUTPUT_CSV = PROJECT_ROOT / "data/processed/trichoptera_scopus_with_abstracts.csv"
SAVE_INTERVAL = 50  # Save every N papers

# Statistics tracking
stats = {
    "openalex": 0,
    "semantic": 0,
    "crossref": 0,
    "pubmed": 0,
    "already_had": 0,
    "failed": 0
}

# Load your Scopus CSV
if OUTPUT_CSV.exists():
    print(f"Resuming from existing file: {OUTPUT_CSV}")
    df = pd.read_csv(OUTPUT_CSV)
    # Count already existing abstracts
    stats["already_had"] = df["Abstract"].notna().sum() if "Abstract" in df.columns else 0
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
                abstract = r.json().get("abstract")
                if abstract:
                    return abstract
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))  # Exponential backoff
                continue
            return None
    return None


# ---------- CrossRef API (Free, no key required) ----------
def get_abstract_crossref(doi, max_retries=3):
    """Fetch abstract from CrossRef API - free, no API key needed"""
    if pd.isna(doi) or not doi:
        return None
    
    # Clean DOI (remove https://doi.org/ prefix if present)
    clean_doi = str(doi).replace("https://doi.org/", "").replace("http://dx.doi.org/", "").strip()
    
    url = f"https://api.crossref.org/works/{clean_doi}"
    headers = {
        "User-Agent": "Trichoptera Bibliometric Study (mailto:your-email@example.com)"  # Polite to include contact
    }
    
    for attempt in range(max_retries):
        try:
            r = requests.get(url, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if "message" in data:
                    # CrossRef returns abstract in message.abstract array or message.abstract-text
                    abstract_text = data["message"].get("abstract")
                    if abstract_text:
                        # Abstract can be HTML or plain text
                        if isinstance(abstract_text, str):
                            return abstract_text
                        elif isinstance(abstract_text, list):
                            # Sometimes it's an array of text chunks
                            return " ".join(str(t) for t in abstract_text)
            elif r.status_code == 404:
                return None  # DOI not found
        except (requests.exceptions.Timeout, requests.exceptions.RequestException) as e:
            if attempt < max_retries - 1:
                time.sleep(1 * (attempt + 1))
                continue
            return None
    return None


# ---------- PubMed API (for biomedical papers) ----------
def get_abstract_pubmed(doi, max_retries=3):
    """Fetch abstract from PubMed API using DOI"""
    if pd.isna(doi) or not doi:
        return None
    
    # Clean DOI
    clean_doi = str(doi).replace("https://doi.org/", "").replace("http://dx.doi.org/", "").strip()
    
    # First, search for the paper by DOI
    search_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    search_params = {
        "db": "pubmed",
        "term": f"{clean_doi}[DOI]",
        "retmode": "json"
    }
    
    try:
        r = requests.get(search_url, params=search_params, timeout=15)
        if r.status_code == 200:
            data = r.json()
            pmids = data.get("esearchresult", {}).get("idlist", [])
            
            if not pmids:
                return None  # Not found in PubMed
            
            # Fetch abstract using PMID
            fetch_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
            fetch_params = {
                "db": "pubmed",
                "id": ",".join(pmids[:1]),  # Just get first match
                "retmode": "xml",
                "rettype": "abstract"
            }
            
            r2 = requests.get(fetch_url, params=fetch_params, timeout=15)
            if r2.status_code == 200:
                # Parse XML to extract abstract
                import xml.etree.ElementTree as ET
                try:
                    root = ET.fromstring(r2.text)
                    # PubMed XML structure: PubmedArticle -> MedlineCitation -> Article -> Abstract -> AbstractText
                    for abstract_text in root.iter("{http://www.ncbi.nlm.nih.gov}AbstractText"):
                        if abstract_text.text:
                            return abstract_text.text
                except ET.ParseError:
                    return None
    except (requests.exceptions.Timeout, requests.exceptions.RequestException):
        return None
    
    return None


# ---------- Main enrichment loop ----------
processed = 0
total = len(df)
needs_abstract = total - stats["already_had"]

print(f"\nStarting abstract fetching...")
print(f"Total papers: {total}")
print(f"Already have abstracts: {stats['already_had']}")
print(f"Need abstracts: {needs_abstract}")
print(f"\nTrying sources in order: OpenAlex → Semantic Scholar → CrossRef → PubMed\n")

for idx, row in tqdm(df.iterrows(), total=total, desc="Fetching abstracts"):
    # Skip if already has abstract
    if isinstance(row.get("Abstract"), str) and row["Abstract"].strip():
        continue  # already filled

    doi = row.get("DOI")
    abstract = None
    source = None

    # Try sources in order of reliability/speed
    # 1. OpenAlex (usually fastest, good coverage)
    abstract = get_abstract_openalex(doi)
    if abstract:
        source = "openalex"
        stats["openalex"] += 1
    else:
        # 2. Semantic Scholar (good for CS/ML papers)
        abstract = get_abstract_semantic(doi)
        if abstract:
            source = "semantic"
            stats["semantic"] += 1
        else:
            # 3. CrossRef (free, good coverage)
            abstract = get_abstract_crossref(doi)
            if abstract:
                source = "crossref"
                stats["crossref"] += 1
            else:
                # 4. PubMed (for biomedical papers)
                abstract = get_abstract_pubmed(doi)
                if abstract:
                    source = "pubmed"
                    stats["pubmed"] += 1
                else:
                    stats["failed"] += 1

    if abstract:
        # Ensure Abstract column is string type to avoid dtype warnings
        if df["Abstract"].dtype != "object":
            df["Abstract"] = df["Abstract"].astype(str)
        df.at[idx, "Abstract"] = abstract

    processed += 1
    
    # Save periodically
    if processed % SAVE_INTERVAL == 0:
        df.to_csv(OUTPUT_CSV, index=False)
        current_found = stats["openalex"] + stats["semantic"] + stats["crossref"] + stats["pubmed"]
        tqdm.write(f"Progress: {processed}/{total} | Found: {current_found} | "
                  f"OpenAlex: {stats['openalex']}, Semantic: {stats['semantic']}, "
                  f"CrossRef: {stats['crossref']}, PubMed: {stats['pubmed']}, "
                  f"Failed: {stats['failed']}")

    time.sleep(0.2)  # be polite to APIs

# Final save
print(f"\nSaving final results to {OUTPUT_CSV}...")
df.to_csv(OUTPUT_CSV, index=False)

# Print summary statistics
print(f"\n{'='*70}")
print("ABSTRACT FETCHING SUMMARY")
print(f"{'='*70}")
print(f"Total papers processed: {total}")
print(f"Already had abstracts: {stats['already_had']}")
print(f"\nAbstracts fetched this run:")
print(f"  OpenAlex:     {stats['openalex']}")
print(f"  Semantic Scholar: {stats['semantic']}")
print(f"  CrossRef:     {stats['crossref']}")
print(f"  PubMed:       {stats['pubmed']}")
print(f"  Failed:       {stats['failed']}")

total_fetched = stats["openalex"] + stats["semantic"] + stats["crossref"] + stats["pubmed"]
total_with_abstract = stats["already_had"] + total_fetched
coverage = (total_with_abstract / total * 100) if total > 0 else 0

print(f"\nFinal coverage: {total_with_abstract}/{total} ({coverage:.1f}%)")
print(f"{'='*70}")
print(f"✓ Complete! Results saved to {OUTPUT_CSV}")
