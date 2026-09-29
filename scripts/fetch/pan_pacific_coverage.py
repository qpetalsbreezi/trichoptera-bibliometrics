"""Pan-Pacific Entomologist coverage check: Scopus records + abstract backfill."""
import os
import sys
from collections import Counter
from pathlib import Path

import pandas as pd
import requests
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "fetch"))
sys.path.insert(0, str(ROOT / "scripts"))
from fetch_abstracts import _fetch_abstract_waterfall  # noqa: E402

OUT = ROOT / "analysis" / "combined" / "pan_pacific_scopus.csv"
QUERY = 'SRCTITLE("Pan-Pacific Entomologist")'


def fetch_scopus():
    h = {"X-ELS-APIKey": os.environ["SCOPUS_API_KEY"], "Accept": "application/json"}
    recs, start = [], 0
    while True:
        r = requests.get(
            "https://api.elsevier.com/content/search/scopus",
            headers=h,
            params={"query": QUERY, "count": 25, "start": start},
            timeout=60,
        )
        r.raise_for_status()
        res = r.json()["search-results"]
        ent = res.get("entry", [])
        if not ent or "error" in ent[0]:
            break
        for e in ent:
            recs.append(
                dict(
                    EID=e.get("eid"),
                    DOI=e.get("prism:doi"),
                    Year=int((e.get("prism:coverDate") or "0")[:4]),
                    Type=e.get("subtypeDescription"),
                    Title=e.get("dc:title"),
                )
            )
        start += len(ent)
        if start >= int(res["opensearch:totalResults"]):
            break
    return pd.DataFrame(recs)


def main():
    if OUT.exists():
        df = pd.read_csv(OUT)
    else:
        df = fetch_scopus()
        df["Abstract"] = None
        df["abstract_source"] = None
    stats = Counter()
    stats["status_counts"] = {}
    session = requests.Session()
    todo = df.index[df["Abstract"].isna()]
    for i, idx in enumerate(tqdm(todo, file=sys.stderr)):
        doi = df.at[idx, "DOI"] if pd.notna(df.at[idx, "DOI"]) else None
        text, source, _, _ = _fetch_abstract_waterfall(
            doi, df.at[idx, "Title"], session, stats, scopus_id=df.at[idx, "EID"]
        )
        if text:
            df.at[idx, "Abstract"] = text
            df.at[idx, "abstract_source"] = source
        if i % 50 == 0:
            df.to_csv(OUT, index=False)
    df.to_csv(OUT, index=False)
    print(Counter(df["abstract_source"].fillna("none")))


if __name__ == "__main__":
    main()
