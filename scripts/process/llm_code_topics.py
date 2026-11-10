"""Second-pass topic coding for group-focused articles (biotic + abiotic).

Uses gpt-4o-mini, temperature 0, title + abstract only.
Writes 0/1 columns per topic; failed/invalid rows are treated as all zeros.
"""

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
from openai import OpenAI
from tqdm import tqdm

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.pipeline import (  # noqa: E402
    PAPER_QUERY_ORDER,
    PAPER_TAXON_LABELS,
    PipelinePaths,
    add_query_arg,
    filter_article_review,
    get_query_config,
    load_dotenv,
)

DEFAULT_MODEL = "gpt-4o-mini"
TEMPERATURE = 0
NUM_THREADS = 8
SAVE_INTERVAL = 50
FOCUS_KEEP = {"Primary focus", "Secondary mention"}
RELEVANCE_FIELD = "Taxon_Relevance"

BIOTIC_TOPICS = [
    "population_size",
    "community_structure",
    "species_richness",
    "diversity",
    "predation_by_fish",
    "predation_by_invertebrates",
    "grazers_scrapers",
    "collectors",
    "herbivores",
    "shredders",
    "filter_feeders",
    "deposit_feeders",
    "detritus",
    "competition",
    "parasites_of_the_insect",
    "disease_of_the_insect",
    "secondary_production",
]

ABIOTIC_TOPICS = [
    "lake",
    "stream",
    "river",
    "wetland",
    "springs",
    "hyporheic",
    "temperature",
    "flow_rate",
    "current",
    "dissolved_oxygen",
    "ph",
    "substrate_type",
    "nutrients",
    "water_chemistry",
    "pollution",
    "climate_change",
]

ALL_TOPICS = BIOTIC_TOPICS + ABIOTIC_TOPICS

TOPIC_LABELS = {
    "population_size": "Population size",
    "community_structure": "Community structure",
    "species_richness": "Species richness",
    "diversity": "Diversity",
    "predation_by_fish": "Predation by fish",
    "predation_by_invertebrates": "Predation by invertebrates",
    "grazers_scrapers": "Grazers/scrapers",
    "collectors": "Collectors",
    "herbivores": "Herbivores",
    "shredders": "Shredders",
    "filter_feeders": "Filter feeders",
    "deposit_feeders": "Deposit feeders",
    "detritus": "Detritus",
    "competition": "Competition",
    "parasites_of_the_insect": "Parasites",
    "disease_of_the_insect": "Disease",
    "secondary_production": "Secondary production",
    "lake": "Lake",
    "stream": "Stream",
    "river": "River",
    "wetland": "Wetland",
    "springs": "Springs",
    "hyporheic": "Hyporheic",
    "temperature": "Temperature",
    "flow_rate": "Flow rate",
    "current": "Current",
    "dissolved_oxygen": "Dissolved oxygen",
    "ph": "pH",
    "substrate_type": "Substrate type",
    "nutrients": "Nutrients",
    "water_chemistry": "Water chemistry",
    "pollution": "Pollution",
    "climate_change": "Climate change",
}

# Expected Group-focused N (Article+Review, Primary/Secondary) for reconciliation.
EXPECTED_N = {
    "mosquitoes": 34717,
    "ephemeroptera": 3098,
    "plecoptera": 2032,
    "trichoptera": 3099,
    "odonata": 4878,
}

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


def parse_retry_after_seconds(exc: Exception):
    import re

    msg = str(exc)
    m = re.search(r"try again in\s+([0-9.]+)\s*s", msg, re.I)
    if m:
        return float(m.group(1))
    m = re.search(r"try again in\s+([0-9.]+)\s*m", msg, re.I)
    if m:
        return float(m.group(1)) * 60.0
    m = re.search(r"try again in\s+([0-9.]+)\s*h", msg, re.I)
    if m:
        return float(m.group(1)) * 3600.0
    return None


def safe_json_loads(text):
    if text is None:
        return None
    s = str(text).strip()
    if not s:
        return None
    if s.startswith("```"):
        s = s[3:].lstrip()
        if s.lower().startswith("json"):
            s = s[4:].lstrip()
        if s.endswith("```"):
            s = s[:-3].rstrip()
    try:
        return json.loads(s)
    except Exception:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(s[start : end + 1])
            except Exception:
                return None
        return None


def zeros_topics() -> dict:
    return {t: 0 for t in ALL_TOPICS}


def normalize_topic_output(parsed: dict) -> tuple[dict, str]:
    """Return (topics, status). status is 'ok' or 'invalid'."""
    if not isinstance(parsed, dict):
        return zeros_topics(), "invalid"
    out = {}
    for t in ALL_TOPICS:
        if t not in parsed:
            return zeros_topics(), "invalid"
        v = parsed[t]
        if v in (0, 1):
            out[t] = int(v)
        elif v in ("0", "1"):
            out[t] = int(v)
        else:
            return zeros_topics(), "invalid"
    return out, "ok"


# Inline clarifications shown in the prompt next to selected topic keys.
TOPIC_PROMPT_NOTES = {
    "species_richness": (
        "species_richness (number of species counted or compared at a site or in a community)"
    ),
    "diversity": "diversity (species or community diversity, not genetic diversity)",
    "predation_by_invertebrates": (
        "predation_by_invertebrates (including these insects as predators)"
    ),
    "secondary_production": (
        "secondary_production (biomass production of the insects over time, e.g., annual production estimates)"
    ),
    "lake": "lake (including ponds and reservoirs)",
    "flow_rate": "flow_rate (including discharge, floods, and droughts)",
    "current": "current (water velocity)",
    "pollution": (
        "pollution (contamination of water or habitat, not insecticides applied for mosquito control)"
    ),
}


def _topic_list_text() -> tuple[str, str]:
    """Prompt topic lines, with inline clarifications where defined."""
    biotic = "\n".join(TOPIC_PROMPT_NOTES.get(t, t) for t in BIOTIC_TOPICS)
    abiotic = "\n".join(TOPIC_PROMPT_NOTES.get(t, t) for t in ABIOTIC_TOPICS)
    return biotic, abiotic


def build_topic_prompt(group: str, title, abstract) -> str:
    title_s = "" if title is None or (isinstance(title, float) and pd.isna(title)) else str(title)
    abs_s = "" if abstract is None or (isinstance(abstract, float) and pd.isna(abstract)) else str(abstract)
    biotic, abiotic = _topic_list_text()
    return f"""You are classifying scientific articles about {group} for a bibliometric study.

Read the title and abstract. For each topic below, answer 1 if the article studies it and 0 if not.

Rules:
- Answer 1 only if the topic is measured, tested, or a main subject. A passing mention gets 0.
- An article can have any number of 1s, including none.
- Use only what the title and abstract say. If there is no abstract, use the title.
- Habitat topics (lake through hyporheic) mean where the study was done.
- Describing new species, new records, or taxonomic revisions does not count as species_richness or diversity.
- disease_of_the_insect and parasites_of_the_insect mean pathogens or parasites that infect the insects themselves. Diseases and pathogens the insects transmit to humans or animals (malaria, dengue, Zika, West Nile, filariasis, Plasmodium, filarial worms) are always 0 for these two topics. This includes studies of these pathogens developing inside the insect, vector competence, and detecting them in insects.

Biotic topics:
{biotic}

Abiotic topics:
{abiotic}

Title: {title_s}

Abstract: {abs_s}

Return only JSON with every topic as a key and 0 or 1 as the value."""


def classify_topics(
    client: OpenAI,
    model: str,
    group: str,
    title,
    abstract,
    max_retries: int = 8,
) -> tuple[dict, str]:
    prompt = build_topic_prompt(group, title, abstract)
    attempt = 0
    while True:
        try:
            response = client.chat.completions.create(
                model=model,
                temperature=TEMPERATURE,
                messages=[
                    {"role": "system", "content": "You are a careful bibliometric classifier. Return only valid JSON."},
                    {"role": "user", "content": prompt},
                ],
            )
            raw = response.choices[0].message.content.strip()
            parsed = safe_json_loads(raw)
            topics, status = normalize_topic_output(parsed)
            if status == "ok":
                return topics, "ok"
            if attempt < max_retries - 1:
                attempt += 1
                time.sleep(0.5)
                continue
            return zeros_topics(), "invalid"
        except Exception as e:
            if is_daily_rate_limit_error(e):
                wait = parse_retry_after_seconds(e)
                if wait is not None and wait <= 120:
                    time.sleep(wait + 1.0)
                    continue
                raise DailyRateLimitExceeded(str(e)) from e
            if attempt < max_retries - 1:
                time.sleep((2**attempt) * 0.5)
                attempt += 1
                continue
            return zeros_topics(), "failed"


def load_focused_frame(query_id: str) -> pd.DataFrame:
    paths = PipelinePaths(query_id)
    coded = paths.coded
    if not coded.is_file():
        raise SystemExit(f"Missing coded CSV: {coded}")
    df = pd.read_csv(coded, low_memory=False)
    df = filter_article_review(df)
    if RELEVANCE_FIELD not in df.columns:
        raise SystemExit(f"Missing {RELEVANCE_FIELD} in {coded}")
    df = df[df[RELEVANCE_FIELD].isin(FOCUS_KEEP)].copy()
    df = df.reset_index(drop=True)
    return df


def out_dir() -> Path:
    d = Path(__file__).resolve().parents[2] / "analysis" / "combined" / "topic_coding"
    d.mkdir(parents=True, exist_ok=True)
    return d


def run_coding(
    query_id: str,
    limit: int | None,
    num_threads: int,
    save_interval: int,
    seed: int,
) -> Path:
    load_dotenv()
    api_key = os.getenv("OPENAI_API_KEY")
    model = os.getenv("OPENAI_MODEL", DEFAULT_MODEL)
    if not api_key:
        raise SystemExit("OPENAI_API_KEY not set.")

    qconf = get_query_config(query_id)
    group = qconf["llm"]["study_descriptor"]
    paper_label = PAPER_TAXON_LABELS[query_id]

    df = load_focused_frame(query_id)
    expected = EXPECTED_N.get(query_id)
    if expected is not None and limit is None and len(df) != expected:
        print(
            f"WARNING: {paper_label} Group-focused N={len(df):,} "
            f"does not match expected {expected:,}"
        )
    else:
        print(f"{paper_label} Group-focused N={len(df):,} (expected {expected})")

    if limit is not None:
        df = df.sample(n=min(limit, len(df)), random_state=seed).reset_index(drop=True)
        print(f"Pilot sample: {len(df)} rows (seed={seed})")

    stem = f"{query_id}_topics"
    if limit is not None:
        stem = f"{query_id}_pilot_{limit}"
    out_path = out_dir() / f"{stem}.csv"

    # Keep text columns only in memory for API calls; checkpoints omit them (too large).
    text_cols = [c for c in ["Title", "Abstract"] if c in df.columns]
    meta_cols = [c for c in ["EID", "DOI", "Year", "Taxon_Relevance", "Type"] if c in df.columns]
    work = df[meta_cols + text_cols].copy()
    work["EID"] = work["EID"].astype(str)
    for t in ALL_TOPICS:
        work[t] = 0
    work["topic_status"] = "pending"
    work["insect_group"] = paper_label
    work["query_id"] = query_id

    save_cols = meta_cols + ALL_TOPICS + ["topic_status", "insect_group", "query_id"]

    def save_checkpoint() -> None:
        tmp = out_path.with_suffix(".csv.tmp")
        work.loc[:, save_cols].to_csv(tmp, index=False)
        tmp.replace(out_path)

    # Resume: vectorized merge of already-ok rows
    if out_path.exists():
        prev = pd.read_csv(out_path, low_memory=False)
        if "topic_status" in prev.columns and "EID" in prev.columns:
            prev_ok = prev[prev["topic_status"] == "ok"].copy()
            prev_ok["EID"] = prev_ok["EID"].astype(str)
            topic_cols = [t for t in ALL_TOPICS if t in prev_ok.columns]
            if len(prev_ok) and topic_cols:
                print(f"Resuming: {len(prev_ok)} already-ok rows in {out_path.name}", flush=True)
                work = work.set_index("EID")
                prev_ok = prev_ok.drop_duplicates("EID").set_index("EID")
                shared = work.index.intersection(prev_ok.index)
                work.loc[shared, topic_cols] = prev_ok.loc[shared, topic_cols]
                work.loc[shared, "topic_status"] = "ok"
                work = work.reset_index()

    client = OpenAI(api_key=api_key, timeout=120.0)
    todo_idx = work.index[work["topic_status"] != "ok"].tolist()
    print(f"To classify: {len(todo_idx)} / {len(work)}", flush=True)
    if not todo_idx:
        save_checkpoint()
        return out_path

    # Snapshot texts only for rows still to classify; drop text from `work` to cut memory.
    titles = {}
    abstracts = {}
    if "Title" in work.columns:
        titles = work.loc[todo_idx, "Title"].to_dict()
    if "Abstract" in work.columns:
        abstracts = work.loc[todo_idx, "Abstract"].to_dict()
    work.drop(columns=[c for c in ("Title", "Abstract") if c in work.columns], inplace=True)

    def process_one(i: int):
        return i, *classify_topics(
            client, model, group, titles.get(i), abstracts.get(i)
        )

    completed = 0
    try:
        with ThreadPoolExecutor(max_workers=num_threads) as ex:
            with tqdm(total=len(todo_idx), desc=f"topics:{query_id}") as pbar:
                for start in range(0, len(todo_idx), num_threads):
                    batch = todo_idx[start : start + num_threads]
                    futs = [ex.submit(process_one, i) for i in batch]
                    for fut in as_completed(futs):
                        try:
                            i, topics, status = fut.result()
                        except DailyRateLimitExceeded:
                            save_checkpoint()
                            raise
                        except Exception as e:
                            print(f"Worker error (continuing): {e}", flush=True)
                            continue
                        for t, v in topics.items():
                            work.at[i, t] = v
                        work.at[i, "topic_status"] = status
                        completed += 1
                        pbar.update(1)
                        if completed % save_interval == 0:
                            save_checkpoint()
                            print(
                                f"checkpoint {completed}/{len(todo_idx)} "
                                f"ok-total={(work.topic_status == 'ok').sum()}",
                                flush=True,
                            )
    except DailyRateLimitExceeded as e:
        save_checkpoint()
        raise SystemExit(f"Daily rate limit hit; progress saved to {out_path}\n{e}") from e

    save_checkpoint()
    n_ok = int((work["topic_status"] == "ok").sum())
    n_fail = int((work["topic_status"] == "failed").sum())
    n_inv = int((work["topic_status"] == "invalid").sum())
    print(f"Wrote {out_path}  ok={n_ok} failed={n_fail} invalid={n_inv}", flush=True)
    return out_path


def pilot_summary(csv_path: Path) -> None:
    df = pd.read_csv(csv_path, low_memory=False)
    n = len(df)
    print(f"\n=== Pilot summary: {csv_path.name} (n={n}) ===")
    print(df["topic_status"].value_counts().to_string())
    print("\nTopic rates (% of pilot sample):")
    rows = []
    for t in ALL_TOPICS:
        pct = 100.0 * df[t].sum() / n if n else 0.0
        rows.append((TOPIC_LABELS[t], t, int(df[t].sum()), pct))
    for label, key, cnt, pct in rows:
        print(f"  {label:32s}  {cnt:4d}  {pct:5.1f}%")
    tagged = df[ALL_TOPICS].sum(axis=1)
    print(f"\nTopics tagged per article: mean={tagged.mean():.2f}  "
          f"median={tagged.median():.0f}  zero-tag={int((tagged==0).sum())} ({100*(tagged==0).mean():.1f}%)")
    # Show a few positive examples
    print("\nSample positives (first 8 articles with ≥1 tag):")
    pos = df[tagged > 0].head(8)
    for _, r in pos.iterrows():
        on = [TOPIC_LABELS[t] for t in ALL_TOPICS if r[t] == 1]
        title = str(r.get("Title", ""))[:100]
        print(f"  - {title}")
        print(f"    → {', '.join(on)}")


def build_tables() -> None:
    """Build Table A/B percent CSVs + raw counts from full (non-pilot) topic files."""
    base = out_dir()
    frames = {}
    report = []
    for qid in PAPER_QUERY_ORDER:
        path = base / f"{qid}_topics.csv"
        if not path.is_file():
            report.append(f"MISSING {path.name}")
            continue
        df = pd.read_csv(path, low_memory=False)
        label = PAPER_TAXON_LABELS[qid]
        n = len(df)
        exp = EXPECTED_N[qid]
        n_fail = int((df["topic_status"] != "ok").sum()) if "topic_status" in df.columns else 0
        report.append(
            f"{label}: N={n:,} expected={exp:,} "
            f"{'OK' if n == exp else 'MISMATCH'}  failed/invalid={n_fail}"
        )
        frames[label] = df
        if n_fail:
            raise SystemExit(
                "Refusing to build tables while any rows are not ok.\n" + "\n".join(report)
            )

    print("\n".join(report))
    if len(frames) != 5:
        raise SystemExit("Need all five *_topics.csv files to build tables.")

    def pct_table(topics: list[str]) -> pd.DataFrame:
        rows = []
        for t in topics:
            row = {"Topic": TOPIC_LABELS[t]}
            for label in PAPER_TAXON_LABELS.values():
                d = frames[label]
                n = len(d)
                pct = 100.0 * d[t].sum() / n if n else 0.0
                row[label] = f"{pct:.1f}%"
            rows.append(row)
        n_row = {"Topic": "N (Group-focused)"}
        for label in PAPER_TAXON_LABELS.values():
            n_row[label] = f"{len(frames[label]):,}"
        rows.append(n_row)
        cols = ["Topic"] + list(PAPER_TAXON_LABELS.values())
        return pd.DataFrame(rows)[cols]

    def count_table() -> pd.DataFrame:
        rows = []
        for t in ALL_TOPICS:
            row = {"topic_key": t, "Topic": TOPIC_LABELS[t], "block": "biotic" if t in BIOTIC_TOPICS else "abiotic"}
            for label in PAPER_TAXON_LABELS.values():
                d = frames[label]
                row[f"{label}_n_tagged"] = int(d[t].sum())
                row[f"{label}_N"] = len(d)
            rows.append(row)
        return pd.DataFrame(rows)

    # Table A omits disease_of_the_insect (kept in coding + raw counts).
    table_a_topics = [t for t in BIOTIC_TOPICS if t != "disease_of_the_insect"]
    a = pct_table(table_a_topics)
    b = pct_table(ABIOTIC_TOPICS)
    raw = count_table()
    a.to_csv(base / "table_a_biotic_pct.csv", index=False)
    b.to_csv(base / "table_b_abiotic_pct.csv", index=False)
    raw.to_csv(base / "topic_counts_raw.csv", index=False)
    print(f"Wrote {base / 'table_a_biotic_pct.csv'}")
    print(f"Wrote {base / 'table_b_abiotic_pct.csv'}")
    print(f"Wrote {base / 'topic_counts_raw.csv'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Topic coding for group-focused articles.")
    add_query_arg(parser)
    parser.add_argument("--limit", type=int, default=None, help="Pilot sample size (random).")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for --limit.")
    parser.add_argument("--threads", type=int, default=NUM_THREADS)
    parser.add_argument("--save-interval", type=int, default=SAVE_INTERVAL)
    parser.add_argument(
        "--build-tables",
        action="store_true",
        help="Build Table A/B CSVs from full *_topics.csv files (no API calls).",
    )
    parser.add_argument(
        "--summary",
        type=str,
        default="",
        help="Print pilot summary for an existing CSV path.",
    )
    args = parser.parse_args()

    if args.build_tables:
        build_tables()
        return
    if args.summary:
        pilot_summary(Path(args.summary))
        return

    path = run_coding(
        query_id=args.query_id,
        limit=args.limit,
        num_threads=args.threads,
        save_interval=args.save_interval,
        seed=args.seed,
    )
    if args.limit is not None:
        pilot_summary(path)


if __name__ == "__main__":
    main()
