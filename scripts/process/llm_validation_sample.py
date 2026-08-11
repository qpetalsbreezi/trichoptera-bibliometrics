#!/usr/bin/env python3
"""Draw and freeze the stratified LLM validation sample (300 papers / group)."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

_SCRIPTS_DIR = Path(__file__).resolve().parent.parent
if str(_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPTS_DIR))

from lib.llm_validation import (  # noqa: E402
    DEFAULT_SAMPLE_N,
    DEFAULT_SEED,
    RELEVANCE_FIELD,
    display_label,
    load_coded_frame,
    sample_manifest_path,
    validation_dir,
    validation_query_ids,
)


def allocate_targets(counts: dict[tuple, int], total: int) -> dict[tuple, int]:
    """Proportional allocation across non-empty strata; remainder by largest remainder."""
    nonempty = {k: c for k, c in counts.items() if c > 0}
    if not nonempty:
        return {}
    pop = sum(nonempty.values())
    raw = {k: (total * c / pop) for k, c in nonempty.items()}
    floors = {k: min(nonempty[k], int(v)) for k, v in raw.items()}
    assigned = sum(floors.values())
    rem = total - assigned
    order = sorted(
        nonempty.keys(),
        key=lambda k: (raw[k] - floors[k], nonempty[k]),
        reverse=True,
    )
    targets = dict(floors)
    i = 0
    while rem > 0 and i < len(order) * 3:
        k = order[i % len(order)]
        if targets[k] < nonempty[k]:
            targets[k] += 1
            rem -= 1
        i += 1
    return targets


def draw_group_sample(df: pd.DataFrame, n: int, seed: int) -> pd.DataFrame:
    work = df.copy()
    work["stratum"] = list(
        zip(work["year_band"].astype(str), work["abstract_available"].map(bool))
    )
    counts = work["stratum"].value_counts().to_dict()
    targets = allocate_targets(counts, min(n, len(work)))
    parts = []
    rng = seed
    for stratum, target in sorted(targets.items(), key=lambda x: str(x[0])):
        pool = work[work["stratum"] == stratum]
        take = min(target, len(pool))
        if take <= 0:
            continue
        parts.append(pool.sample(n=take, random_state=rng))
        rng += 1
    out = pd.concat(parts, ignore_index=True) if parts else work.head(0)
    if len(out) < n and len(work) > len(out):
        leftover = work[~work["row_key"].isin(out["row_key"])]
        need = min(n - len(out), len(leftover))
        if need:
            out = pd.concat(
                [out, leftover.sample(n=need, random_state=seed + 99)],
                ignore_index=True,
            )
    return out.head(n)


def build_manifest(query_ids: list[str], n: int, seed: int) -> pd.DataFrame:
    rows = []
    for i, qid in enumerate(query_ids):
        df = load_coded_frame(qid)
        sample = draw_group_sample(df, n=n, seed=seed + i * 17)
        keep_cols = [
            "row_key",
            "EID",
            "DOI",
            "Title",
            "Year",
            "year_band",
            "abstract_available",
            "Source",
            "Abstract",
            "Author_Affiliations",
            RELEVANCE_FIELD,
            "Research_Theme",
            "Country",
            "Region_Global",
        ]
        for col in keep_cols:
            if col not in sample.columns:
                sample[col] = ""
        part = sample[keep_cols].copy()
        part.insert(0, "query_id", qid)
        part.insert(1, "taxon_label", display_label(qid))
        part = part.rename(
            columns={
                RELEVANCE_FIELD: "Taxon_Relevance_A",
                "Research_Theme": "Research_Theme_A",
                "Country": "Country_A",
                "Region_Global": "Region_Global_A",
            }
        )
        part["model_A"] = "gpt-4o-mini"
        rows.append(part)
        print(
            f"{display_label(qid):14s}  sampled {len(part):3d} / {len(df):,}  "
            f"bands={part['year_band'].value_counts().to_dict()}  "
            f"abs={part['abstract_available'].value_counts().to_dict()}"
        )
    return pd.concat(rows, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description="Freeze stratified LLM validation sample")
    parser.add_argument("--n-per-group", type=int, default=DEFAULT_SAMPLE_N)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--query-id",
        action="append",
        dest="query_ids",
        help="Limit to one or more query_ids (repeatable). Default: all five groups.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing sample_manifest.csv",
    )
    args = parser.parse_args()

    out = sample_manifest_path()
    if out.exists() and not args.force:
        raise SystemExit(
            f"Sample already frozen at {out}. Pass --force to redraw (will invalidate Gemini outputs)."
        )

    query_ids = args.query_ids or validation_query_ids()
    manifest = build_manifest(query_ids, n=args.n_per_group, seed=args.seed)
    validation_dir().mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out, index=False)
    print(f"\n✓ Wrote {len(manifest):,} rows to {out}")


if __name__ == "__main__":
    main()
