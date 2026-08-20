#!/usr/bin/env bash
# Wait for in-flight plecoptera recode, then recode remaining groups (8 threads).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/scripts"

WAIT_PID="${1:-}"
if [[ -n "$WAIT_PID" ]] && kill -0 "$WAIT_PID" 2>/dev/null; then
  echo "[$(date -Iseconds)] Waiting for plecoptera pid $WAIT_PID..."
  while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 30; done
  echo "[$(date -Iseconds)] Plecoptera process exited."
fi

is_complete() {
  local coded="$1"
  python3 - <<PY
import sys
import pandas as pd

def is_placeholder(row) -> bool:
    t = str(row.get("Research_Theme", "")).strip().lower()
    r = str(row.get("Taxon_Relevance", row.get("Trichoptera_Relevance", ""))).strip().lower()
    return t in ("", "not specified", "nan") and r in ("", "not specified", "nan")

df = pd.read_csv("${coded}", low_memory=False)
n = int(df.apply(is_placeholder, axis=1).sum())
print(f"Placeholder rows remaining: {n}/{len(df)}")
sys.exit(0 if n == 0 else 2)
PY
}

groups=(trichoptera ephemeroptera odonata mosquitoes)
for Q in "${groups[@]}"; do
  CODED="data/processed/${Q}/scopus_api_coded.csv"
  BACKUP="data/processed/${Q}/scopus_api_coded_pre_no_peripheral_20260819.csv"

  if [[ -f "$CODED" ]] && is_complete "$CODED"; then
    echo "[$(date -Iseconds)] $Q already complete — skipping."
    continue
  fi

  echo "[$(date -Iseconds)] ========== $Q =========="
  if [[ -f "$CODED" && ! -f "$BACKUP" ]]; then
    cp "$CODED" "$BACKUP"
    echo "Backed up -> $BACKUP"
  fi
  if [[ -f "$CODED" ]]; then
    rm "$CODED"
  fi
  python3 scripts/process/llm_code_taxon.py --query-id "$Q" --threads 8 --save-interval 50
  if is_complete "$CODED"; then
    echo "[$(date -Iseconds)] $Q complete."
  else
    echo "[$(date -Iseconds)] $Q incomplete (likely rate limit). Stopping chain; rerun this script to resume."
    exit 2
  fi
done
echo "[$(date -Iseconds)] All groups recoded."
