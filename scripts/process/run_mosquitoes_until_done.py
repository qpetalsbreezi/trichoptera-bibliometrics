"""Detached supervisor: resume Culicidae topic coding until complete, then build tables."""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "analysis" / "combined" / "topic_coding"
CSV = OUT / "mosquitoes_topics.csv"
AUTO = OUT / "mosquitoes_auto.log"
LOG = OUT / "mosquitoes_run_stdout.log"
PY = sys.executable
SCRIPT = Path(__file__).resolve().parent / "llm_code_topics.py"


def log(msg: str) -> None:
    AUTO.parent.mkdir(parents=True, exist_ok=True)
    line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} {msg}\n"
    with AUTO.open("a", encoding="utf-8") as f:
        f.write(line)
    print(line, end="", flush=True)


def ok_count() -> tuple[int, int]:
    import pandas as pd

    d = pd.read_csv(CSV, low_memory=False)
    return int((d["topic_status"] == "ok").sum()), len(d)


def main() -> None:
    os.chdir(ROOT)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    while True:
        ok, n = ok_count()
        log(f"loop ok={ok}/{n}")
        if ok >= n:
            log("COMPLETE — building tables")
            r = subprocess.run(
                [PY, str(SCRIPT), "--build-tables"],
                cwd=ROOT,
                env=env,
                capture_output=True,
                text=True,
            )
            (OUT / "tables_watch.log").open("a").write(r.stdout + r.stderr)
            log(f"tables exit={r.returncode}")
            return
        with LOG.open("a", encoding="utf-8") as logf:
            r = subprocess.run(
                [
                    PY,
                    str(SCRIPT),
                    "--query-id",
                    "mosquitoes",
                    "--threads",
                    "8",
                    "--save-interval",
                    "100",
                ],
                cwd=ROOT,
                env=env,
                stdout=logf,
                stderr=subprocess.STDOUT,
            )
        log(f"classifier exited code={r.returncode}")
        time.sleep(10)


if __name__ == "__main__":
    main()
