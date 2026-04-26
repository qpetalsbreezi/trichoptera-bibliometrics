"""
Central path and config resolution for multi-query bibliometric runs.

All scripts take --query-id (default: config default_query_id or env QUERY_ID).
Data and analysis outputs live under namespaces derived from query_id.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH = PROJECT_ROOT / "config" / "queries.json"


def project_root() -> Path:
    return PROJECT_ROOT


def load_queries_config() -> dict:
    if not CONFIG_PATH.exists():
        raise FileNotFoundError(f"Missing queries config: {CONFIG_PATH}")
    with open(CONFIG_PATH, encoding="utf-8") as f:
        return json.load(f)


def default_query_id() -> str:
    env = os.environ.get("QUERY_ID", "").strip()
    if env:
        return env
    cfg = load_queries_config()
    return str(cfg.get("default_query_id", "trichoptera"))


def get_query_config(query_id: str) -> dict:
    cfg = load_queries_config()
    queries = cfg.get("queries") or {}
    if query_id not in queries:
        raise SystemExit(
            f"Unknown query_id {query_id!r}. Defined in config/queries.json: {list(queries.keys())}"
        )
    return queries[query_id]


def load_dotenv(project_root: Path | None = None) -> None:
    """Load KEY=VALUE lines from .env at project root into os.environ (no override)."""
    root = project_root or PROJECT_ROOT
    env_file = root / ".env"
    if not env_file.exists():
        return
    with open(env_file, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                key, value = line.split("=", 1)
                key = key.strip()
                if key not in os.environ:
                    os.environ[key] = value.strip()


def add_query_arg(parser) -> None:
    parser.add_argument(
        "--query-id",
        default=default_query_id(),
        help="Query namespace (see config/queries.json). Env QUERY_ID sets the default when omitted.",
    )


class PipelinePaths:
    """Resolved paths for one query_id. Keeps the same pipeline layout for every query."""

    def __init__(self, query_id: str):
        self.query_id = query_id
        r = PROJECT_ROOT
        self.raw_scopus_api = r / "data" / "raw" / "scopus_api" / query_id
        self.raw_pop = r / "data" / "raw" / "scopus_publish_or_perish" / query_id
        self.raw_google_scholar = r / "data" / "raw" / "google_scholar" / query_id
        self.processed = r / "data" / "processed" / query_id
        self.analysis = r / "analysis" / query_id
        self.schema = r / "data" / "taxon_schema.json"

        self.combined_scopus_api = self.processed / "scopus_api_combined_2010_2025.csv"
        self.with_abstracts = self.processed / "scopus_api_with_abstracts.csv"
        self.with_authors = self.processed / "scopus_api_with_authors.csv"
        self.coded = self.processed / "scopus_api_coded.csv"
        self.pop_combined = self.processed / "scopus_pop_combined_2010_2025.csv"

    def rq_dir(self, name: str) -> Path:
        """e.g. name='rq1_coverage' -> analysis/{query_id}/rq1_coverage"""
        return self.analysis / name
