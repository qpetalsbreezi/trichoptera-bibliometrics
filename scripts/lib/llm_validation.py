"""Shared helpers for GPT vs Gemini LLM validation."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from lib.pipeline import (
    PAPER_QUERY_ORDER,
    PipelinePaths,
    normalize_research_theme,
    paper_taxon_label,
    project_root,
)

RELEVANCE_FIELD = "Taxon_Relevance"
LEGACY_RELEVANCE_FIELD = "Trichoptera_Relevance"
COMPARE_FIELDS = ("Taxon_Relevance", "Research_Theme")
ABSTRACT_UNAVAILABLE = "__ABSTRACT_UNAVAILABLE__"
RELEVANCE_TIERS = (
    "Primary focus",
    "Secondary mention",
    "Peripheral",
    "Not target-taxon-focused",
)
RELEVANCE_OFF_TARGET = "Not target-taxon-focused"
RELEVANCE_ALLOWED_VALUES = list(RELEVANCE_TIERS)
YEAR_BANDS = (
    ("2010-2015", 2010, 2015),
    ("2016-2019", 2016, 2019),
    ("2020-2025", 2020, 2025),
)
DEFAULT_SAMPLE_N = 300
DEFAULT_SEED = 20260810
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
SPOTCHECK_N = 30


def validation_dir() -> Path:
    return project_root() / "analysis" / "combined" / "llm_validation"


def sample_manifest_path() -> Path:
    return validation_dir() / "sample_manifest.csv"


def gemini_coded_path() -> Path:
    return validation_dir() / "gemini_coded.csv"


def agreement_metrics_path() -> Path:
    return validation_dir() / "agreement_metrics.csv"


def disagreement_review_path() -> Path:
    return validation_dir() / "disagreements_for_review.csv"


def spotcheck_path() -> Path:
    return validation_dir() / "spotcheck_agreements.csv"


def norm(v) -> str:
    if pd.isna(v):
        return ""
    return str(v).strip()


def abstract_text_ok(abstract) -> bool:
    """True when abstract text is present and not the unavailable placeholder."""
    if abstract is None or (isinstance(abstract, float) and pd.isna(abstract)):
        return False
    s = str(abstract).strip()
    if not s or s == ABSTRACT_UNAVAILABLE:
        return False
    return True


def make_row_key(row) -> str:
    eid = norm(row.get("EID", ""))
    if eid:
        return f"eid::{eid}"
    scopus_id = norm(row.get("ScopusID", ""))
    if scopus_id:
        return f"scopus::{scopus_id}"
    doi = norm(row.get("DOI", "")).lower()
    if doi:
        return f"doi::{doi}"
    title = norm(row.get("Title", "")).lower()
    year = norm(row.get("Year", ""))
    journal = norm(row.get("Source", row.get("Journal", ""))).lower()
    return f"title-year-source::{title}::{year}::{journal}"


def relevance_value(row) -> str:
    return norm(row.get(RELEVANCE_FIELD, row.get(LEGACY_RELEVANCE_FIELD, "")))


def year_band(year) -> str:
    try:
        y = int(year)
    except (TypeError, ValueError):
        return "unknown"
    for label, lo, hi in YEAR_BANDS:
        if lo <= y <= hi:
            return label
    return "unknown"


def abstract_available_flag(row) -> bool:
    keys = set(row.index) if hasattr(row, "index") else set(row.keys())
    if "abstract_available" in keys:
        val = row.get("abstract_available")
        if isinstance(val, bool):
            return val
        s = norm(val).lower()
        if s in ("true", "1", "yes"):
            return True
        if s in ("false", "0", "no"):
            return False
    return abstract_text_ok(row.get("Abstract", ""))


def load_schema_for_llm(schema_path: Path | None = None) -> tuple[dict, str]:
    path = schema_path or (project_root() / "data" / "taxon_schema.json")
    with open(path, encoding="utf-8") as f:
        schema = json.load(f)

    schema_columns = schema.get("columns", {})
    llm_coded_fields = ["Country", "Region_Global", "Research_Theme", RELEVANCE_FIELD]
    llm_schema: dict = {}
    for col, spec in schema_columns.items():
        if col in llm_coded_fields:
            if "allowed_values" in spec:
                llm_schema[col] = spec["allowed_values"]
            else:
                llm_schema[col] = "short free-text"
    if RELEVANCE_FIELD not in llm_schema:
        llm_schema[RELEVANCE_FIELD] = list(RELEVANCE_ALLOWED_VALUES)
    return llm_schema, json.dumps(llm_schema, indent=2)


def normalize_taxon_relevance(value) -> str:
    """Map missing/Not Specified relevance to the single off-target label."""
    s = norm(value)
    if not s or s.lower() in ("not specified", "nan", "none"):
        return RELEVANCE_OFF_TARGET
    return s


def taxon_relevance_is_ns(parsed: dict) -> bool:
    """True when Taxon_Relevance is missing or Not Specified (invalid under new schema)."""
    val = norm(parsed.get(RELEVANCE_FIELD, parsed.get(LEGACY_RELEVANCE_FIELD, "")))
    return (not val) or val.lower() == "not specified"


def load_coded_frame(query_id: str) -> pd.DataFrame:
    paths = PipelinePaths(query_id)
    if not paths.coded.exists():
        raise FileNotFoundError(f"Missing coded file for {query_id}: {paths.coded}")
    df = pd.read_csv(paths.coded)
    if "Author_Affiliations" not in df.columns and paths.with_authors.exists():
        authors = pd.read_csv(paths.with_authors)
        if "Author_Affiliations" in authors.columns and "Title" in authors.columns:
            df = df.merge(
                authors[["Title", "Author_Affiliations"]],
                on="Title",
                how="left",
                suffixes=("", "_auth"),
            )
    if "Author_Affiliations" not in df.columns:
        df["Author_Affiliations"] = None
    df["row_key"] = df.apply(make_row_key, axis=1)
    df["year_band"] = df["Year"].map(year_band)
    df["abstract_available"] = df.apply(abstract_available_flag, axis=1)
    df[RELEVANCE_FIELD] = df.apply(relevance_value, axis=1)
    if "Research_Theme" in df.columns:
        df["Research_Theme"] = df["Research_Theme"].map(normalize_research_theme)
    return df


def validation_query_ids() -> list[str]:
    return list(PAPER_QUERY_ORDER)


def cohens_kappa(y_true: list[str], y_pred: list[str]) -> float:
    """Cohen's κ for categorical labels (no sklearn dependency)."""
    n = len(y_true)
    if n == 0:
        return float("nan")
    labels = sorted(set(y_true) | set(y_pred))
    idx = {lab: i for i, lab in enumerate(labels)}
    k = len(labels)
    matrix = [[0] * k for _ in range(k)]
    for a, b in zip(y_true, y_pred):
        matrix[idx[a]][idx[b]] += 1

    po = sum(matrix[i][i] for i in range(k)) / n
    row_m = [sum(matrix[i][j] for j in range(k)) for i in range(k)]
    col_m = [sum(matrix[i][j] for i in range(k)) for j in range(k)]
    pe = sum((row_m[i] / n) * (col_m[i] / n) for i in range(k))
    if pe >= 1.0:
        return 1.0 if po >= 1.0 else float("nan")
    return (po - pe) / (1.0 - pe)


def abstract_snippet(text, max_chars: int = 400) -> str:
    s = norm(text)
    if len(s) <= max_chars:
        return s
    return s[: max_chars - 1].rstrip() + "…"


def display_label(query_id: str) -> str:
    return paper_taxon_label(query_id)


def relevance_adjacent(a: str, b: str) -> bool:
    a, b = norm(a), norm(b)
    if a not in RELEVANCE_TIERS or b not in RELEVANCE_TIERS:
        return False
    return abs(RELEVANCE_TIERS.index(a) - RELEVANCE_TIERS.index(b)) == 1


def is_all_not_specified(parsed: dict, llm_schema: dict) -> bool:
    """True when every schema field is missing or 'Not Specified' (incl. empty Country)."""
    for col in llm_schema:
        val = norm(parsed.get(col, ""))
        if col == "Country":
            if val == "" or val.lower() == "not specified":
                continue
            return False
        if val.lower() != "not specified":
            return False
    return True


def is_gemini_all_ns_raw(raw_json: str) -> bool:
    """Detect Gemini rows where Taxon_Relevance, Research_Theme, and Country are all NS."""
    s = str(raw_json)
    return (
        '"Taxon_Relevance": "Not Specified"' in s
        and '"Research_Theme": "Not Specified"' in s
        and '"Country": "Not Specified"' in s
    )


def classify_disagreement_type(row) -> str:
    """Single primary label for a GPT vs Gemini disagreement row."""
    rel_a = norm(row.get("Taxon_Relevance_A", row.get("rel_A", "")))
    rel_b = norm(row.get("Taxon_Relevance_B", row.get("rel_B", "")))
    theme_a = norm(row.get("Research_Theme_A", row.get("theme_A", "")))
    theme_b = norm(row.get("Research_Theme_B", row.get("theme_B", "")))

    rel_dis = rel_a != rel_b
    theme_dis = theme_a != theme_b

    if rel_dis:
        if rel_b == "Not Specified":
            return "gemini_ns_relevance"
        if rel_a == "Not Specified":
            return "gpt_ns_relevance"
        if relevance_adjacent(rel_a, rel_b):
            return "adjacent_relevance"

    if theme_dis:
        if theme_b == "Not Specified":
            return "gemini_ns_theme"
        if theme_a == "Not Specified":
            return "gpt_ns_theme"

    if rel_dis and theme_dis:
        return "substantive_both"
    if rel_dis:
        return "substantive_relevance"
    if theme_dis:
        return "substantive_theme"
    return "agree"
