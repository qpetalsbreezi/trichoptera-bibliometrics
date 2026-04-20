"""Shared OpenAlex URL, headers, and DOI normalization for fetch scripts."""

from __future__ import annotations

import os
from urllib.parse import urlencode

import pandas as pd
import requests

# (connect, read) seconds — avoids hanging indefinitely on bad networks / TLS stalls.
OPENALEX_TIMEOUT = (10.0, 25.0)


def clean_doi(doi) -> str | None:
    if pd.isna(doi) or not doi:
        return None
    return (
        str(doi)
        .replace("https://doi.org/", "")
        .replace("http://dx.doi.org/", "")
        .strip()
    ) or None


def openalex_headers() -> dict:
    mail = (os.environ.get("OPENALEX_CONTACT_EMAIL") or "").strip()
    if not mail:
        mail = "your-email@example.com"
    return {"User-Agent": f"Bibliometric-pipeline/1.0 (mailto:{mail})"}


def openalex_work_url(clean_doi: str) -> str:
    """Work-by-DOI URL with mailto / api_key query params when set."""
    base = f"https://api.openalex.org/works/https://doi.org/{clean_doi}"
    params: dict[str, str] = {}
    mail = (os.environ.get("OPENALEX_CONTACT_EMAIL") or "").strip()
    if mail:
        params["mailto"] = mail
    key = (os.environ.get("OPENALEX_API_KEY") or "").strip()
    if key:
        params["api_key"] = key
    if not params:
        return base
    return f"{base}?{urlencode(params)}"


def retry_after_seconds(response: requests.Response) -> float | None:
    raw = response.headers.get("Retry-After")
    if not raw:
        return None
    raw = raw.strip()
    try:
        return float(min(max(int(raw), 5), 120))
    except ValueError:
        return 60.0


def crossref_mailto_ua() -> str:
    mail = (os.environ.get("OPENALEX_CONTACT_EMAIL") or "").strip()
    if not mail:
        mail = "your-email@example.com"
    return f"Bibliometric-pipeline (mailto:{mail})"
