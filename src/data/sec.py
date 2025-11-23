from __future__ import annotations

import os
from functools import lru_cache
from typing import Dict, Iterable, List, Optional
from urllib.parse import urljoin

import requests


DEFAULT_SEC_UA = os.getenv("SEC_USER_AGENT", "Quant13ResearchBot/1.0 (contact@example.com)")


def _headers(accept: str | None = None) -> Dict[str, str]:
    headers = {
        "User-Agent": DEFAULT_SEC_UA,
        "Accept-Encoding": "gzip, deflate",
    }
    if accept:
        headers["Accept"] = accept
    return headers


@lru_cache(maxsize=1)
def _ticker_cik_map() -> Dict[str, str]:
    url = "https://www.sec.gov/include/ticker.txt"
    response = requests.get(url, headers=_headers("text/plain"), timeout=30)
    response.raise_for_status()
    mapping: Dict[str, str] = {}
    for line in response.text.splitlines():
        if not line or line.startswith("#"):
            continue
        if "|" in line:
            symbol, cik = line.split("|", 1)
        elif "\t" in line:
            symbol, cik = line.split("\t", 1)
        else:
            parts = line.split()
            if len(parts) != 2:
                continue
            symbol, cik = parts
        mapping[symbol.strip().lower()] = cik.strip()
    return mapping


def lookup_cik(ticker: str) -> Optional[str]:
    return _ticker_cik_map().get(ticker.lower())


def fetch_recent_filings(ticker: str, forms: Iterable[str] | None = None, limit_per_form: int = 1) -> List[Dict[str, str]]:
    desired_forms = list(forms) if forms else ["10-K", "10-Q"]
    cik = lookup_cik(ticker)
    if not cik:
        return []
    padded_cik = f"{int(cik):010d}"
    cik_int = str(int(cik))
    url = f"https://data.sec.gov/submissions/CIK{padded_cik}.json"
    try:
        response = requests.get(url, headers=_headers("application/json"), timeout=30)
        response.raise_for_status()
    except requests.RequestException:
        return []

    data = response.json()
    recent = data.get("filings", {}).get("recent", {})
    forms_list = recent.get("form", [])
    accessions = recent.get("accessionNumber", [])
    filing_dates = recent.get("filingDate", [])
    report_dates = recent.get("reportDate", [])
    documents = recent.get("primaryDocument", [])

    counts = {form: 0 for form in desired_forms}
    results: List[Dict[str, str]] = []
    for form, accession, filing_date, report_date, primary_doc in zip(forms_list, accessions, filing_dates, report_dates, documents):
        if form not in desired_forms:
            continue
        if counts.get(form, 0) >= limit_per_form:
            continue

        accession_key = accession.replace("-", "")
        base_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_key}/"
        document_url = urljoin(base_url, primary_doc)
        text_url = document_url
        if "." in primary_doc:
            stem, ext = primary_doc.rsplit(".", 1)
            text_url = urljoin(base_url, f"{stem}.txt")

        results.append({
            "form": form,
            "filingDate": filing_date,
            "reportDate": report_date,
            "accessionNumber": accession,
            "cik": padded_cik,
            "documentUrl": document_url,
            "textUrl": text_url,
        })
        counts[form] = counts.get(form, 0) + 1
        if all(counts.get(form, 0) >= limit_per_form for form in desired_forms):
            break

    return results


def fetch_document_text(urls: Iterable[str]) -> str:
    for url in urls:
        if not url:
            continue
        try:
            response = requests.get(url, headers=_headers("text/html, text/plain"), timeout=30)
            response.raise_for_status()
        except requests.RequestException:
            continue
        content_type = response.headers.get("Content-Type", "")
        if "text" in content_type or "html" in content_type:
            return response.text
    return ""
