"""
Phase 1a — Data Ingestion
=========================
Pulls two things from the free openFDA FAERS (drug adverse event) API:

1. `counts`  -> aggregated daily serious-AE counts per drug.
               This IS the Phase 3 forecasting target — one API call
               per drug gives you years of daily counts for free.
2. `reports` -> individual case reports, flattened into a
               `patient_summary_text` field that Phase 1b (LLM
               categorization) and Phase 2 (embeddings) will consume.

Design goals (so this never gets you "stuck"):
- Every HTTP response is cached to disk by its request hash. Re-running
  the same command costs zero extra API calls — safe to stop/resume.
- Automatic retry with exponential backoff on network errors / 429s.
- A fixed sleep between requests, calibrated to openFDA's published
  rate limits, so you never trigger a 429 in the first place.

Usage
-----
    python src/data_ingestion.py counts  --drugs metformin lisinopril --start 2022-01-01 --end 2024-12-31
    python src/data_ingestion.py reports --drugs metformin lisinopril --start 2022-01-01 --end 2024-12-31 --per-drug-limit 2000
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("data_ingestion")

PAGE_SIZE = 100  # openFDA allows up to 1000, but 100 keeps memory + retries cheap


# --------------------------------------------------------------------------
# Low-level HTTP helpers (cached + retried + rate-limited)
# --------------------------------------------------------------------------
class OpenFDAError(Exception):
    """Raised for non-retryable openFDA API errors (e.g. bad query syntax)."""


def _cache_key(params: dict[str, Any]) -> Path:
    raw = json.dumps(params, sort_keys=True)
    digest = hashlib.sha256(raw.encode()).hexdigest()[:24]
    return config.CACHE_DIR / f"{digest}.json"


@retry(
    reraise=True,
    stop=stop_after_attempt(5),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=retry_if_exception_type((requests.exceptions.RequestException,)),
)
def _get(params: dict[str, Any]) -> dict[str, Any]:
    """GET the openFDA endpoint once, with retry on transient errors."""
    if config.OPENFDA_API_KEY:
        params = {**params, "api_key": config.OPENFDA_API_KEY}
    resp = requests.get(config.OPENFDA_BASE_URL, params=params, timeout=30)

    if resp.status_code == 429:
        # openFDA sends 429 when you exceed rate limits — back off harder.
        log.warning("Rate limited (429). Sleeping 10s before retry...")
        time.sleep(10)
        resp.raise_for_status()  # triggers tenacity retry

    if resp.status_code == 404:
        # openFDA returns 404 (not 200 + empty list) when a query matches
        # zero records. That's a valid "no results," not an error.
        return {"results": []}

    resp.raise_for_status()
    return resp.json()


def fetch_cached(params: dict[str, Any]) -> dict[str, Any]:
    """Fetch a result, transparently caching to disk by request signature."""
    path = _cache_key(params)
    if path.exists():
        return json.loads(path.read_text())

    data = _get(params)
    path.write_text(json.dumps(data))
    time.sleep(config.OPENFDA_SLEEP_SECONDS)  # stay under rate limit even on cache misses
    return data


def _fda_date(date_str: str) -> str:
    """Convert 'YYYY-MM-DD' -> openFDA's 'YYYYMMDD' format."""
    return date_str.replace("-", "")


def _date_range_query(start: str, end: str) -> str:
    # Use a real space around TO — requests will URL-encode it correctly.
    # (A literal "+" here gets percent-escaped to %2B by requests and openFDA
    # can't parse it, which is what caused the 500 errors.)
    return f"receivedate:[{_fda_date(start)} TO {_fda_date(end)}]"


# --------------------------------------------------------------------------
# Subcommand 1: counts
# --------------------------------------------------------------------------
def fetch_counts_for_drug(drug: str, start: str, end: str) -> pd.DataFrame:
    """
    One aggregated API call returns the FULL daily time series — this is
    the single most efficient call in the project (no pagination needed).
    """
    search = f'patient.drug.medicinalproduct:"{drug}" AND serious:1 AND {_date_range_query(start, end)}'
    params = {"search": search, "count": "receivedate"}
    data = fetch_cached(params)
    results = data.get("results", [])
    if not results:
        log.warning(f"[counts] No results for '{drug}' — check spelling or date range.")
        return pd.DataFrame(columns=["date", "count", "drug"])

    df = pd.DataFrame(results).rename(columns={"time": "date"})
    df["date"] = pd.to_datetime(df["date"], format="%Y%m%d")
    df["drug"] = drug
    return df[["date", "count", "drug"]]


def run_counts(drugs: list[str], start: str, end: str) -> None:
    all_rows = []
    for drug in tqdm(drugs, desc="Fetching daily counts"):
        df = fetch_counts_for_drug(drug, start, end)
        if not df.empty:
            out_path = config.RAW_COUNTS_DIR / f"{drug}_counts.csv"
            df.to_csv(out_path, index=False)
            log.info(f"[counts] {drug}: {len(df)} days, {int(df['count'].sum())} total serious AE reports -> {out_path}")
        all_rows.append(df)

    combined = pd.concat(all_rows, ignore_index=True) if all_rows else pd.DataFrame()
    combined_path = config.PROCESSED_DIR / "daily_counts.csv"
    combined.to_csv(combined_path, index=False)
    log.info(f"[counts] Combined dataset ({len(combined)} rows) -> {combined_path}")
    log.info("This file is your Phase 3 forecasting input, ready as-is.")


# --------------------------------------------------------------------------
# Subcommand 2: reports
# --------------------------------------------------------------------------
def _flatten_report(raw: dict[str, Any], drug: str) -> dict[str, Any]:
    """
    Turn one raw openFDA case-report JSON blob into a flat row, including
    a human-readable `patient_summary_text` for the LLM (Phase 1b) and
    embeddings (Phase 2) to consume.
    """
    patient = raw.get("patient", {})
    reactions = [r.get("reactionmeddrapt", "") for r in patient.get("reaction", []) if r.get("reactionmeddrapt")]
    drugs_involved = [
        d.get("medicinalproduct", "") for d in patient.get("drug", []) if d.get("medicinalproduct")
    ]
    age = patient.get("patientonsetage")
    sex_map = {"1": "male", "2": "female"}
    sex = sex_map.get(str(patient.get("patientsex")), "unknown")

    seriousness_flags = [
        label
        for key, label in [
            ("seriousnessdeath", "death"),
            ("seriousnesshospitalization", "hospitalization"),
            ("seriousnesslifethreatening", "life-threatening"),
            ("seriousnessdisabling", "disabling"),
        ]
        if raw.get(key) == "1"
    ]

    summary = (
        f"A {age or 'unknown-age'} year old {sex} patient taking "
        f"{', '.join(drugs_involved) or drug} experienced: {', '.join(reactions) or 'unspecified reaction(s)'}. "
        f"Seriousness: {', '.join(seriousness_flags) or 'non-serious/other'}. "
        f"Report received {raw.get('receivedate', 'unknown date')}."
    )

    return {
        "safetyreportid": raw.get("safetyreportid"),
        "drug": drug,
        "receivedate": raw.get("receivedate"),
        "patient_sex": sex,
        "patient_age": age,
        "reactions": "; ".join(reactions),
        "drugs_involved": "; ".join(drugs_involved),
        "seriousness_flags": "; ".join(seriousness_flags),
        "patient_summary_text": summary,
    }


def fetch_reports_for_drug(drug: str, start: str, end: str, per_drug_limit: int) -> list[dict[str, Any]]:
    search = f'patient.drug.medicinalproduct:"{drug}" AND {_date_range_query(start, end)}'
    rows: list[dict[str, Any]] = []
    skip = 0

    with tqdm(total=per_drug_limit, desc=f"Fetching reports: {drug}", leave=False) as pbar:
        while len(rows) < per_drug_limit:
            page_limit = min(PAGE_SIZE, per_drug_limit - len(rows))
            params = {"search": search, "limit": page_limit, "skip": skip}
            data = fetch_cached(params)
            results = data.get("results", [])
            if not results:
                break  # exhausted all available reports for this drug/date range

            for raw in results:
                rows.append(_flatten_report(raw, drug))
            pbar.update(len(results))
            skip += page_limit

    return rows


def run_reports(drugs: list[str], start: str, end: str, per_drug_limit: int) -> None:
    all_rows: list[dict[str, Any]] = []
    for drug in drugs:
        rows = fetch_reports_for_drug(drug, start, end, per_drug_limit)
        log.info(f"[reports] {drug}: {len(rows)} reports pulled")
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows).drop_duplicates(subset="safetyreportid")
    out_path = config.PROCESSED_DIR / "patient_summaries.parquet"
    df.to_parquet(out_path, index=False)
    log.info(f"[reports] Total unique reports: {len(df)} -> {out_path}")
    log.info("This file feeds directly into Phase 1b (LLM categorization).")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1a — openFDA data ingestion")
    sub = parser.add_subparsers(dest="command", required=True)

    p_counts = sub.add_parser("counts", help="Pull aggregated daily serious-AE counts")
    p_counts.add_argument("--drugs", nargs="+", default=config.DRUGS)
    p_counts.add_argument("--start", default=config.START_DATE)
    p_counts.add_argument("--end", default=config.END_DATE)

    p_reports = sub.add_parser("reports", help="Pull individual case reports")
    p_reports.add_argument("--drugs", nargs="+", default=config.DRUGS)
    p_reports.add_argument("--start", default=config.START_DATE)
    p_reports.add_argument("--end", default=config.END_DATE)
    p_reports.add_argument("--per-drug-limit", type=int, default=2000)

    args = parser.parse_args()

    if not config.OPENFDA_API_KEY:
        log.warning(
            "No OPENFDA_API_KEY set — running at the free anonymous limit "
            "(40 req/min, 1,000 req/day). Get a free key: "
            "https://open.fda.gov/apis/authentication/"
        )

    if args.command == "counts":
        run_counts(args.drugs, args.start, args.end)
    elif args.command == "reports":
        run_reports(args.drugs, args.start, args.end, args.per_drug_limit)


if __name__ == "__main__":
    main()
