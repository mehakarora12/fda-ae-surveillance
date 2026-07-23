"""
Phase 4 — Explanatory RAG Diagnostics
========================================
For every high-confidence anomaly from Phase 3, retrieves the categorized
reports that actually occurred in that time window (a lightweight
retrieval step over the Phase 1b output — no vector search needed here
since we're filtering by drug + date, not semantic similarity), and asks
an LLM to synthesize a plain-English explanation of what pattern of
adverse events is driving the spike.

Why this counts as RAG:
  The LLM never free-associates a diagnosis — every claim it makes is
  grounded in the actual retrieved reports we hand it in the prompt. This
  is the "R" (retrieval) constraining the "G" (generation), same core
  idea as the vector-search RAG in Phase 2, just with structured metadata
  filtering instead of semantic search (a legitimate, often underrated,
  retrieval strategy worth mentioning in interviews).

Why one call per anomaly (not batched, unlike Phase 1b):
  There are only ~30-40 anomalies total vs. ~13,000 reports — volume is
  low enough that batching would save negligible time, and each call
  needs more context (many reports at once) and more careful reasoning
  than the short categorization task in Phase 1b.

Usage
-----
    python src/rag_diagnostics.py run
    python src/rag_diagnostics.py status
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path

import pandas as pd
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

import data_ingestion as di  # noqa: E402 — reuse the tested, cached, rate-limited openFDA client

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("rag_diagnostics")

ANOMALIES_PATH = config.PROCESSED_DIR / "anomalies.csv"
EXPLANATIONS_DIR = config.PROCESSED_DIR / "anomaly_explanations"
EXPLANATIONS_DIR.mkdir(parents=True, exist_ok=True)
FINAL_PATH = config.PROCESSED_DIR / "anomaly_explanations.csv"

CONTEXT_WINDOW_DAYS = 3    # +/- days around the anomaly date to pull reports from
MAX_CONTEXT_REPORTS = 8    # kept small — real reports can be verbose (long comedication
                           # lists), and staying well under Groq's 6,000 TPM free-tier
                           # ceiling in a single call matters more than a larger sample
SLEEP_SECONDS = 5          # low call volume here, light sleep is enough

SEVERITY_VOCAB = ["mild", "moderate", "severe", "life-threatening", "fatal"]

SYSTEM_PROMPT = f"""You are a pharmacovigilance analyst writing a root-cause note for a \
statistically anomalous change in adverse event reports for a drug. You will be given \
the anomaly's stats — including an explicit Direction field — and a sample of the \
actual case reports from that time window.

Base every claim ONLY on the reports provided. Do not invent details. If the reports \
don't clearly explain the anomaly, say so honestly rather than speculating.

CRITICAL: Use the given Direction field exactly as stated (SPIKE = higher than expected, \
DROP = lower than expected). Do not infer or guess the direction yourself from the numbers.

For "dominant_severity", choose exactly ONE value from this list: {SEVERITY_VOCAB}.

Respond ONLY with a JSON object of this exact shape:
{{"summary": "1-2 sentence plain-English summary of what happened, correctly reflecting the given Direction",
 "dominant_organ_systems": ["...", "..."],
 "dominant_severity": "one of {SEVERITY_VOCAB}",
 "notable_pattern": "1-2 sentences on any recurring theme (e.g. co-administered drug, demographic skew, dose-related signal) or 'no clear pattern beyond the aggregate stats' if none exists",
 "confidence_caveat": "1 sentence noting this is an exploratory signal from a limited sample, not a clinical conclusion"}}

No prose, no markdown fences, no text outside that JSON object.
"""


# --------------------------------------------------------------------------
# Retrieval: LIVE, targeted openFDA query for this specific anomaly window
# (not the Phase 1b sample, which — as this bug revealed — doesn't have
# even coverage across the full date range)
# --------------------------------------------------------------------------
def get_context_reports(drug: str, date: pd.Timestamp) -> list[str]:
    window_start = (date - pd.Timedelta(days=CONTEXT_WINDOW_DAYS)).strftime("%Y-%m-%d")
    window_end = (date + pd.Timedelta(days=CONTEXT_WINDOW_DAYS)).strftime("%Y-%m-%d")

    search = f'patient.drug.medicinalproduct:"{drug}" AND {di._date_range_query(window_start, window_end)}'
    params = {"search": search, "limit": MAX_CONTEXT_REPORTS}
    data = di.fetch_cached(params)
    results = data.get("results", [])

    return [di._flatten_report(raw, drug)["patient_summary_text"] for raw in results]


# --------------------------------------------------------------------------
# Groq call (retried, cached per-anomaly on disk — same idempotent pattern
# as Phase 1b, so reruns never repeat work or waste quota)
# --------------------------------------------------------------------------
def _client() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set. Get one free at https://console.groq.com/keys")
    return Groq(api_key=config.GROQ_API_KEY)


def _anomaly_id(row: pd.Series) -> str:
    raw = f"{row['drug']}_{row['date']}"
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
def _explain_anomaly(client: Groq, row: pd.Series, context_reports: list[str]) -> dict:
    numbered_reports = "\n".join(f"{i+1}. {text}" for i, text in enumerate(context_reports))
    direction = "SPIKE (higher than expected)" if row["zscore"] >= 0 else "DROP (lower than expected)"
    stats_block = (
        f"Drug: {row['drug']}\n"
        f"Date: {row['date']}\n"
        f"Direction: {direction}\n"
        f"Actual report count that day: {row['count']}\n"
        f"Expected count (Holt-Winters forecast): {row['expected_count']}\n"
        f"Z-score: {row['zscore']}\n"
        f"Sample of {len(context_reports)} reports from +/-{CONTEXT_WINDOW_DAYS} days:\n"
        + numbered_reports
    )

    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        max_tokens=400,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": stats_block},
        ],
    )
    return json.loads(response.choices[0].message.content)


# --------------------------------------------------------------------------
# Subcommand: run
# --------------------------------------------------------------------------
def _already_done_ids() -> set[str]:
    return {f.stem for f in EXPLANATIONS_DIR.glob("*.json")}


def run_diagnostics() -> None:
    if not ANOMALIES_PATH.exists():
        raise FileNotFoundError(f"{ANOMALIES_PATH} not found. Run Phase 3 first.")

    anomalies = pd.read_csv(ANOMALIES_PATH, parse_dates=["date"])
    anomalies = anomalies[anomalies["is_anomaly_high_confidence"]].copy()

    done_ids = _already_done_ids()
    anomalies["_id"] = anomalies.apply(_anomaly_id, axis=1)
    pending = anomalies[~anomalies["_id"].isin(done_ids)]
    log.info(f"{len(done_ids)} anomalies already explained, {len(pending)} pending.")

    if pending.empty:
        log.info("Nothing to do.")
        merge_results()
        return

    client = _client()
    for _, row in tqdm(pending.iterrows(), total=len(pending), desc="Explaining anomalies"):
        context_reports = get_context_reports(row["drug"], row["date"])
        if not context_reports:
            log.warning(f"No context reports found for {row['drug']} on {row['date']} — skipping.")
            continue

        try:
            explanation = _explain_anomaly(client, row, context_reports)
        except Exception as e:  # noqa: BLE001
            log.error(f"Failed to explain {row['drug']} {row['date']}: {e}")
            continue

        out = {
            "drug": row["drug"],
            "date": str(row["date"].date()),
            "count": int(row["count"]),
            "expected_count": float(row["expected_count"]),
            "zscore": float(row["zscore"]),
            "n_context_reports": len(context_reports),
            **explanation,
        }
        (EXPLANATIONS_DIR / f"{row['_id']}.json").write_text(json.dumps(out))
        time.sleep(SLEEP_SECONDS)

    merge_results()


def _normalize_severity(value: str) -> str:
    """Prompt instructions alone don't guarantee the LLM sticks to the
    controlled vocabulary (same lesson as Phase 1b) — map compound/loose
    answers like 'Hospitalization and death' or 'non-serious/other' to the
    single closest controlled value. Severe indicators are checked BEFORE
    mild ones, so an ambiguous compound answer (e.g. 'non-serious/other and
    hospitalization') resolves to the more severe reading — the safer
    default for a safety-signal tool."""
    s = str(value).strip().lower()
    if "fatal" in s or "death" in s:
        return "fatal"
    if "life-threatening" in s or "life threatening" in s:
        return "life-threatening"
    if "hospitalization" in s or "disabling" in s or "severe" in s:
        return "severe"
    if "non-serious" in s or "non serious" in s:
        return "mild"
    if "serious" in s:
        return "moderate"
    if "mild" in s:
        return "mild"
    return "moderate"  # safe default for anything unrecognized


def merge_results() -> None:
    rows = [json.loads(f.read_text()) for f in EXPLANATIONS_DIR.glob("*.json")]
    if not rows:
        log.info("No explanations yet.")
        return
    df = pd.DataFrame(rows).sort_values(["drug", "date"])

    if "dominant_severity" in df.columns:
        before = df["dominant_severity"].copy()
        df["dominant_severity"] = df["dominant_severity"].apply(_normalize_severity)
        n_fixed = (before.str.lower().str.strip() != df["dominant_severity"]).sum()
        if n_fixed:
            log.info(f"Normalized {n_fixed} non-standard severity labels.")

    if "dominant_organ_systems" in df.columns:
        df["dominant_organ_systems"] = df["dominant_organ_systems"].apply(
            lambda v: [str(x).strip().lower() for x in v] if isinstance(v, list) else v
        )

    df.to_csv(FINAL_PATH, index=False)
    log.info(f"Saved {len(df)} anomaly explanations -> {FINAL_PATH}")


def print_status() -> None:
    if not ANOMALIES_PATH.exists():
        log.info("Phase 3 not run yet.")
        return
    anomalies = pd.read_csv(ANOMALIES_PATH, parse_dates=["date"])
    total = int(anomalies["is_anomaly_high_confidence"].sum())
    done = len(_already_done_ids())
    log.info(f"Explained {done}/{total} high-confidence anomalies.")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 4 — RAG anomaly diagnostics")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Explain all pending high-confidence anomalies")
    sub.add_parser("status", help="Show progress")
    sub.add_parser("merge", help="Rebuild anomaly_explanations.csv from cache")

    args = parser.parse_args()
    if args.command == "run":
        run_diagnostics()
    elif args.command == "status":
        print_status()
    elif args.command == "merge":
        merge_results()


if __name__ == "__main__":
    main()