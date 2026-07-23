"""
Phase 1b — LLM Clinical Categorization
=======================================
Takes the `patient_summary_text` produced by Phase 1a and asks an LLM to
categorize each report: primary organ system affected, severity, most
likely causal drug, and a few short tags. This structured output feeds:
  - Phase 2  (category becomes embedding metadata / filter)
  - Phase 3  (anomaly detection can run per-category, not just per-drug)
  - Phase 5  (compare LLM tags against ~100 hand-labeled reports)

Why batched, cached calls to Groq (free tier):
  Groq's llama-3.1-8b-instant free tier = 30 req/min, 14,400 req/day,
  but only ~6,000 tokens/min. With potentially 10,000+ reports, one call
  per report would blow the token budget in minutes. Batching ~20 reports
  per call cuts total requests by ~20x, which is what actually keeps this
  project inside free-tier limits.

Crash-safety:
  Every batch's parsed result is written to its own file in
  data/processed/categorized_batches/, named by the report IDs it covers.
  On (re)start, we scan that folder, build the set of already-done IDs,
  and only send the *remaining* reports in new batches. Kill the script
  anytime — rerunning `run` never repeats work or wastes API calls.

Usage
-----
    python src/extraction.py run --batch-size 20 --sleep 8
    python src/extraction.py status
    python src/extraction.py merge     # rebuild the final parquet from cached batches
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
from groq import Groq
from tenacity import retry, stop_after_attempt, wait_exponential
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("extraction")

BATCHES_DIR = config.PROCESSED_DIR / "categorized_batches"
BATCHES_DIR.mkdir(parents=True, exist_ok=True)
FINAL_PATH = config.PROCESSED_DIR / "categorized_reports.parquet"
SUMMARIES_PATH = config.PROCESSED_DIR / "patient_summaries.parquet"

ORGAN_SYSTEMS = [
    "cardiovascular", "neurological", "gastrointestinal", "dermatological",
    "psychiatric", "hepatic", "renal", "hematologic", "respiratory",
    "musculoskeletal", "endocrine/metabolic", "other/unclassified",
]
SEVERITIES = ["mild", "moderate", "severe", "life-threatening", "fatal"]

SYSTEM_PROMPT = f"""You are a clinical pharmacovigilance assistant categorizing FDA adverse event reports.

For EACH report given, return an object with these exact keys:
- "id": the report's id, copied exactly as given
- "primary_organ_system": ONE of {ORGAN_SYSTEMS}
- "severity": ONE of {SEVERITIES}
- "likely_causal_drug": the drug name most plausibly responsible (from the ones listed in the report)
- "category_tags": 1-3 short lowercase tags, e.g. ["known_side_effect", "dose_related", "drug_interaction_suspected"]
- "rationale": ONE short sentence explaining your reasoning

Respond ONLY with a JSON object of the exact shape:
{{"reports": [{{"id": "...", "primary_organ_system": "...", "severity": "...", "likely_causal_drug": "...", "category_tags": [...], "rationale": "..."}}]}}

No prose, no markdown fences, no text outside that JSON object.
"""


# --------------------------------------------------------------------------
# Groq call (retried, cached per-batch on disk)
# --------------------------------------------------------------------------
class GroqExtractionError(Exception):
    pass


def _client() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError(
            "GROQ_API_KEY not set. Get a free key at https://console.groq.com/keys "
            "and paste it into your .env file."
        )
    return Groq(api_key=config.GROQ_API_KEY)


@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
def _call_groq(client: Groq, batch: list[dict[str, Any]]) -> dict[str, Any]:
    user_content = json.dumps(
        [{"id": r["safetyreportid"], "summary": r["patient_summary_text"]} for r in batch]
    )
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        max_tokens=2000,
        response_format={"type": "json_object"},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_content},
        ],
    )
    raw = response.choices[0].message.content
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError as e:
        raise GroqExtractionError(f"Model did not return valid JSON: {e}\nRaw: {raw[:300]}")
    if "reports" not in parsed:
        raise GroqExtractionError(f"Missing 'reports' key in response: {raw[:300]}")
    return parsed


def _batch_cache_path(batch: list[dict[str, Any]]) -> Path:
    ids = sorted(r["safetyreportid"] for r in batch)
    digest = hashlib.sha256(json.dumps(ids).encode()).hexdigest()[:24]
    return BATCHES_DIR / f"{digest}.json"


# --------------------------------------------------------------------------
# Progress tracking
# --------------------------------------------------------------------------
def _already_done_ids() -> set[str]:
    done: set[str] = set()
    for f in BATCHES_DIR.glob("*.json"):
        try:
            data = json.loads(f.read_text())
            done.update(r["id"] for r in data.get("reports", []))
        except (json.JSONDecodeError, KeyError):
            log.warning(f"Skipping unreadable cache file: {f}")
    return done


def _load_pending(max_per_drug: int | None = None) -> pd.DataFrame:
    if not SUMMARIES_PATH.exists():
        raise FileNotFoundError(f"{SUMMARIES_PATH} not found. Run Phase 1a `reports` first.")
    df = pd.read_parquet(SUMMARIES_PATH)
    df["safetyreportid"] = df["safetyreportid"].astype(str)
    done_ids = _already_done_ids()
    pending = df[~df["safetyreportid"].isin(done_ids)]

    if max_per_drug is not None:
        # Stratified sample: same cap per drug, so no single drug dominates
        # the categorized set. Deterministic (random_state) so reruns keep
        # sampling the same reports rather than drifting each time.
        # (Built with a plain loop + concat rather than groupby().apply() —
        # newer pandas versions silently drop the grouping column there.)
        sampled_parts = [
            g.sample(n=min(len(g), max_per_drug), random_state=42)
            for _, g in pending.groupby("drug")
        ]
        pending = pd.concat(sampled_parts, ignore_index=True) if sampled_parts else pending
        log.info(f"Sampled down to {max_per_drug}/drug -> {len(pending)} reports to categorize.")

    log.info(f"{len(done_ids)} reports already categorized, {len(pending)} pending.")
    return pending


# --------------------------------------------------------------------------
# Subcommand: run
# --------------------------------------------------------------------------
def run_extraction(batch_size: int, sleep_seconds: float, max_per_drug: int | None) -> None:
    pending = _load_pending(max_per_drug)
    if pending.empty:
        log.info("Nothing pending — everything is already categorized. Run `merge` if needed.")
        return

    client = _client()
    records = pending.to_dict("records")
    batches = [records[i : i + batch_size] for i in range(0, len(records), batch_size)]

    for batch in tqdm(batches, desc="Categorizing batches"):
        cache_path = _batch_cache_path(batch)
        if cache_path.exists():
            continue  # already processed in a previous run

        try:
            result = _call_groq(client, batch)
        except Exception as e:  # noqa: BLE001
            log.error(f"Batch failed permanently after retries, skipping for now: {e}")
            continue  # move on; rerunning `run` later will retry this batch

        cache_path.write_text(json.dumps(result))
        time.sleep(sleep_seconds)  # stay under Groq's tokens/min free-tier limit

    merge_results()


# --------------------------------------------------------------------------
# Subcommand: merge
# --------------------------------------------------------------------------
SEVERITY_NORMALIZATION = {
    # The LLM didn't strictly follow the 5-value controlled vocabulary from
    # the system prompt (a known LLM-categorization failure mode: schema
    # drift under free-form generation). Normalize the stray values here
    # rather than in the prompt, so this is auditable and reproducible.
    "hospitalization": "severe",
    "non-serious": "mild",
    "non-serious/other": "mild",
    "unknown": "moderate",
}


def _normalize_severity(value: str) -> str:
    v = str(value).strip().lower()
    return SEVERITY_NORMALIZATION.get(v, v)


def merge_results() -> None:
    if not SUMMARIES_PATH.exists():
        log.warning("No patient_summaries.parquet found yet — nothing to merge into.")
        return

    all_rows = []
    for f in BATCHES_DIR.glob("*.json"):
        data = json.loads(f.read_text())
        all_rows.extend(data.get("reports", []))

    if not all_rows:
        log.info("No categorized batches yet.")
        return

    cat_df = pd.DataFrame(all_rows).rename(columns={"id": "safetyreportid"})
    cat_df["safetyreportid"] = cat_df["safetyreportid"].astype(str)
    cat_df = cat_df.drop_duplicates(subset="safetyreportid")

    before = cat_df["severity"].copy()
    cat_df["severity"] = cat_df["severity"].apply(_normalize_severity)
    n_fixed = (before.str.lower().str.strip() != cat_df["severity"]).sum()
    if n_fixed:
        log.info(f"Normalized {n_fixed} non-standard severity labels (see SEVERITY_NORMALIZATION).")

    summaries = pd.read_parquet(SUMMARIES_PATH)
    summaries["safetyreportid"] = summaries["safetyreportid"].astype(str)

    merged = summaries.merge(cat_df, on="safetyreportid", how="inner")
    merged.to_parquet(FINAL_PATH, index=False)
    log.info(f"Merged {len(merged)} categorized reports -> {FINAL_PATH}")


# --------------------------------------------------------------------------
# Subcommand: status
# --------------------------------------------------------------------------
def print_status() -> None:
    if not SUMMARIES_PATH.exists():
        log.info("Phase 1a reports not run yet — nothing to categorize.")
        return
    total = len(pd.read_parquet(SUMMARIES_PATH))
    done = len(_already_done_ids())
    pct = (done / total * 100) if total else 0
    log.info(f"Categorized {done}/{total} reports ({pct:.1f}%).")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 1b — LLM clinical categorization")
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="Categorize all pending reports")
    p_run.add_argument("--batch-size", type=int, default=20)
    p_run.add_argument(
        "--sleep", type=float, default=12.0,
        help="Seconds between batches. Increase if you see 429 rate-limit errors.",
    )
    p_run.add_argument(
        "--max-per-drug", type=int, default=None,
        help="Cap reports categorized per drug (stratified sample). "
             "E.g. --max-per-drug 350 categorizes ~2,800 total instead of everything.",
    )

    sub.add_parser("status", help="Show categorization progress")
    sub.add_parser("merge", help="Rebuild categorized_reports.parquet from cached batches")

    args = parser.parse_args()

    if args.command == "run":
        run_extraction(args.batch_size, args.sleep, args.max_per_drug)
    elif args.command == "status":
        print_status()
    elif args.command == "merge":
        merge_results()


if __name__ == "__main__":
    main()