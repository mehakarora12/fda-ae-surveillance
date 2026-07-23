"""
Phase 5 — Evaluation Framework
================================
Turns the pipeline's output into the three quantified metrics your resume
bullets need:

1. Categorization accuracy — Phase 1b's LLM tags vs. a hand-labeled sample.
2. Anomaly precision — Phase 3's flagged anomalies vs. real, independently
   verifiable FDA safety signals.
3. RAG faithfulness — does Phase 4's generated explanation actually follow
   from the retrieved reports, or does it drift/hallucinate?

Design note: (1) and (2) need a human in the loop — that's not a limitation
to route around, it's the correct design. An LLM grading its own categorization
labels would be circular evidence, not evaluation. This script generates the
labeling templates and scores them once you've filled them in; it never
scores itself using itself.

(3) IS automatable: an LLM-as-judge checking claim-vs-source support is a
legitimate, standard technique (the same idea behind RAGAS's faithfulness
metric) and doesn't require grading its own prior output blind.

Usage
-----
    python src/evaluation.py sample-for-labeling --n 100
    # ... open outputs/reports/categorization_labeling_template.csv, fill in
    # true_organ_system / true_severity by reading patient_summary_text ...
    python src/evaluation.py score-categorization

    python src/evaluation.py sample-anomalies
    # ... open outputs/reports/anomaly_ground_truth_template.csv, cross-check
    # each anomaly against https://www.fda.gov/drugs/drug-safety-and-availability ...
    python src/evaluation.py score-anomalies

    python src/evaluation.py score-faithfulness
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
import rag_diagnostics as rd  # noqa: E402 — reuse the same live context-retrieval

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("evaluation")

CATEGORIZED_PATH = config.PROCESSED_DIR / "categorized_reports.parquet"
ANOMALY_EXPLANATIONS_PATH = config.PROCESSED_DIR / "anomaly_explanations.csv"

REPORTS_DIR = config.ROOT_DIR / "outputs" / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)
LABELING_TEMPLATE_PATH = REPORTS_DIR / "categorization_labeling_template.csv"
ANOMALY_GT_TEMPLATE_PATH = REPORTS_DIR / "anomaly_ground_truth_template.csv"
CATEGORIZATION_SCORE_PATH = REPORTS_DIR / "categorization_accuracy.txt"
ANOMALY_SCORE_PATH = REPORTS_DIR / "anomaly_precision.txt"

FAITHFULNESS_DIR = config.PROCESSED_DIR / "faithfulness_scores"
FAITHFULNESS_DIR.mkdir(parents=True, exist_ok=True)
FAITHFULNESS_CSV = config.PROCESSED_DIR / "faithfulness_scores.csv"

# Verified during this project — see NIH LiverTox: amoxicillin-clavulanate is
# the leading documented cause of clinically apparent drug-induced liver
# injury in the US/Europe, with onset days-to-weeks after starting therapy.
# This seeds row 2 of the anomaly ground-truth template as a real, citable
# example of how to fill in the rest.
SEEDED_GROUND_TRUTH = {
    ("amoxicillin", "2022-11-22"): {
        "is_real_signal": "Y",
        "fda_reference_url": "https://www.ncbi.nlm.nih.gov/books/NBK548517/ (NIH LiverTox)",
        "notes": "Amoxicillin-clavulanate is documented as the leading cause of "
                 "clinically apparent drug-induced liver injury in the US/Europe "
                 "(NIH LiverTox). Matches this anomaly's cluster of DILI + TEN reports.",
    }
}


# --------------------------------------------------------------------------
# 1. Categorization accuracy (Phase 1b LLM tags vs. hand-labeled sample)
# --------------------------------------------------------------------------
def sample_for_labeling(n: int, seed: int = 42) -> None:
    if not CATEGORIZED_PATH.exists():
        raise FileNotFoundError(f"{CATEGORIZED_PATH} not found. Run Phase 1b first.")
    df = pd.read_parquet(CATEGORIZED_PATH)

    # Stratified per-drug sample (loop + concat, not groupby().apply() —
    # newer pandas drops the grouping column there, a bug we already hit once
    # in Phase 1b's sampling code)
    n_drugs = df["drug"].nunique()
    per_drug = max(1, n // n_drugs)
    parts = [g.sample(n=min(len(g), per_drug), random_state=seed) for _, g in df.groupby("drug")]
    sample = pd.concat(parts, ignore_index=True)

    out = sample[["safetyreportid", "drug", "patient_summary_text", "primary_organ_system", "severity"]].rename(
        columns={"primary_organ_system": "predicted_organ_system", "severity": "predicted_severity"}
    )
    out["true_organ_system"] = ""
    out["true_severity"] = ""
    out["notes"] = ""
    out.to_csv(LABELING_TEMPLATE_PATH, index=False)

    log.info(f"Wrote {len(out)} reports to label -> {LABELING_TEMPLATE_PATH}")
    log.info(
        "Open this in Excel. For each row, read patient_summary_text and fill in "
        "true_organ_system / true_severity using the SAME controlled vocabulary the "
        "model uses (see ORGAN_SYSTEMS / SEVERITIES in src/extraction.py). Leave blank "
        "to skip a row. Then run `score-categorization`."
    )


def score_categorization() -> None:
    if not LABELING_TEMPLATE_PATH.exists():
        raise FileNotFoundError("Run `sample-for-labeling` first.")
    df = pd.read_csv(LABELING_TEMPLATE_PATH, dtype=str).fillna("")

    labeled = df[df["true_organ_system"].str.strip() != ""].copy()
    if labeled.empty:
        log.warning("No rows labeled yet — fill in true_organ_system/true_severity and rerun.")
        return

    def _norm(s: pd.Series) -> pd.Series:
        return s.str.strip().str.lower()

    organ_acc = (_norm(labeled["true_organ_system"]) == _norm(labeled["predicted_organ_system"])).mean()

    severity_labeled = labeled[labeled["true_severity"].str.strip() != ""]
    severity_acc = (
        (_norm(severity_labeled["true_severity"]) == _norm(severity_labeled["predicted_severity"])).mean()
        if not severity_labeled.empty else float("nan")
    )

    mismatches = labeled[_norm(labeled["true_organ_system"]) != _norm(labeled["predicted_organ_system"])]

    report = (
        f"Categorization Accuracy Report\n{'=' * 40}\n"
        f"Labeled sample size: {len(labeled)}\n"
        f"Organ system accuracy: {organ_acc:.1%}\n"
        f"Severity accuracy: {severity_acc:.1%}\n\n"
        f"Mismatches ({len(mismatches)}):\n"
    )
    for _, row in mismatches.head(20).iterrows():
        report += f"  - {row['safetyreportid']}: predicted='{row['predicted_organ_system']}' true='{row['true_organ_system']}'\n"

    CATEGORIZATION_SCORE_PATH.write_text(report)
    print(report)
    log.info(f"Saved -> {CATEGORIZATION_SCORE_PATH}")


# --------------------------------------------------------------------------
# 2. Anomaly precision (Phase 3 flags vs. real FDA safety signals)
# --------------------------------------------------------------------------
def sample_anomalies_for_ground_truth() -> None:
    if not ANOMALY_EXPLANATIONS_PATH.exists():
        raise FileNotFoundError(f"{ANOMALY_EXPLANATIONS_PATH} not found. Run Phase 4 first.")
    df = pd.read_csv(ANOMALY_EXPLANATIONS_PATH, dtype=str)

    df["is_real_signal"] = ""
    df["fda_reference_url"] = ""
    df["notes"] = ""

    for i, row in df.iterrows():
        key = (row["drug"], row["date"])
        if key in SEEDED_GROUND_TRUTH:
            df.loc[i, list(SEEDED_GROUND_TRUTH[key].keys())] = list(SEEDED_GROUND_TRUTH[key].values())

    df.to_csv(ANOMALY_GT_TEMPLATE_PATH, index=False)
    n_seeded = len(SEEDED_GROUND_TRUTH)
    log.info(f"Wrote {len(df)} anomalies to label -> {ANOMALY_GT_TEMPLATE_PATH} ({n_seeded} pre-filled as an example)")
    log.info(
        "For each remaining row, check https://www.fda.gov/drugs/drug-safety-and-availability "
        "(and general medical literature search) for that drug around that date. Mark "
        "is_real_signal as Y/N. Then run `score-anomalies`."
    )


def score_anomalies() -> None:
    if not ANOMALY_GT_TEMPLATE_PATH.exists():
        raise FileNotFoundError("Run `sample-anomalies` first.")
    df = pd.read_csv(ANOMALY_GT_TEMPLATE_PATH, dtype=str).fillna("")

    labeled = df[df["is_real_signal"].str.strip().str.upper().isin(["Y", "N"])]
    if labeled.empty:
        log.warning("No rows labeled yet — fill in is_real_signal (Y/N) and rerun.")
        return

    n_true = (labeled["is_real_signal"].str.strip().str.upper() == "Y").sum()
    precision = n_true / len(labeled)

    report = (
        f"Anomaly Precision Report\n{'=' * 40}\n"
        f"Labeled anomalies: {len(labeled)} / {len(df)} total flagged\n"
        f"Confirmed real signals: {n_true}\n"
        f"Precision: {precision:.1%}\n\n"
        f"Note: this measures precision (of flagged anomalies, how many are real), "
        f"not recall (we don't have an independent list of ALL real signals in this "
        f"window to check against) — be precise about which metric you cite.\n"
    )
    ANOMALY_SCORE_PATH.write_text(report)
    print(report)
    log.info(f"Saved -> {ANOMALY_SCORE_PATH}")


# --------------------------------------------------------------------------
# 3. RAG faithfulness (LLM-as-judge — automated, no self-grading)
# --------------------------------------------------------------------------
def _client() -> Groq:
    if not config.GROQ_API_KEY:
        raise RuntimeError("GROQ_API_KEY not set.")
    return Groq(api_key=config.GROQ_API_KEY)


JUDGE_PROMPT = """You are a strict fact-checker reviewing a pharmacovigilance analyst's CLINICAL \
claims against a sample of real case reports.

IMPORTANT CONTEXT — do not penalize for these, they are correct by design, not errors:
- The sources are a sample of reports from a period of several days AROUND the event date,
  not necessarily from the exact date itself. A source dated a few days before or after is
  still valid supporting context.
- You are being given ONLY the qualitative/clinical claims (organ systems affected, severity,
  notable patterns) — NOT any report counts, dates, or statistical figures, because those come
  from a separate aggregate database, not from these individual sample reports. Do not comment
  on missing counts or dates; that is out of scope for this check.

Your ONLY job: does the sample of reports support the CLINICAL content of the claim (the organ
systems, severity level, and notable pattern described)?

Respond ONLY with JSON: {"score": <float 0.0-1.0>, "justification": "<1 sentence>"}
1.0 = the clinical claim is directly supported by the sources.
0.5 = partially supported, or plausible but not explicitly stated.
0.0 = contradicted by the sources, or describes clinical content absent from them entirely.
"""


@retry(reraise=True, stop=stop_after_attempt(4), wait=wait_exponential(multiplier=2, min=4, max=60))
def _judge_faithfulness(client: Groq, sources: list[str], claim: str) -> dict:
    numbered = "\n".join(f"{i+1}. {s}" for i, s in enumerate(sources))
    user_content = f"SOURCES:\n{numbered}\n\nCLAIM:\n{claim}"
    response = client.chat.completions.create(
        model=config.LLM_MODEL,
        temperature=0,
        max_tokens=150,
        response_format={"type": "json_object"},
        messages=[{"role": "system", "content": JUDGE_PROMPT}, {"role": "user", "content": user_content}],
    )
    return json.loads(response.choices[0].message.content)


def _faithfulness_id(drug: str, date: str) -> str:
    return hashlib.sha256(f"{drug}_{date}".encode()).hexdigest()[:16]


def score_faithfulness() -> None:
    if not ANOMALY_EXPLANATIONS_PATH.exists():
        raise FileNotFoundError(f"{ANOMALY_EXPLANATIONS_PATH} not found. Run Phase 4 first.")
    df = pd.read_csv(ANOMALY_EXPLANATIONS_PATH)

    done_ids = {f.stem for f in FAITHFULNESS_DIR.glob("*.json")}
    client = _client()

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Judging faithfulness"):
        fid = _faithfulness_id(row["drug"], row["date"])
        if fid in done_ids:
            continue

        # Re-fetch the same context this anomaly's explanation was grounded in
        # (instant — already cached from Phase 4's run, no new API calls)
        sources = rd.get_context_reports(row["drug"], pd.Timestamp(row["date"]))
        if not sources:
            continue

        claim = (
            f"Organ systems affected: {row['dominant_organ_systems']}. "
            f"Severity: {row['dominant_severity']}. "
            f"Notable pattern: {row['notable_pattern']}"
        )
        try:
            result = _judge_faithfulness(client, sources, claim)
        except Exception as e:  # noqa: BLE001
            log.error(f"Failed to judge {row['drug']} {row['date']}: {e}")
            continue

        out = {"drug": row["drug"], "date": row["date"], **result}
        (FAITHFULNESS_DIR / f"{fid}.json").write_text(json.dumps(out))
        time.sleep(5)

    rows = [json.loads(f.read_text()) for f in FAITHFULNESS_DIR.glob("*.json")]
    if not rows:
        log.info("No faithfulness scores yet.")
        return
    scores_df = pd.DataFrame(rows)
    scores_df.to_csv(FAITHFULNESS_CSV, index=False)
    log.info(f"Mean faithfulness score: {scores_df['score'].mean():.2f} (n={len(scores_df)}) -> {FAITHFULNESS_CSV}")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 5 — evaluation framework")
    sub = parser.add_subparsers(dest="command", required=True)

    p_sample = sub.add_parser("sample-for-labeling", help="Generate categorization ground-truth template")
    p_sample.add_argument("--n", type=int, default=100)

    sub.add_parser("score-categorization", help="Score categorization accuracy from filled-in template")
    sub.add_parser("sample-anomalies", help="Generate anomaly ground-truth template")
    sub.add_parser("score-anomalies", help="Score anomaly precision from filled-in template")
    sub.add_parser("score-faithfulness", help="Score RAG faithfulness via LLM-as-judge")

    args = parser.parse_args()
    if args.command == "sample-for-labeling":
        sample_for_labeling(args.n)
    elif args.command == "score-categorization":
        score_categorization()
    elif args.command == "sample-anomalies":
        sample_anomalies_for_ground_truth()
    elif args.command == "score-anomalies":
        score_anomalies()
    elif args.command == "score-faithfulness":
        score_faithfulness()


if __name__ == "__main__":
    main()