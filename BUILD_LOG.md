# Build Log — FDA Adverse Event Surveillance & Root-Cause RAG System

Living document. Fill in metrics as you run each phase — this becomes your
resume bullet source material.

---

## Setup (one-time)

- [ ] `python -m venv venv && source venv/bin/activate`
- [ ] `pip install -r requirements.txt`
- [ ] `cp .env.example .env` → paste `OPENAI_API_KEY`
- [ ] (Optional, recommended) Get a free openFDA API key at
      https://open.fda.gov/apis/authentication/ → paste into `.env` as
      `OPENFDA_API_KEY` (raises daily limit from 1,000 → 120,000 requests)
- [ ] Pick 5–10 drugs to analyze. Good starting set (mix of common +
      known real safety-signal history, useful for Phase 5 ground truth):
      `metformin lisinopril atorvastatin ibuprofen sertraline
      hydrochlorothiazide amoxicillin omeprazole`

---

## Phase 1a — Data Ingestion

**Status:** code complete (`src/data_ingestion.py`), not yet run.

**What it pulls:**
- `counts` subcommand → openFDA's aggregated `count=receivedate` endpoint,
  directly giving daily serious-AE counts per drug (this IS the Phase 3
  forecasting target, pulled in a handful of API calls).
- `reports` subcommand → individual case reports (bounded per-drug limit),
  flattened into `patient_summary_text` for Phase 1b + Phase 2.

**How to run:**
```
python src/data_ingestion.py counts  --drugs metformin lisinopril atorvastatin --start 2022-01-01 --end 2024-12-31
python src/data_ingestion.py reports --drugs metformin lisinopril atorvastatin --start 2022-01-01 --end 2024-12-31 --per-drug-limit 2000
```

**Metrics to fill in:**
- [ ] Drugs pulled: ___
- [ ] Date range: ___
- [ ] Total serious-AE report-weeks (counts): ___
- [ ] Total individual case reports pulled: ___

---

## Phase 1b — LLM Clinical Categorization

**Status:** code complete (`src/extraction.py`), not yet run.

**Model/API:** `gpt-4o-mini` via OpenAI Batch API, strict JSON-schema
structured outputs, 15 reports/request, checkpointed (idempotent — safe to
run `submit → status → fetch` across multiple sessions).

**How to run:**
```
python src/extraction.py submit
python src/extraction.py status     # poll until completed
python src/extraction.py fetch
```

**Metrics to fill in:**
- [ ] Reports categorized: ___
- [ ] Actual $ cost (OpenAI usage dashboard): ___
- [ ] Categorization accuracy vs. ~100 hand-labeled reports (Phase 5): ___

---

## Phase 2 — Vector Space Construction
**Status:** not started.

## Phase 3 — Forecasting & Anomaly Detection
**Status:** not started.

## Phase 4 — Explanatory RAG Diagnostics
**Status:** not started.

## Phase 5 — Evaluation Framework
**Status:** not started.
**Reminder:** cross-check flagged anomalies against real FDA safety
communications / label changes for the chosen drugs — searchable at
https://www.fda.gov/drugs/drug-safety-and-availability — this is your
independently checkable ground truth, stronger than an informal eyeball
check.

## Phase 6 — Streamlit Dashboard
**Status:** not started.

---

## Resume bullet drafts (fill in once metrics exist)

- Built an end-to-end drug-safety surveillance pipeline over real FDA FAERS
  data (via openFDA API), combining LLM structured categorization,
  Prophet/ARIMA forecasting, and unsupervised anomaly detection (Z-score +
  Isolation Forest) to flag emerging safety signals across **___ drugs**
  and **___ reports**.
- Designed a cost-optimized, rate-limit-proof LLM categorization pipeline
  (OpenAI Batch API + structured outputs) achieving **___% accuracy** at
  **<$___** total cost.
- Built a RAG-based root-cause explainer triggered automatically on
  statistical anomalies, validated against real FDA safety communications
  (**___% precision/recall**) and evaluated with RAGAS (**___
  faithfulness**).
