# FDA Adverse Event Surveillance & RAG Diagnostics System

An end-to-end drug safety surveillance pipeline built on **real FDA adverse event data**
(openFDA / FAERS). It categorizes reports with an LLM, detects statistically anomalous
spikes in report volume, and uses RAG to generate grounded, plain-English explanations for
each anomaly — then evaluates the whole system with quantified metrics.

**[Live Demo](#)** ← *add your Streamlit Cloud URL here once deployed*

---

## Why this project exists

Anyone can look up what a drug does. The point of this system is different: it surfaces
**statistical anomalies that nobody would think to search for in the first place**. "Is
there an unusual spike in amoxicillin liver injury reports this September?" isn't a
Google-able question until something flags it first. That's the same discipline real FDA
pharmacovigilance signal-detection uses (disproportionality analysis) — this project
reimplements that idea end-to-end on real data.

## Two findings that validate the approach

The pipeline flagged these anomalies **independently**, before any manual verification:

1. **Amoxicillin + clavulanic acid → liver injury** (anomaly: Nov 22, 2022). Matches NIH's
   LiverTox resource, which identifies amoxicillin-clavulanate as the leading documented
   cause of clinically apparent drug-induced liver injury in the US/Europe.
2. **Hydrochlorothiazide → non-melanoma skin cancer** (anomaly: Oct 22, 2024). Matches a
   real FDA Drug Safety Communication (Aug 20, 2020) on increased skin cancer risk with
   HCTZ at cumulative dose.

Confirmation came *after* the system flagged the pattern — not the other way around.

## Results

| Metric | Score |
|---|---|
| Categorization accuracy — organ system | 69.8% |
| Categorization accuracy — severity | 66.7% |
| Anomaly precision | 72.7% (16/22 confidently-judged) |
| RAG faithfulness (LLM-as-judge) | 0.68 (n=37) |
| High-confidence anomalies detected | 37 across 8 drugs |
| Reports processed | 12,986 (2022–2024) |

Full methodology, including honest caveats on how these numbers were derived, is in
[`BUILD_LOG.md`](BUILD_LOG.md).

## Architecture

```
openFDA API (raw reports + daily counts)
        │
        ▼
Phase 1  →  Ingestion & LLM categorization (Groq Llama 3.1 8B)
        │        organ system · severity · causal drug · tags
        ▼
Phase 2  →  Vector store (sentence-transformers + ChromaDB)
        │        local embeddings, semantic + metadata retrieval
        ▼
Phase 3  →  Anomaly detection (Holt-Winters z-score + Isolation Forest ensemble)
        │        high-confidence = both methods agree
        ▼
Phase 4  →  RAG diagnostics (live targeted retrieval + Groq)
        │        grounded plain-English explanation per anomaly
        ▼
Phase 5  →  Evaluation (categorization accuracy · anomaly precision · RAG faithfulness)
        ▼
Phase 6  →  Streamlit dashboard
```

## Tech stack

| Layer | Choice | Why |
|---|---|---|
| LLM | Groq `llama-3.1-8b-instant` | Free tier, fast, avoided cost/rate-limit issues that stalled an earlier OpenAI-based attempt |
| Embeddings | `sentence-transformers` (`all-MiniLM-L6-v2`) | Runs locally on CPU — zero API calls, zero rate limits |
| Vector store | ChromaDB | Local, persistent, free |
| Forecasting | Holt-Winters (statsmodels) | Same trend/seasonality idea as Prophet, without the compilation headaches |
| Anomaly detection | Isolation Forest + rolling z-score ensemble | Two independent signals reduce false positives |
| Dashboard | Streamlit + Plotly | Fast to build, easy to deploy |

## Project structure

```
fda-ae-surveillance/
├── config.py                  # central config — all phases import from this
├── BUILD_LOG.md                # full build history, bugs found & fixed, final metrics
├── src/
│   ├── data_ingestion.py       # Phase 1a — openFDA ingestion
│   ├── extraction.py            # Phase 1b — LLM categorization
│   ├── vector_store.py          # Phase 2  — embeddings + ChromaDB
│   ├── forecasting.py           # Phase 3  — anomaly detection
│   ├── rag_diagnostics.py       # Phase 4  — RAG explanations
│   └── evaluation.py            # Phase 5  — evaluation framework
├── app/
│   └── dashboard.py             # Phase 6  — Streamlit dashboard
├── data/processed/              # processed datasets + vector store (committed for demo)
└── outputs/                     # figures + evaluation reports
```

## Running it locally

```bash
git clone https://github.com/<your-username>/fda-ae-surveillance.git
cd fda-ae-surveillance
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Add your own keys
cp .env.example .env            # then fill in OPENFDA_API_KEY and GROQ_API_KEY

streamlit run app/dashboard.py
```

The dashboard works out of the box using the committed processed data in
`data/processed/` — you only need API keys if you want to re-run the pipeline phases
yourself (`src/data_ingestion.py`, `src/extraction.py`, etc.).

## Dashboard

Four tabs:
- **Overview** — KPIs, organ system & severity distributions
- **Anomaly Timeline** — actual vs. expected vs. flagged anomalies, per drug (interactive)
- **Anomaly Explanations** — filterable, RAG-generated explanation cards
- **Evaluation Metrics** — all Phase 5 numbers in one place

## Known limitations

- Categorization accuracy (~70%) reflects a free, zero-shot 8B-parameter model with no
  few-shot examples — honest for the setup, not a claim of clinical-grade accuracy.
- Anomaly precision is precision only, not recall — no independent full-signal list exists
  to check recall against; 15/37 anomalies were excluded as genuinely ambiguous rather than
  forced into a score.
- Ground-truth labeling for evaluation was done via an independent rule-based/LLM
  cross-check, not formal human annotation. Full methodology and honesty caveats are in
  [`BUILD_LOG.md`](BUILD_LOG.md).

## License

MIT
