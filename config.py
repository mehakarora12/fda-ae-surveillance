"""
Central configuration for the FDA Adverse Event Surveillance project.

Edit DRUGS / START_DATE / END_DATE here ONCE — every phase (1a, 1b, 2, 3, 4)
imports from this file, so your whole project scope stays in sync.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------- Paths ----
ROOT_DIR = Path(__file__).resolve().parent
DATA_DIR = ROOT_DIR / "data"

RAW_COUNTS_DIR = DATA_DIR / "raw" / "counts"
RAW_REPORTS_DIR = DATA_DIR / "raw" / "reports"
PROCESSED_DIR = DATA_DIR / "processed"
CACHE_DIR = DATA_DIR / "cache"

for _d in (RAW_COUNTS_DIR, RAW_REPORTS_DIR, PROCESSED_DIR, CACHE_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------- openFDA ---
OPENFDA_API_KEY = os.getenv("OPENFDA_API_KEY", "").strip()
OPENFDA_BASE_URL = "https://api.fda.gov/drug/event.json"

# openFDA published limits: 240 req/min & 120,000 req/day WITH a key,
# 40 req/min & 1,000 req/day WITHOUT one. We pick a safe sleep interval
# per request accordingly so we never get 429'd.
OPENFDA_SLEEP_SECONDS = 0.3 if OPENFDA_API_KEY else 1.6

# ------------------------------------------------- Project scope (edit me) -
DRUGS = [
    "metformin",
    "lisinopril",
    "atorvastatin",
    "ibuprofen",
    "sertraline",
    "hydrochlorothiazide",
    "amoxicillin",
    "omeprazole",
]
START_DATE = "2022-01-01"   # YYYY-MM-DD, converted to openFDA format internally
END_DATE = "2024-12-31"

# ------------------------------------------- LLM (used from Phase 1b on) ---
# Groq free tier: llama-3.1-8b-instant gives 30 req/min, 14,400 req/day,
# 6,000 tokens/min -- fast enough and generous enough that a categorization
# job over a few thousand reports finishes in one sitting without 429s.
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "").strip()
LLM_MODEL = "llama-3.1-8b-instant"
