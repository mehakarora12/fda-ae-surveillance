"""
Phase 2 — Vector Space Construction
=====================================
Embeds every categorized report's `patient_summary_text` into a vector
using a local, free embedding model (no API, no rate limits — a welcome
change after Phase 1b), and stores it in ChromaDB alongside its
structured metadata (drug, organ system, severity, tags).

Why this design:
  - Local embeddings (sentence-transformers, CPU) mean this phase is
    bottlenecked only by your machine, not a free-tier ceiling.
  - Storing rich metadata alongside each vector enables HYBRID retrieval
    in Phase 4: semantic similarity search *combined with* structured
    filters (e.g. "similar to this report, but only severe cardiovascular
    cases") — this is the detail worth mentioning in interviews, plain
    similarity search alone is a much weaker RAG design.
  - `upsert` (not `add`) makes reruns idempotent: safe to stop and
    re-run the build without duplicating vectors.

Usage
-----
    python src/vector_store.py build
    python src/vector_store.py status
    python src/vector_store.py query --text "severe liver damage after starting medication" --k 5
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import chromadb
import pandas as pd
from tqdm import tqdm

sys.path.append(str(Path(__file__).resolve().parent.parent))
import config  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
log = logging.getLogger("vector_store")

CATEGORIZED_PATH = config.PROCESSED_DIR / "categorized_reports.parquet"
VECTOR_STORE_DIR = config.DATA_DIR / "vector_store"
VECTOR_STORE_DIR.mkdir(parents=True, exist_ok=True)
COLLECTION_NAME = "adverse_event_reports"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"  # 384-dim, fast on CPU, free, no API key needed
EMBED_BATCH_SIZE = 64


# --------------------------------------------------------------------------
# Model + client (lazy-loaded so `status`/`query` don't pay model load cost
# unless they actually need it)
# --------------------------------------------------------------------------
_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer

        log.info(f"Loading local embedding model '{EMBEDDING_MODEL_NAME}' (first run downloads it once)...")
        _model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    return _model


def _get_collection():
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_DIR))
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},  # cosine similarity fits sentence embeddings best
    )


def _safe_join_tags(value) -> str:
    """
    category_tags round-trips through parquet as a numpy array, not a list —
    `value or []` throws on numpy arrays because truthiness of an array with
    more than one element is ambiguous. Check for None explicitly instead.
    """
    if value is None:
        return ""
    return "; ".join(str(t) for t in value)


# --------------------------------------------------------------------------
# Subcommand: build
# --------------------------------------------------------------------------
def build_vector_store() -> None:
    if not CATEGORIZED_PATH.exists():
        raise FileNotFoundError(f"{CATEGORIZED_PATH} not found. Run Phase 1b `merge` first.")

    df = pd.read_parquet(CATEGORIZED_PATH)
    df["safetyreportid"] = df["safetyreportid"].astype(str)
    collection = _get_collection()

    # Skip reports already embedded (idempotent resume, same pattern as Phase 1a/1b)
    existing_ids = set(collection.get(include=[])["ids"])
    pending = df[~df["safetyreportid"].isin(existing_ids)]
    log.info(f"{len(existing_ids)} reports already embedded, {len(pending)} pending.")
    if pending.empty:
        log.info("Nothing to do — vector store is already up to date.")
        return

    model = _get_model()
    records = pending.to_dict("records")

    for i in tqdm(range(0, len(records), EMBED_BATCH_SIZE), desc="Embedding batches"):
        batch = records[i : i + EMBED_BATCH_SIZE]
        texts = [r["patient_summary_text"] for r in batch]

        vectors = model.encode(texts, show_progress_bar=False, normalize_embeddings=True).tolist()

        collection.upsert(
            ids=[r["safetyreportid"] for r in batch],
            embeddings=vectors,
            documents=texts,
            metadatas=[
                {
                    "drug": r.get("drug", ""),
                    "primary_organ_system": r.get("primary_organ_system", ""),
                    "severity": r.get("severity", ""),
                    "likely_causal_drug": r.get("likely_causal_drug", ""),
                    "category_tags": _safe_join_tags(r.get("category_tags")),
                    "receivedate": str(r.get("receivedate", "")),
                }
                for r in batch
            ],
        )

    log.info(f"Vector store now contains {collection.count()} reports -> {VECTOR_STORE_DIR}")


# --------------------------------------------------------------------------
# Subcommand: status
# --------------------------------------------------------------------------
def print_status() -> None:
    collection = _get_collection()
    total_categorized = len(pd.read_parquet(CATEGORIZED_PATH)) if CATEGORIZED_PATH.exists() else 0
    log.info(f"Vector store: {collection.count()} / {total_categorized} categorized reports embedded.")


# --------------------------------------------------------------------------
# Subcommand: query (manual sanity-check of retrieval quality)
# --------------------------------------------------------------------------
def search(query_text: str, k: int = 5, where: dict | None = None) -> None:
    model = _get_model()
    collection = _get_collection()
    query_vector = model.encode([query_text], normalize_embeddings=True).tolist()

    results = collection.query(
        query_embeddings=query_vector,
        n_results=k,
        where=where,  # e.g. {"primary_organ_system": "hepatic"}
    )

    print(f"\nTop {k} results for: {query_text!r}\n" + "-" * 60)
    for doc, meta, dist in zip(results["documents"][0], results["metadatas"][0], results["distances"][0]):
        print(f"[similarity={1 - dist:.3f}] ({meta['drug']} | {meta['primary_organ_system']} | {meta['severity']})")
        print(f"  {doc}\n")


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(description="Phase 2 — vector space construction")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("build", help="Embed all pending categorized reports into the vector store")
    sub.add_parser("status", help="Show how many reports are embedded")

    p_query = sub.add_parser("query", help="Run a test semantic search")
    p_query.add_argument("--text", required=True)
    p_query.add_argument("--k", type=int, default=5)
    p_query.add_argument("--organ-system", default=None, help="Optional metadata filter")

    args = parser.parse_args()

    if args.command == "build":
        build_vector_store()
    elif args.command == "status":
        print_status()
    elif args.command == "query":
        where = {"primary_organ_system": args.organ_system} if args.organ_system else None
        search(args.text, args.k, where)


if __name__ == "__main__":
    main()