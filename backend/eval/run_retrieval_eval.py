"""Run a small offline retrieval benchmark over the imported eval corpus."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, List

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env")

from services.database_service import get_db_service
from eval.retrieval_ranking import rank_chunks, rank_documents

DEFAULT_QUERY_PATH = Path(__file__).with_name("retrieval_smoke_queries.jsonl")


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    queries = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            queries.append(json.loads(line))
    return queries


def _contains_expected_terms(chunk: Dict[str, Any], expected_terms: Iterable[str]) -> bool:
    content = f"{chunk.get('title', '')}\n{chunk.get('content', '')}"
    return all(term.lower() in content.lower() for term in expected_terms)


def _first_relevant_rank(
    chunks: List[Dict[str, Any]],
    expected_document_ids: Iterable[str],
    expected_terms: Iterable[str],
) -> int | None:
    expected_docs = set(expected_document_ids)
    for rank, chunk in enumerate(chunks, 1):
        if chunk.get("document_id") not in expected_docs:
            continue
        if _contains_expected_terms(chunk, expected_terms):
            return rank
    return None


def run_eval(query_path: Path, top_k: int) -> Dict[str, Any]:
    db = get_db_service()
    queries = _load_queries(query_path)
    documents = db.list_ready_documents_for_retrieval()
    chunks = db.get_chunks_by_documents([doc["id"] for doc in documents])
    chunks = [chunk for chunk in chunks if chunk.get("chunk_type") != "parent"]

    results = []
    doc_hits = 0
    chunk_hits = 0
    reciprocal_ranks = []
    category_buckets: Dict[str, Dict[str, float]] = {}

    for item in queries:
        ranked_documents = rank_documents(item["query"], documents, top_k=top_k)
        ranked_chunks = rank_chunks(item["query"], chunks, top_k=top_k)

        expected_docs = set(item["expected_document_ids"])
        returned_doc_ids = [doc["id"] for doc in ranked_documents]
        doc_hit = bool(expected_docs.intersection(returned_doc_ids))
        first_rank = _first_relevant_rank(
            ranked_chunks,
            expected_docs,
            item["expected_terms"],
        )
        chunk_hit = first_rank is not None

        doc_hits += int(doc_hit)
        chunk_hits += int(chunk_hit)
        reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)
        bucket = category_buckets.setdefault(
            item["category"],
            {"queries": 0, "document_hits": 0, "chunk_hits": 0, "reciprocal_rank_sum": 0.0},
        )
        bucket["queries"] += 1
        bucket["document_hits"] += int(doc_hit)
        bucket["chunk_hits"] += int(chunk_hit)
        bucket["reciprocal_rank_sum"] += 0.0 if first_rank is None else 1.0 / first_rank
        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "document_hit": doc_hit,
                "chunk_hit": chunk_hit,
                "first_relevant_rank": first_rank,
                "top_documents": returned_doc_ids,
                "top_chunks": [chunk["id"] for chunk in ranked_chunks],
            }
        )

    total = len(queries)
    summary = {
        "queries": total,
        "documents": len(documents),
        "chunks": len(chunks),
        f"document_hit@{top_k}": round(doc_hits / total, 4) if total else 0.0,
        f"chunk_hit@{top_k}": round(chunk_hits / total, 4) if total else 0.0,
        "mrr": round(sum(reciprocal_ranks) / total, 4) if total else 0.0,
        "by_category": {
            category: {
                "queries": int(bucket["queries"]),
                f"document_hit@{top_k}": round(bucket["document_hits"] / bucket["queries"], 4),
                f"chunk_hit@{top_k}": round(bucket["chunk_hits"] / bucket["queries"], 4),
                "mrr": round(bucket["reciprocal_rank_sum"] / bucket["queries"], 4),
            }
            for category, bucket in sorted(category_buckets.items())
        },
    }
    return {"summary": summary, "results": results}


def main() -> None:
    os.environ.setdefault("DOCUMENT_EMBEDDING_PROVIDER", "local_hash")
    os.environ.setdefault("DOCUMENT_VECTOR_STORE", "sqlite")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERY_PATH)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("retrieval_eval_results.json"))
    args = parser.parse_args()

    result = run_eval(args.queries, args.top_k)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
