"""Run formal retrieval evaluation against the aligned gold JSONL set."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env")

from services.database_service import get_db_service
from eval.retrieval_ranking import rank_chunks, rank_documents

DEFAULT_QUERIES = Path(__file__).with_name("gold_queries_aligned.jsonl")


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _first_rank(returned_ids: List[str], relevant_ids: List[str]) -> int | None:
    relevant = set(relevant_ids)
    for rank, item_id in enumerate(returned_ids, 1):
        if item_id in relevant:
            return rank
    return None


def run_eval(query_path: Path, top_k: int) -> Dict[str, Any]:
    db = get_db_service()
    queries = _load_queries(query_path)
    documents = db.list_ready_documents_for_retrieval()
    chunks = db.get_chunks_by_documents([doc["id"] for doc in documents])
    chunks = [chunk for chunk in chunks if chunk.get("chunk_type") != "parent"]

    category_buckets: Dict[str, Dict[str, float]] = {}
    results = []
    doc_hits = 0
    chunk_hits = 0
    reciprocal_ranks = []
    abstention_correct = 0
    abstention_total = 0

    for item in queries:
        ranked_documents = rank_documents(item["query"], documents, top_k=top_k)
        ranked_chunks = rank_chunks(item["query"], chunks, top_k=top_k)
        returned_doc_ids = [doc["id"] for doc in ranked_documents]
        returned_chunk_ids = [chunk["id"] for chunk in ranked_chunks]

        negative_case = bool(item.get("negative_case"))
        if negative_case:
            doc_hit = False
            chunk_hit = False
            first_rank = None
            abstention_total += 1
            abstention_correct += int(not item.get("relevant_document_ids") and not item.get("relevant_chunk_ids"))
        else:
            doc_hit = bool(set(item.get("relevant_document_ids", [])).intersection(returned_doc_ids))
            first_rank = _first_rank(returned_chunk_ids, item.get("relevant_chunk_ids", []))
            chunk_hit = first_rank is not None
            doc_hits += int(doc_hit)
            chunk_hits += int(chunk_hit)
            reciprocal_ranks.append(0.0 if first_rank is None else 1.0 / first_rank)

        bucket = category_buckets.setdefault(
            item["category"],
            {
                "queries": 0,
                "document_hits": 0,
                "chunk_hits": 0,
                "reciprocal_rank_sum": 0.0,
                "positive_queries": 0,
            },
        )
        bucket["queries"] += 1
        if not negative_case:
            bucket["positive_queries"] += 1
            bucket["document_hits"] += int(doc_hit)
            bucket["chunk_hits"] += int(chunk_hit)
            bucket["reciprocal_rank_sum"] += 0.0 if first_rank is None else 1.0 / first_rank

        results.append(
            {
                "id": item["id"],
                "category": item["category"],
                "negative_case": negative_case,
                "document_hit": doc_hit,
                "chunk_hit": chunk_hit,
                "first_relevant_rank": first_rank,
                "top_documents": returned_doc_ids,
                "top_chunks": returned_chunk_ids,
            }
        )

    positive_total = sum(1 for item in queries if not item.get("negative_case"))
    summary = {
        "queries": len(queries),
        "positive_queries": positive_total,
        "negative_queries": abstention_total,
        "documents": len(documents),
        "chunks": len(chunks),
        f"document_hit@{top_k}": round(doc_hits / positive_total, 4) if positive_total else 0.0,
        f"chunk_hit@{top_k}": round(chunk_hits / positive_total, 4) if positive_total else 0.0,
        "mrr": round(sum(reciprocal_ranks) / positive_total, 4) if positive_total else 0.0,
        "no_answer_label_accuracy": round(abstention_correct / abstention_total, 4) if abstention_total else None,
        "by_category": {
            category: {
                "queries": int(bucket["queries"]),
                "positive_queries": int(bucket["positive_queries"]),
                f"document_hit@{top_k}": (
                    round(bucket["document_hits"] / bucket["positive_queries"], 4)
                    if bucket["positive_queries"]
                    else None
                ),
                f"chunk_hit@{top_k}": (
                    round(bucket["chunk_hits"] / bucket["positive_queries"], 4)
                    if bucket["positive_queries"]
                    else None
                ),
                "mrr": (
                    round(bucket["reciprocal_rank_sum"] / bucket["positive_queries"], 4)
                    if bucket["positive_queries"]
                    else None
                ),
            }
            for category, bucket in sorted(category_buckets.items())
        },
    }
    return {"summary": summary, "results": results}


def main() -> None:
    os.environ.setdefault("DOCUMENT_EMBEDDING_PROVIDER", "local_hash")
    os.environ.setdefault("DOCUMENT_VECTOR_STORE", "sqlite")

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("gold_retrieval_eval_results.json"))
    args = parser.parse_args()

    result = run_eval(args.queries, args.top_k)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
