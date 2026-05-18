"""Run image-vector and optional multimodal rerank smoke evaluation."""

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
from services.document_multimodal_embedding_service import get_document_multimodal_embedding_service
from services.document_vector_store_service import get_document_vector_store_service
from services.multimodal_rerank_service import get_multimodal_rerank_service

DEFAULT_QUERIES = Path(__file__).with_name("multimodal_smoke_queries.jsonl")


def _load_queries(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_eval(query_path: Path, top_k: int) -> Dict[str, Any]:
    db = get_db_service()
    service = get_document_multimodal_embedding_service()
    vector_store = get_document_vector_store_service()
    reranker = get_multimodal_rerank_service()
    documents = db.list_ready_documents_for_retrieval()
    images = []
    for document in documents:
        images.extend(db.get_images_by_document(document["id"]))
    chunks = db.get_chunks_by_documents([document["id"] for document in documents])
    image_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "image"]
    chunk_by_image_key = {
        (chunk.get("document_id"), chunk.get("image_index")): chunk
        for chunk in image_chunks
    }

    results = []
    vector_hits = 0
    rerank_hits = 0
    for query in _load_queries(query_path):
        vector_results = vector_store.query_images(query["query"], images, service, top_k=top_k)
        if not vector_results:
            vector_results = service.rank_images(query["query"], images, top_k=top_k)
        vector_ids = [Path(item.get("image_path") or "").name for item in vector_results]
        vector_hit = query["expected_image_file"] in vector_ids
        vector_hits += int(vector_hit)

        candidate_chunks = []
        for image in vector_results:
            key = (image.get("document_id"), image.get("image_index"))
            chunk = chunk_by_image_key.get(key)
            if chunk:
                candidate_chunks.append(chunk)
        reranked = reranker.rerank_images(query["query"], candidate_chunks, images, top_k=top_k)
        reranked_files = []
        image_by_key = {(image["document_id"], image["image_index"]): image for image in images}
        for chunk in reranked:
            image = image_by_key.get((chunk.get("document_id"), chunk.get("image_index")))
            reranked_files.append(Path((image or {}).get("image_path") or "").name)
        rerank_hit = query["expected_image_file"] in reranked_files
        rerank_hits += int(rerank_hit)
        results.append(
            {
                "id": query["id"],
                "vector_hit": vector_hit,
                "rerank_hit": rerank_hit,
                "vector_top_images": vector_ids,
                "rerank_top_images": reranked_files,
            }
        )

    total = len(results)
    return {
        "summary": {
            "queries": total,
            f"image_vector_hit@{top_k}": round(vector_hits / total, 4) if total else 0.0,
            f"image_rerank_hit@{top_k}": round(rerank_hits / total, 4) if total else 0.0,
            "multimodal_available": service.is_available(),
            "rerank_available": reranker.is_available(),
        },
        "results": results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--queries", type=Path, default=DEFAULT_QUERIES)
    parser.add_argument("--top-k", type=int, default=3)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("multimodal_eval_results.json"))
    args = parser.parse_args()
    result = run_eval(args.queries, args.top_k)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result["summary"], ensure_ascii=False))
    print(args.output)


if __name__ == "__main__":
    main()
