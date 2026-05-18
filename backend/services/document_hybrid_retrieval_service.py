"""
Hybrid retrieval utilities for document chunks.

Provides deterministic BM25 keyword recall and reciprocal rank fusion (RRF)
so callers can combine exact-term and semantic retrieval before reranking.
"""
import math
from collections import Counter
from typing import Any, Dict, List, Optional

from services.document_embedding_service import DocumentEmbeddingService


class DocumentHybridRetrievalService:
    """BM25 recall plus RRF fusion for document chunks."""

    def __init__(self, k1: float = 1.5, b: float = 0.75, rrf_k: int = 60):
        self.k1 = k1
        self.b = b
        self.rrf_k = rrf_k

    def rank_bm25(
        self,
        query: str,
        chunks: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Return top-k chunks ranked by BM25 over heading/title/content."""
        if not query or not chunks:
            return []

        query_terms = DocumentEmbeddingService._tokenize(query)
        if not query_terms:
            return []

        tokenized_docs = [
            DocumentEmbeddingService._tokenize(
                DocumentEmbeddingService._chunk_text(chunk)
            )
            for chunk in chunks
        ]
        doc_count = len(tokenized_docs)
        avg_doc_len = (
            sum(len(tokens) for tokens in tokenized_docs) / doc_count
            if doc_count else 0.0
        )
        if avg_doc_len == 0:
            return []

        document_frequency = Counter()
        for tokens in tokenized_docs:
            document_frequency.update(set(tokens))

        scored = []
        for chunk, tokens in zip(chunks, tokenized_docs):
            if not tokens:
                continue

            term_frequency = Counter(tokens)
            score = 0.0
            doc_len = len(tokens)
            for term in query_terms:
                tf = term_frequency.get(term, 0)
                if not tf:
                    continue
                df = document_frequency.get(term, 0)
                idf = math.log(1 + (doc_count - df + 0.5) / (df + 0.5))
                denominator = tf + self.k1 * (
                    1 - self.b + self.b * doc_len / avg_doc_len
                )
                score += idf * (tf * (self.k1 + 1)) / denominator

            if score <= 0:
                continue

            item = dict(chunk)
            item["_bm25_score"] = round(score, 6)
            scored.append(item)

        scored.sort(key=lambda item: item.get("_bm25_score", 0.0), reverse=True)
        for rank, item in enumerate(scored, 1):
            item["_bm25_rank"] = rank
        return scored[:top_k]

    def reciprocal_rank_fusion(
        self,
        ranked_lists: Dict[str, List[Dict[str, Any]]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Fuse multiple ranked lists with reciprocal rank fusion."""
        fused: Dict[str, Dict[str, Any]] = {}

        for source, items in ranked_lists.items():
            for rank, item in enumerate(items, 1):
                chunk_id = item.get("id")
                if not chunk_id:
                    continue

                key = str(chunk_id)
                if key not in fused:
                    fused[key] = dict(item)
                    fused[key]["_rrf_score"] = 0.0
                    fused[key]["_retrieval_sources"] = []

                merged = fused[key]
                merged.update(item)
                merged["_rrf_score"] += 1.0 / (self.rrf_k + rank)
                if source not in merged["_retrieval_sources"]:
                    merged["_retrieval_sources"].append(source)
                merged[f"_{source}_rank"] = rank

        results = list(fused.values())
        for item in results:
            item["_rrf_score"] = round(item.get("_rrf_score", 0.0), 6)
        results.sort(key=lambda item: item.get("_rrf_score", 0.0), reverse=True)
        return results[:top_k]


_document_hybrid_retrieval_service: Optional[DocumentHybridRetrievalService] = None


def get_document_hybrid_retrieval_service() -> DocumentHybridRetrievalService:
    global _document_hybrid_retrieval_service
    if _document_hybrid_retrieval_service is None:
        _document_hybrid_retrieval_service = DocumentHybridRetrievalService()
    return _document_hybrid_retrieval_service
