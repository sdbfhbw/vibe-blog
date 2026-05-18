"""Shared ranking helpers used by retrieval evaluation scripts."""

from typing import Any, Dict, List

from services.document_embedding_service import get_document_embedding_service
from services.document_hybrid_retrieval_service import (
    get_document_hybrid_retrieval_service,
)
from services.document_vector_store_service import get_document_vector_store_service


def rank_documents(
    query: str,
    documents: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    embedding_service = get_document_embedding_service()
    vector_store = get_document_vector_store_service()
    ranked = vector_store.query_documents(
        query,
        documents,
        embedding_service,
        top_k=top_k,
    )
    if ranked:
        return ranked
    return embedding_service.rank_documents(query, documents, top_k=top_k)


def rank_chunks(
    query: str,
    chunks: List[Dict[str, Any]],
    top_k: int,
) -> List[Dict[str, Any]]:
    embedding_service = get_document_embedding_service()
    vector_store = get_document_vector_store_service()
    hybrid_service = get_document_hybrid_retrieval_service()

    vector_results = vector_store.query_chunks(
        query,
        chunks,
        embedding_service,
        top_k=top_k,
    )
    if not vector_results:
        vector_results = embedding_service.rank_chunks(query, chunks, top_k=top_k)

    bm25_results = hybrid_service.rank_bm25(query, chunks, top_k=top_k)
    return hybrid_service.reciprocal_rank_fusion(
        {"vector": vector_results, "bm25": bm25_results},
        top_k=top_k,
    )
