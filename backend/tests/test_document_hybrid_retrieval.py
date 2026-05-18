from services.document_hybrid_retrieval_service import DocumentHybridRetrievalService


def _chunk(chunk_id: str, title: str, content: str):
    return {
        "id": chunk_id,
        "document_id": "doc_1",
        "title": title,
        "content": content,
        "heading_path": ["root", title],
    }


def test_bm25_prefers_exact_term_match():
    service = DocumentHybridRetrievalService()
    chunks = [
        _chunk("redis", "Redis 缓存击穿", "互斥锁 热点 key 缓存击穿"),
        _chunk("overview", "Redis 概览", "Redis 数据结构"),
    ]

    ranked = service.rank_bm25("Redis 缓存击穿", chunks, top_k=2)

    assert ranked[0]["id"] == "redis"
    assert ranked[0]["_bm25_score"] > ranked[-1].get("_bm25_score", 0)


def test_rrf_rewards_chunks_seen_by_multiple_retrievers():
    service = DocumentHybridRetrievalService(rrf_k=60)
    vector_ranked = [
        {"id": "vector_only"},
        {"id": "shared"},
    ]
    bm25_ranked = [
        {"id": "shared"},
        {"id": "bm25_only"},
    ]

    fused = service.reciprocal_rank_fusion(
        {"vector": vector_ranked, "bm25": bm25_ranked},
        top_k=3,
    )

    assert fused[0]["id"] == "shared"
    assert fused[0]["_retrieval_sources"] == ["vector", "bm25"]
    assert fused[0]["_vector_rank"] == 2
    assert fused[0]["_bm25_rank"] == 1
