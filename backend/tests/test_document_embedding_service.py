from services.document_embedding_service import DocumentEmbeddingService


def test_rank_chunks_batches_missing_embeddings(monkeypatch):
    service = DocumentEmbeddingService()
    calls = []

    def fake_embed_texts(texts):
        calls.append(list(texts))
        return [[1.0, 0.0] for _ in texts]

    monkeypatch.setattr(service, "embed_texts", fake_embed_texts)

    ranked = service.rank_chunks(
        "query",
        [
            {"id": "a", "title": "A", "content": "first"},
            {"id": "b", "title": "B", "content": "second"},
        ],
        top_k=2,
    )

    assert [item["id"] for item in ranked] == ["a", "b"]
    assert len(calls) == 2
    assert calls[0] == ["query"]
    assert calls[1] == ["A\nfirst", "B\nsecond"]
