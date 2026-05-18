from services.document_multimodal_embedding_service import DocumentMultimodalEmbeddingService


def test_rank_images_uses_shared_multimodal_space(monkeypatch):
    service = DocumentMultimodalEmbeddingService()
    monkeypatch.setattr(service, "is_available", lambda: True)
    monkeypatch.setattr(service, "embed_texts", lambda texts: [[1.0, 0.0]])

    images = [
        {
            "id": "diagram",
            "image_embedding": "[1.0, 0.0]",
        },
        {
            "id": "photo",
            "image_embedding": "[0.0, 1.0]",
        },
    ]

    ranked = service.rank_images("architecture diagram", images, top_k=2)

    assert [item["id"] for item in ranked] == ["diagram", "photo"]
    assert ranked[0]["image_relevance_score"] > ranked[1]["image_relevance_score"]


def test_embed_texts_uses_dashscope_api_when_configured(monkeypatch):
    monkeypatch.setenv("DOCUMENT_MULTIMODAL_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("DOCUMENT_MULTIMODAL_EMBEDDING_PROVIDER", "dashscope")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    service = DocumentMultimodalEmbeddingService()
    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output": {
                    "embeddings": [
                        {"index": 0, "embedding": [1.0, 0.0], "type": "text"},
                    ]
                }
            }

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse()

    monkeypatch.setattr(
        "services.document_multimodal_embedding_service.requests.post",
        fake_post,
    )

    vectors = service.embed_texts(["architecture"])

    assert vectors == [[1.0, 0.0]]
    assert captured["url"].endswith(
        "/services/embeddings/multimodal-embedding/multimodal-embedding"
    )
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "qwen3-vl-embedding"
    assert captured["json"]["input"]["contents"][0]["text"] == "architecture"


def test_embed_texts_falls_back_to_local_model_on_api_failure(monkeypatch):
    monkeypatch.setenv("DOCUMENT_MULTIMODAL_RETRIEVAL_ENABLED", "true")
    monkeypatch.setenv("DOCUMENT_MULTIMODAL_EMBEDDING_PROVIDER", "dashscope")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")

    service = DocumentMultimodalEmbeddingService()
    monkeypatch.setattr(service, "_embed_dashscope_texts", lambda texts: (_ for _ in ()).throw(RuntimeError("boom")))
    monkeypatch.setattr(service, "_embed_local_texts", lambda texts: [[0.25, 0.75]])

    vectors = service.embed_texts(["architecture"])

    assert vectors == [[0.25, 0.75]]
    assert service.provider == "local_clip"
