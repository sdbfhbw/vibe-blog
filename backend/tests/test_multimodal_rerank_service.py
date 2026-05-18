from services.multimodal_rerank_service import MultimodalRerankService


def test_rerank_images_uses_dashscope_multimodal_api(monkeypatch, tmp_path):
    monkeypatch.setenv("DOCUMENT_MULTIMODAL_RERANK_ENABLED", "true")
    monkeypatch.setenv("DOCUMENT_MULTIMODAL_RERANK_PROVIDER", "dashscope")
    monkeypatch.setenv("OPENAI_API_KEY", "test-key")
    monkeypatch.setenv(
        "OPENAI_API_BASE",
        "https://dashscope.aliyuncs.com/compatible-mode/v1",
    )

    image_path = tmp_path / "diagram.png"
    image_path.write_bytes(b"fake-image")

    chunks = [
        {
            "id": "chunk_image_0",
            "document_id": "doc_1",
            "image_index": 0,
            "_retrieval_sources": ["image_vector"],
        },
        {
            "id": "chunk_image_1",
            "document_id": "doc_1",
            "image_index": 1,
            "_retrieval_sources": ["image_vector"],
        },
    ]
    images = [
        {
            "document_id": "doc_1",
            "image_index": 0,
            "image_path": str(image_path),
        },
        {
            "document_id": "doc_1",
            "image_index": 1,
            "image_path": str(image_path),
        },
    ]

    captured = {}

    class FakeResponse:
        def raise_for_status(self):
            return None

        def json(self):
            return {
                "output": {
                    "results": [
                        {"index": 1, "relevance_score": 0.92},
                        {"index": 0, "relevance_score": 0.41},
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
        "services.multimodal_rerank_service.requests.post",
        fake_post,
    )

    service = MultimodalRerankService()
    reranked = service.rerank_images("architecture diagram", chunks, images, top_k=2)

    assert [item["id"] for item in reranked] == ["chunk_image_1", "chunk_image_0"]
    assert reranked[0]["_image_rerank_score"] == 0.92
    assert reranked[0]["_image_rerank_model"] == "qwen3-vl-rerank"
    assert reranked[0]["_retrieval_sources"] == ["image_vector", "image_rerank"]
    assert captured["url"].endswith("/services/rerank/text-rerank/text-rerank")
    assert captured["headers"]["Authorization"] == "Bearer test-key"
    assert captured["json"]["model"] == "qwen3-vl-rerank"
    assert captured["json"]["input"]["query"] == "architecture diagram"
    assert captured["json"]["input"]["documents"][0]["type"] == "image"
    assert captured["json"]["input"]["documents"][0]["image"].startswith("data:image/png;base64,")
