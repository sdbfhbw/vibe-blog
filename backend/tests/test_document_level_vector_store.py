import json

from services.document_embedding_service import DocumentEmbeddingService
from services.document_vector_store_service import DocumentVectorStoreService


class _FakeCollection:
    def __init__(self, query_result=None):
        self.upserts = []
        self.query_result = query_result or {"ids": [[]], "distances": [[]]}

    def upsert(self, **kwargs):
        self.upserts.append(kwargs)

    def query(self, **kwargs):
        return self.query_result


class _FakeEmbeddingService:
    def embed_texts(self, texts):
        return [[0.1, 0.2, 0.3] for _ in texts]


def _document(document_id: str, filename: str, summary: str):
    return {
        "id": document_id,
        "filename": filename,
        "file_type": "md",
        "status": "ready",
        "summary": summary,
        "markdown_content": "",
        "embedding": json.dumps([0.1, 0.2, 0.3]),
        "embedding_model": "test-model",
    }


def test_upsert_documents_writes_document_summary_embeddings(monkeypatch):
    service = DocumentVectorStoreService()
    collection = _FakeCollection()
    monkeypatch.setattr(service, "is_available", lambda: True)
    monkeypatch.setattr(service, "_get_document_collection", lambda: collection)

    success = service.upsert_documents([
        _document("doc_1", "redis.md", "Redis cache breakdown"),
    ])

    assert success is True
    assert len(collection.upserts) == 1
    payload = collection.upserts[0]
    assert payload["ids"] == ["doc_1"]
    assert payload["documents"] == ["redis.md\nRedis cache breakdown"]
    assert payload["metadatas"][0]["filename"] == "redis.md"


def test_query_documents_hydrates_hnsw_results_from_sqlite_rows(monkeypatch):
    service = DocumentVectorStoreService()
    collection = _FakeCollection(
        query_result={
            "ids": [["doc_2", "stale_doc"]],
            "distances": [[0.1, 0.2]],
        }
    )
    monkeypatch.setattr(service, "is_available", lambda: True)
    monkeypatch.setattr(service, "_get_document_collection", lambda: collection)
    documents = [
        _document("doc_1", "redis.md", "Redis"),
        _document("doc_2", "langgraph.md", "LangGraph workflow"),
    ]

    ranked = service.query_documents(
        "LangGraph",
        documents,
        _FakeEmbeddingService(),
        top_k=2,
    )

    assert [doc["id"] for doc in ranked] == ["doc_2"]
    assert ranked[0]["document_relevance_score"] == 0.9
    assert ranked[0]["_vector_store"] == "chroma_document"
