"""
Document chunk embedding and retrieval service.

Default provider uses the OpenAI-compatible embeddings endpoint when an API key
is configured, and falls back to deterministic local hash embeddings otherwise.
"""
import hashlib
import json
import logging
import math
import os
import re
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


def cosine_similarity(a: List[float], b: List[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(x * x for x in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


class DocumentEmbeddingService:
    """Generate and search embeddings for documents and document chunks."""

    def __init__(self):
        explicit_provider = os.getenv("DOCUMENT_EMBEDDING_PROVIDER", os.getenv("EMBEDDING_PROVIDER", "")).strip()
        has_api_key = bool(self._get_api_key())
        self.provider = (explicit_provider or ("openai" if has_api_key else "local_hash")).lower()
        default_model = "text-embedding-v4" if self.provider == "openai" else "local-hash-v1"
        self.model_name = os.getenv("DOCUMENT_EMBEDDING_MODEL", os.getenv("EMBEDDING_MODEL", default_model))
        self.dim = int(os.getenv("DOCUMENT_EMBEDDING_DIM", "384"))
        self.batch_size = max(1, int(os.getenv("DOCUMENT_EMBEDDING_BATCH_SIZE", "16")))
        self.timeout = int(os.getenv("DOCUMENT_EMBEDDING_TIMEOUT", "60"))

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        if self.provider == "openai":
            return self._embed_openai(texts)
        return [self._embed_local_hash(text) for text in texts]

    def enrich_chunks(self, chunks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach embedding JSON metadata to chunk dicts before database save."""
        retrievable_chunks = [
            chunk for chunk in chunks if chunk.get("chunk_type") != "parent"
        ]
        texts = [self._chunk_text(chunk) for chunk in retrievable_chunks]
        embeddings = self.embed_texts(texts)
        embedding_by_identity = {
            id(chunk): embedding
            for chunk, embedding in zip(retrievable_chunks, embeddings)
        }
        enriched = []
        for chunk in chunks:
            item = dict(chunk)
            embedding = embedding_by_identity.get(id(chunk))
            if embedding is not None:
                item["embedding"] = json.dumps(embedding, ensure_ascii=False)
                item["embedding_model"] = self.model_name
                item["embedding_dim"] = len(embedding)
            enriched.append(item)
        return enriched

    def enrich_document(self, document: Dict[str, Any]) -> Dict[str, Any]:
        """Attach document-level embedding metadata before database save."""
        embedding = self.embed_texts([self._document_text(document)])[0]
        item = dict(document)
        item["embedding"] = json.dumps(embedding, ensure_ascii=False)
        item["embedding_model"] = self.model_name
        item["embedding_dim"] = len(embedding)
        return item

    def rank_chunks(self, query: str, chunks: List[Dict[str, Any]], top_k: int = 8) -> List[Dict[str, Any]]:
        """Return top-k chunks by cosine similarity to query."""
        if not query or not chunks:
            return []

        return self._rank_items(
            query=query,
            items=chunks,
            top_k=top_k,
            text_builder=self._chunk_text,
            score_key="relevance_score",
        )

    def rank_documents(
        self,
        query: str,
        documents: List[Dict[str, Any]],
        top_k: int = 3,
    ) -> List[Dict[str, Any]]:
        """Return top-k documents by cosine similarity to the query."""
        if not query or not documents:
            return []

        return self._rank_items(
            query=query,
            items=documents,
            top_k=top_k,
            text_builder=self._document_text,
            score_key="document_relevance_score",
        )

    def _rank_items(
        self,
        query: str,
        items: List[Dict[str, Any]],
        top_k: int,
        text_builder,
        score_key: str,
    ) -> List[Dict[str, Any]]:
        query_embedding = self.embed_texts([query])[0]
        embeddings = self._load_or_embed_missing(items, text_builder)

        scored = []
        for item, embedding in zip(items, embeddings):
            if embedding is None:
                continue
            ranked_item = dict(item)
            ranked_item[score_key] = round(
                cosine_similarity(query_embedding, embedding),
                6,
            )
            scored.append(ranked_item)

        scored.sort(key=lambda item: item.get(score_key, 0.0), reverse=True)
        return scored[:top_k]

    def _load_or_embed_missing(self, items, text_builder) -> List[Optional[List[float]]]:
        embeddings: List[Optional[List[float]]] = [
            self._load_embedding(item) for item in items
        ]
        missing_indices = [
            idx for idx, embedding in enumerate(embeddings) if embedding is None
        ]
        if not missing_indices:
            return embeddings

        missing_embeddings = self.embed_texts(
            [text_builder(items[idx]) for idx in missing_indices]
        )
        for idx, embedding in zip(missing_indices, missing_embeddings):
            embeddings[idx] = embedding
        return embeddings

    def _embed_openai(self, texts: List[str]) -> List[List[float]]:
        api_key = self._get_api_key()
        api_base = self._get_api_base()
        if not api_key:
            logger.warning("Document embedding provider is openai but API key is missing; falling back to local hash")
            self.provider = "local_hash"
            self.model_name = "local-hash-v1"
            return [self._embed_local_hash(text) for text in texts]

        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        url = f"{api_base}/embeddings"
        embeddings: List[List[float]] = []

        try:
            for start in range(0, len(texts), self.batch_size):
                batch = texts[start:start + self.batch_size]
                payload: Dict[str, Any] = {
                    "model": self.model_name,
                    "input": batch,
                }
                dimensions = os.getenv("DOCUMENT_EMBEDDING_DIMENSIONS", "").strip()
                if dimensions:
                    payload["dimensions"] = int(dimensions)

                response = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
                response.raise_for_status()
                body = response.json()
                data = body.get("data", [])
                if len(data) != len(batch):
                    raise ValueError(f"embedding response size mismatch: expected {len(batch)}, got {len(data)}")

                data = sorted(data, key=lambda item: item.get("index", 0))
                for item in data:
                    embedding = item.get("embedding")
                    if not isinstance(embedding, list):
                        raise ValueError("embedding response item missing vector")
                    embeddings.append([float(v) for v in embedding])

            return embeddings
        except Exception as exc:
            logger.warning("OpenAI-compatible document embedding failed; falling back to local hash: %s", exc)
            self.provider = "local_hash"
            self.model_name = "local-hash-v1"
            return [self._embed_local_hash(text) for text in texts]

    @staticmethod
    def _get_api_key() -> str:
        return (
            os.getenv("DOCUMENT_EMBEDDING_API_KEY")
            or os.getenv("EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or ""
        ).strip()

    @staticmethod
    def _get_api_base() -> str:
        return (
            os.getenv("DOCUMENT_EMBEDDING_API_BASE")
            or os.getenv("EMBEDDING_API_BASE")
            or os.getenv("OPENAI_API_BASE")
            or "https://api.openai.com/v1"
        ).rstrip("/")

    def _embed_local_hash(self, text: str) -> List[float]:
        vec = [0.0] * self.dim
        tokens = self._tokenize(text)
        if not tokens:
            return vec

        for token in tokens:
            digest = hashlib.md5(token.encode("utf-8")).hexdigest()
            idx = int(digest[:8], 16) % self.dim
            sign = 1.0 if int(digest[8:10], 16) % 2 == 0 else -1.0
            vec[idx] += sign

        norm = math.sqrt(sum(x * x for x in vec))
        if norm:
            vec = [x / norm for x in vec]
        return vec

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        text = (text or "").lower()
        word_tokens = re.findall(r"[a-z0-9_+\-./#]{2,}", text)
        cjk_tokens = re.findall(r"[\u4e00-\u9fff]", text)
        return word_tokens + cjk_tokens

    @staticmethod
    def _chunk_text(chunk: Dict[str, Any]) -> str:
        heading_path = chunk.get("heading_path", [])
        if isinstance(heading_path, str):
            try:
                heading_path = json.loads(heading_path)
            except Exception:
                heading_path = [heading_path]
        heading = " > ".join(str(part) for part in heading_path if str(part).strip())
        title = chunk.get("title", "")
        content = chunk.get("content", "")
        return f"{heading}\n{title}\n{content}".strip()

    @staticmethod
    def _document_text(document: Dict[str, Any]) -> str:
        """Build compact text used for document-level retrieval."""
        filename = document.get("filename", "")
        summary = (document.get("summary") or "").strip()
        markdown = document.get("markdown_content", "") or ""
        body = summary or markdown[:4000]
        return f"{filename}\n{body}".strip()

    @staticmethod
    def _load_embedding(chunk: Dict[str, Any]) -> Optional[List[float]]:
        raw = chunk.get("embedding")
        if not raw:
            return None
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(value, list):
                return [float(v) for v in value]
        except Exception:
            return None
        return None


_document_embedding_service: Optional[DocumentEmbeddingService] = None


def get_document_embedding_service() -> DocumentEmbeddingService:
    global _document_embedding_service
    if _document_embedding_service is None:
        _document_embedding_service = DocumentEmbeddingService()
    return _document_embedding_service
