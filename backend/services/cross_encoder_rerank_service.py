"""
Optional cross-encoder reranker for document chunks.

Enable with CROSS_ENCODER_RERANK_ENABLED=true. If sentence-transformers or the
configured model is unavailable, callers should fall back to deterministic
rerank.
"""
import logging
import os
from typing import Any, Dict, List, Optional

from services.document_embedding_service import DocumentEmbeddingService

logger = logging.getLogger(__name__)


class CrossEncoderRerankService:
    """Rerank candidate chunks with a sentence-transformers CrossEncoder."""

    def __init__(self):
        self.enabled = os.getenv("CROSS_ENCODER_RERANK_ENABLED", "false").lower() == "true"
        self.model_name = os.getenv("CROSS_ENCODER_RERANK_MODEL", "BAAI/bge-reranker-base")
        self.max_length = int(os.getenv("CROSS_ENCODER_MAX_LENGTH", "512"))
        self._model = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self._available is not None:
            return self._available
        try:
            self._get_model()
            self._available = True
        except Exception as exc:
            logger.warning("Cross-encoder rerank 不可用，回退规则 rerank: %s", exc)
            self._available = False
        return self._available

    def rerank(self, query: str, chunks: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        if not query or not chunks or not self.is_available():
            return []

        pairs = [(query, DocumentEmbeddingService._chunk_text(chunk)) for chunk in chunks]
        scores = self._get_model().predict(pairs)

        scored = []
        for chunk, score in zip(chunks, scores):
            item = dict(chunk)
            item["_cross_encoder_score"] = round(float(score), 6)
            item["_rerank_score"] = item["_cross_encoder_score"]
            scored.append(item)

        scored.sort(key=lambda item: item.get("_cross_encoder_score", 0.0), reverse=True)
        return scored[:top_k]

    def _get_model(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import CrossEncoder

        self._model = CrossEncoder(self.model_name, max_length=self.max_length)
        return self._model


_cross_encoder_rerank_service: Optional[CrossEncoderRerankService] = None


def get_cross_encoder_rerank_service() -> CrossEncoderRerankService:
    global _cross_encoder_rerank_service
    if _cross_encoder_rerank_service is None:
        _cross_encoder_rerank_service = CrossEncoderRerankService()
    return _cross_encoder_rerank_service
