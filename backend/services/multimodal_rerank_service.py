"""
Optional multimodal reranker for document images.

Image recall starts with multimodal embeddings. This service adds a second-stage
Qwen3-VL rerank pass that scores the original images against the text query
before those image candidates enter hybrid fusion.
"""
import base64
import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional

import requests

logger = logging.getLogger(__name__)


class MultimodalRerankService:
    """Rerank image chunks with DashScope qwen3-vl-rerank."""

    def __init__(self):
        self.enabled = os.getenv(
            "DOCUMENT_MULTIMODAL_RERANK_ENABLED",
            "false",
        ).lower() == "true"
        self.provider = os.getenv(
            "DOCUMENT_MULTIMODAL_RERANK_PROVIDER",
            "dashscope",
        ).strip().lower()
        self.model_name = os.getenv(
            "DOCUMENT_MULTIMODAL_RERANK_MODEL",
            "qwen3-vl-rerank",
        )
        self.api_base = self._get_api_base()
        self.timeout = int(os.getenv("DOCUMENT_MULTIMODAL_RERANK_TIMEOUT", "60"))
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if not self.enabled or self.provider != "dashscope":
            return False
        if self._available is None:
            self._available = bool(self._get_api_key())
            if not self._available:
                logger.warning("Multimodal rerank API key is missing; image rerank disabled")
        return self._available

    def rerank_images(
        self,
        query: str,
        image_chunks: List[Dict[str, Any]],
        document_images: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Rerank image chunks against the raw images."""
        if not query or not image_chunks or not self.is_available():
            return []

        image_by_key = {
            (image.get("document_id"), image.get("image_index")): image
            for image in document_images
        }

        documents = []
        chunk_by_position = {}
        for chunk in image_chunks:
            key = (chunk.get("document_id"), chunk.get("image_index"))
            image = image_by_key.get(key)
            image_path = (image or {}).get("image_path") or (image or {}).get("path")
            if not image_path or not os.path.exists(image_path):
                continue

            media_type = mimetypes.guess_type(image_path)[0] or "image/png"
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("ascii")

            position = len(documents)
            documents.append(
                {
                    "type": "image",
                    "image": f"data:{media_type};base64,{encoded}",
                }
            )
            chunk_by_position[position] = chunk

        if not documents:
            return []

        response = requests.post(
            f"{self.api_base}/services/rerank/text-rerank/text-rerank",
            headers={
                "Authorization": f"Bearer {self._get_api_key()}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": {
                    "query": query,
                    "documents": documents,
                },
                "parameters": {
                    "return_documents": False,
                    "top_n": min(top_k, len(documents)),
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        results = body.get("output", {}).get("results", [])

        reranked = []
        for item in results:
            idx = item.get("index")
            chunk = chunk_by_position.get(idx)
            if chunk is None:
                continue
            score = item.get("relevance_score")
            enriched = dict(chunk)
            enriched["_image_rerank_score"] = round(float(score or 0.0), 6)
            enriched["_image_rerank_model"] = self.model_name
            sources = list(enriched.get("_retrieval_sources", []))
            if "image_rerank" not in sources:
                sources.append("image_rerank")
            enriched["_retrieval_sources"] = sources
            reranked.append(enriched)
        return reranked

    @staticmethod
    def _get_api_key() -> str:
        return (
            os.getenv("DOCUMENT_MULTIMODAL_RERANK_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or ""
        ).strip()

    @staticmethod
    def _get_api_base() -> str:
        return (
            os.getenv("DOCUMENT_MULTIMODAL_RERANK_API_BASE")
            or MultimodalRerankService._normalize_dashscope_api_base(
                os.getenv("OPENAI_API_BASE", "")
            )
            or os.getenv("DASHSCOPE_API_BASE")
            or "https://dashscope.aliyuncs.com/api/v1"
        ).rstrip("/")

    @staticmethod
    def _normalize_dashscope_api_base(api_base: str) -> str:
        api_base = (api_base or "").rstrip("/")
        compatible_suffix = "/compatible-mode/v1"
        if api_base.endswith(compatible_suffix):
            return api_base[: -len(compatible_suffix)] + "/api/v1"
        return api_base


_multimodal_rerank_service: Optional[MultimodalRerankService] = None


def get_multimodal_rerank_service() -> MultimodalRerankService:
    global _multimodal_rerank_service
    if _multimodal_rerank_service is None:
        _multimodal_rerank_service = MultimodalRerankService()
    return _multimodal_rerank_service
