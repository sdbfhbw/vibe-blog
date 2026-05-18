"""
Optional multimodal embedding service for document images.

Text chunks keep using `DocumentEmbeddingService`. This service is dedicated to
models that embed images and text into the same vector space, enabling
text-to-image retrieval in addition to caption/OCR retrieval.

The default path is API-first:
- use Qwen3-VL multimodal embeddings when an API key is configured
- fall back to a local CLIP/SigLIP-style model when the remote path is not
  configured or fails at runtime
"""
import base64
import json
import logging
import mimetypes
import os
from typing import Any, Dict, List, Optional

import requests
from PIL import Image

from services.document_embedding_service import cosine_similarity

logger = logging.getLogger(__name__)


class DocumentMultimodalEmbeddingService:
    """Generate text/image embeddings in one shared multimodal space."""

    def __init__(self):
        self.enabled = os.getenv(
            "DOCUMENT_MULTIMODAL_RETRIEVAL_ENABLED",
            "false",
        ).lower() == "true"
        explicit_provider = os.getenv(
            "DOCUMENT_MULTIMODAL_EMBEDDING_PROVIDER",
            "",
        ).strip()
        has_api_key = bool(self._get_api_key())
        self.provider = (
            explicit_provider or ("dashscope" if has_api_key else "local_clip")
        ).lower()
        self.model_name = os.getenv(
            "DOCUMENT_MULTIMODAL_EMBEDDING_MODEL",
            "qwen3-vl-embedding" if self.provider == "dashscope" else "clip-ViT-B-32",
        )
        self.api_base = self._get_api_base()
        self.batch_size = max(
            1,
            int(os.getenv("DOCUMENT_MULTIMODAL_EMBEDDING_BATCH_SIZE", "8")),
        )
        self.timeout = int(os.getenv("DOCUMENT_MULTIMODAL_EMBEDDING_TIMEOUT", "60"))
        self._model = None
        self._available: Optional[bool] = None

    def is_available(self) -> bool:
        if not self.enabled:
            return False
        if self._available is not None:
            return self._available
        if self.provider == "dashscope":
            self._available = bool(self._get_api_key())
            if not self._available:
                logger.warning(
                    "Multimodal embedding provider is dashscope but API key is missing; "
                    "falling back to local model",
                )
                self._switch_to_local_model()
            else:
                return True
        try:
            self._get_model()
            self._available = True
        except Exception as exc:
            logger.warning("Multimodal embedding unavailable, image-vector recall disabled: %s", exc)
            self._available = False
        return self._available

    def embed_texts(self, texts: List[str]) -> List[List[float]]:
        if not texts or not self.is_available():
            return []
        if self.provider == "dashscope":
            try:
                return self._embed_dashscope_texts(texts)
            except Exception as exc:
                logger.warning(
                    "Qwen3-VL text embedding failed; falling back to local model: %s",
                    exc,
                )
                self._switch_to_local_model()
        try:
            return self._embed_local_texts(texts)
        except Exception as exc:
            logger.warning("Local multimodal text embedding unavailable: %s", exc)
            self._available = False
            return []

    def enrich_images(self, images: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Attach image embedding metadata before database save."""
        if not images or not self.is_available():
            return images

        valid_pairs = []
        for image in images:
            image_path = image.get("image_path") or image.get("path")
            if not image_path or not os.path.exists(image_path):
                continue
            valid_pairs.append((image, image_path))

        if not valid_pairs:
            return images

        if self.provider == "dashscope":
            try:
                vectors = self._embed_dashscope_images([pair[1] for pair in valid_pairs])
            except Exception as exc:
                logger.warning(
                    "Qwen3-VL image embedding failed; falling back to local model: %s",
                    exc,
                )
                self._switch_to_local_model()
                try:
                    vectors = self._embed_local_images([pair[1] for pair in valid_pairs])
                except Exception as local_exc:
                    logger.warning("Local multimodal image embedding unavailable: %s", local_exc)
                    self._available = False
                    return images
        else:
            try:
                vectors = self._embed_local_images([pair[1] for pair in valid_pairs])
            except Exception as exc:
                logger.warning("Local multimodal image embedding unavailable: %s", exc)
                self._available = False
                return images

        enriched_by_identity = {
            id(image): [float(v) for v in vector]
            for (image, _), vector in zip(valid_pairs, vectors)
        }

        enriched = []
        for image in images:
            item = dict(image)
            vector = enriched_by_identity.get(id(image))
            if vector is not None:
                item["image_embedding"] = json.dumps(vector, ensure_ascii=False)
                item["image_embedding_model"] = self.model_name
                item["image_embedding_dim"] = len(vector)
            enriched.append(item)
        return enriched

    def rank_images(
        self,
        query: str,
        images: List[Dict[str, Any]],
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Return top-k document images by text-to-image cosine similarity."""
        if not query or not images or not self.is_available():
            return []

        query_vectors = self.embed_texts([query])
        if not query_vectors:
            return []
        query_embedding = query_vectors[0]

        scored = []
        for image in images:
            embedding = self._load_image_embedding(image)
            if embedding is None:
                continue
            item = dict(image)
            item["image_relevance_score"] = round(
                cosine_similarity(query_embedding, embedding),
                6,
            )
            scored.append(item)

        scored.sort(
            key=lambda item: item.get("image_relevance_score", 0.0),
            reverse=True,
        )
        return scored[:top_k]

    def _embed_dashscope_texts(self, texts: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for text in texts:
            vectors.extend(self._post_dashscope([{"text": text}]))
        return vectors

    def _embed_dashscope_images(self, image_paths: List[str]) -> List[List[float]]:
        vectors: List[List[float]] = []
        for image_path in image_paths:
            media_type = mimetypes.guess_type(image_path)[0] or "image/png"
            with open(image_path, "rb") as image_file:
                encoded = base64.b64encode(image_file.read()).decode("ascii")
            vectors.extend(
                self._post_dashscope(
                    [{"image": f"data:{media_type};base64,{encoded}"}]
                )
            )
        return vectors

    def _post_dashscope(
        self,
        contents: List[Dict[str, Any]],
    ) -> List[List[float]]:
        api_key = self._get_api_key()
        if not api_key:
            raise RuntimeError("missing multimodal embedding API key")

        response = requests.post(
            f"{self.api_base}/services/embeddings/multimodal-embedding/multimodal-embedding",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json={
                "model": self.model_name,
                "input": {
                    "contents": contents,
                },
            },
            timeout=self.timeout,
        )
        response.raise_for_status()
        body = response.json()
        data = body.get("output", {}).get("embeddings", [])
        if len(data) != len(contents):
            raise ValueError(
                f"multimodal embedding response size mismatch: expected {len(contents)}, got {len(data)}"
            )

        vectors = []
        for item in sorted(data, key=lambda row: row.get("index", 0)):
            vector = item.get("embedding")
            if not isinstance(vector, list):
                raise ValueError("multimodal embedding response item missing vector")
            vectors.append([float(v) for v in vector])
        return vectors

    def _embed_local_texts(self, texts: List[str]) -> List[List[float]]:
        vectors = self._get_model().encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return [[float(v) for v in vector] for vector in vectors]

    def _embed_local_images(self, image_paths: List[str]) -> List[List[float]]:
        valid_images = []
        for image_path in image_paths:
            try:
                with Image.open(image_path) as img:
                    valid_images.append(img.convert("RGB").copy())
            except Exception as exc:
                logger.warning("Image load failed for multimodal embedding [%s]: %s", image_path, exc)
                valid_images.append(None)

        vectors = self._get_model().encode(
            [image for image in valid_images if image is not None],
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        vector_iter = iter(vectors)
        resolved = []
        for image in valid_images:
            if image is None:
                resolved.append([])
            else:
                resolved.append([float(v) for v in next(vector_iter)])
        return resolved

    def _switch_to_local_model(self):
        self.provider = "local_clip"
        self.model_name = os.getenv(
            "DOCUMENT_MULTIMODAL_LOCAL_MODEL",
            "clip-ViT-B-32",
        )
        self._available = None

    def _get_model(self):
        if self._model is not None:
            return self._model

        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(self.model_name)
        return self._model

    @staticmethod
    def _get_api_key() -> str:
        return (
            os.getenv("DOCUMENT_MULTIMODAL_EMBEDDING_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("DASHSCOPE_API_KEY")
            or ""
        ).strip()

    @staticmethod
    def _get_api_base() -> str:
        return (
            os.getenv("DOCUMENT_MULTIMODAL_EMBEDDING_API_BASE")
            or DocumentMultimodalEmbeddingService._normalize_dashscope_api_base(
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

    @staticmethod
    def _load_image_embedding(image: Dict[str, Any]) -> Optional[List[float]]:
        raw = image.get("image_embedding")
        if not raw:
            return None
        try:
            value = json.loads(raw) if isinstance(raw, str) else raw
            if isinstance(value, list):
                return [float(v) for v in value]
        except Exception:
            return None
        return None


_document_multimodal_embedding_service: Optional[DocumentMultimodalEmbeddingService] = None


def get_document_multimodal_embedding_service() -> DocumentMultimodalEmbeddingService:
    global _document_multimodal_embedding_service
    if _document_multimodal_embedding_service is None:
        _document_multimodal_embedding_service = DocumentMultimodalEmbeddingService()
    return _document_multimodal_embedding_service
