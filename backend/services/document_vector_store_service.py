"""
Optional Chroma vector store for document, chunk, and image embeddings.

SQLite remains the source of truth for text and metadata. Chroma is used only
as a retrieval index when DOCUMENT_VECTOR_STORE=chroma:
- document summaries: coarse document recall
- document chunks: fine-grained chunk recall
- document images: multimodal text-to-image recall
"""
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

from services.document_embedding_service import DocumentEmbeddingService

logger = logging.getLogger(__name__)


class DocumentVectorStoreService:
    """Persist and query document and chunk embeddings with Chroma."""

    def __init__(self):
        self.store = os.getenv("DOCUMENT_VECTOR_STORE", "sqlite").lower()
        self.chunk_collection_name = os.getenv("CHROMA_COLLECTION_NAME", "document_chunks")
        self.document_collection_name = os.getenv(
            "CHROMA_DOCUMENT_COLLECTION_NAME",
            "document_summaries",
        )
        self.image_collection_name = os.getenv(
            "CHROMA_IMAGE_COLLECTION_NAME",
            "document_images",
        )
        default_dir = Path(__file__).resolve().parents[1] / "data" / "chroma"
        self.persist_dir = os.getenv("CHROMA_PERSIST_DIR", "").strip() or str(default_dir)
        self._client = None
        self._chunk_collection = None
        self._document_collection = None
        self._image_collection = None
        self._available: Optional[bool] = None

    def is_enabled(self) -> bool:
        return self.store == "chroma"

    def is_available(self) -> bool:
        if not self.is_enabled():
            return False
        if self._available is not None:
            return self._available
        try:
            self._get_chunk_collection()
            self._get_document_collection()
            self._get_image_collection()
            self._available = True
        except Exception as exc:
            logger.warning("Chroma 向量库不可用，回退 SQLite 全量扫描: %s", exc)
            self._available = False
        return self._available

    def upsert_chunks(self, chunks: List[Dict[str, Any]]) -> bool:
        if not chunks or not self.is_available():
            return False

        ids = []
        documents = []
        embeddings = []
        metadatas = []
        for chunk in chunks:
            if chunk.get("chunk_type") == "parent":
                continue
            chunk_id = chunk.get("id")
            embedding = DocumentEmbeddingService._load_embedding(chunk)
            if not chunk_id or embedding is None:
                continue
            ids.append(str(chunk_id))
            documents.append(DocumentEmbeddingService._chunk_text(chunk))
            embeddings.append(embedding)
            metadatas.append(self._metadata(chunk))

        if not ids:
            return False

        collection = self._get_chunk_collection()
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Chroma 向量索引已更新: chunks=%s", len(ids))
        return True

    def upsert_documents(self, documents: List[Dict[str, Any]]) -> bool:
        """Persist document-level embeddings for coarse HNSW recall."""
        if not documents or not self.is_available():
            return False

        ids = []
        texts = []
        embeddings = []
        metadatas = []
        for document in documents:
            document_id = document.get("id")
            embedding = DocumentEmbeddingService._load_embedding(document)
            if not document_id or embedding is None:
                continue
            ids.append(str(document_id))
            texts.append(DocumentEmbeddingService._document_text(document))
            embeddings.append(embedding)
            metadatas.append(self._document_metadata(document))

        if not ids:
            return False

        collection = self._get_document_collection()
        collection.upsert(
            ids=ids,
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Chroma document summary index updated: documents=%s", len(ids))
        return True

    def upsert_images(self, images: List[Dict[str, Any]]) -> bool:
        """Persist multimodal image embeddings for text-to-image recall."""
        if not images or not self.is_available():
            return False

        from services.document_multimodal_embedding_service import (
            DocumentMultimodalEmbeddingService,
        )

        ids = []
        documents = []
        embeddings = []
        metadatas = []
        for image in images:
            image_id = image.get("id")
            embedding = DocumentMultimodalEmbeddingService._load_image_embedding(image)
            if not image_id or embedding is None:
                continue
            ids.append(str(image_id))
            documents.append(self._image_text(image))
            embeddings.append(embedding)
            metadatas.append(self._image_metadata(image))

        if not ids:
            return False

        collection = self._get_image_collection()
        collection.upsert(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
        )
        logger.info("Chroma image index updated: images=%s", len(ids))
        return True

    def query_chunks(
        self,
        query: str,
        all_chunks: List[Dict[str, Any]],
        embedding_service,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Query Chroma and hydrate results from SQLite-loaded chunks."""
        if not query or not all_chunks or not self.is_available():
            return []

        chunk_by_id = {str(chunk.get("id")): chunk for chunk in all_chunks if chunk.get("id")}
        if not chunk_by_id:
            return []

        selected_doc_ids = sorted({
            str(chunk.get("document_id"))
            for chunk in all_chunks
            if chunk.get("document_id")
        })
        if not selected_doc_ids:
            logger.warning("Chroma 查询缺少 document_id，回退到 SQLite 全量扫描")
            return []

        query_embedding = embedding_service.embed_texts([query])[0]
        collection = self._get_chunk_collection()
        n_results = min(max(top_k * 5, top_k), max(len(chunk_by_id), top_k))
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            where={"document_id": {"$in": selected_doc_ids}},
            include=["distances"],
        )

        result_ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        scored = []
        for idx, chunk_id in enumerate(result_ids):
            chunk = chunk_by_id.get(str(chunk_id))
            if not chunk:
                continue
            distance = distances[idx] if idx < len(distances) else None
            item = dict(chunk)
            if distance is not None:
                item["relevance_score"] = round(max(0.0, 1.0 - float(distance)), 6)
            item["_vector_store"] = "chroma"
            scored.append(item)
            if len(scored) >= top_k:
                break

        return scored

    def query_documents(
        self,
        query: str,
        all_documents: List[Dict[str, Any]],
        embedding_service,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Query the document-level HNSW index and hydrate from SQLite rows."""
        if not query or not all_documents or not self.is_available():
            return []

        document_by_id = {
            str(document.get("id")): document
            for document in all_documents
            if document.get("id")
        }
        if not document_by_id:
            return []

        query_embedding = embedding_service.embed_texts([query])[0]
        collection = self._get_document_collection()
        n_results = min(max(top_k * 5, top_k), max(len(document_by_id), top_k))
        result = collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["distances"],
        )

        result_ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        scored = []
        for idx, document_id in enumerate(result_ids):
            document = document_by_id.get(str(document_id))
            if not document:
                continue
            distance = distances[idx] if idx < len(distances) else None
            item = dict(document)
            if distance is not None:
                item["document_relevance_score"] = round(
                    max(0.0, 1.0 - float(distance)),
                    6,
                )
            item["_vector_store"] = "chroma_document"
            scored.append(item)
            if len(scored) >= top_k:
                break

        return scored

    def query_images(
        self,
        query: str,
        all_images: List[Dict[str, Any]],
        multimodal_service,
        top_k: int,
    ) -> List[Dict[str, Any]]:
        """Query the multimodal image index using a text embedding."""
        if not query or not all_images or not self.is_available():
            return []

        image_by_id = {
            str(image.get("id")): image
            for image in all_images
            if image.get("id")
        }
        if not image_by_id:
            return []

        selected_doc_ids = sorted({
            str(image.get("document_id"))
            for image in all_images
            if image.get("document_id")
        })
        if not selected_doc_ids:
            return []

        query_embeddings = multimodal_service.embed_texts([query])
        if not query_embeddings:
            return []

        collection = self._get_image_collection()
        n_results = min(max(top_k * 5, top_k), max(len(image_by_id), top_k))
        result = collection.query(
            query_embeddings=[query_embeddings[0]],
            n_results=n_results,
            where={"document_id": {"$in": selected_doc_ids}},
            include=["distances"],
        )

        result_ids = (result.get("ids") or [[]])[0]
        distances = (result.get("distances") or [[]])[0]
        scored = []
        for idx, image_id in enumerate(result_ids):
            image = image_by_id.get(str(image_id))
            if not image:
                continue
            distance = distances[idx] if idx < len(distances) else None
            item = dict(image)
            if distance is not None:
                item["image_relevance_score"] = round(max(0.0, 1.0 - float(distance)), 6)
            item["_vector_store"] = "chroma_image"
            scored.append(item)
            if len(scored) >= top_k:
                break
        return scored

    def _get_client(self):
        if self._client is not None:
            return self._client

        import chromadb

        Path(self.persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=self.persist_dir)
        return self._client

    def _get_chunk_collection(self):
        if self._chunk_collection is not None:
            return self._chunk_collection

        self._chunk_collection = self._get_client().get_or_create_collection(
            name=self.chunk_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._chunk_collection

    def _get_document_collection(self):
        if self._document_collection is not None:
            return self._document_collection

        self._document_collection = self._get_client().get_or_create_collection(
            name=self.document_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._document_collection

    def _get_image_collection(self):
        if self._image_collection is not None:
            return self._image_collection

        self._image_collection = self._get_client().get_or_create_collection(
            name=self.image_collection_name,
            metadata={"hnsw:space": "cosine"},
        )
        return self._image_collection

    @staticmethod
    def _metadata(chunk: Dict[str, Any]) -> Dict[str, Any]:
        heading_path = chunk.get("heading_path", [])
        if not isinstance(heading_path, str):
            heading_path = json.dumps(heading_path, ensure_ascii=False)
        return {
            "document_id": str(chunk.get("document_id", "")),
            "parent_id": str(chunk.get("parent_id", "")),
            "chunk_type": str(chunk.get("chunk_type", "")),
            "chunk_index": int(chunk.get("chunk_index") or 0),
            "image_index": int(chunk.get("image_index") or -1),
            "title": str(chunk.get("title", "")),
            "heading_path": heading_path,
            "embedding_model": str(chunk.get("embedding_model", "")),
        }

    @staticmethod
    def _document_metadata(document: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "filename": str(document.get("filename", "")),
            "file_type": str(document.get("file_type", "")),
            "status": str(document.get("status", "")),
            "embedding_model": str(document.get("embedding_model", "")),
        }

    @staticmethod
    def _image_text(image: Dict[str, Any]) -> str:
        caption = image.get("caption", "")
        ocr_text = image.get("ocr_text", "")
        return f"{caption}\n{ocr_text}".strip()

    @staticmethod
    def _image_metadata(image: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "document_id": str(image.get("document_id", "")),
            "image_index": int(image.get("image_index") or 0),
            "page_num": int(image.get("page_num") or 0),
            "image_embedding_model": str(image.get("image_embedding_model", "")),
        }


_document_vector_store_service: Optional[DocumentVectorStoreService] = None


def get_document_vector_store_service() -> DocumentVectorStoreService:
    global _document_vector_store_service
    if _document_vector_store_service is None:
        _document_vector_store_service = DocumentVectorStoreService()
    return _document_vector_store_service
