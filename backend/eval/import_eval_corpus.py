"""Import the 15-document offline RAG corpus with stable document ids."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))
load_dotenv(BACKEND_DIR / ".env")

from services.database_service import get_db_service
from services.document_embedding_service import get_document_embedding_service
from services.document_multimodal_embedding_service import get_document_multimodal_embedding_service
from services.document_vector_store_service import get_document_vector_store_service
from services.file_parser_service import get_file_parser, init_file_parser


def _parse_frontmatter(markdown: str) -> Dict[str, str]:
    match = re.match(r"^---\s*\n(.*?)\n---\s*\n", markdown, re.DOTALL)
    if not match:
        return {}
    metadata: Dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        metadata[key.strip()] = value.strip()
    return metadata


def _load_image_placeholders(images_dir: Path) -> Dict[str, Dict[str, Any]]:
    placeholders: Dict[str, Dict[str, Any]] = {}
    if not images_dir.exists():
        return placeholders
    for path in images_dir.glob("*.json"):
        data = json.loads(path.read_text(encoding="utf-8"))
        file_name = data.get("file_name")
        if not file_name:
            continue
        placeholders[file_name] = data
    return placeholders


def _images_for_document(
    markdown: str,
    placeholders: Dict[str, Dict[str, Any]],
    real_images_dir: Path | None = None,
) -> List[Dict[str, Any]]:
    images: List[Dict[str, Any]] = []
    for alt_text, image_ref in re.findall(r"!\[(.*?)\]\((.*?)\)", markdown):
        file_name = Path(image_ref).name
        placeholder = placeholders.get(file_name)
        if not placeholder:
            continue
        real_path = (real_images_dir / file_name) if real_images_dir else None
        images.append(
            {
                "filename": file_name,
                "url": image_ref,
                "path": str(real_path) if real_path and real_path.exists() else "",
                "image_path": str(real_path) if real_path and real_path.exists() else "",
                "caption": placeholder.get("caption", ""),
                "ocr_text": placeholder.get("ocr_text", ""),
                "alt_text": alt_text or placeholder.get("alt_text", ""),
                "page_num": 0,
            }
        )
    return images


def _reset_document_rows(doc_id: str) -> None:
    db = get_db_service()
    with db.get_connection() as conn:
        conn.execute("DELETE FROM knowledge_chunks WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM document_images WHERE document_id = ?", (doc_id,))
        conn.execute("DELETE FROM documents WHERE id = ?", (doc_id,))


def import_corpus(corpus_root: Path, real_images_dir: Path | None = None) -> Dict[str, int]:
    manifest = json.loads((corpus_root / "corpus_manifest.json").read_text(encoding="utf-8"))
    placeholders = _load_image_placeholders(corpus_root / "images")

    db = get_db_service()
    parser = get_file_parser()
    if parser is None:
        parser = init_file_parser(mineru_token="")
    embedding_service = get_document_embedding_service()
    multimodal_service = get_document_multimodal_embedding_service()
    vector_store = get_document_vector_store_service()

    imported_docs = 0
    imported_chunks = 0
    imported_images = 0

    for item in manifest:
        source_path = corpus_root / item["file"]
        markdown = source_path.read_text(encoding="utf-8")
        metadata = _parse_frontmatter(markdown)
        doc_id = metadata.get("doc_id") or item["doc_id"]
        _reset_document_rows(doc_id)

        db.create_document(
            doc_id=doc_id,
            filename=item["file"],
            file_path=str(source_path),
            file_size=source_path.stat().st_size,
            file_type="md",
        )
        db.save_parse_result(doc_id, markdown, None)

        images = _images_for_document(markdown, placeholders, real_images_dir)
        images = multimodal_service.enrich_images(images)
        if images:
            db.save_images(doc_id, images)

        chunks = parser.chunk_markdown(markdown, images=images)
        chunks = embedding_service.enrich_chunks(chunks)
        db.save_chunks(doc_id, chunks)
        saved_chunks = db.get_chunks_by_document(doc_id)
        vector_store.upsert_chunks(saved_chunks)

        summary = item.get("title", "")
        db.update_document_summary(doc_id, summary)
        enriched_document = embedding_service.enrich_document(
            {
                "id": doc_id,
                "filename": item["file"],
                "file_type": "md",
                "status": "ready",
                "summary": summary,
                "markdown_content": markdown,
            }
        )
        db.update_document_embedding(
            doc_id=doc_id,
            embedding=enriched_document["embedding"],
            embedding_model=enriched_document["embedding_model"],
            embedding_dim=enriched_document["embedding_dim"],
        )
        db.update_document_status(doc_id, "ready")

        imported_docs += 1
        imported_chunks += len(saved_chunks)
        imported_images += len(images)

    ready_documents = db.list_ready_documents_for_retrieval()
    vector_store.upsert_documents(ready_documents)
    all_images = []
    for document in ready_documents:
        all_images.extend(db.get_images_by_document(document["id"]))
    vector_store.upsert_images(all_images)
    return {
        "documents": imported_docs,
        "chunks": imported_chunks,
        "images": imported_images,
    }


def _extract_corpus(zip_path: Path) -> Path:
    temp_dir = Path(tempfile.mkdtemp(prefix="rag_eval_corpus_"))
    with zipfile.ZipFile(zip_path) as archive:
        archive.extractall(temp_dir)
    children = [path for path in temp_dir.iterdir() if path.is_dir()]
    if len(children) != 1:
        raise ValueError("Expected a single top-level corpus directory in the zip file")
    return children[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("corpus", type=Path, help="Corpus directory or zip archive")
    parser.add_argument(
        "--images-dir",
        type=Path,
        default=Path(__file__).with_name("images"),
        help="Directory containing real bitmap assets matching the placeholder names",
    )
    args = parser.parse_args()

    os.environ.setdefault("DOCUMENT_EMBEDDING_PROVIDER", "local_hash")
    os.environ.setdefault("DOCUMENT_VECTOR_STORE", "sqlite")

    corpus_path = args.corpus
    if corpus_path.suffix.lower() == ".zip":
        corpus_path = _extract_corpus(corpus_path)
    if not (corpus_path / "corpus_manifest.json").exists():
        raise FileNotFoundError(f"Missing corpus_manifest.json under {corpus_path}")

    result = import_corpus(corpus_path, real_images_dir=args.images_dir)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
