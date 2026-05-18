"""Export the actual chunk layout created by the document parser."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.database_service import get_db_service

OUTPUT_PATH = Path(__file__).with_name("chunk_manifest.csv")


def export_manifest() -> int:
    db = get_db_service()
    documents = db.list_ready_documents_for_retrieval()
    rows = []
    for document in documents:
        for chunk in db.get_chunks_by_document(document["id"]):
            rows.append(
                {
                    "document_id": document["id"],
                    "chunk_id": chunk["id"],
                    "chunk_index": chunk["chunk_index"],
                    "chunk_type": chunk["chunk_type"],
                    "parent_id": chunk.get("parent_id") or "",
                    "heading_path": chunk.get("heading_path") or "",
                    "title": chunk.get("title") or "",
                    "content_preview": (chunk.get("content") or "").replace("\n", " ")[:180],
                }
            )

    with OUTPUT_PATH.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()) if rows else [
            "document_id",
            "chunk_id",
            "chunk_index",
            "chunk_type",
            "parent_id",
            "heading_path",
            "title",
            "content_preview",
        ])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def main() -> None:
    count = export_manifest()
    print(f"exported_chunks={count}")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
