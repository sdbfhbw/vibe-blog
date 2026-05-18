"""
Rebuild embeddings for existing knowledge chunks.

Usage from repo root:
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\reembed_document_chunks.py

Optional:
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\reembed_document_chunks.py --doc-id <document_id>
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\reembed_document_chunks.py --only-missing
    backend\\.venv\\Scripts\\python.exe backend\\scripts\\reembed_document_chunks.py --dry-run
"""
import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

BACKEND_DIR = Path(__file__).resolve().parents[1]
if str(BACKEND_DIR) not in sys.path:
    sys.path.insert(0, str(BACKEND_DIR))

from services.database_service import DatabaseService
from services.document_embedding_service import get_document_embedding_service


def parse_args():
    parser = argparse.ArgumentParser(description="Rebuild document chunk embeddings.")
    parser.add_argument("--doc-id", help="Only rebuild chunks for one document_id.")
    parser.add_argument("--only-missing", action="store_true", help="Only rebuild chunks without embedding.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be updated without writing.")
    return parser.parse_args()


def fetch_chunks(db: DatabaseService, doc_id: str = None, only_missing: bool = False):
    clauses = []
    params = []
    if doc_id:
        clauses.append("document_id = ?")
        params.append(doc_id)
    if only_missing:
        clauses.append("(embedding IS NULL OR embedding = '')")

    where_sql = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    with db.get_connection() as conn:
        cursor = conn.execute(
            f"""
            SELECT *
            FROM knowledge_chunks
            {where_sql}
            ORDER BY document_id, chunk_index
            """,
            params,
        )
        return [dict(row) for row in cursor.fetchall()]


def update_embeddings(db: DatabaseService, chunks):
    service = get_document_embedding_service()
    enriched = service.enrich_chunks(chunks)
    rows = [
        (
            chunk.get("embedding"),
            chunk.get("embedding_model"),
            chunk.get("embedding_dim"),
            chunk.get("id"),
        )
        for chunk in enriched
    ]
    with db.get_connection() as conn:
        conn.executemany(
            """
            UPDATE knowledge_chunks
            SET embedding = ?, embedding_model = ?, embedding_dim = ?
            WHERE id = ?
            """,
            rows,
        )
    return enriched


def main():
    load_dotenv(BACKEND_DIR / ".env")
    args = parse_args()
    db = DatabaseService()
    chunks = fetch_chunks(db, doc_id=args.doc_id, only_missing=args.only_missing)
    service = get_document_embedding_service()

    print(f"provider={service.provider}, model={service.model_name}")
    print(f"chunks={len(chunks)}, doc_id={args.doc_id or '*'}, only_missing={args.only_missing}")

    if args.dry_run or not chunks:
        return

    updated = update_embeddings(db, chunks)
    try:
        from services.document_vector_store_service import get_document_vector_store_service
        get_document_vector_store_service().upsert_chunks(updated)
    except Exception as exc:
        print(f"chroma_upsert_skipped={exc}")
    dim = updated[0].get("embedding_dim") if updated else None
    print(f"updated={len(updated)}, embedding_dim={dim}")


if __name__ == "__main__":
    main()
