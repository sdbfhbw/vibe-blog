"""Build a 120-item gold set grounded in the current imported chunk manifest."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List

MANIFEST_PATH = Path(__file__).with_name("chunk_manifest.csv")
RAW_OUTPUT = Path(__file__).with_name("gold_queries_raw.jsonl")
ALIGNED_OUTPUT = Path(__file__).with_name("gold_queries_aligned.jsonl")


def _load_manifest() -> List[Dict[str, str]]:
    with MANIFEST_PATH.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _difficulty(index: int) -> str:
    return ("easy", "medium", "hard")[index % 3]


def _snippet(row: Dict[str, str]) -> str:
    return (row.get("content_preview") or row.get("title") or row["chunk_id"]).strip()


def _base_sample(
    index: int,
    category: str,
    row: Dict[str, str],
    query: str,
    answer: str,
    content_types: List[str],
    *,
    requires_parent_context: bool = False,
    requires_multimodal: bool = False,
    image_evidence_type: str | None = None,
) -> Dict[str, Any]:
    return {
        "id": f"eval_{index:03d}",
        "category": category,
        "difficulty": _difficulty(index),
        "query": query,
        "expected_answer": answer,
        "relevant_document_ids": [row["document_id"]],
        "relevant_chunk_ids": [row["chunk_id"]],
        "required_content_types": content_types,
        "requires_parent_context": requires_parent_context,
        "requires_multimodal": requires_multimodal,
        "negative_case": False,
        "image_evidence_type": image_evidence_type,
        "evaluation_focus": f"Evaluate {category} retrieval against real imported chunks.",
    }


def _take(rows: Iterable[Dict[str, str]], count: int) -> List[Dict[str, str]]:
    return list(rows)[:count]


def build_samples(rows: List[Dict[str, str]]) -> List[Dict[str, Any]]:
    by_type: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    by_doc: Dict[str, List[Dict[str, str]]] = defaultdict(list)
    for row in rows:
        by_type[row["chunk_type"]].append(row)
        by_doc[row["document_id"]].append(row)

    samples: List[Dict[str, Any]] = []
    idx = 1

    for row in _take(by_type["section"], 35):
        samples.append(
            _base_sample(
                idx,
                "text",
                row,
                f"请根据文档回答这一段主要说明什么：{row.get('title') or _snippet(row)[:36]}",
                _snippet(row),
                ["text"],
            )
        )
        idx += 1

    for row in _take(by_type["code"], 20):
        samples.append(
            _base_sample(
                idx,
                "code",
                row,
                f"请定位与这段代码最相关的实现：{_snippet(row)[:48]}",
                _snippet(row),
                ["code"],
            )
        )
        idx += 1

    for row in _take(by_type["table"], 15):
        samples.append(
            _base_sample(
                idx,
                "table",
                row,
                f"请找到包含该表格信息的 chunk：{row.get('title') or _snippet(row)[:36]}",
                _snippet(row),
                ["table"],
            )
        )
        idx += 1

    for row in _take(by_type["image"], 15):
        samples.append(
            _base_sample(
                idx,
                "image",
                row,
                f"请找到与这张图最相关的 image chunk：{row.get('title') or _snippet(row)[:36]}",
                _snippet(row),
                ["image_caption", "image_ocr"],
                requires_multimodal=True,
                image_evidence_type="caption+OCR",
            )
        )
        idx += 1

    parent_rows = [row for row in by_type["section"] if row.get("parent_id")]
    for row in _take(parent_rows, 15):
        samples.append(
            _base_sample(
                idx,
                "parent_child",
                row,
                f"这个子段落属于哪个上下文主题：{_snippet(row)[:42]}",
                _snippet(row),
                ["text"],
                requires_parent_context=True,
            )
        )
        idx += 1

    for document_id, doc_rows in list(by_doc.items())[:10]:
        seed = next((row for row in doc_rows if row["chunk_type"] == "section"), doc_rows[0])
        samples.append(
            _base_sample(
                idx,
                "document_recall",
                seed,
                f"我要查阅与 {document_id} 对应主题最相关的资料，应优先召回哪篇文档？",
                document_id,
                ["text"],
            )
        )
        idx += 1

    no_answer_queries = [
        "这些资料有没有说明如何把 Chroma 部署到 Kubernetes 集群？",
        "文档里是否给出了 Redis Streams 替代 SSE 的完整实现？",
        "资料是否证明 image embedding 在所有任务上都优于 OCR？",
        "文档有没有给出 GPT-5.1 的官方 RAG 基准分数？",
        "这些资料中是否包含 Vue 前端完整组件代码？",
    ]
    for query in no_answer_queries:
        samples.append(
            {
                "id": f"eval_{idx:03d}",
                "category": "no_answer",
                "difficulty": _difficulty(idx),
                "query": query,
                "expected_answer": "NOT_FOUND",
                "relevant_document_ids": [],
                "relevant_chunk_ids": [],
                "required_content_types": ["text"],
                "requires_parent_context": False,
                "requires_multimodal": False,
                "negative_case": True,
                "image_evidence_type": None,
                "evaluation_focus": "Evaluate abstention on unsupported questions.",
            }
        )
        idx += 1

    generation_specs = [
        (
            "生成一篇博客：混合检索为什么优于只用向量检索。",
            ["doc_bm25_sparse_retrieval", "doc_embedding_retrieval", "doc_rag_retrieval_pipeline"],
        ),
        (
            "生成一篇博客：从 PDF 解析到 chunk schema 的完整链路。",
            ["doc_pdf_markdown_parser", "doc_markdown_chunking_strategy", "doc_python_ast_chunking"],
        ),
        (
            "生成一篇博客：LangGraph 如何编排 retrieve、rerank、generate。",
            ["doc_langgraph_blog_agent", "doc_cross_encoder_rerank"],
        ),
        (
            "生成一篇博客：多模态 RAG 中图片 chunk 如何设计和检索。",
            ["doc_multimodal_retrieval", "doc_pdf_markdown_parser"],
        ),
        (
            "生成一篇博客：SQLite 和 Chroma 在本地 RAG 中分别负责什么。",
            ["doc_sqlite_chunk_storage", "doc_chroma_vector_store"],
        ),
    ]
    for query, document_ids in generation_specs:
        aligned_rows = [
            row
            for document_id in document_ids
            for row in by_doc[document_id]
            if row["chunk_type"] in {"section", "code", "table", "image"}
        ][:4]
        samples.append(
            {
                "id": f"eval_{idx:03d}",
                "category": "generation_topic",
                "difficulty": _difficulty(idx),
                "query": query,
                "expected_answer": "应基于相关资料生成结构化长文，并为关键结论提供可追踪证据。",
                "relevant_document_ids": document_ids,
                "relevant_chunk_ids": [row["chunk_id"] for row in aligned_rows],
                "required_content_types": sorted({row["chunk_type"] for row in aligned_rows}),
                "requires_parent_context": True,
                "requires_multimodal": any(row["chunk_type"] == "image" for row in aligned_rows),
                "negative_case": False,
                "image_evidence_type": "caption+OCR" if any(row["chunk_type"] == "image" for row in aligned_rows) else None,
                "evaluation_focus": "Evaluate multi-document evidence coverage for long-form generation.",
            }
        )
        idx += 1

    if len(samples) != 120:
        raise ValueError(f"Expected 120 samples, got {len(samples)}")
    return samples


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    samples = build_samples(_load_manifest())
    _write_jsonl(RAW_OUTPUT, samples)
    _write_jsonl(ALIGNED_OUTPUT, samples)
    print(
        json.dumps(
            {
                "samples": len(samples),
                "raw_output": str(RAW_OUTPUT),
                "aligned_output": str(ALIGNED_OUTPUT),
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
