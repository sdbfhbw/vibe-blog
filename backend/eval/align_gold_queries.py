"""Align virtual gold labels to actual imported chunk ids."""

from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any, Dict, List

DEFAULT_MANIFEST = Path(__file__).with_name("chunk_manifest.csv")


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9_+\-./#]+|[\u4e00-\u9fff]", (text or "").lower()))


def _load_manifest(path: Path) -> List[Dict[str, str]]:
    with path.open(encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def _score(sample: Dict[str, Any], chunk: Dict[str, str]) -> float:
    text = " ".join(
        [
            sample.get("query", ""),
            sample.get("expected_answer", ""),
            " ".join(sample.get("required_content_types", [])),
        ]
    )
    query_tokens = _tokenize(text)
    chunk_tokens = _tokenize(
        " ".join(
            [
                chunk.get("title", ""),
                chunk.get("heading_path", ""),
                chunk.get("content_preview", ""),
            ]
        )
    )
    overlap = len(query_tokens.intersection(chunk_tokens))
    requested_types = set(sample.get("required_content_types", []))
    chunk_type = chunk.get("chunk_type", "")
    type_bonus = 2.0 if chunk_type in requested_types else 0.0
    if "text" in requested_types and chunk_type == "section":
        type_bonus = 2.0
    return overlap + type_bonus


def align_samples(samples: List[Dict[str, Any]], manifest: List[Dict[str, str]], top_n: int = 3) -> List[Dict[str, Any]]:
    aligned = []
    for sample in samples:
        item = dict(sample)
        docs = set(item.get("relevant_document_ids", []))
        candidates = [
            chunk
            for chunk in manifest
            if chunk.get("document_id") in docs and chunk.get("chunk_type") != "parent"
        ]
        ranked = sorted(candidates, key=lambda chunk: _score(item, chunk), reverse=True)
        item["relevant_chunk_ids"] = [chunk["chunk_id"] for chunk in ranked[:top_n] if _score(item, chunk) > 0]
        item["_alignment_candidates"] = [
            {
                "chunk_id": chunk["chunk_id"],
                "chunk_type": chunk["chunk_type"],
                "score": _score(item, chunk),
                "title": chunk["title"],
            }
            for chunk in ranked[:5]
        ]
        aligned.append(item)
    return aligned


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Raw gold JSONL with stable document ids")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("gold_queries_aligned.jsonl"))
    args = parser.parse_args()

    aligned = align_samples(_load_jsonl(args.input), _load_manifest(args.manifest))
    with args.output.open("w", encoding="utf-8") as handle:
        for item in aligned:
            handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(json.dumps({"aligned_samples": len(aligned), "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
