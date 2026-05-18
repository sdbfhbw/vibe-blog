"""Generate simple real bitmap assets for multimodal offline evaluation."""

from __future__ import annotations

import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

OUTPUT_DIR = Path(__file__).with_name("images")

ASSETS = {
    "rag_pipeline.png": ["Document Recall", "BM25 / Vector", "RRF", "Rerank", "Parent-child", "Citations"],
    "bm25_terms.png": ["BM25", "Exact terms", "function_name", "error_code"],
    "embedding_space.png": ["Query", "Embedding Space", "Semantic Neighbors"],
    "hnsw_layers.png": ["Layer 2", "Layer 1", "Layer 0", "Entry Point"],
    "cross_encoder_rerank.png": ["Query + Candidate", "Cross-Encoder", "Relevance Score"],
    "langgraph_workflow.png": ["retrieve", "rerank", "generate", "citations"],
    "sse_stream_ui.png": ["Streaming Output", "data: token", "Markdown Preview"],
    "python_ast_tree.png": ["Module", "ClassDef", "FunctionDef", "Assign"],
    "markdown_heading_tree.png": ["# Title", "## Section", "### Child"],
    "table_header_error.png": ["metric", "MRR@10", "Recall@10", "Latency_ms"],
    "pdf_parse_flow.png": ["PDF", "Markdown", "text", "code", "table", "image"],
    "multimodal_retrieval_arch.png": ["Caption/OCR", "Image Embedding", "Fusion", "Rerank"],
    "sqlite_schema_error.png": ["SQLITE_BUSY", "database is locked"],
    "chroma_collection_error.png": ["collection", "rag/blog:demo", "invalid name"],
    "eval_dashboard.png": ["Hit@K", "MRR", "nDCG", "Citation Accuracy"],
}


def _font(size: int) -> ImageFont.ImageFont:
    for candidate in (
        "C:/Windows/Fonts/msyh.ttc",
        "C:/Windows/Fonts/arial.ttf",
    ):
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def generate_images() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    title_font = _font(30)
    text_font = _font(24)
    count = 0
    for file_name, labels in ASSETS.items():
        image = Image.new("RGB", (1200, 720), "#f7f8fb")
        draw = ImageDraw.Draw(image)
        draw.rounded_rectangle((40, 40, 1160, 680), radius=24, outline="#22324d", width=4, fill="#ffffff")
        draw.text((70, 70), file_name.replace(".png", "").replace("_", " ").title(), fill="#111827", font=title_font)

        if len(labels) >= 4:
            x = 90
            y = 290
            for idx, label in enumerate(labels):
                box = (x, y, x + 150, y + 90)
                draw.rounded_rectangle(box, radius=12, fill="#dbeafe", outline="#2563eb", width=3)
                draw.multiline_text((x + 12, y + 22), label, fill="#1e3a8a", font=text_font, spacing=4)
                if idx < len(labels) - 1:
                    draw.line((x + 150, y + 45, x + 185, y + 45), fill="#334155", width=4)
                    draw.polygon([(x + 185, y + 45), (x + 170, y + 36), (x + 170, y + 54)], fill="#334155")
                x += 185
        else:
            y = 220
            for label in labels:
                draw.rounded_rectangle((110, y, 1090, y + 82), radius=12, fill="#ecfeff", outline="#0891b2", width=3)
                draw.text((140, y + 24), label, fill="#164e63", font=text_font)
                y += 110

        image.save(OUTPUT_DIR / file_name)
        count += 1
    return count


def main() -> None:
    print(json.dumps({"generated_images": generate_images()}, ensure_ascii=False))


if __name__ == "__main__":
    main()
