from services.blog_generator.agents.researcher import ResearcherAgent
from services.document_embedding_service import DocumentEmbeddingService
from services.file_parser_service import FileParserService


def test_chunk_markdown_persists_parent_and_child_chunks(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_CHUNK_TOKEN_SIZE", "8")
    monkeypatch.setenv("KNOWLEDGE_CHUNK_TOKEN_OVERLAP", "1")
    parser = FileParserService(mineru_token="")
    markdown = (
        "# Redis\n\n"
        "cache penetration bloom filter mutex lock\n\n"
        "cache breakdown hot key logical expiration"
    )

    chunks = parser.chunk_markdown(markdown, chunk_size=24, chunk_overlap=3)

    parents = [chunk for chunk in chunks if chunk.get("chunk_type") == "parent"]
    children = [chunk for chunk in chunks if chunk.get("chunk_type") != "parent"]

    assert len(parents) == 1
    assert len(children) >= 2
    assert parents[0]["parent_id"] == children[0]["parent_id"]
    assert parents[0]["content"].strip() == markdown.strip()


def test_parent_chunks_are_not_embedded(monkeypatch):
    monkeypatch.setenv("DOCUMENT_EMBEDDING_PROVIDER", "local_hash")
    service = DocumentEmbeddingService()
    chunks = [
        {
            "chunk_type": "parent",
            "title": "Redis",
            "content": "full section",
            "parent_id": "parent_1",
        },
        {
            "chunk_type": "paragraph",
            "title": "Redis Part 1",
            "content": "cache breakdown",
            "parent_id": "parent_1",
        },
    ]

    enriched = service.enrich_chunks(chunks)

    assert "embedding" not in enriched[0]
    assert enriched[1]["embedding"]


def test_expand_parent_chunks_returns_parent_once_for_multiple_child_hits():
    parent = {
        "id": "parent_row",
        "document_id": "doc_1",
        "chunk_type": "parent",
        "parent_id": "parent_1",
        "content": "full parent section",
    }
    child_a = {
        "id": "child_a",
        "document_id": "doc_1",
        "chunk_type": "paragraph",
        "parent_id": "parent_1",
        "_retrieval_sources": ["vector"],
        "_rrf_score": 0.02,
    }
    child_b = {
        "id": "child_b",
        "document_id": "doc_1",
        "chunk_type": "paragraph",
        "parent_id": "parent_1",
        "_retrieval_sources": ["bm25"],
        "_rrf_score": 0.03,
    }

    expanded = ResearcherAgent._expand_parent_chunks(
        [child_a, child_b],
        [parent, child_a, child_b],
    )

    assert len(expanded) == 1
    assert expanded[0]["id"] == "parent_row"
    assert expanded[0]["_parent_expanded"] is True
    assert expanded[0]["_matched_child_ids"] == ["child_a", "child_b"]
    assert expanded[0]["_retrieval_sources"] == ["vector", "bm25"]
    assert expanded[0]["_rrf_score"] == 0.03


def test_expand_parent_chunks_falls_back_to_neighbors_for_legacy_chunks():
    child_a = {
        "id": "child_a",
        "document_id": "doc_1",
        "chunk_type": "paragraph",
        "parent_id": "parent_legacy",
        "chunk_index": 1,
    }
    child_b = {
        "id": "child_b",
        "document_id": "doc_1",
        "chunk_type": "paragraph",
        "parent_id": "parent_legacy",
        "chunk_index": 2,
    }

    expanded = ResearcherAgent._expand_parent_chunks(
        [child_a],
        [child_a, child_b],
        fallback_neighbor_window=1,
    )

    assert [chunk["id"] for chunk in expanded] == ["child_a", "child_b"]
    assert all(chunk["_parent_expanded"] is False for chunk in expanded)


def test_chunk_markdown_splits_python_code_by_ast_boundaries():
    parser = FileParserService(mineru_token="")
    markdown = (
        "# Demo\n\n"
        "```python\n"
        "import os\n\n"
        "def first():\n"
        "    return 1\n\n"
        "class Worker:\n"
        "    def run(self):\n"
        "        return os.getcwd()\n"
        "```\n"
    )

    chunks = parser.chunk_markdown(markdown)
    code_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "code"]

    assert len(code_chunks) == 2
    assert "def first" in code_chunks[0]["content"]
    assert "class Worker" in code_chunks[1]["content"]
    assert "import os" in code_chunks[0]["content"]
    assert "import os" in code_chunks[1]["content"]


def test_chunk_markdown_preserves_table_header_when_split(monkeypatch):
    monkeypatch.setenv("KNOWLEDGE_CHUNK_TOKEN_SIZE", "12")
    parser = FileParserService(mineru_token="")
    markdown = (
        "# Metrics\n\n"
        "| name | value |\n"
        "| --- | --- |\n"
        "| alpha | 1 |\n"
        "| beta | 2 |\n"
        "| gamma | 3 |\n"
    )

    chunks = parser.chunk_markdown(markdown, chunk_size=36, chunk_overlap=3)
    table_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "table"]

    assert len(table_chunks) >= 2
    assert all(chunk["content"].startswith("| name | value |\n| --- | --- |") for chunk in table_chunks)


def test_chunk_markdown_builds_image_chunks_from_caption_and_ocr():
    parser = FileParserService(mineru_token="")
    markdown = "# Diagram\n\n![cache flow](/files/mineru/x/diagram.png)\n"
    images = [{
        "url": "/files/mineru/x/diagram.png",
        "filename": "diagram.png",
        "caption": "A cache flow diagram",
        "ocr_text": "GET /items",
    }]

    chunks = parser.chunk_markdown(markdown, images=images)
    image_chunks = [chunk for chunk in chunks if chunk.get("chunk_type") == "image"]

    assert len(image_chunks) == 1
    assert image_chunks[0]["image_index"] == 0
    assert "Alt text: cache flow" in image_chunks[0]["content"]
    assert "Caption: A cache flow diagram" in image_chunks[0]["content"]
    assert "OCR: GET /items" in image_chunks[0]["content"]


def test_expand_context_units_keeps_structured_chunks_direct():
    parent = {
        "id": "parent_row",
        "document_id": "doc_1",
        "chunk_type": "parent",
        "parent_id": "parent_1",
        "content": "full parent section",
    }
    paragraph = {
        "id": "child_text",
        "document_id": "doc_1",
        "chunk_type": "paragraph",
        "parent_id": "parent_1",
    }
    code = {
        "id": "child_code",
        "document_id": "doc_1",
        "chunk_type": "code",
        "parent_id": "parent_1",
    }

    expanded = ResearcherAgent._expand_context_units(
        [paragraph, code],
        [parent, paragraph, code],
    )

    assert [chunk["id"] for chunk in expanded] == ["child_code", "parent_row"]
    assert expanded[0]["_context_unit"] == "direct"
    assert expanded[1]["_context_unit"] == "parent"


def test_hydrate_image_vector_chunks_maps_images_to_image_chunks():
    image_chunk = {
        "id": "chunk_image",
        "document_id": "doc_1",
        "chunk_type": "image",
        "image_index": 0,
        "relevance_score": 0.2,
    }
    image_hit = {
        "id": "img_doc_1_0",
        "document_id": "doc_1",
        "image_index": 0,
        "image_relevance_score": 0.91,
        "_vector_store": "chroma_image",
    }

    hydrated = ResearcherAgent._hydrate_image_vector_chunks([image_hit], [image_chunk])

    assert len(hydrated) == 1
    assert hydrated[0]["id"] == "chunk_image"
    assert hydrated[0]["relevance_score"] == 0.91
    assert hydrated[0]["_image_vector_score"] == 0.91
    assert hydrated[0]["_retrieval_sources"] == ["image_vector"]
