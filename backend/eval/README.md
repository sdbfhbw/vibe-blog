# RAG Offline Evaluation

This directory contains a repeatable local workflow for evaluating the document
retrieval stack against the 15-document corpus.

## Workflow

1. Generate the real bitmap assets:

   ```powershell
   .\backend\.venv\Scripts\python.exe backend/eval/generate_eval_images.py
   ```

2. Import the corpus with stable `doc_id` values:

   ```powershell
   .\backend\.venv\Scripts\python.exe backend/eval/import_eval_corpus.py d:\Chrome\rag_blog_eval_corpus_15_docs.zip
   ```

3. Export the actual chunk layout produced by the parser:

   ```powershell
   .\backend\.venv\Scripts\python.exe backend/eval/export_chunk_manifest.py
   ```

4. Build the formal 120-item gold set from the current manifest:

   ```powershell
   .\backend\.venv\Scripts\python.exe backend/eval/build_gold_queries.py
   .\backend\.venv\Scripts\python.exe backend/eval/align_gold_queries.py backend/eval/gold_queries_raw.jsonl --output backend/eval/gold_queries_aligned.jsonl
   ```

5. Run the smoke retrieval benchmark:

   ```powershell
   .\backend\.venv\Scripts\python.exe backend/eval/run_retrieval_eval.py
   ```

6. Run the formal 120-item retrieval benchmark:

   ```powershell
   .\backend\.venv\Scripts\python.exe backend/eval/run_gold_retrieval_eval.py
   ```

7. Run multimodal smoke evaluation:

   ```powershell
   .\backend\.venv\Scripts\python.exe backend/eval/run_multimodal_eval.py
   ```

## Notes

- The original bundle contains image metadata placeholders. `generate_eval_images.py`
  creates deterministic PNG assets with the same names, so importing the corpus
  exercises both text metadata and true image-vector retrieval.
- The smoke benchmark is intentionally small and deterministic. It validates the
  end-to-end offline pipeline before a larger gold dataset is aligned to the
  generated chunk ids.
- `align_gold_queries.py` converts a raw JSONL gold set that references stable
  document ids into a concrete JSONL with actual imported `chunk_id` values:

  ```powershell
  .\backend\.venv\Scripts\python.exe backend/eval/align_gold_queries.py path\to\gold_queries_raw.jsonl
  ```
- By default the scripts use the repository services exactly as configured by the
  environment. For a fully offline deterministic run, set:

  ```powershell
  $env:DOCUMENT_EMBEDDING_PROVIDER = "local_hash"
  $env:DOCUMENT_VECTOR_STORE = "sqlite"
  ```
