1. [DONE] Answer are given in raw markdown format without rendering. (Added marked.js + full CSS styling for tables, headers, lists, and code blocks).
2. [DONE] Test the file uploading.
3. [DONE] Is the output size limited by llm's max output tokens? (Explained num_ctx 8192 buffer, prompt vs output split, and 4500-token capacity).
4. [DONE] Has our RAG a reranker, what is it and how does it work? (Implemented FlashRank Cross-Encoder in src/reranker.py & 2-stage retriever pipeline).
5. [DONE] Is it possible to make progress bar while a query is processed. (Implemented real-time SSE stage progress bar in backend & Web UI).
6. [DONE] Has Qudrant its knowladge vector graph and if so can we visialise it? (Implemented 2D PCA projection endpoint /api/graph and interactive HTML5 vector map with pan/zoom/hover tooltips in Web UI).
7. [DONE] Dash board about documents DB statistics. (Implemented /api/stats endpoint & interactive modal dashboard with volume, vector counts, categories, and document inventory).
8. [DONE] Test comparison between embedders installed. (Conducted benchmark on BGE-M3 vs Nomic-Embed vs FRIDA; documented findings in evaluation_and_benchmarks.md).
9. [DONE] Prompt enginnering & LLM-as-a-Judge Guardrail. (Implemented strict prompt engineering rules and built src/judge.py single-purpose LLM-as-a-Judge fact-checking audit in stream pipeline and Web UI).
10. [DONE] Check to_improve.md for improvement. (Audited roadmap; updated completed features: Reranker, 3D Graph, SSE Progress, Judge Guardrail, DB Stats; refined next high-impact backlog items).
11. [DONE] Full codebase audit, linting, typing & optimization. (Applied comprehensive refactoring across all 8 src/ modules: unified module-level imports, added proper type annotations, extracted _build_sources helper, batched Qdrant upserts, path-traversal guard in /api/documents, flashrank top-level import, empty-text guard in chunker, _extract_token_from_sse helper in api.py, _fallback_result method in judge.py, _resolve_dimension classmethod in embeddings.py. All files passed py_compile syntax check).
12. [DONE] Plan A (Dual-Search Query Rewriter) + Two-Stage Dynamic Slot-Filling Clarifier + As-You-Type Autocomplete. (Implemented src/query_rewriter.py for Dual-Search candidate pooling, src/query_clarifier.py with 15ms Qdrant vector pre-search, fast LLM slot-filling analysis, <1ms LRU warm cache, and debounced as-you-type autocomplete with Tab completion in Web UI).
13. [DONE] Migrate Qdrant to autonomous server daemon / container architecture. (Deployed standalone pure-Rust Qdrant daemon v1.13.4 on ports 6333/6334 with dedicated NVMe storage, streamed all 32,772 points without data loss in 15s, updated src/indexer.py and src/retriever.py with auto-connecting client and local-mode fallback, and eliminated >20k points warning and file lock conflicts).
14. [x] **Excel parsing & DFMEA/BOM Ingestion**:
   - Implemented cell-unmerging propagation for complex hierarchical table headers.
   - Preserved multi-row table headers (`Текущие средства контроля [Предупреждение]`, `ПЧР (RPN)`).
   - Atomic micro-chunking for tabular rows (bypassing destructive sliding-window word chunker).
   - Added Exact-Document Keyword Boosting & Hybrid Multilingual Dense + Cross-Encoder scoring.
15. [x] **Investigate Page Assist RAG Architecture on GitHub**:
   - Analyzed `@langchain/community` `MemoryVectorStore`, client-side `pdfjs-dist`, `Readability.js`, and browser `IndexedDB` vs our enterprise server Qdrant HNSW backend.
16. [x] **Comprehensive Technical Architecture Documentation**:
   - Created `project_docs/technical_documentation.md` detailing system architecture, Qdrant daemon, BGE-M3, Cross-Encoder reranker, OllamaOCR, Co-Pilot, and API specifications.
17. [x] **Adaptive OllamaOCR via LLaMA-3.2-Vision (11B VLM) & CJK Unicode Filter**:
   - Upgraded OCR from legacy `minicpm-v` to Meta's `llama3.2-vision:11b` running locally on the RTX A6000 GPU with zero-temperature greedy decoding.
   - Built an active regex filter in `src/parsers/pdf_parser.py` stripping accidental CJK/Asian character hallucinations.
   - Cleaned all legacy OCR Markdown files and re-embedded vector points in Qdrant with `bge-m3`. Database is 100% verified with 0 foreign character artifacts.
18. [x] **In-Browser Universal Document Viewer Modal**:
   - Implemented inline PDF/Excel/Document serving in `/api/documents/{filename}` with `Content-Disposition: inline`.
   - Built a sleek glassmorphism modal viewer directly in `web_client/index.html` with automatic link interception, `#page=N` direct targeting, and ESC/fullscreen support. Eliminates local downloads and browser tab freezing.
19. [x] **Small-to-Big Parent Page Hydration (Full PageIndex Pattern)**:
   - Implemented `hydrate_parent_page` in `src/retriever.py` to automatically reconstruct the 100% full parent page text (with headers, footnotes, tables, and units) for all winning candidate chunks.
   - Preserves atomic row structures for Excel DFMEA/BOM.
   - Enforces a 12,000-character safety context budget with feature toggle `enable_parent_page_hydration: bool = True`. 
20. [x] **Universal In-Browser Document & Spreadsheet Previewer (`/api/documents/preview/{filename}`)**:
   - Implemented native server-side HTML conversion for `.xlsx` spreadsheets (with multi-sheet tab switching, sticky frozen headers, and instant cell search via `openpyxl`).
   - Implemented formatted HTML conversion for `.docx` documents (preserving headings, tables, and typography via `mammoth`).
   - Integrated with the Web UI modal viewer so clicking any Word, Excel, or PDF document immediately displays the formatted content inside the modal with zero download prompts.
21. [x] **Integrated In-Browser Stage 1 Markdown Quality Inspector (`/api/documents/markdown-preview/{filename}`) & UX Analytics**:
   - Built backend endpoint rendering extracted structured Markdown in Monokai theme with `marked.js`, table styling, copy button, and KPI header (character count, word count, chunks, table lines).
   - Replaced file downloads in Document Registry with direct in-browser inspection (`📝 Верификация MD` button).
   - Built persistent telemetry engine in `src/analytics.py` (SQLite `data/analytics.db`) tracking query latencies, retrieval times, verification ratios, real Client IP leaderboards, and daily bar chart trends.
22. [ ] **Investigate SaaS «ТехЭксперт» Integration Pipeline**:
   - Explore connecting desktop/enterprise installation of TechExpert (ТехЭксперт) via standard export / file-sync connector to GASlight-Me RAG. 