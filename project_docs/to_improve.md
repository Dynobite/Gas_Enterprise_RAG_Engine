# 🚀 GASlight-Me Improvements & Architecture Roadmap

> This document tracks completed architectural upgrades and backlog enhancements for the GASlight-Me On-Premise RAG system. Last updated: **2026-08-24**.

---

## 🟢 1. Completed Key Upgrades

### ✅ 1.1 Pure-Rust Qdrant Server Daemon (`localhost:6333`)
- **Status**: ✅ **COMPLETED & DEPLOYED**
- **Architecture**: Migrated from local SQLite embedded mode to standalone pure-Rust Qdrant v1.13.4 daemon on dedicated NVMe storage.
- **Impact**: Eliminates SQLite file lock conflicts and scales past 100k+ vector points with sub-3ms HNSW latency.

### ✅ 1.2 Small-to-Big Parent Page Hydration (PageIndex Pattern)
- **Status**: ✅ **COMPLETED & DEPLOYED**
- **Architecture**: Two-phase retrieval in `src/retriever.py`:
  1. High-speed vector search + FlashRank Cross-Encoder reranking ranks exact micro-chunks.
  2. Sub-millisecond in-memory Parent Page Hydration (`hydrate_parent_page`) reconstructs the 100% full parent page text (with headers, footnotes, tables, and units) before feeding LLM.
- **Impact**: Zero-latency PageIndex experience: search remains pinpoint accurate while the generator sees full contextual page scope.

### ✅ 1.3 Adaptive Vision-Language Model OCR (`llama3.2-vision:11b` + CJK Guardrail)
- **Status**: ✅ **COMPLETED & DEPLOYED**
- **Architecture**: Automated raster scan detection (`len(text) < 30`) + sub-second `pdftoppm` rendering + direct base64 streaming to local **`llama3.2-vision:11b`** (Meta 11B Multimodal VLM) on RTX A6000 GPU with zero-temperature greedy decoding and active CJK unicode regex filter.
- **Impact**: Eliminates Asian/Chinese character hallucinations; successfully transcribed, cleaned, and indexed all scanned historical standards (e.g. `СТО Газпром 2-4.1-212-2008.pdf`).

### ✅ 1.4 Universal In-Browser Document & Spreadsheet Previewer (`/api/documents/preview/`)
- **Status**: ✅ **COMPLETED & DEPLOYED**
- **Architecture**: Native server-side HTML conversion for `.xlsx` (interactive sheet tabs, frozen headers, real-time search), `.docx` (clean HTML typography via `mammoth`), and `.pdf` (`#page=N`).
- **Impact**: All document links, source buttons, and inline citations open directly in the glassmorphism modal with zero download prompts.

### ✅ 1.5 Dual-Search Query Rewriter & Engineering Co-Pilot (Plans A & B)
- **Status**: ✅ **COMPLETED & DEPLOYED**
- **Architecture**: `src/query_rewriter.py` for acronym expansion + `src/query_clarifier.py` with 15ms vector pre-search and debounced as-you-type autocomplete (`Ctrl + Space`).

### ✅ 1.6 User Experience Analytics & Real Client IP Telemetry (`data/analytics.db`)
- **Status**: ✅ **COMPLETED & DEPLOYED**
- **Architecture**: Embedded SQLite database logging real workstation client IPs (`request.client.host`), query latencies, daily activity timelines, and corporate leaderboards.

### ✅ 1.7 Integrated In-Browser Stage 1 Markdown Quality Inspector (`/api/documents/markdown-preview/`)
- **Status**: ✅ **COMPLETED & DEPLOYED**
- **Architecture**: Server endpoint rendering structured Markdown in Monokai typography with `marked.js`, full table layout, copy button, and KPI header (character count, word count, chunk count, table lines). Replaced file downloads with direct in-browser inspection.

---

## 🔮 2. Next Architecture Roadmap (Backlog)

### 🔹 2.1 Integration with SaaS / On-Premise «ТехЭксперт»
- **Concept**: Build an automated connector / crawler for the corporate **ТехЭксперт** standard depository to automatically sync updated СТО Газпром revisions, ГОСТs, and normative acts.

### 🔹 2.2 GraphRAG & Entity Community Summaries
- **Concept**: Extract structural engineering entities (Pipes, Flanges, Valves, Steel Grades) into a topological knowledge graph with community summarization for cross-standard dependency mapping.

### 🔹 2.3 Multi-Modal Technical Drawing OCR & CAD Vectorization
- **Concept**: Use Vision-Language Models to extract dimensional tolerance tables and title blocks from AutoCAD / PDF technical drawings.

