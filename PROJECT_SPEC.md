# Project Spec

## Project Core
**Name:** GASlight-Me 🪔 (Инженерная База Знаний ОИиР)  
**Goal:** Enterprise Local RAG system for semantic search, instant verification, and expert question answering over Gas Pipeline Standards (СТО Газпром, ГОСТ, РД), Engineering Drawings, DFMEA Risk Matrices, and Valve BOM Specifications with zero hallucinations.  
**Environment:** Linux Ubuntu (Remote Host `c14753` @ `localhost`) + Dedicated NVIDIA RTX A6000 (48 GB VRAM) / Windows Development Client  
**Project Root:** `c:\Users\Lead-Engineer\Desktop\Gas_RAG` (Remote: `/home/ollamauser/RAG_SYSTEM`)  
**Runtime:** Python 3.10+ (Miniconda) / FastAPI / Ollama LLM Runtime / Pure-Rust Qdrant Vector Daemon  
**Entry Point:** `src/api.py` (Production URL: `http://localhost:8000`)

---

## 1. Architecture & Infrastructure

### 1.1 Vector Storage & Retrieval Layer
- **Qdrant Server Daemon:** Standalone pure-Rust daemon v1.13.4 running on ports `6333` (REST) and `6334` (gRPC) with NVMe storage. Eliminates SQLite lock contention and scales beyond 100k+ points.
- **Dense Embedding Model:** `bge-m3` (1024 dimensions, multilingual, Cosine metric, 8192 token context) served locally via Ollama.
- **Stage 2 Reranker:** `FlashRank` (Cross-Encoder `ms-marco-MiniLM-L-12-v2`) providing deep attention scoring in 15–20 ms.
- **Small-to-Big Parent Page Hydration:** FlashRank ranks precise micro-chunks, and `GasRagRetriever.hydrate_parent_page()` dynamically assembles the 100% full parent page text (with headers, footnotes, tables, and units) from Qdrant in < 1 ms before feeding LLM generator.

### 1.2 LLM & Vision Engine (Local GPU Hardware)
- **Primary Generator:** `qwen3.6:35b` (MoE) / `huihui_ai/deepseek-r1-abliterated:32b` running locally on NVIDIA RTX A6000 (48 GB VRAM).
- **Fast Interactive Co-Pilot & Quick Summary LLM:** `gpt-oss:20b` (20.9B parameter model) for ultra-fast pre-retrieval slot-filling analysis (~1.2s) and high-speed query summaries.
- **Vision-Language Model (OllamaOCR):** `llama3.2-vision:11b` (Meta 11B Multimodal VLM) for zero-hallucination OCR of raster/scanned PDF standards with zero-temperature greedy decoding and active CJK unicode filtering.
- **LLM-as-a-Judge Fact Auditor:** `src/judge.py` running in streaming pipeline to compute NLI Entailment, Grounding Ratio (%), and emit certification badges.

### 1.3 Pre-Retrieval & User Experience Subsystems
- **Dual-Search Query Rewriter (Plan A):** `src/query_rewriter.py` expands acronyms, standard codes, and expands candidate search pool.
- **Engineering Co-Pilot & Clarifier (Plan B):** `src/query_clarifier.py` provides 15 ms vector pre-search, fast slot-filling analysis, and 1-click query refine chips.
- **In-Browser Document & Spreadsheet Previewer:** `/api/documents/preview/{filename}` converts `.xlsx` (multi-sheet tabs & search), `.docx` (HTML typography), and `.pdf` (`#page=N`) natively without download prompts.
- **Integrated In-Browser Stage 1 Markdown Inspector:** `/api/documents/markdown-preview/{filename}` renders extracted structured Markdown in Monokai typography with KPI telemetry bar (characters, words, chunks, table stats) and copy button.
- **UX Analytics & Telemetry Engine:** `src/analytics.py` (SQLite `data/analytics.db`) tracking real Client IP addresses, query latencies, daily activity timelines, and leaderboard rankings.

---

## 2. Data Dictionary & Schemas

### 2.1 Qdrant Vector Collection: `gas_rag_standards`
- **Vector Dimension:** 1024 (Cosine distance).
- **Point Payload Schema:**
  - `text` (str): Processed markdown content or full reconstructed parent page text.
  - `source` (str): Raw filename (e.g., `СТО Газпром (ч1).pdf`, `ДПМА.491435.ЗД1 DFMEA.xlsx`).
  - `page` (int, optional): 1-indexed document page number.
  - `sheet` (str, optional): Excel worksheet name.
  - `format` (str): File extension (`pdf`, `xlsx`, `docx`, `txt`).
  - `doc_type` (str): Document classification (`Gazprom Standard (СТО)`, `Engineering Drawing`, `BOM Specification`, `Risk Matrix (DFMEA)`).
  - `verification_link` (str): Direct URL (`http://localhost:8000/api/documents/preview/{source}#page={page}`).

### 2.2 Telemetry Database (`data/analytics.db`)
- **`queries` Table:** `id`, `timestamp`, `date`, `user_id` (Client IP), `query_text`, `model_id`, `retrieval_ms`, `generation_ms`, `total_ms`, `grounding_ratio`, `judge_verdict`.
- **`uploads` Table:** `id`, `timestamp`, `date`, `user_id` (Client IP), `filename`, `format`, `size_bytes`, `chunks_count`.

---

## 3. Configuration & Endpoints

### 3.1 REST API Routes (`src/api.py`)
- `POST /api/query/stream`: Server-Sent Events (SSE) streaming pipeline (Rewriter $\rightarrow$ Dual-Search $\rightarrow$ FlashRank $\rightarrow$ Parent Page Hydration $\rightarrow$ Generator $\rightarrow$ LLM Judge Audit $\rightarrow$ Suggestions).
- `POST /api/query/clarify`: Real-time engineering Co-Pilot slot-filling recommendations.
- `POST /api/ingest`: Dynamic document upload and vectorization.
- `GET /api/documents/preview/{filename}`: Universal In-Browser HTML renderer for PDF, XLSX, and DOCX.
- `GET /api/documents/markdown-preview/{filename}`: Integrated In-Browser Stage 1 Markdown Quality Inspector.
- `GET /api/documents/{filename}`: Raw file stream with `Content-Disposition: inline`.
- `GET /api/stats`: Knowledge base vector counts and file inventory.
- `GET /api/analytics`: UX telemetry, daily activity timeline charts, and Client IP leaderboards.
- `GET /api/graph`: 3D HNSW semantic graph projection.
- `GET /health`: System daemon health and hardware status.

---

## 4. Credentials & Environment

> **NEVER commit plain-text passwords to public repositories.**

| Variable / Parameter | Setting | Purpose |
| :--- | :--- | :--- |
| `SERVER_IP` | `localhost` (Hostname: `c14753`, Port: `22`) | Remote AI compute node |
| `SSH_USER` | `ollamauser` | Remote execution user |
| `OLLAMA_HOST` | `http://localhost:11434` | Local GPU Ollama instance |
| `QDRANT_URL` | `http://localhost:6333` | Standalone Qdrant server daemon |
| `SYSTEMD_SERVICE`| `gas_rag.service` (user-level, lingering enabled) | 24/7 server persistence |

---

## 5. Global Rules & Constraints

1. **Zero Hallucination Mandate:** All engineering outputs must be strictly verifiable against retrieved standards with direct page citations (`[ИСТОЧНИК #X: Название, Стр. Y]`).
2. **Atomic Tabular Preservation:** Never apply word-based sliding-window chunking to Excel DFMEA or BOM files. Cells and hierarchical multi-row headers must be unmerged and preserved row-by-row.
3. **No File Downloads for Previews:** All document and spreadsheet clicks must open natively inside the browser modal viewer (`/api/documents/preview/`).
4. **Zero-Latency Parent Page Context:** High-speed micro-chunk vector ranking must always be paired with sub-millisecond in-memory Parent Page Hydration.
5. **Real Client IP Identification:** Telemetry and activity metrics must track real network workstation IP addresses (`request.client.host`).

---

## 6. Current Roadmap & Open Items

- [x] Item 1–13: Core Hybrid RAG, Refactoring, Dual-Search Rewriter, and Pure-Rust Qdrant Daemon Migration.
- [x] Item 14: Hierarchical Excel Parsing & DFMEA/BOM Ingestion.
- [x] Item 15–16: Page Assist Architecture Benchmark & Technical Documentation.
- [x] Item 17: Adaptive OllamaOCR via `llama3.2-vision:11b` + CJK Unicode Filter (transcribed 92-page scanned `СТО Газпром 2-4.1-212-2008`).
- [x] Item 18: In-Browser Universal Document Modal Viewer (`#page=N`).
- [x] Item 19: Small-to-Big Parent Page Hydration (Full PageIndex Pattern).
- [x] Item 20: Monokai Dark Web Client Redesign & As-You-Type Pre-Retrieval Slot Filling.
- [x] Item 21: Integrated In-Browser Stage 1 Markdown Quality Inspector (`/api/documents/markdown-preview/`).
- [ ] Item 22: SaaS «ТехЭксперт» Integration Pipeline (Export/Sync API).
- [x] Item 20: Native In-Browser DOCX and XLSX HTML Document Previewer.
- [x] Item 21: User Experience Analytics, Telemetry & Real Client IP Leaderboards.
- [ ] Item 22: *[Awaiting confirmation]* Investigate integration with SaaS informational system **ТехЭксперт** (Desktop API / Connector).

---

*Last updated: 2026-08-21. Managed by master_spec agent.*
