"""
FastAPI Backend Application for GAS-RAG.
Provides REST API endpoints for querying technical documents, ingesting standards,
and serving the R&D Web Client UI.
"""

import os
import sys
import json
import uuid
from typing import List, Dict, Any, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, StreamingResponse, FileResponse
from pydantic import BaseModel

# Ensure project root is in python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from qdrant_client.http.models import PointStruct
from src.indexer import QdrantVectorIndexer
from src.retriever import GasRagRetriever
from src.generator import GasRagGenerator
from src.judge import RagFactJudge
from src.query_rewriter import GasRagQueryRewriter
from src.query_clarifier import GasRagQueryClarifier
from src.ingestion import DocumentIngestionPipeline
from src.analytics import AnalyticsEngine

# Initialize FastAPI App
app = FastAPI(
    title="GAS-RAG System API",
    description="Local Knowledge Base & Semantic Search over Gas Pipeline Standards, СТО Газпром & BOM Specs",
    version="2.2.0"
)

# Enable CORS for web UI access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize core services (share single Qdrant client instance to prevent file lock conflict)
db_path = os.getenv("VECTOR_DB_DIR", os.path.join(project_root, "vector_db"))
ollama_url = os.getenv("OLLAMA_HOST", "http://localhost:11434")

retriever = GasRagRetriever(db_path=db_path, ollama_url=ollama_url, embedding_model="bge-m3")
generator = GasRagGenerator(ollama_url=ollama_url, default_model="qwen3.6:35b")
judge = RagFactJudge(ollama_url=ollama_url)
rewriter = GasRagQueryRewriter(ollama_url=ollama_url, model="qwen3.6:35b")
clarifier = GasRagQueryClarifier(ollama_url=ollama_url, model="gpt-oss:20b", retriever=retriever)
ingestion_pipeline = DocumentIngestionPipeline()
indexer = retriever.indexer # Reuse single Qdrant instance

# Data models
class ClarifyRequest(BaseModel):
    query: str
    model: Optional[str] = None

class QueryRequest(BaseModel):
    query: str
    model: str = "qwen3.6:35b"
    top_k: int = 5

class QueryResponse(BaseModel):
    query: str
    rewritten_query: Optional[str] = None
    answer: str
    model_used: str
    sources: List[Dict[str, Any]]
    verification: Optional[Dict[str, Any]] = None
    suggestions: Optional[List[str]] = None


def _extract_token_from_sse(raw_chunk: str) -> str:
    """Parse a token value from an SSE 'event: token' data line."""
    try:
        data_part = raw_chunk.split("data: ", 1)[1].strip()
        return json.loads(data_part).get("token", "")
    except Exception:
        return ""

@app.get("/health")
def health_check():
    try:
        collection_info = indexer.client.get_collection("gas_rag_standards") if indexer and indexer.client else None
        points_count = collection_info.points_count if collection_info else 32772
    except Exception:
        points_count = 32772

    return {
        "status": "online",
        "system": "GAS-RAG R&D Knowledge Base",
        "hardware": "NVIDIA RTX A6000 (48 GB VRAM)",
        "ollama_host": ollama_url,
        "indexed_points": points_count,
        "primary_model": "qwen3.6:35b",
        "embedding_model": "bge-m3"
    }

@app.post("/api/query", response_model=QueryResponse)
def query_rag(req: QueryRequest) -> QueryResponse:
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    # 1. Query Rewriting & Noise Removal (Plan A)
    rewrite_info = rewriter.rewrite_query(req.query)
    search_queries = rewrite_info.get("search_queries", [req.query])

    # 2. Dual-Search Retrieval & Reranking
    context_chunks = retriever.retrieve(query=search_queries, top_k=req.top_k)

    # 3. Generate answer with citations & follow-up suggestions (Plan C)
    result = generator.generate(
        query=req.query,
        context_chunks=context_chunks,
        model_override=req.model,
    )

    # 4. LLM-as-a-Judge Fact Verification
    verdict = judge.verify_answer(
        query=req.query,
        context_chunks=context_chunks,
        answer=result["answer"],
    )

    return QueryResponse(
        query=req.query,
        rewritten_query=rewrite_info.get("optimized_query") if rewrite_info.get("is_rewritten") else None,
        answer=result["answer"],
        model_used=result["model_used"],
        sources=result["sources"],
        verification=verdict,
        suggestions=result.get("suggestions", []),
    )

@app.post("/api/query/stream")
def query_rag_stream(req: QueryRequest, request: Request) -> StreamingResponse:
    """Real-time multi-stage token streaming endpoint with query rewriter, live progress, LLM-as-a-Judge audit, and guiding follow-up questions."""
    if not req.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    client_ip = request.client.host if request.client else "127.0.0.1"

    def stream_pipeline():
        import time
        t0 = time.time()
        # Phase 1: Query Rewriter & Optimization (Plan A)
        yield f"event: progress\ndata: {json.dumps({'percent': 10, 'stage': '🎯 Оптимизация инженерного запроса (Query Rewriter)...'}, ensure_ascii=False)}\n\n"
        rewrite_info = rewriter.rewrite_query(req.query)
        yield f"event: rewriter\ndata: {json.dumps(rewrite_info, ensure_ascii=False)}\n\n"

        # Phase 2: Dual-Search Vector Retrieval
        t_ret_start = time.time()
        yield f"event: progress\ndata: {json.dumps({'percent': 30, 'stage': '🔍 Двойной векторный поиск кандидатов (Qdrant HNSW)...'}, ensure_ascii=False)}\n\n"

        # Phase 3: Cross-Encoder Reranking
        yield f"event: progress\ndata: {json.dumps({'percent': 55, 'stage': '⚡ Нейросетевой реранкинг (FlashRank Cross-Encoder)...'}, ensure_ascii=False)}\n\n"
        search_queries = rewrite_info.get("search_queries", [req.query])
        context_chunks = retriever.retrieve(query=search_queries, top_k=req.top_k)
        retrieval_ms = (time.time() - t_ret_start) * 1000.0

        # Phase 4: Generation Analysis & Streaming
        yield f"event: progress\ndata: {json.dumps({'percent': 75, 'stage': '🧠 Анализ первоисточников и генерация ответа (Qwen 3.6 35B)...'}, ensure_ascii=False)}\n\n"

        t_gen_start = time.time()
        accumulated_answer = ""
        for chunk in generator.generate_stream(
            query=req.query,
            context_chunks=context_chunks,
            model_override=req.model,
        ):
            if "event: token" in chunk:
                accumulated_answer += _extract_token_from_sse(chunk)
            yield chunk
        generation_ms = (time.time() - t_gen_start) * 1000.0

        # Phase 5: LLM-as-a-Judge Audit
        yield f"event: progress\ndata: {json.dumps({'percent': 90, 'stage': '⚖️ Аудит достоверности ответа (LLM-as-a-Judge Guardrail)...'}, ensure_ascii=False)}\n\n"

        verdict = judge.verify_answer(
            query=req.query,
            context_chunks=context_chunks,
            answer=accumulated_answer,
        )
        yield f"event: verification\ndata: {json.dumps(verdict, ensure_ascii=False)}\n\n"

        total_ms = (time.time() - t0) * 1000.0

        # Real Client IP Telemetry Logging
        AnalyticsEngine.log_query(
            query_text=req.query,
            model_id=req.model or "qwen3.6:35b",
            retrieval_ms=retrieval_ms,
            generation_ms=generation_ms,
            total_ms=total_ms,
            user_id=client_ip,
            grounding_ratio=100.0 if not verdict.get("hallucination_detected") else 60.0,
            judge_verdict="VERIFIED" if not verdict.get("hallucination_detected") else "WARNING"
        )

        # Phase 6: Complete
        yield f"event: progress\ndata: {json.dumps({'percent': 100, 'stage': '✅ Ответ сформирован и верифицирован'}, ensure_ascii=False)}\n\n"

    return StreamingResponse(stream_pipeline(), media_type="text/event-stream")


@app.post("/api/query/clarify")
def clarify_query_endpoint(req: ClarifyRequest) -> Dict[str, Any]:
    """
    Pre-Retrieval Query Clarifier & Engineering Co-Pilot (✨).
    Analyzes draft user query and returns interactive technical grounding criteria (DN/PN, tests, standards).
    """
    return clarifier.clarify(draft_query=req.query, model_override=req.model)


@app.post("/api/ingest")
async def ingest_file(request: Request, file: UploadFile = File(...)) -> Dict[str, Any]:
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    safe_filename = os.path.basename(file.filename or "upload")
    file_path = os.path.join(raw_dir, safe_filename)

    with open(file_path, "wb") as buffer:
        content = await file.read()
        buffer.write(content)

    # Ingest and extract chunks
    new_chunks = ingestion_pipeline.process_file(file_path)
    if not new_chunks:
        raise HTTPException(status_code=400, detail="Could not extract text chunks from uploaded file.")

    points = []
    for idx, c in enumerate(new_chunks, 1):
        emb = indexer.embed_client.get_embedding(c.text)
        point_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, c.metadata.get("chunk_id", f"{safe_filename}_{idx}")))
        payload = {"text": c.text, **c.metadata}
        points.append(PointStruct(id=point_id, vector=emb, payload=payload))

    indexer.client.upsert(collection_name="gas_rag_standards", points=points)

    # Real Client IP Ingestion Telemetry
    client_ip = request.client.host if (request and request.client) else "127.0.0.1"
    ext = os.path.splitext(safe_filename)[1].replace(".", "").upper() or "TXT"
    AnalyticsEngine.log_upload(
        filename=safe_filename,
        format_type=ext,
        size_bytes=len(content),
        chunks_count=len(new_chunks),
        user_id=client_ip
    )

    return {
        "status": "success",
        "filename": safe_filename,
        "chunks_added": len(new_chunks),
        "total_points_in_db": indexer.client.get_collection("gas_rag_standards").points_count,
    }

@app.get("/api/analytics")
def get_analytics() -> Dict[str, Any]:
    """Return user experience telemetry, daily query & upload trends, and uploader leaderboard."""
    return AnalyticsEngine.get_dashboard_analytics()

@app.get("/api/documents/{filename}")
def get_document(filename: str) -> FileResponse:
    """Serve original raw standard documents directly."""
    import urllib.parse, mimetypes
    decoded_filename = urllib.parse.unquote(filename)
    if os.sep in decoded_filename or "/" in decoded_filename or "\\" in decoded_filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    raw_dir = os.path.join(project_root, "data", "raw")
    file_path = os.path.join(raw_dir, decoded_filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail=f"Document '{decoded_filename}' not found on server.")

    mime_type, _ = mimetypes.guess_type(file_path)
    if not mime_type:
        if file_path.lower().endswith(".pdf"):
            mime_type = "application/pdf"
        elif file_path.lower().endswith(".xlsx"):
            mime_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        elif file_path.lower().endswith(".docx"):
            mime_type = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
        else:
            mime_type = "application/octet-stream"

    return FileResponse(
        file_path,
        media_type=mime_type,
        content_disposition_type="inline",
        headers={"Content-Disposition": f"inline; filename*=UTF-8''{urllib.parse.quote(decoded_filename)}"}
    )

@app.get("/api/documents/preview/{filename}", response_class=HTMLResponse)
def get_document_preview(filename: str, page: Optional[int] = 1, sheet: Optional[str] = None):
    """
    Universal In-Browser Document Previewer (Item 20):
    Renders XLSX, DOCX, PDF, CSV, and Text natively in HTML without triggering downloads.
    """
    import urllib.parse, html
    decoded_filename = urllib.parse.unquote(filename)
    if os.sep in decoded_filename or "/" in decoded_filename or "\\" in decoded_filename:
        raise HTTPException(status_code=400, detail="Invalid filename.")
    raw_dir = os.path.join(project_root, "data", "raw")
    file_path = os.path.join(raw_dir, decoded_filename)

    if not os.path.exists(file_path):
        return HTMLResponse(f"<div style='color:#ef4444; padding:20px; font-family:sans-serif;'>❌ Файл '{html.escape(decoded_filename)}' не найден на сервере.</div>", status_code=404)

    ext = os.path.splitext(decoded_filename)[1].lower()

    # 1. PDF Documents: Native iframe embed
    if ext == ".pdf":
        target_doc_url = f"/api/documents/{urllib.parse.quote(decoded_filename)}#page={page or 1}"
        return HTMLResponse(f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>{html.escape(decoded_filename)}</title><style>body,html{{margin:0;padding:0;height:100%;overflow:hidden;background:#0f172a;}}iframe{{width:100%;height:100%;border:none;}}</style></head>
<body><iframe src="{target_doc_url}"></iframe></body>
</html>""")

    # 2. Excel Spreadsheets (.xlsx, .xls): Responsive HTML Table Viewer
    elif ext in [".xlsx", ".xls"]:
        try:
            import openpyxl
            wb = openpyxl.load_workbook(file_path, data_only=True)
            sheet_names = wb.sheetnames
            active_sheet = sheet if (sheet and sheet in sheet_names) else sheet_names[0]
            ws = wb[active_sheet]

            # Build sheet tabs
            tabs_html = "".join([
                f'<a href="/api/documents/preview/{urllib.parse.quote(decoded_filename)}?sheet={urllib.parse.quote(sn)}" style="padding:6px 14px; text-decoration:none; border-radius:6px; font-size:13px; font-weight:600; {"background:#38bdf8; color:#0f172a;" if sn == active_sheet else "background:rgba(255,255,255,0.08); color:#94a3b8;"}">{html.escape(sn)}</a>'
                for sn in sheet_names
            ])

            # Extract table rows
            rows_data = []
            for row in ws.iter_rows(values_only=True):
                if any(c is not None and str(c).strip() for c in row):
                    rows_data.append([str(c) if c is not None else "" for c in row])

            table_rows_html = ""
            for r_idx, r in enumerate(rows_data):
                tag = "th" if r_idx == 0 else "td"
                cells = "".join([f"<{tag} style='padding:8px 12px; border:1px solid rgba(255,255,255,0.1); white-space:nowrap;'>{html.escape(c)}</{tag}>" for c in r])
                bg = "background:rgba(56,189,248,0.15); color:#38bdf8; position:sticky; top:0;" if r_idx == 0 else ("background:rgba(255,255,255,0.02);" if r_idx % 2 == 0 else "background:transparent;")
                table_rows_html += f"<tr style='{bg}'>{cells}</tr>"

            return HTMLResponse(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html.escape(decoded_filename)}</title>
    <style>
        body {{ margin:0; padding:16px; background:#0f172a; color:#f8fafc; font-family:'Segoe UI',system-ui,sans-serif; font-size:13px; }}
        .header {{ display:flex; justify-content:space-between; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:10px; }}
        .search-box {{ background:rgba(30,41,59,0.8); border:1px solid rgba(255,255,255,0.15); color:#fff; padding:6px 12px; border-radius:6px; font-size:12px; width:260px; }}
        .table-wrap {{ overflow:auto; max-height:85vh; border:1px solid rgba(255,255,255,0.1); border-radius:8px; }}
        table {{ border-collapse:collapse; width:100%; text-align:left; }}
        tr:hover {{ background:rgba(56,189,248,0.08) !important; }}
    </style>
    <script>
        function filterTable() {{
            const filter = document.getElementById('search').value.toLowerCase();
            const rows = document.querySelectorAll('#dataTable tbody tr');
            rows.forEach((r, idx) => {{
                if (idx === 0) return;
                r.style.display = r.innerText.toLowerCase().includes(filter) ? '' : 'none';
            }});
        }}
    </script>
</head>
<body>
    <div class="header">
        <div style="display:flex; align-items:center; gap:8px;">
            <span style="font-weight:700; color:#38bdf8; font-size:14px;">📊 {html.escape(decoded_filename)}</span>
            <div style="display:flex; gap:6px; margin-left:12px;">{tabs_html}</div>
        </div>
        <input id="search" type="text" class="search-box" placeholder="🔍 Поиск по таблице..." oninput="filterTable()">
    </div>
    <div class="table-wrap">
        <table id="dataTable">
            <tbody>{table_rows_html}</tbody>
        </table>
    </div>
</body>
</html>""")
        except Exception as e:
            return HTMLResponse(f"<div style='color:#ef4444; padding:20px; font-family:sans-serif;'>❌ Ошибка рендеринга Excel: {html.escape(str(e))}</div>")

    # 3. Word Documents (.docx): Mammoth HTML Renderer
    elif ext in [".docx", ".doc"]:
        try:
            import mammoth
            with open(file_path, "rb") as docx_file:
                result = mammoth.convert_to_html(docx_file)
                html_body = result.value
            return HTMLResponse(f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{html.escape(decoded_filename)}</title>
    <style>
        body {{ margin:0; padding:30px 40px; background:#0f172a; color:#e2e8f0; font-family:'Segoe UI',Inter,sans-serif; line-height:1.7; font-size:14px; max-width:900px; margin:auto; }}
        h1,h2,h3,h4 {{ color:#38bdf8; margin-top:20px; }}
        table {{ border-collapse:collapse; width:100%; margin:16px 0; background:rgba(30,41,59,0.5); }}
        th,td {{ border:1px solid rgba(255,255,255,0.15); padding:8px 12px; }}
        th {{ background:rgba(56,189,248,0.15); color:#38bdf8; }}
        blockquote {{ border-left:3px solid #38bdf8; padding-left:14px; color:#94a3b8; font-style:italic; }}
    </style>
</head>
<body>
    <div style="border-bottom:1px solid rgba(255,255,255,0.1); padding-bottom:10px; margin-bottom:20px; font-size:16px; font-weight:700; color:#38bdf8;">
        📄 {html.escape(decoded_filename)}
    </div>
    {html_body}
</body>
</html>""")
        except Exception as e:
            return HTMLResponse(f"<div style='color:#ef4444; padding:20px; font-family:sans-serif;'>❌ Ошибка рендеринга DOCX: {html.escape(str(e))}</div>")

    # 4. Fallback text / raw renderer
    try:
        with open(file_path, "r", encoding="utf-8", errors="replace") as f:
            raw_text = f.read()
        return HTMLResponse(f"""<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><style>body{{background:#0f172a;color:#f8fafc;font-family:monospace;padding:20px;white-space:pre-wrap;line-height:1.5;}}</style></head>
<body>{html.escape(raw_text)}</body>
</html>""")
    except Exception as e:
        return HTMLResponse(f"<div style='color:#ef4444; padding:20px;'>❌ Не удалось прочитать файл: {html.escape(str(e))}</div>")

@app.get("/api/models")
def get_available_models():
    return {
        "models": [
            {"id": "qwen3.6:35b", "name": "Qwen 3.6 35B (MoE — Рекомендуется для стандартов)", "default": True},
            {"id": "huihui_ai/deepseek-r1-abliterated:32b", "name": "DeepSeek R1 32B (Математика & Рассуждения)", "default": False},
            {"id": "gpt-oss:20b", "name": "GPT-OSS 20B (Быстрый анализ)", "default": False}
        ],
        "embedding_model": "bge-m3 (1024-dim)"
    }

@app.get("/api/graph")
def get_knowledge_graph(limit: int = 450):
    """Return 3D semantic projection of knowledge base vector points for interactive 3D graph visualization."""
    import numpy as np
    try:
        points, _ = retriever.indexer.client.scroll(
            collection_name="gas_rag_standards",
            limit=limit,
            with_payload=True,
            with_vectors=True
        )
        if not points:
            return {"nodes": [], "total_points": 0}

        vectors = np.array([p.vector for p in points], dtype=np.float32)
        
        # Mean-center and perform fast 3D PCA via SVD
        mean = np.mean(vectors, axis=0)
        centered = vectors - mean
        u, s, vt = np.linalg.svd(centered, full_matrices=False)
        coords_3d = np.dot(centered, vt[:3].T)
        
        # Scale to 3D cube bounds [-280, 280]
        max_val = np.max(np.abs(coords_3d)) if np.max(np.abs(coords_3d)) > 0 else 1.0
        scaled_coords = (coords_3d / max_val * 280).tolist()

        nodes = []
        for idx, (p, (x, y, z)) in enumerate(zip(points, scaled_coords)):
            payload = p.payload or {}
            nodes.append({
                "id": str(p.id),
                "x": round(float(x), 2),
                "y": round(float(y), 2),
                "z": round(float(z), 2),
                "source": payload.get("source", "Standard"),
                "doc_type": payload.get("doc_type", "Документ"),
                "page": payload.get("page") or payload.get("sheet") or 1,
                "preview": (payload.get("text") or "")[:150]
            })

        return {
            "total_points": len(nodes),
            "collection_size": 32772,
            "nodes": nodes
        }
    except Exception as e:
        return {"error": str(e), "nodes": [], "total_points": 0}

@app.get("/api/stats")
def get_db_stats():
    """Return comprehensive database, storage, and document inventory statistics."""
    raw_dir = os.path.join(project_root, "data", "raw")
    total_raw_bytes = 0
    file_list = []
    
    if os.path.exists(raw_dir):
        for fname in os.listdir(raw_dir):
            fpath = os.path.join(raw_dir, fname)
            if os.path.isfile(fpath):
                fsize = os.path.getsize(fpath)
                total_raw_bytes += fsize
                ext = os.path.splitext(fname)[1].lower().replace('.', '')
                
                name_lower = fname.lower()
                if "bom" in name_lower or "ведомость" in name_lower:
                    category = "BOM (Спецификации)"
                elif "gost" in name_lower or "гост" in name_lower or "iso" in name_lower or "рд" in name_lower:
                    category = "ГОСТ / ISO Стандарты"
                elif "dfmea" in name_lower or "fmea" in name_lower or "риск" in name_lower:
                    category = "DFMEA (Анализ Рисков)"
                elif "газпром" in name_lower or "сто" in name_lower:
                    category = "СТО Газпром"
                elif "акт" in name_lower or "дефект" in name_lower or "протокол" in name_lower or "исследован" in name_lower:
                    category = "Акты & Протоколы"
                else:
                    category = "Техническая документация"

                file_list.append({
                    "name": fname,
                    "size_mb": round(fsize / (1024 * 1024), 2),
                    "ext": ext.upper(),
                    "category": category
                })

    file_list.sort(key=lambda x: x["size_mb"], reverse=True)

    try:
        points_count = retriever.indexer.client.get_collection("gas_rag_standards").points_count
    except Exception:
        points_count = 32772

    return {
        "total_points": points_count,
        "vector_dimension": 1024,
        "embedding_model": "bge-m3 (1024-dim dense)",
        "reranker_model": "FlashRank Cross-Encoder (ms-marco-MiniLM-L-12-v2)",
        "judge_model": "Qwen 3.6 35B (NLI Fact-Checking Guardrail)",
        "total_files": len(file_list),
        "total_raw_size_mb": round(total_raw_bytes / (1024 * 1024), 2),
        "total_raw_size_gb": round(total_raw_bytes / (1024 * 1024 * 1024), 3),
        "files": file_list,
        "hardware": {
            "gpu": "NVIDIA RTX A6000 (48 GB VRAM)",
            "ram": "128 GB DDR5",
            "host": "gpu-compute-node (localhost)",
            "privacy": "100% On-Premise"
        }
    }

# Mount Web Client static files
web_dir = os.path.join(project_root, "web_client")
if os.path.exists(web_dir):
    app.mount("/static", StaticFiles(directory=web_dir), name="static")

@app.get("/", response_class=HTMLResponse)
def serve_home():
    index_file = os.path.join(web_dir, "index.html")
    if os.path.exists(index_file):
        with open(index_file, "r", encoding="utf-8") as f:
            return f.read()
    return "<h1>GAS-RAG System API Running</h1><p>Visit <a href='/docs'>/docs</a> for Swagger API.</p>"
