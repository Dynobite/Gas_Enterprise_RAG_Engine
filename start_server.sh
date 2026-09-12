#!/bin/bash
# Autonomous daemon runner for GASlight-Me FastAPI server
export OLLAMA_HOST=http://localhost:11434
export VECTOR_DB_DIR=/home/user/RAG_SYSTEM/vector_db
export PYTHONUNBUFFERED=1

pkill -9 -f uvicorn 2>/dev/null || true
sleep 1
cd /home/user/RAG_SYSTEM
nohup /home/user/miniconda3/bin/python3 -m uvicorn src.api:app --host 0.0.0.0 --port 8000 > /home/user/RAG_SYSTEM/server.log 2>&1 &
sleep 2
echo "Server started with PID $(pgrep -f 'uvicorn src.api:app')"
