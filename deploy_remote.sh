#!/bin/bash
# ==============================================================================
# Remote Deployment Script for GAS-RAG System
# Target Workstation: ollamauser@192.168.152.38 (Host: c14753)
# ==============================================================================

set -e

echo "🚀 Setting up GAS-RAG project environment on remote workstation c14753..."

TARGET_DIR="/home/ollamauser/RAG_SYSTEM"

# Step 1: Create isolated project directories
echo "📁 Step 1: Creating project folder structure at ${TARGET_DIR}..."
mkdir -p ${TARGET_DIR}/{vector_db,data/raw,data/processed,src,web_client}

# Step 2: Create Conda virtual environment 'rag_env'
echo "🐍 Step 2: Setting up Conda virtual environment 'rag_env' (Python 3.11)..."
if ~/miniconda3/bin/conda info --envs | grep -q "rag_env"; then
    echo "  -> Environment 'rag_env' already exists."
else
    ~/miniconda3/bin/conda create -n rag_env python=3.11 -y
fi

# Step 3: Activate environment and install dependencies
echo "📦 Step 3: Installing core RAG dependencies..."
source ~/miniconda3/bin/activate rag_env
pip install --upgrade pip
pip install -r ${TARGET_DIR}/requirements.txt

# Step 4: Verify Ollama service & models
echo "🦙 Step 4: Verifying Ollama service on port 11434..."
curl -s http://localhost:11434/api/tags | grep -q "qwen3.6:35b" && echo "  -> Model qwen3.6:35b is available." || echo "  -> Pulling qwen3.6:35b..."

# Step 5: Start FastAPI server
echo "⚡ Step 5: GAS-RAG system setup complete!"
echo "To run the backend server:"
echo "  source ~/miniconda3/bin/activate rag_env"
echo "  cd /home/ollamauser/RAG_SYSTEM"
echo "  uvicorn api:app --host 0.0.0.0 --port 8000 --reload"
