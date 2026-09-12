#!/bin/bash
# Install and run vLLM via Docker on Machine 2 (c11171 / 192.168.152.74)
# Make sure to run this on the GPU machine!

set -e

echo "🚀 Preparing vLLM installation for RTX A6000..."

docker pull vllm/vllm-openai:latest

# By default, we use Qwen2.5-32B AWQ which fits easily in 48GB with huge KV cache
MODEL_NAME="Qwen/Qwen2.5-32B-Instruct-AWQ"

echo "Starting vLLM container for $MODEL_NAME on port 11434 (replacing Ollama)..."

# Stop Ollama if it's running
sudo systemctl stop ollama || sudo pkill -f ollama || true

# Run vLLM
# --max-model-len 16384 for long context RAG
docker run -d --name vllm-server \
  --gpus all \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -p 11434:8000 \
  --ipc=host \
  --restart unless-stopped \
  vllm/vllm-openai:latest \
  --model $MODEL_NAME \
  --quantization awq \
  --max-model-len 16384 \
  --gpu-memory-utilization 0.95

echo "✅ vLLM container started! It might take a few minutes to download the model."
echo "You can check logs with: docker logs -f vllm-server"
