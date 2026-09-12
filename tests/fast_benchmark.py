"""
Targeted Model Comparison Benchmark: Qwen 3.6 35B vs GPT-OSS 20B (100% GPU VRAM)
"""

import os
import sys
import time
import json
import urllib.request

# Ensure UTF-8
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

api_url = "http://localhost:8000"
test_set_path = "/home/ollamauser/RAG_SYSTEM/tests/golden_test_set.json"

with open(test_set_path, "r", encoding="utf-8") as f:
    test_cases = json.load(f)

models = [
    {"id": "qwen3.6:35b", "name": "Qwen 3.6 35B (MoE — Full GPU)"},
    {"id": "gpt-oss:20b", "name": "GPT-OSS 20B (Fast — Full GPU)"}
]

print("==================================================================================")
print("🚀 FAST BENCHMARK: QWEN 3.6 35B vs GPT-OSS 20B (100% VRAM)")
print("==================================================================================")

results = []

for m in models:
    model_id = m["id"]
    model_name = m["name"]
    print(f"\n🧠 Testing: {model_name}")
    
    latencies = []
    kw_recalls = []
    source_hits = []

    for tc in test_cases:
        qid = tc["id"]
        q = tc["question"]
        doc = tc["document"]
        keywords = tc["expected_keywords"]

        payload = {"query": q, "model": model_id, "top_k": 3}
        t0 = time.time()
        try:
            req = urllib.request.Request(
                f"{api_url}/api/query",
                data=json.dumps(payload).encode("utf-8"),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
                dt = time.time() - t0
                ans = data.get("answer", "").lower()
                sources = [s.get("source", "") for s in data.get("sources", [])]
        except Exception as e:
            dt = time.time() - t0
            ans = f"Error: {e}"
            sources = []

        latencies.append(dt)
        hit = any(doc.lower() in s.lower() for s in sources)
        source_hits.append(1 if hit else 0)

        matched = [kw for kw in keywords if kw.lower() in ans]
        kw_rec = len(matched) / len(keywords) if keywords else 1.0
        kw_recalls.append(kw_rec)

        print(f"  [{qid}] Latency: {dt:.2f}s | Source Hit: {'✅' if hit else '❌'} | Fact Recall: {kw_rec*100:.0f}%")

    avg_lat = sum(latencies) / len(latencies)
    avg_rec = (sum(kw_recalls) / len(kw_recalls)) * 100
    avg_src = (sum(source_hits) / len(source_hits)) * 100

    results.append({
        "name": model_name,
        "id": model_id,
        "avg_latency": round(avg_lat, 2),
        "fact_recall": round(avg_rec, 1),
        "source_precision": round(avg_src, 1)
    })

print("\n==================================================================================")
print("📊 BENCHMARK COMPARISON TABLE")
print("==================================================================================")
print(f"{'Model':<35} | {'Avg Latency':<12} | {'Fact Recall':<12} | {'Doc Precision':<12}")
print("-" * 78)
for r in results:
    print(f"{r['name']:<35} | {r['avg_latency']:>8.2f} s   | {r['fact_recall']:>9.1f} %  | {r['source_precision']:>9.1f} %")
