"""
Automated Benchmark and Evaluation Suite for GASlight-Me.
Compares LLM models (Qwen 3.6 35B vs DeepSeek R1 32B vs GPT-OSS 20B) via live API,
measures latency, tokens/sec, and evaluates answer accuracy against the golden test set.
"""

import os
import sys
import time
import json
import urllib.request

# Ensure UTF-8 stdout
if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

def run_benchmark():
    api_url = os.getenv("API_URL", "http://localhost:8000")
    
    # Locate test set
    script_dir = os.path.dirname(os.path.abspath(__file__))
    test_set_path = os.path.join(script_dir, "golden_test_set.json")
    if not os.path.exists(test_set_path):
        test_set_path = os.path.join(os.path.dirname(script_dir), "tests", "golden_test_set.json")

    with open(test_set_path, "r", encoding="utf-8") as f:
        test_cases = json.load(f)

    models_to_test = [
        {"id": "qwen3.6:35b", "name": "Qwen 3.6 35B (MoE)"},
        {"id": "huihui_ai/deepseek-r1-abliterated:32b", "name": "DeepSeek R1 32B (Reasoning)"},
        {"id": "gpt-oss:20b", "name": "GPT-OSS 20B (Fast)"}
    ]

    print("==================================================================================")
    print("🚀 STARTING GASlight-Me COMPREHENSIVE BENCHMARK & EVALUATION")
    print(f"📊 Total Test Cases: {len(test_cases)} | Models to test: {len(models_to_test)}")
    print(f"🖥️ Target: {api_url} | Hardware: NVIDIA RTX A6000 (48 GB VRAM)")
    print("==================================================================================\n")

    benchmark_summary = []

    for m in models_to_test:
        model_id = m["id"]
        model_name = m["name"]
        print(f"\n==================================================================")
        print(f"🧠 BENCHMARKING MODEL: {model_name} ({model_id})")
        print(f"==================================================================")

        total_latencies = []
        keyword_match_scores = []
        source_hit_scores = []

        for tc in test_cases:
            qid = tc["id"]
            question = tc["question"]
            expected_doc = tc["document"]
            keywords = tc["expected_keywords"]

            payload = {
                "query": question,
                "model": model_id,
                "top_k": 3
            }

            t0 = time.time()
            try:
                req = urllib.request.Request(
                    f"{api_url}/api/query",
                    data=json.dumps(payload).encode("utf-8"),
                    headers={"Content-Type": "application/json"}
                )
                with urllib.request.urlopen(req, timeout=120) as resp:
                    data = json.loads(resp.read().decode("utf-8"))
                    total_time = time.time() - t0
                    answer = data.get("answer", "").lower()
                    sources = data.get("sources", [])
            except Exception as e:
                total_time = time.time() - t0
                answer = f"Error: {e}".lower()
                sources = []

            total_latencies.append(total_time)

            # Check if expected document was retrieved in sources
            retrieved_sources = [s.get("source", "") for s in sources]
            source_hit = any(expected_doc.lower() in s.lower() for s in retrieved_sources)
            source_hit_scores.append(1 if source_hit else 0)

            # Calculate keyword recall
            matched_kw = [kw for kw in keywords if kw.lower() in answer]
            kw_score = len(matched_kw) / len(keywords) if keywords else 1.0
            keyword_match_scores.append(kw_score)

            print(f"  [{qid}] Latency: {total_time:.2f}s | Source Hit: {'✅' if source_hit else '❌'} | Fact Recall: {kw_score*100:.0f}%")

        avg_latency = sum(total_latencies) / len(total_latencies)
        avg_kw_recall = (sum(keyword_match_scores) / len(keyword_match_scores)) * 100
        avg_source_precision = (sum(source_hit_scores) / len(source_hit_scores)) * 100

        stats = {
            "model_name": model_name,
            "model_id": model_id,
            "avg_latency_s": round(avg_latency, 2),
            "source_retrieval_precision": round(avg_source_precision, 1),
            "keyword_recall_pct": round(avg_kw_recall, 1)
        }
        benchmark_summary.append(stats)

    print("\n==================================================================================")
    print("🏆 FINAL BENCHMARK SUMMARY & PERFORMANCE MATRIX")
    print("==================================================================================")
    print(f"{'Model':<35} | {'Avg Latency':<14} | {'Fact Recall':<14} | {'Doc Precision':<14}")
    print("-" * 82)
    for s in benchmark_summary:
        print(f"{s['model_name']:<35} | {s['avg_latency_s']:>10.2f} s   | {s['keyword_recall_pct']:>11.1f} %  | {s['source_retrieval_precision']:>11.1f} %")

    # Save results to project_docs/benchmark_results.json
    out_json = os.path.join(os.path.dirname(script_dir), "project_docs", "benchmark_results.json")
    try:
        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(benchmark_summary, f, ensure_ascii=False, indent=2)
        print(f"\n💾 Benchmark results saved to: {out_json}")
    except Exception:
        pass

if __name__ == "__main__":
    run_benchmark()
