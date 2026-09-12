import time
import json
import urllib.request
import os

api_url = 'http://localhost:8000'
test_file = '/home/ollamauser/RAG_SYSTEM/tests/golden_test_set.json'

with open(test_file, 'r', encoding='utf-8') as f:
    test_cases = json.load(f)

print('==================================================================================')
print('🚀 WARM BENCHMARK: QWEN 3.6 35B (100% GPU VRAM on RTX A6000)')
print('==================================================================================\n')

latencies = []
fact_recalls = []
source_hits = []

for tc in test_cases:
    qid = tc['id']
    q = tc['question']
    doc = tc['document']
    keywords = tc['expected_keywords']

    payload = {'query': q, 'model': 'qwen3.6:35b', 'top_k': 3}
    t0 = time.time()
    req = urllib.request.Request(
        f'{api_url}/api/query',
        data=json.dumps(payload).encode('utf-8'),
        headers={'Content-Type': 'application/json'}
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        data = json.loads(resp.read().decode('utf-8'))
        dt = time.time() - t0
        ans = data.get('answer', '').lower()
        sources = [s.get('source', '') for s in data.get('sources', [])]

    latencies.append(dt)
    hit = any(doc.lower() in s.lower() for s in sources)
    source_hits.append(1 if hit else 0)

    matched = [kw for kw in keywords if kw.lower() in ans]
    rec = len(matched) / len(keywords) if keywords else 1.0
    fact_recalls.append(rec)

    print(f"[{qid}] Latency: {dt:.2f}s | Source Hit: {'[OK]' if hit else '[FAIL]'} | Fact Recall: {rec*100:.0f}%")

avg_lat = sum(latencies) / len(latencies)
avg_rec = (sum(fact_recalls) / len(fact_recalls)) * 100
avg_src = (sum(source_hits) / len(source_hits)) * 100

print('\n==================================================================================')
print('🏆 PERFORMANCE RESULTS FOR QWEN 3.6 35B')
print('==================================================================================')
print(f'Average Query Latency:            {avg_lat:.2f} seconds')
print(f'Document Retrieval Precision:     {avg_src:.1f}%')
print(f'Fact Recall (Technical Accuracy): {avg_rec:.1f}%')
