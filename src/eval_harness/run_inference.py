import json
import time
import requests
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))

from schema import load_test_cases
from prompts import build_prompt

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "llama3.1:8b"
KB_DIR = Path("data/knowledge_base")
TEST_CASES_PATH = "data/test_cases/test_cases.jsonl"
OUTPUT_PATH = Path("data/results/raw_outputs.jsonl")

def call_ollama(prompt: str) -> dict:
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    return resp.json()

def main():
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    test_cases = load_test_cases(TEST_CASES_PATH, str(KB_DIR))
    print(f"Running inference on {len(test_cases)} test cases against {MODEL_NAME}...")
    
    results = []
    for i, tc in enumerate(test_cases, 1):
        context = (KB_DIR / tc.doc).read_text(encoding="utf-8")
        prompt = build_prompt(context, tc.question)
        start = time.time()
        response = call_ollama(prompt)
        elapsed = round(time.time() - start, 2)
        
        model_answer = response.get("response", "").strip()
        result = {
            "id": tc.id,
            "doc": tc.doc,
            "category": tc.category,
            "question": tc.question,
            "expected_answer": tc.expected_answer,
            "model_answer": model_answer,
            "latency_sec": elapsed,
        }
        results.append(result)
        print(f"[{i}/{len(test_cases)}] {tc.id} ({tc.category}) — {elapsed}s")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            
    print(f"\nSaved {len(results)} raw outputs to {OUTPUT_PATH}")

if __name__ == "__main__":
    main()
