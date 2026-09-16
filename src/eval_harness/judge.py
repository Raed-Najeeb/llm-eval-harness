import json
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
JUDGE_MODEL = "llama3.1:8b"  # same model as judge — fine for v1, note as a limitation later

JUDGE_SYSTEM = """You are a strict grading assistant. You will be given a
question, the expected (ground-truth) answer, and a model's actual answer.

Decide if the model's answer is semantically CORRECT or INCORRECT relative
to the expected answer. Minor wording, phrasing, or grammatical differences
that preserve the same meaning are fine (e.g. "60 days" vs "within 60 days"
is CORRECT). Missing a substantive fact, qualifier, or condition present in
the expected answer, contradicting it, or dodging the question when a real
answer exists counts as INCORRECT.

Respond with ONLY valid JSON, no other text, in this exact format:
{"verdict": "correct", "reasoning": "one short sentence"}
or
{"verdict": "incorrect", "reasoning": "one short sentence"}
"""

def build_judge_prompt(question: str, expected_answer: str, model_answer: str) -> str:
    return f"""{JUDGE_SYSTEM}

Question: {question}
Expected answer: {expected_answer}
Model's actual answer: {model_answer}

JSON verdict:"""


def call_judge(question: str, expected_answer: str, model_answer: str) -> dict:
    prompt = build_judge_prompt(question, expected_answer, model_answer)
    payload = {
        "model": JUDGE_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0.0},
    }
    resp = requests.post(OLLAMA_URL, json=payload, timeout=120)
    resp.raise_for_status()
    raw = resp.json().get("response", "").strip()

    try:
        start = raw.index("{")
        end = raw.rindex("}") + 1
        parsed = json.loads(raw[start:end])
        if parsed.get("verdict") not in ("correct", "incorrect"):
            raise ValueError("bad verdict value")
        return parsed
    except (ValueError, json.JSONDecodeError):
        return {"verdict": "judge_error", "reasoning": f"unparseable judge output: {raw[:200]}"}