import json
from dataclasses import dataclass
from pathlib import Path

VALID_CATEGORIES = {"answerable", "unanswerable", "adversarial_conflicting"}

@dataclass
class TestCase:
    id: str
    doc: str
    question: str
    expected_answer: str
    category: str

def load_test_cases(jsonl_path: str, kb_dir: str) -> list[TestCase]:
    kb_path = Path(kb_dir)
    cases = []
    seen_ids = set()
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line_num, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            tc = TestCase(**obj)

            if tc.id in seen_ids:
                raise ValueError(f"Line {line_num}: duplicate id '{tc.id}'")
            seen_ids.add(tc.id)

            if tc.category not in VALID_CATEGORIES:
                raise ValueError(f"Line {line_num}: invalid category '{tc.category}'")

            if not (kb_path / tc.doc).exists():
                raise ValueError(f"Line {line_num}: doc file '{tc.doc}' not found in {kb_dir}")

            cases.append(tc)
    return cases

if __name__ == "__main__":
    cases = load_test_cases("data/test_cases/test_cases.jsonl", "data/knowledge_base")
    print(f"Loaded {len(cases)} valid test cases.")
    from collections import Counter
    print(Counter(c.category for c in cases))
