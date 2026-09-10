import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from eval_harness.schema import load_test_cases

def test_all_test_cases_valid():
    cases = load_test_cases("data/test_cases/test_cases.jsonl", "data/knowledge_base")
    assert len(cases) >= 30, "Need at least 30 test cases for meaningful coverage"

def test_category_coverage():
    cases = load_test_cases("data/test_cases/test_cases.jsonl", "data/knowledge_base")
    categories = {c.category for c in cases}
    assert {"answerable", "unanswerable", "adversarial_conflicting"}.issubset(categories)
