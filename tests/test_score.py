import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src" / "eval_harness"))

from score import is_refusal, score_one


def test_is_refusal_detects_exact_phrase():
    assert is_refusal("Not specified in the document.") is True
    assert is_refusal("NOT SPECIFIED IN THE DOCUMENT") is True

def test_is_refusal_false_on_real_answer():
    assert is_refusal("18 days.") is False

def test_unanswerable_correct_when_refused():
    result = {
        "id": "t1", "category": "unanswerable", "question": "q",
        "expected_answer": "Not specified in the document",
        "model_answer": "Not specified in the document.",
    }
    scored = score_one(result)
    assert scored["verdict"] == "correct"
    assert scored["judge_used"] is False

def test_unanswerable_incorrect_when_answered():
    result = {
        "id": "t2", "category": "unanswerable", "question": "q",
        "expected_answer": "Not specified in the document",
        "model_answer": "It offers 10 sick days.",
    }
    scored = score_one(result)
    assert scored["verdict"] == "incorrect"

def test_adversarial_incorrect_when_refused():
    result = {
        "id": "t3", "category": "adversarial_conflicting", "question": "q",
        "expected_answer": "No, max is 6 days",
        "model_answer": "Not specified in the document.",
    }
    scored = score_one(result)
    assert scored["verdict"] == "incorrect"
    assert scored["judge_used"] is False