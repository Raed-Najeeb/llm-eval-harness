import json
import sys
from pathlib import Path
from collections import defaultdict

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORED_PATH = REPO_ROOT / "data" / "results" / "scored_outputs.jsonl"

# Thresholds are deliberately set from your actual Phase 4 baseline run.
# A future change that drops below these fails the build.
MIN_ACCURACY_BY_CATEGORY = {
    "unanswerable": 1.00,           # must stay perfect
    "answerable": 0.70,             # allow minor fluctuation below observed 78%
    "adversarial_conflicting": 0.00,  # known failure mode, tracked not gated (see note below)
}


def main():
    if not SCORED_PATH.exists():
        print(f"FAIL: {SCORED_PATH} not found. Run score.py first.")
        sys.exit(1)

    by_category = defaultdict(list)
    with open(SCORED_PATH, encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            by_category[r["category"]].append(r["verdict"] == "correct")

    failed = False
    print("=== REGRESSION GATE ===")
    for category, min_acc in MIN_ACCURACY_BY_CATEGORY.items():
        results = by_category.get(category, [])
        if not results:
            print(f"FAIL: no results found for category '{category}'")
            failed = True
            continue
        acc = sum(results) / len(results)
        status = "OK" if acc >= min_acc else "FAIL"
        if status == "FAIL":
            failed = True
        print(f"[{status}] {category}: {acc:.0%} (min required: {min_acc:.0%})")

    if failed:
        print("\nRegression gate FAILED.")
        sys.exit(1)
    print("\nRegression gate PASSED.")


if __name__ == "__main__":
    main()