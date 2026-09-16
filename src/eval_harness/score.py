import json
from pathlib import Path
from judge import call_judge
from datetime import datetime, timezone

RUN_HISTORY_PATH = Path("data/results/run_history.jsonl") if not 'REPO_ROOT' in dir() else REPO_ROOT / "data" / "results" / "run_history.jsonl"

RAW_OUTPUTS_PATH = Path("data/results/raw_outputs.jsonl")
SCORED_OUTPUTS_PATH = Path("data/results/scored_outputs.jsonl")

REFUSAL_PHRASE = "not specified in the document"


def is_refusal(model_answer: str) -> bool:
    return REFUSAL_PHRASE in model_answer.lower()


def score_one(result: dict) -> dict:
    category = result["category"]
    model_answer = result["model_answer"]
    refused = is_refusal(model_answer)

    if category == "unanswerable":
        verdict = "correct" if refused else "incorrect"
        reasoning = "Correctly refused." if refused else "Should have refused but gave an answer (possible hallucination)."
        judge_used = False
    else:
        if refused:
            verdict = "incorrect"
            reasoning = "Refused to answer even though the context supports a real answer."
            judge_used = False
        else:
            judged = call_judge(result["question"], result["expected_answer"], model_answer)
            verdict = judged["verdict"]
            reasoning = judged["reasoning"]
            judge_used = True

    return {
        **result,
        "refused": refused,
        "verdict": verdict,
        "verdict_reasoning": reasoning,
        "judge_used": judge_used,
    }


def main():
    with open(RAW_OUTPUTS_PATH, encoding="utf-8") as f:
        results = [json.loads(line) for line in f]

    scored = []
    for i, r in enumerate(results, 1):
        s = score_one(r)
        scored.append(s)
        print(f"[{i}/{len(results)}] {s['id']} ({s['category']}) -> {s['verdict']}")

    SCORED_OUTPUTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(SCORED_OUTPUTS_PATH, "w", encoding="utf-8") as f:
        for s in scored:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print("\n=== SUMMARY ===")
    categories = sorted(set(r["category"] for r in scored))
    for cat in categories:
        cat_results = [r for r in scored if r["category"] == cat]
        correct = sum(1 for r in cat_results if r["verdict"] == "correct")
        total = len(cat_results)
        print(f"{cat}: {correct}/{total} correct ({100*correct/total:.0f}%)")

    overall_correct = sum(1 for r in scored if r["verdict"] == "correct")
    print(f"\nOverall: {overall_correct}/{len(scored)} correct ({100*overall_correct/len(scored):.0f}%)")

        # Append this run's summary to history for drift tracking
    run_record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "overall_accuracy": overall_correct / len(scored),
        "by_category": {
            cat: sum(1 for r in scored if r["category"] == cat and r["verdict"] == "correct") / 
                 sum(1 for r in scored if r["category"] == cat)
            for cat in categories
        },
        "total_cases": len(scored),
    }
    RUN_HISTORY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(RUN_HISTORY_PATH, "a", encoding="utf-8") as f:
        f.write(json.dumps(run_record) + "\n")
    print(f"\nAppended run summary to {RUN_HISTORY_PATH}")


if __name__ == "__main__":
    main()