# LLM Eval & Red-Team Harness

An automated evaluation framework that scores an LLM's answers for faithfulness,
tests it against adversarial edge cases, tracks accuracy drift across runs, and
gates deployments in CI — built to surface failure modes that a single overall
accuracy number would hide.

## Headline Finding

Running Llama 3.1 8B (via Ollama) against 30 labeled test cases over a fictional
knowledge base:

| Category | Accuracy | What it measures |
|---|---|---|
| `unanswerable` | **100%** | Correctly refuses when the answer isn't in context |
| `answerable` | **78%** | Correctly answers straightforward lookup questions |
| `adversarial_conflicting` | **0%** | Corrects a false premise using the provided context |

A flat "63% accurate" summary would have hidden a complete, systematic failure
mode: the model reliably refuses to answer when it *should*, but also reliably
**fails to correct a false premise even when the correcting fact is right there
in the context** — it defaults to "not specified in the document" instead of
pushing back. That gap only shows up because the eval set is category-aware
instead of one flat pass/fail bucket.

## Architecture

```
Test cases (JSONL, labeled by category)
        │
        ▼
Local inference — Llama 3.1 8B via Ollama (grounded, temperature=0)
        │
        ▼
Category-aware scoring:
  - unanswerable            → refusal-string match
  - answerable / adversarial → LLM-as-judge (same model), strict rubric
        │
        ▼
Scored results + run history (appended, for drift tracking)
        │
        ├──► GitHub Actions: unit tests + regression gate on every push
        └──► Streamlit dashboard: category accuracy, drift over time, failing-case explorer
```

## How to run locally

Requires [Ollama](https://ollama.com/) installed and `llama3.1:8b` pulled.

```bash
git clone https://github.com/<your-username>/llm-eval-harness.git
cd llm-eval-harness
python -m venv venv
source venv/Scripts/activate   # Windows Git Bash
pip install -r requirements.txt

python src/eval_harness/run_inference.py   # generate raw model outputs
python src/eval_harness/score.py           # score + append to run history
python -m streamlit run src/eval_harness/dashboard.py   # view dashboard
```

## CI

Every push runs unit tests against the scoring logic and a regression gate
against the committed baseline results (`data/results/scored_outputs.jsonl`).
CI does **not** re-run full model inference (no GPU on GitHub's hosted
runners, and Ollama isn't preinstalled) — it checks logic correctness and
guards against silent regressions in checked-in results. See
`.github/workflows/eval-ci.yml`.

## Limitations

- **Same model serves as both subject and judge** (Llama 3.1 8B for both).
  This risks self-preference bias — a model may rate its own answer style
  more favorably than an independent judge would. A production version
  would use a stronger or different model as judge.
- **Small test set** (30 cases across 6 fictional documents). Enough to
  demonstrate the method and surface a real failure mode, not enough for
  statistically robust category-level accuracy figures.
- **Fictional knowledge base by design** — guarantees unambiguous ground
  truth (the model has no prior knowledge to hallucinate from other than
  what's in the doc), but doesn't test real-world document messiness
  (formatting noise, contradictory sources, very long context).
- **Dashboard reflects the last locally-run evaluation**, not live scoring —
  the deployed version can't call a local Ollama server, so it's a snapshot
  of whatever was last pushed.
- **Refusal detection is a simple string match** (`"not specified in the
  document"`), not a semantic check — a model refusing in different wording
  would currently be misclassified as a non-refusal.

## What I'd build next

- A second, independent judge model to cross-check the primary judge and
  quantify self-preference bias.
- A prompt-design experiment: does explicitly instructing the model to
  "correct false premises using the context" fix the 0% adversarial_conflicting
  score, or does the failure run deeper than prompting?
- Semantic refusal detection instead of string matching.