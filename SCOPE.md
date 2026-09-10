# Scope

**System under test:** RAG-style Q&A over a small document set, answered by Llama 3.1 8B (served locally via Ollama).

**What this harness evaluates:**
1. Faithfulness — does the answer stay grounded in the retrieved context, or does it hallucinate facts not present in the source?
2. Robustness — does the model degrade gracefully on adversarial/edge-case inputs (ambiguous questions, irrelevant context, conflicting context)?
3. Drift — do faithfulness scores degrade over repeated runs or config changes?

**Out of scope (for now):** retrieval quality itself (assume a fixed, pre-built context per test case), multi-turn conversation, non-English input.
