import json
from pathlib import Path
import streamlit as st
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
SCORED_PATH = REPO_ROOT / "data" / "results" / "scored_outputs.jsonl"
HISTORY_PATH = REPO_ROOT / "data" / "results" / "run_history.jsonl"

st.set_page_config(page_title="LLM Eval Harness Dashboard", layout="wide")
st.title("LLM Eval & Red-Team Harness — Dashboard")
st.caption("Model under test: llama3.1:8b (via Ollama) | Judge: llama3.1:8b")

# --- Load data ---
with open(SCORED_PATH, encoding="utf-8") as f:
    scored = [json.loads(line) for line in f]
df = pd.DataFrame(scored)

with open(HISTORY_PATH, encoding="utf-8") as f:
    history = [json.loads(line) for line in f]

# --- Current run summary ---
st.header("Latest Run — Accuracy by Category")
summary = df.groupby("category")["verdict"].apply(lambda x: (x == "correct").mean()).reset_index()
summary.columns = ["category", "accuracy"]
st.bar_chart(summary.set_index("category")["accuracy"])
st.dataframe(summary.style.format({"accuracy": "{:.0%}"}), use_container_width=True)

# --- Drift over time ---
st.header("Accuracy Over Time (Drift Tracking)")
if len(history) >= 2:
    hist_rows = []
    for h in history:
        row = {"timestamp": h["timestamp"], "overall": h["overall_accuracy"]}
        row.update(h["by_category"])
        hist_rows.append(row)
    hist_df = pd.DataFrame(hist_rows)
    hist_df["timestamp"] = pd.to_datetime(hist_df["timestamp"])
    hist_df = hist_df.set_index("timestamp")
    st.line_chart(hist_df)
    st.caption(f"{len(history)} runs logged. A meaningful drop here would flag a regression in model, prompt, or eval design.")
else:
    st.info("Only one run logged so far — run score.py again after any change to start seeing drift trends.")

# --- Failing cases explorer ---
st.header("Failing Cases")
category_filter = st.selectbox("Filter by category", ["All"] + sorted(df["category"].unique().tolist()))
failures = df[df["verdict"] == "incorrect"]
if category_filter != "All":
    failures = failures[failures["category"] == category_filter]

st.write(f"{len(failures)} failing case(s)")
for _, row in failures.iterrows():
    with st.expander(f"{row['id']} ({row['category']}) — {row['question'][:60]}..."):
        st.write("**Question:**", row["question"])
        st.write("**Expected:**", row["expected_answer"])
        st.write("**Model said:**", row["model_answer"])
        st.write("**Verdict reasoning:**", row["verdict_reasoning"])