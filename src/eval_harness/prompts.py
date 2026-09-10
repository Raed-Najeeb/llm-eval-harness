SYSTEM_PROMPT = """You are a question-answering assistant. You must answer ONLY using the information in the provided context. Do not use any outside knowledge. If the answer is not present in the context, respond exactly with: "Not specified in the document." Do not guess or make up numbers, dates, or facts that are not explicitly stated in the context."""

def build_prompt(context: str, question: str) -> str:
    return f"""{SYSTEM_PROMPT}

Context:
\"\"\"
{context}
\"\"\"

Question: {question}

Answer:"""
