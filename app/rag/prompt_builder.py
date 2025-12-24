# app/rag/prompt_builder.py

from typing import List


def build_prompt(context_chunks: List[str]) -> str:
    """
    Build constrained evaluation prompt (RAG).
    """

    context = "\n\n".join(context_chunks)

    return f"""
SYSTEM:
You are an AI evaluator.

CONTEXT:
{context}

TASK:
Evaluate the candidate strictly based on the context above.
Return structured evaluation only.
""".strip()
