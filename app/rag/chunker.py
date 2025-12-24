# app/rag/chunker.py

from typing import List


def chunk_text(text: str, source: str, chunk_size: int = 300) -> List[str]:
    """
    Very naive text chunking.
    """
    chunks = []
    current = ""

    for line in text.splitlines():
        if len(current) + len(line) > chunk_size:
            chunks.append(f"[{source}] {current.strip()}")
            current = ""
        current += line + " "

    if current.strip():
        chunks.append(f"[{source}] {current.strip()}")

    return chunks
