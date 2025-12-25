# app/rag/chunker.py

from typing import List, Dict, Any


def chunk_text(
    text: str, 
    source: str, 
    doc_type: str = "general",
    chunk_size: int = 300
) -> List[str]:
    """
    Enhanced text chunking with metadata tracking.
    Returns chunks with embedded metadata tags.
    """
    chunks = []
    current = ""
    
    for line in text.splitlines():
        if len(current) + len(line) > chunk_size:
            # Add chunk with metadata tags
            chunk_with_meta = f"[{source}|{doc_type}] {current.strip()}"
            chunks.append(chunk_with_meta)
            current = ""
        current += line + " "
    
    if current.strip():
        chunk_with_meta = f"[{source}|{doc_type}] {current.strip()}"
        chunks.append(chunk_with_meta)
    
    return chunks


def chunk_text_with_metadata(
    text: str,
    metadata: Dict[str, Any],
    chunk_size: int = 300
) -> List[Dict[str, Any]]:
    """
    Alternative: Return chunks with separate metadata.
    Returns list of {"text": chunk_text, "metadata": metadata}
    """
    chunks_text = []
    current = ""
    
    for line in text.splitlines():
        if len(current) + len(line) > chunk_size:
            chunks_text.append(current.strip())
            current = ""
        current += line + " "
    
    if current.strip():
        chunks_text.append(current.strip())
    
    # Return chunks with metadata
    return [{"text": chunk, "metadata": metadata} for chunk in chunks_text]
