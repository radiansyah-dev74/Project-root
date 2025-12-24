# app/rag/vector_db.py

from typing import List
import math


class SimpleVectorDB:
    """
    Extremely simple in-memory vector DB.
    Token overlap similarity (no embeddings).
    """

    def __init__(self):
        self.documents: List[str] = []

    def add_documents(self, docs: List[str]):
        self.documents.extend(docs)

    def similarity_search(self, query: str, top_k: int = 3) -> List[str]:
        query_tokens = set(query.lower().split())

        scored = []
        for doc in self.documents:
            doc_tokens = set(doc.lower().split())
            score = len(query_tokens & doc_tokens)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]
