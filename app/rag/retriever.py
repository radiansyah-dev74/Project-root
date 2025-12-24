# app/rag/retriever.py

from typing import List
from app.rag.vector_db import SimpleVectorDB
from app.rag.chunker import chunk_text


class Retriever:
    """
    Very simple retriever abstraction.
    """

    def __init__(self, vector_db: SimpleVectorDB):
        self.vector_db = vector_db

    def search(self, query: str, top_k: int = 3) -> List[str]:
        return self.vector_db.similarity_search(query, top_k=top_k)


def build_retriever(cv_text: str, job_text: str) -> Retriever:
    """
    Build retriever from CV + Job Description text.
    """

    chunks = []

    chunks.extend(chunk_text(cv_text, source="cv"))
    chunks.extend(chunk_text(job_text, source="job"))

    vector_db = SimpleVectorDB()
    vector_db.add_documents(chunks)

    return Retriever(vector_db)
