# app/rag/retriever.py

from typing import List
from app.rag.vector_db import SimpleVectorDB, global_vector_db
from app.rag.chunker import chunk_text


class Retriever:
    """
    Enhanced retriever with context filtering for CV vs Project evaluation.
    """

    def __init__(self, vector_db: SimpleVectorDB):
        self.vector_db = vector_db

    def search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Generic search (backward compatible).
        """
        return self.vector_db.similarity_search(query, top_k=top_k)

    def search_for_cv_evaluation(self, query: str, top_k: int = 5) -> List[str]:
        """
        NEW: Search ONLY in CV context documents.
        CV context = Job Description + CV Scoring Rubric
        """
        return self.vector_db.search_with_filter(
            query=query,
            filter_dict={"doc_type": {"$in": ["job_description", "cv_rubric"]}},
            top_k=top_k
        )

    def search_for_project_evaluation(self, query: str, top_k: int = 5) -> List[str]:
        """
        NEW: Search ONLY in Project context documents.
        Project context = Case Study Brief + Project Scoring Rubric
        """
        return self.vector_db.search_with_filter(
            query=query,
            filter_dict={"doc_type": {"$in": ["case_study", "project_rubric"]}},
            top_k=top_k
        )


def build_retriever(cv_text: str, job_text: str) -> Retriever:
    """
    Build retriever from CV + Job Description text (for uploaded files).
    """
    chunks = []
    metadatas = []

    # CV chunks with metadata
    cv_chunks = chunk_text(cv_text, source="cv")
    chunks.extend(cv_chunks)
    metadatas.extend([{"source": "cv", "doc_type": "candidate_cv"} for _ in cv_chunks])

    # Job Description chunks with metadata (from uploaded file)
    job_chunks = chunk_text(job_text, source="job")
    chunks.extend(job_chunks)
    metadatas.extend([{"source": "job", "doc_type": "uploaded_jd"} for _ in job_chunks])

    vector_db = SimpleVectorDB()
    vector_db.add_documents(chunks, metadatas)

    return Retriever(vector_db)


def get_global_retriever() -> Retriever:
    """
    Get retriever for internal documents (global instance).
    """
    return Retriever(global_vector_db)


# Initialize global retriever
global_retriever = get_global_retriever()
