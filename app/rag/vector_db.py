# app/rag/vector_db.py

from typing import List, Dict, Any, Optional
import math


class SimpleVectorDB:
    """
    Enhanced in-memory vector DB with metadata support.
    Token overlap similarity (no embeddings).
    """

    def __init__(self):
        self.documents: List[str] = []
        self.metadatas: List[Dict[str, Any]] = []  # NEW: Metadata for each document

    def add_documents(self, docs: List[str], metadatas: Optional[List[Dict]] = None):
        """
        Add documents with optional metadata.
        """
        self.documents.extend(docs)
        
        if metadatas:
            self.metadatas.extend(metadatas)
        else:
            # Default metadata if not provided
            self.metadatas.extend([{} for _ in range(len(docs))])

    def similarity_search(self, query: str, top_k: int = 3) -> List[str]:
        """
        Search all documents (backward compatible).
        """
        query_tokens = set(query.lower().split())

        scored = []
        for doc in self.documents:
            doc_tokens = set(doc.lower().split())
            score = len(query_tokens & doc_tokens)
            scored.append((score, doc))

        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]

    def search_with_filter(
        self, 
        query: str, 
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = 3
    ) -> List[str]:
        """
        NEW: Search documents with metadata filtering.
        
        filter_dict example: {"doc_type": "cv_context"} 
        or {"doc_type": {"$in": ["cv_context", "project_context"]}}
        """
        # Step 1: Filter documents based on metadata
        filtered_indices = []
        for idx, metadata in enumerate(self.metadatas):
            if self._matches_filter(metadata, filter_dict):
                filtered_indices.append(idx)
        
        # Step 2: Search only in filtered documents
        query_tokens = set(query.lower().split())
        scored = []
        
        for idx in filtered_indices:
            doc = self.documents[idx]
            doc_tokens = set(doc.lower().split())
            score = len(query_tokens & doc_tokens)
            scored.append((score, doc))
        
        scored.sort(key=lambda x: x[0], reverse=True)
        return [doc for score, doc in scored[:top_k]]

    def _matches_filter(self, metadata: Dict, filter_dict: Optional[Dict]) -> bool:
        """
        Check if metadata matches filter criteria.
        """
        if not filter_dict:
            return True
        
        for key, value in filter_dict.items():
            if key not in metadata:
                return False
            
            if isinstance(value, dict) and "$in" in value:
                # Handle {"doc_type": {"$in": ["cv_context", "project_context"]}}
                if metadata[key] not in value["$in"]:
                    return False
            elif metadata[key] != value:
                return False
        
        return True

    def get_all_documents(self) -> List[Dict[str, Any]]:
        """
        Get all documents with their metadata.
        """
        return [
            {"document": doc, "metadata": meta}
            for doc, meta in zip(self.documents, self.metadatas)
        ]


# Global instance for internal documents
global_vector_db = SimpleVectorDB()
