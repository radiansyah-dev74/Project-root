#!/usr/bin/env python3
"""
Script to ingest internal documents into global vector database.
Run this once: python scripts/ingest_internal.py
"""

import sys
import os

# Add project root to Python path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.rag.vector_db import global_vector_db
from app.rag.chunker import chunk_text


def read_text_file(filepath: str) -> str:
    """Read text file."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"  ❌ Error reading {filepath}: {e}")
        return ""


def ingest_internal_docs():
    """Load all internal documents into vector DB."""
    print("📥 Ingesting internal documents into vector DB...")
    
    # Get absolute paths
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    internal_dir = os.path.join(base_dir, "internal_docs")
    
    docs_to_ingest = [
        {
            "filename": "job_description.txt",
            "doc_type": "job_description",
            "source": "internal"
        },
        {
            "filename": "case_study_brief.txt", 
            "doc_type": "case_study",
            "source": "internal"
        },
        {
            "filename": "cv_scoring_rubric.txt",
            "doc_type": "cv_rubric", 
            "source": "internal"
        },
        {
            "filename": "project_scoring_rubric.txt",
            "doc_type": "project_rubric",
            "source": "internal"
        }
    ]
    
    all_chunks = []
    all_metadatas = []
    
    for doc_info in docs_to_ingest:
        filename = doc_info["filename"]
        filepath = os.path.join(internal_dir, filename)
        doc_type = doc_info["doc_type"]
        source = doc_info["source"]
        
        print(f"  Processing: {filename}")
        
        # Check if file exists
        if not os.path.exists(filepath):
            print(f"  ⚠️ File not found: {filepath}")
            # Try alternative path (relative to script)
            alt_path = os.path.join(base_dir, "internal_docs", filename)
            if os.path.exists(alt_path):
                filepath = alt_path
                print(f"  Found at: {alt_path}")
            else:
                print(f"  ❌ Skipping: File not found")
                continue
        
        # Read the file
        text = read_text_file(filepath)
        if not text:
            print(f"  ⚠️ Empty or unreadable: {filename}")
            continue
        
        # Chunk the text
        chunks = chunk_text(text, source=source, doc_type=doc_type)
        
        # Create metadata for each chunk
        for chunk in chunks:
            all_chunks.append(chunk)
            all_metadatas.append({
                "doc_type": doc_type,
                "source": source,
                "filename": filename
            })
        
        print(f"    Added {len(chunks)} chunks")
    
    # Add to global vector DB
    if all_chunks:
        global_vector_db.add_documents(all_chunks, all_metadatas)
        print(f"\n✅ Successfully added {len(all_chunks)} chunks to vector DB")
        print("📊 Document types breakdown:")
        
        # Count by doc_type
        from collections import Counter
        doc_type_counts = Counter([m["doc_type"] for m in all_metadatas])
        
        for doc_type, count in doc_type_counts.items():
            print(f"  - {doc_type}: {count} chunks")
        
        # Verify by searching
        print("\n🔍 Verification search:")
        test_queries = [
            ("backend developer", {"doc_type": "job_description"}),
            ("case study requirements", {"doc_type": "case_study"}),
            ("technical skills match", {"doc_type": "cv_rubric"}),
            ("correctness code quality", {"doc_type": "project_rubric"})
        ]
        
        for query, filter_dict in test_queries:
            results = global_vector_db.search_with_filter(query, filter_dict, top_k=1)
            if results:
                print(f"  ✓ '{query}': Found {len(results)} result(s)")
            else:
                print(f"  ⚠️ '{query}': No results")
                
    else:
        print("⚠️ No documents were ingested")
        print("\n💡 Troubleshooting:")
        print("1. Check files exist in internal_docs/ directory")
        print("2. Files should be: job_description.txt, case_study_brief.txt, cv_scoring_rubric.txt, project_scoring_rubric.txt")
        print("3. Check file permissions")


if __name__ == "__main__":
    # Change to project root directory
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    ingest_internal_docs()
