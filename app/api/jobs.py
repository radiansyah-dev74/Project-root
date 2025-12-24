# app/api/jobs.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path

from app.core.job_manager import JobManager
from app.core.worker import AsyncWorker
from app.rag.retriever import build_retriever
from app.rag.prompt_builder import build_prompt
from app.ai.llm_client import call_llm

router = APIRouter()

# Singletons (acceptable for case study)
job_manager = JobManager()
worker = AsyncWorker(job_manager)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def rag_pipeline(cv_text: str, job_text: str):
    """
    Full pipeline:
    PDF → Text → RAG → LLM Evaluation
    """

    # 1. Build retriever (simple vector DB)
    retriever = build_retriever(cv_text, job_text)

    # 2. Retrieve relevant context
    query = "Evaluate backend engineer suitability"
    context_chunks = retriever.search(query)

    # 3. Build constrained prompt (RAG)
    prompt = build_prompt(context_chunks)

    # 4. Call real LLM
    llm_output = call_llm(prompt)

    return {
        "llm_output": llm_output,
        "prompt_used": prompt
    }


@router.post("/jobs/upload")
def upload_job(
    cv_pdf: UploadFile = File(...),
    job_pdf: UploadFile = File(...)
):
    if not cv_pdf.filename.endswith(".pdf") or not job_pdf.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    job_id = job_manager.create_job()

    cv_path = UPLOAD_DIR / f"{job_id}_cv.pdf"
    job_path = UPLOAD_DIR / f"{job_id}_job.pdf"

    with open(cv_path, "wb") as f:
        f.write(cv_pdf.file.read())

    with open(job_path, "wb") as f:
        f.write(job_pdf.file.read())

    # Run async evaluation
    worker.run_job(
        job_id=job_id,
        cv_pdf_path=str(cv_path),
        job_pdf_path=str(job_path),
        task_fn=rag_pipeline
    )

    return {
        "job_id": job_id,
        "status": "queued"
    }


@router.get("/jobs/{job_id}")
def get_job(job_id: str):
    job = job_manager.get_job(job_id)

    if not job:
        raise HTTPException(status_code=404, detail="Job not found")

    return job
