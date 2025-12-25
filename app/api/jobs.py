# app/api/jobs.py

from fastapi import APIRouter, UploadFile, File, HTTPException
from pathlib import Path
import json

from app.core.job_manager import JobManager
from app.core.worker import AsyncWorker
from app.rag.retriever import global_retriever, build_retriever
from app.rag.prompt_builder import (
    build_cv_evaluation_prompt, 
    build_project_evaluation_prompt,
    build_final_summary_prompt
)
from app.ai.llm_client import call_llm
from app.utils.pdf_reader import extract_text_from_pdf

router = APIRouter()

# Singletons
job_manager = JobManager()
worker = AsyncWorker(job_manager)

UPLOAD_DIR = Path("data/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)


def parse_llm_json_response(response_text: str) -> dict:
    """Parse LLM JSON response with fallback."""
    try:
        # Try to extract JSON if wrapped in other text
        start_idx = response_text.find('{')
        end_idx = response_text.rfind('}') + 1
        if start_idx != -1 and end_idx != 0:
            json_str = response_text[start_idx:end_idx]
            return json.loads(json_str)
        return json.loads(response_text)
    except json.JSONDecodeError:
        # Fallback parsing
        return {"error": "Failed to parse LLM response", "raw": response_text[:200]}


def evaluate_cv_pipeline(cv_text: str, job_title: str) -> dict:
    """
    CV Evaluation Pipeline.
    Uses: Job Description + CV Rubric as context.
    """
    try:
        # Retrieve context for CV evaluation
        context_chunks = global_retriever.search_for_cv_evaluation(
            query=f"{job_title} requirements technical skills"
        )
        
        # Build CV evaluation prompt
        prompt = build_cv_evaluation_prompt(cv_text, context_chunks, job_title)
        
        # Call LLM
        llm_output = call_llm(prompt)
        
        # Parse response
        result = parse_llm_json_response(llm_output)
        
        # Ensure match_rate is float 0-1
        match_rate = result.get("match_rate", 0.5)
        if isinstance(match_rate, str):
            try:
                match_rate = float(match_rate)
            except:
                match_rate = 0.5
        
        return {
            "match_rate": min(1.0, max(0.0, match_rate)),  # Clamp 0-1
            "feedback": result.get("feedback", llm_output[:500]),
            "raw_scores": result.get("scores", {}),
            "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt
        }
    except Exception as e:
        return {
            "match_rate": 0.0,
            "feedback": f"Error in CV evaluation: {str(e)}",
            "raw_scores": {},
            "error": str(e)
        }


def evaluate_project_pipeline(project_text: str) -> dict:
    """
    Project Evaluation Pipeline.
    Uses: Case Study Brief + Project Rubric as context.
    """
    try:
        # Retrieve context for project evaluation
        context_chunks = global_retriever.search_for_project_evaluation(
            query="project evaluation case study requirements"
        )
        
        # Build project evaluation prompt
        prompt = build_project_evaluation_prompt(project_text, context_chunks)
        
        # Call LLM
        llm_output = call_llm(prompt)
        
        # Parse response
        result = parse_llm_json_response(llm_output)
        
        # Ensure project_score is float 1-5
        project_score = result.get("project_score", 3.0)
        if isinstance(project_score, str):
            try:
                project_score = float(project_score)
            except:
                project_score = 3.0
        
        return {
            "project_score": min(5.0, max(1.0, project_score)),  # Clamp 1-5
            "feedback": result.get("feedback", llm_output[:500]),
            "raw_scores": result.get("scores", {}),
            "prompt_used": prompt[:200] + "..." if len(prompt) > 200 else prompt
        }
    except Exception as e:
        return {
            "project_score": 1.0,
            "feedback": f"Error in project evaluation: {str(e)}",
            "raw_scores": {},
            "error": str(e)
        }


def create_final_summary(cv_result: dict, project_result: dict) -> str:
    """
    Create final summary from both evaluations.
    """
    try:
        prompt = build_final_summary_prompt(cv_result, project_result)
        summary = call_llm(prompt)
        return summary.strip()
    except Exception as e:
        return f"Summary unavailable due to error: {str(e)}"


def full_evaluation_pipeline(cv_text: str, project_text: str, job_title: str) -> dict:
    """
    Full 3-stage evaluation pipeline.
    """
    print(f"Starting evaluation pipeline for: {job_title}")
    
    # 1. CV Evaluation
    print("  Stage 1: CV Evaluation")
    cv_result = evaluate_cv_pipeline(cv_text, job_title)
    
    # 2. Project Evaluation
    print("  Stage 2: Project Evaluation")
    project_result = evaluate_project_pipeline(project_text)
    
    # 3. Final Summary
    print("  Stage 3: Final Summary")
    overall_summary = create_final_summary(cv_result, project_result)
    
    # Combine results
    final_result = {
        "cv_match_rate": cv_result["match_rate"],
        "cv_feedback": cv_result["feedback"],
        "project_score": project_result["project_score"],
        "project_feedback": project_result["feedback"],
        "overall_summary": overall_summary,
        "cv_details": cv_result.get("raw_scores", {}),
        "project_details": project_result.get("raw_scores", {})
    }
    
    print(f"  ✅ Pipeline completed. CV match: {cv_result['match_rate']}, Project score: {project_result['project_score']}")
    return final_result


@router.post("/jobs/upload")
async def upload_job(
    cv_pdf: UploadFile = File(...),
    project_report: UploadFile = File(...)  # CHANGED: job_pdf → project_report
):
    """
    Upload CV and Project Report (not Job Description).
    """
    if not cv_pdf.filename.endswith(".pdf") or not project_report.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    job_id = job_manager.create_job()
    
    # Save files with new naming convention
    cv_path = UPLOAD_DIR / f"{job_id}_cv.pdf"
    project_path = UPLOAD_DIR / f"{job_id}_project.pdf"  # CHANGED: _job → _project
    
    with open(cv_path, "wb") as f:
        f.write(cv_pdf.file.read())
    
    with open(project_path, "wb") as f:
        f.write(project_report.file.read())
    
    # Run async evaluation
    worker.run_job(
        job_id=job_id,
        cv_pdf_path=str(cv_path),
        project_pdf_path=str(project_path),  # CHANGED parameter
        task_fn=full_evaluation_pipeline  # CHANGED: rag_pipeline → full_evaluation_pipeline
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "message": "CV and Project Report uploaded. Evaluation in progress."
    }


@router.post("/jobs/evaluate")
async def evaluate_job(
    job_title: str,
    cv_document_id: str,      # NEW parameter
    project_document_id: str   # NEW parameter
):
    """
    Trigger evaluation with specific document IDs.
    This is a simplified version that doesn't require re-upload.
    """
    # In a real system, you'd look up the documents by ID
    # For now, we'll create a new job and process
    job_id = job_manager.create_job()
    
    # For demo: assume IDs are filenames in uploads directory
    cv_path = UPLOAD_DIR / f"{cv_document_id}_cv.pdf"
    project_path = UPLOAD_DIR / f"{project_document_id}_project.pdf"
    
    if not cv_path.exists() or not project_path.exists():
        raise HTTPException(status_code=404, detail="Document not found")
    
    worker.run_job(
        job_id=job_id,
        cv_pdf_path=str(cv_path),
        project_pdf_path=str(project_path),
        task_fn=full_evaluation_pipeline
    )
    
    return {
        "job_id": job_id,
        "status": "queued",
        "job_title": job_title,
        "cv_document_id": cv_document_id,
        "project_document_id": project_document_id
    }


@router.get("/jobs/{job_id}")
def get_job_result(job_id: str):
    """
    Get evaluation result with standardized format.
    """
    job = job_manager.get_job(job_id)
    
    if not job:
        raise HTTPException(status_code=404, detail="Job not found")
    
    # Standardize response format
    response = {
        "id": job_id,
        "status": job.get("status", "unknown"),
        "created_at": job.get("created_at"),
        "updated_at": job.get("updated_at")
    }
    
    if job.get("status") == "completed" and "result" in job:
        result = job["result"]
        response["result"] = {
            "cv_match_rate": result.get("cv_match_rate", 0.0),
            "cv_feedback": result.get("cv_feedback", ""),
            "project_score": result.get("project_score", 0.0),
            "project_feedback": result.get("project_feedback", ""),
            "overall_summary": result.get("overall_summary", "")
        }
    
    elif job.get("status") == "failed":
        response["error"] = job.get("error", "Unknown error")
    
    return response
