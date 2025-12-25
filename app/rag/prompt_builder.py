# app/rag/prompt_builder.py

from typing import List
import os


def build_prompt(context_chunks: List[str]) -> str:
    """
    Generic prompt builder (backward compatible).
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


def build_cv_evaluation_prompt(cv_text: str, context_chunks: List[str], job_title: str) -> str:
    """
    Build prompt for CV evaluation with 4 specific parameters.
    """
    context = "\n".join(context_chunks)
    
    return f"""
SYSTEM:
You are an AI CV evaluator for a {job_title} position.
Evaluate the candidate's CV based on these 4 parameters:

1. TECHNICAL SKILLS MATCH (backend, databases, APIs, cloud, AI/LLM exposure)
2. EXPERIENCE LEVEL (years, project complexity)
3. RELEVANT ACHIEVEMENTS (impact, scale)
4. CULTURAL FIT (communication, learning attitude)

Score each parameter 1-5, then calculate overall match rate (0-1).
Provide specific feedback for each parameter.

CONTEXT (Job Description & CV Rubric):
{context}

CANDIDATE CV:
{cv_text}

TASK:
1. Score each parameter 1-5
2. Calculate weighted match rate (0-1 scale)
3. Provide detailed feedback for each parameter
4. Format as JSON: {{"scores": {{"technical_skills": X, "experience": X, "achievements": X, "cultural_fit": X}}, "match_rate": 0.X, "feedback": "..."}}
""".strip()


def build_project_evaluation_prompt(project_text: str, context_chunks: List[str]) -> str:
    """
    Build prompt for Project evaluation with 5 specific parameters.
    """
    context = "\n".join(context_chunks)
    
    return f"""
SYSTEM:
You are an AI project evaluator.
Evaluate the project report based on these 5 parameters (score 1-5 each):

1. CORRECTNESS (meets requirements: prompt design, chaining, RAG, handling errors)
2. CODE QUALITY (clean, modular, testable)
3. RESILIENCE (handles failures, retries)
4. DOCUMENTATION (clear README, explanation of trade-offs)
5. CREATIVITY / BONUS (optional improvements like authentication, deployment, dashboards)

CONTEXT (Case Study Brief & Project Rubric):
{context}

PROJECT REPORT:
{project_text}

TASK:
1. Score each parameter 1-5
2. Calculate average project score (1-5 scale)
3. Provide detailed feedback for each parameter
4. Format as JSON: {{"scores": {{"correctness": X, "code_quality": X, "resilience": X, "documentation": X, "creativity": X}}, "project_score": X.X, "feedback": "..."}}
""".strip()


def build_final_summary_prompt(cv_result: dict, project_result: dict) -> str:
    """
    Build prompt for final synthesis of both evaluations.
    """
    return f"""
SYSTEM:
You are an AI hiring assistant.
Synthesize the CV and Project evaluations into a concise overall summary.

CV EVALUATION:
- Match Rate: {cv_result.get('match_rate', 'N/A')}
- Feedback: {cv_result.get('feedback', 'No feedback')}

PROJECT EVALUATION:
- Score: {project_result.get('project_score', 'N/A')}
- Feedback: {project_result.get('feedback', 'No feedback')}

TASK:
Create a concise overall summary (2-3 paragraphs) that:
1. Highlights the candidate's strengths
2. Mentions areas for improvement
3. Provides a final recommendation
4. Keep it professional and actionable

OUTPUT:
Only the summary text, no JSON formatting.
""".strip()
