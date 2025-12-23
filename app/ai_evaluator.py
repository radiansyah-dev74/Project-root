import os
import json
import time
import random
from typing import Dict


class EvaluationError(Exception):
    """
    Custom exception for AI evaluation failures.
    """
    pass


MAX_RETRIES = 3


def evaluate_cv(cv_text: str, job_description: str) -> Dict:
    """
    Evaluate CV against job description using an LLM (or mock).
    Includes retry and output validation.
    """

    for attempt in range(1, MAX_RETRIES + 1):
        try:
            response = _call_llm(cv_text, job_description)
            _validate_response(response)
            return response

        except Exception as e:
            if attempt == MAX_RETRIES:
                raise EvaluationError(
                    f"AI evaluation failed after {MAX_RETRIES} attempts. Last error: {str(e)}"
                )
            time.sleep(1)  # simple backoff


def _call_llm(cv_text: str, job_description: str) -> Dict:
    """
    Call external LLM provider if API key exists,
    otherwise return a mock response.
    """

    api_key = os.getenv("OPENROUTER_API_KEY")

    if not api_key:
        # Fallback mock response (for reproducibility)
        return _mock_response(cv_text, job_description)

    # ---- REAL LLM CALL PLACEHOLDER ----
    # This is intentionally minimal.
    # You are NOT required to make a real API call.
    #
    # If implemented, this function would:
    # - send prompt + context to OpenRouter
    # - receive structured JSON output
    #
    # For this assignment, mock behavior is sufficient.
    raise RuntimeError("Real LLM call not implemented (mock mode recommended).")


def _mock_response(cv_text: str, job_description: str) -> Dict:
    """
    Simulated AI response.
    Randomized slightly to mimic non-determinism.
    """

    score = round(random.uniform(0.6, 0.9), 2)

    return {
        "cv_match_score": score,
        "strengths": "Strong backend fundamentals and relevant experience.",
        "gaps": "Limited exposure to large-scale AI system integration."
    }


def _validate_response(response: Dict):
    """
    Validate AI output format and value ranges.
    """

    required_fields = ["cv_match_score", "strengths", "gaps"]

    for field in required_fields:
        if field not in response:
            raise ValueError(f"Missing field in AI response: {field}")

    score = response["cv_match_score"]

    if not isinstance(score, (int, float)):
        raise ValueError("cv_match_score must be a number")

    if score < 0 or score > 1:
        raise ValueError("cv_match_score must be between 0 and 1")
