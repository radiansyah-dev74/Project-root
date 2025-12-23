import json
from ai_evaluator import evaluate_cv, EvaluationError


def load_text(path: str) -> str:
    """
    Load text file safely.
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if not content:
                raise ValueError("File is empty")
            return content
    except Exception as e:
        raise RuntimeError(f"Failed to load file: {path}. Reason: {str(e)}")


def run_evaluation():
    """
    Main system workflow:
    - Load inputs
    - Validate inputs
    - Run AI evaluation
    - Handle success or failure
    """

    print("Starting AI-based CV screening evaluation...\n")

    try:
        # 1. Load input data (dummy / sample)
        cv_text = load_text("sample_data/sample_cv.txt")
        job_description = load_text("sample_data/sample_job_description.txt")

        # 2. Run AI evaluation (Test 3)
        result = evaluate_cv(
            cv_text=cv_text,
            job_description=job_description
        )

        # 3. Store / display result
        print("Evaluation completed successfully.\n")
        print(json.dumps(result, indent=2))

        # Optional: save result
        with open("output_result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

        print("\nResult saved to output_result.json")

    except EvaluationError as e:
        # Controlled AI-related failure
        print("\nEvaluation failed due to AI error.")
        print(f"Reason: {str(e)}")

    except Exception as e:
        # Unexpected system failure
        print("\nSystem error occurred.")
        print(f"Reason: {str(e)}")


if __name__ == "__main__":
    run_evaluation()
