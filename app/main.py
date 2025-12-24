# app/main.py

from fastapi import FastAPI
from app.api.jobs import router as jobs_router

app = FastAPI(
    title="AI CV Screening Backend",
    description="Async job-based CV screening system with RAG architecture",
    version="1.0.0"
)

# Register API routers
app.include_router(jobs_router)


@app.get("/health")
def health_check():
    """
    Simple health check endpoint.
    Used for monitoring & deployment validation.
    """
    return {"status": "ok"}
