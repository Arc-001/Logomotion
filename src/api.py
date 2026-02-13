"""
FastAPI server for the Manim Graph RAG Agent.

Provides HTTP endpoints for video generation with full control
over length, depth, and topic.
"""

import os
import uuid
from typing import Optional, Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field, model_validator

app = FastAPI(
    title="Manim Graph RAG Agent",
    description="Generate mathematical animations using AI",
    version="1.0.0",
)


# ---------------------------------------------------------------------------
# Request / Response models
# ---------------------------------------------------------------------------

class GenerateRequest(BaseModel):
    """Request body for video generation."""
    prompt: Optional[str] = Field(
        None,
        description="Description of the animation (alias: topic)",
    )
    topic: Optional[str] = Field(
        None,
        description="Topic for the animation — used as prompt if prompt is not set",
    )
    title: Optional[str] = Field(None, description="Scene title")
    length: float = Field(
        float(os.getenv("VIDEO_LENGTH", "1.0")),
        ge=0.1,
        le=30.0,
        description="Target video length in minutes",
    )
    depth: Literal["basic", "detailed", "comprehensive"] = Field(
        os.getenv("EXPLANATION_DEPTH", "detailed"),
        description="How detailed the explanation should be",
    )
    orientation: Literal["landscape", "portrait"] = Field(
        os.getenv("VIDEO_ORIENTATION", "landscape"),
        description="Video orientation: landscape (16:9) or portrait (9:16)",
    )

    @model_validator(mode="after")
    def require_prompt_or_topic(self):
        if not self.prompt and not self.topic:
            raise ValueError("Either 'prompt' or 'topic' must be provided")
        # Normalise: ensure prompt is always populated
        if not self.prompt:
            self.prompt = self.topic
        return self


class GenerateResponse(BaseModel):
    """Response from video generation."""
    job_id: str
    status: str
    message: str


class JobStatus(BaseModel):
    """Status of a generation job."""
    job_id: str
    status: str  # pending, running, completed, failed
    video_path: Optional[str] = None
    code: Optional[str] = None
    error: Optional[str] = None


# Simple in-memory job store
jobs: dict[str, JobStatus] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Manim Graph RAG Agent"}


@app.get("/health")
async def health():
    """Explicit health-check."""
    return {"status": "ok"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start a video generation job.

    Accepts **topic** (or **prompt**), **length** (minutes), and **depth**
    (basic / detailed / comprehensive).  Returns immediately with a job ID
    that can be polled via ``GET /jobs/{job_id}``.
    """
    job_id = str(uuid.uuid4())[:8]

    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
    )

    background_tasks.add_task(
        _run_generation_job,
        job_id,
        request.prompt,
        request.title,
        request.length,
        request.depth,
        request.orientation,
    )

    return GenerateResponse(
        job_id=job_id,
        status="pending",
        message="Job queued for processing",
    )


@app.get("/jobs/{job_id}", response_model=JobStatus)
async def get_job_status(job_id: str):
    """Get the status of a generation job."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return jobs[job_id]


@app.get("/jobs/{job_id}/download")
async def download_video(job_id: str):
    """Download the generated video file."""
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail="Job not found")

    job = jobs[job_id]

    if job.status != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Job is not completed yet (status: {job.status})",
        )

    if not job.video_path or not Path(job.video_path).is_file():
        raise HTTPException(status_code=404, detail="Video file not found on disk")

    return FileResponse(
        path=job.video_path,
        media_type="video/mp4",
        filename=f"{job_id}.mp4",
    )


@app.get("/search")
async def search_examples(query: str, limit: int = 5):
    """Search for similar Manim examples."""
    from .graph_rag.retriever import ManimRetriever

    retriever = ManimRetriever()
    try:
        results = retriever.hybrid_search(query=query, limit=limit)
        return {
            "query": query,
            "results": [
                {
                    "id": r.example_id,
                    "prompt": r.prompt[:200],
                    "score": r.score,
                    "classes": r.used_classes,
                    "animations": r.used_animations,
                }
                for r in results
            ],
        }
    finally:
        retriever.close()


# ---------------------------------------------------------------------------
# Background job runner
# ---------------------------------------------------------------------------

async def _run_generation_job(
    job_id: str,
    prompt: str,
    title: Optional[str],
    length: float,
    depth: str,
    orientation: str,
):
    """Background task to run video generation."""
    from .agent.graph import generate_video

    jobs[job_id].status = "running"

    try:
        result = await generate_video(
            scene_title=title or "Generated Scene",
            scene_prompt_description=prompt,
            scene_length=length,
            explanation_depth=depth,
            orientation=orientation,
        )

        video_path = result.get("final_output_path") or result.get("rendered_video_path")

        if video_path:
            jobs[job_id] = JobStatus(
                job_id=job_id,
                status="completed",
                video_path=video_path,
                code=result.get("code"),
            )
        else:
            jobs[job_id] = JobStatus(
                job_id=job_id,
                status="failed",
                error=result.get("error", "Unknown error"),
            )
    except Exception as e:
        jobs[job_id] = JobStatus(
            job_id=job_id,
            status="failed",
            error=str(e),
        )
