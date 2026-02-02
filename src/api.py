"""
FastAPI server for the Manim Graph RAG Agent.

Provides HTTP endpoints for video generation.
"""

from typing import Optional
from fastapi import FastAPI, HTTPException, BackgroundTasks
from pydantic import BaseModel, Field
import os
import uuid
from pathlib import Path

app = FastAPI(
    title="Manim Graph RAG Agent",
    description="Generate mathematical animations using AI",
    version="1.0.0",
)


class GenerateRequest(BaseModel):
    """Request body for video generation."""
    prompt: str = Field(..., description="Description of the animation")
    title: Optional[str] = Field(None, description="Scene title")
    length: float = Field(1.0, description="Target length in minutes")


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
    code_path: Optional[str] = None
    error: Optional[str] = None


# Simple in-memory job store
jobs: dict[str, JobStatus] = {}


@app.get("/")
async def root():
    """Health check endpoint."""
    return {"status": "ok", "service": "Manim Graph RAG Agent"}


@app.post("/generate", response_model=GenerateResponse)
async def generate_video(request: GenerateRequest, background_tasks: BackgroundTasks):
    """
    Start a video generation job.
    
    Returns immediately with a job ID that can be used to check status.
    """
    job_id = str(uuid.uuid4())[:8]
    
    jobs[job_id] = JobStatus(
        job_id=job_id,
        status="pending",
    )
    
    background_tasks.add_task(
        run_generation_job,
        job_id,
        request.prompt,
        request.title,
        request.length,
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


async def run_generation_job(
    job_id: str,
    prompt: str,
    title: Optional[str],
    length: float,
):
    """Background task to run video generation."""
    from .agent.graph import generate_video
    
    jobs[job_id].status = "running"
    
    try:
        result = await generate_video(
            scene_title=title or "Generated Scene",
            scene_prompt_description=prompt,
            scene_length=length,
        )
        
        video_path = result.get("final_output_path") or result.get("rendered_video_path")
        
        if video_path:
            jobs[job_id] = JobStatus(
                job_id=job_id,
                status="completed",
                video_path=video_path,
                code_path=result.get("temp_code_path"),
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
            ]
        }
    finally:
        retriever.close()
