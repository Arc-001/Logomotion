"""
FastAPI server for the Manim Graph RAG Agent.

Provides HTTP endpoints for video generation with full control
over length, depth, and topic.
"""

import uuid
from typing import Optional, Literal
from pathlib import Path

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from .config import get_settings, normalize_render_quality

STATIC_DIR = Path(__file__).parent / "static"

app = FastAPI(
    title="Manim Graph RAG Agent",
    description="Generate mathematical animations using AI",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=list(get_settings().cors_allow_origins),
    allow_methods=["*"],
    allow_headers=["*"],
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
    length: Optional[float] = Field(
        default=None,
        ge=0.1,
        le=30.0,
        description="Target video length in minutes",
    )
    depth: Literal["basic", "detailed", "comprehensive"] = Field(
        default=None,
        description="How detailed the explanation should be",
    )
    orientation: Literal["landscape", "portrait"] = Field(
        default=None,
        description="Video orientation: landscape (16:9) or portrait (9:16)",
    )
    duration_mode: Literal["strict", "guide"] = Field(
        default=None,
        description="Duration enforcement: 'guide' = soft hint (default), 'strict' = ffmpeg speed adjust",
    )
    web_search: bool = Field(
        default=False,
        description="Fetch latest web / Wikipedia data before generating the animation",
    )
    quality: Optional[Literal["low", "medium", "high", "l", "m", "h", "p", "k"]] = Field(
        default=None,
        description="Render quality (default from settings; low ≈ 480p, medium ≈ 720p, high ≈ 1080p)",
    )
    fps: Optional[int] = Field(
        default=None,
        ge=5,
        le=60,
        description="Frame rate override (default: manim's default for the quality)",
    )
    visual_qa: Optional[bool] = Field(
        default=None,
        description="Review rendered frames with the multimodal LLM and auto-fix layout problems (adds latency)",
    )

    @model_validator(mode="after")
    def apply_defaults_and_validate(self):
        settings = get_settings()
        if not self.prompt and not self.topic:
            raise ValueError("Either 'prompt' or 'topic' must be provided")
        if not self.prompt:
            self.prompt = self.topic
        if self.length is None:
            self.length = settings.video_length
        if self.depth is None:
            self.depth = settings.explanation_depth
        if self.orientation is None:
            self.orientation = settings.video_orientation
        if self.duration_mode is None:
            self.duration_mode = settings.duration_mode
        if self.quality is None:
            self.quality = settings.render_quality
        else:
            self.quality = normalize_render_quality(self.quality)
        if self.visual_qa is None:
            self.visual_qa = settings.visual_qa_enabled
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
    warnings: Optional[list[str]] = None  # non-fatal pipeline issues
    web_sources: Optional[list] = None  # populated when web_search was used


# Simple in-memory job store
_MAX_JOBS = 1000
jobs: dict[str, JobStatus] = {}


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def root():
    """Serve the frontend SPA."""
    index_file = STATIC_DIR / "index.html"
    if index_file.is_file():
        return HTMLResponse(content=index_file.read_text(), status_code=200)
    return HTMLResponse(content="<h1>Manim Graph RAG Agent</h1><p>Frontend not found.</p>", status_code=200)


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
    job_id = str(uuid.uuid4())[:12]

    # Evict oldest completed/failed jobs when store is full
    if len(jobs) >= _MAX_JOBS:
        to_remove = [
            jid for jid, j in jobs.items()
            if j.status in ("completed", "failed")
        ]
        for jid in to_remove[:len(jobs) - _MAX_JOBS + 1]:
            del jobs[jid]

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
        request.duration_mode,
        request.web_search,
        request.quality,
        request.fps,
        request.visual_qa,
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
    duration_mode: str,
    web_search: bool = False,
    quality: Optional[str] = None,
    fps: Optional[int] = None,
    visual_qa: Optional[bool] = None,
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
            duration_mode=duration_mode,
            web_search_enabled=web_search,
            render_quality=quality,
            render_fps=fps,
            visual_qa=visual_qa,
        )

        video_path = result.get("final_output_path") or result.get("rendered_video_path")
        warnings = result.get("pipeline_warnings") or None

        if video_path:
            jobs[job_id] = JobStatus(
                job_id=job_id,
                status="completed",
                video_path=video_path,
                code=result.get("code"),
                warnings=warnings,
                web_sources=result.get("web_sources") or None,
            )
        else:
            error = result.get("error")
            if not error and warnings:
                error = "; ".join(warnings)
            jobs[job_id] = JobStatus(
                job_id=job_id,
                status="failed",
                error=error or "Unknown error",
                warnings=warnings,
            )
    except Exception as e:
        jobs[job_id] = JobStatus(
            job_id=job_id,
            status="failed",
            error=str(e),
        )


# ---------------------------------------------------------------------------
# Static file serving (must be after all API routes)
# ---------------------------------------------------------------------------

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# The Vite build references its hashed bundles as /assets/..., so the
# assets directory must be mounted at that exact path.
_ASSETS_DIR = STATIC_DIR / "assets"
if _ASSETS_DIR.is_dir():
    app.mount("/assets", StaticFiles(directory=str(_ASSETS_DIR)), name="assets")
