"""
State definitions for the LangGraph video generation workflow.

Based on the architecture diagram:
- video_code_gen.py with input/output structure
- Code generation with error correction loop
- Transcript processing for audio
- Video synchronization and merging
"""

from typing import Optional, Annotated
from typing_extensions import TypedDict
from operator import add

from ..config import get_settings


class VideoGenInput(TypedDict):
    """Input structure for video generation."""
    system_message: str  # Optimized system message for code generation
    scene_title: str
    scene_prompt_description: str
    scene_length: float  # Length in minutes


class VideoGenOutput(TypedDict):
    """Output structure from video generation."""
    code: str
    transcript: dict  # {timestamp_in_seconds: text_to_be_said}


class TranscriptSection(TypedDict):
    """A section of the transcript with timestamp."""
    timestamp: float  # Seconds into video
    text: str
    audio_path: Optional[str]


class VideoGenState(TypedDict):
    """
    Complete state for the video generation workflow.
    
    This state flows through all nodes in the LangGraph:
    1. video_code_gen -> code + transcript
    2. code_executor -> rendered video or error
    3. recorrector -> fixed code (if error)
    4. transcript_processor -> sectioned transcript with audio
    5. render_checker -> validated video
    6. synchronizer -> aligned video with transcript
    7. audio_video_merger -> final output
    """
    system_message: str
    scene_title: str
    scene_prompt_description: str
    scene_length: float
    explanation_depth: str  # "basic", "detailed", or "comprehensive"
    orientation: str  # "landscape" or "portrait"
    duration_mode: str  # "guide" = soft hint, "strict" = ffmpeg speed adjust
    render_quality: str  # manim quality flag: l, m, h, p, or k
    render_fps: Optional[int]  # frame rate override (None = manim default)
    
    # Web research (optional — toggled by the API request)
    web_search_enabled: bool          # Whether to run web research before code gen
    web_context: str                  # Formatted research brief injected into code-gen prompt
    web_sources: list[dict]           # Serialised WebSource list for job metadata

    retrieved_examples: list[str]  # Code examples from Graph RAG
    retrieved_context: str  # Formatted context for LLM
    
    code: str
    scene_class_name: str
    
    transcript: dict  # {timestamp: text}
    transcript_sections: list[TranscriptSection]
    
    error: Optional[str]
    error_count: Annotated[int, add]  # Tracks retry attempts
    max_retries: int
    
    temp_code_path: Optional[str]
    rendered_video_path: Optional[str]
    render_logs: str
    
    video_valid: bool
    validation_errors: list[str]
    
    checked_video_path: Optional[str]
    synced_video_path: Optional[str]
    
    audio_segments: list[str]  # Paths to audio files

    temp_dirs: Annotated[list[str], add]  # Per-job temp dirs, removed after final merge

    actual_duration: Optional[float]  # Measured duration of rendered video (seconds)
    target_duration: float  # Computed target duration (scene_length * 60)
    duration_adjusted: bool  # Whether video was time-stretched/trimmed
    duration_factor: Optional[float]  # Factor applied to adjust duration
    
    final_output_path: Optional[str]

    pipeline_warnings: Annotated[list[str], add]  # Non-fatal issues surfaced to the user

    messages: Annotated[list[dict], add]  # Chat history for debugging


def create_initial_state(
    scene_title: str,
    scene_prompt_description: str,
    scene_length: float = 1.0,
    explanation_depth: str = "detailed",
    orientation: str = "landscape",
    duration_mode: str = "guide",
    web_search_enabled: bool = False,
    system_message: Optional[str] = None,
    max_retries: Optional[int] = None,
    render_quality: Optional[str] = None,
    render_fps: Optional[int] = None,
) -> VideoGenState:
    """Create initial state for the workflow."""
    settings = get_settings()
    if max_retries is None:
        max_retries = settings.max_retries
    if render_quality is None:
        render_quality = settings.render_quality
    if render_fps is None:
        render_fps = settings.render_fps
    
    if orientation == "portrait":
        frame_desc = "8 units wide (-4 to +4) and 14.2 units tall (-7.1 to +7.1)"
        layout_hint = "Screen layout: UP for titles, CENTER for visuals, DOWN for equations. Content is tall and narrow."
    else:
        frame_desc = "14.2 units wide (-7.1 to +7.1) and 8 units tall (-4 to +4)"
        layout_hint = "Screen layout: UP for titles, LEFT for visuals, RIGHT for equations."
    
    default_system = f"""You are an expert Manim animator creating educational math videos. Follow these CRITICAL guidelines:

## VIDEO LENGTH
- Target video duration is specified in minutes. Calculate total time needed.
- Use self.wait(seconds) generously between animations for pacing.
- Add at minimum 2-3 seconds wait after each major concept.
- Total animation time should match the target length.

## ELEMENT POSITIONING (CRITICAL - AVOID OVERLAPS)

### RULE 1 — THE "CLEAR DESK" RULE (CRITICAL)
    - Before starting a new major topic or showing a new full-screen text list, you MUST clear ALL previous elements first.
    - REQUIRED: `self.play(FadeOut(Group(*self.mobjects)))` OR `self.clear()` — use one of these two patterns.
    - NEVER render new text or visuals on top of existing elements that are already on screen.

### RULE 2 — FORBIDDEN SYNTAX
    - DO NOT use raw absolute coordinate arrays: NO `move_to([2, -1, 0])`, NO `move_to(np.array([x, y, 0]))`.
    - These always produce miscalculated spacing and overlapping elements.

### RULE 3 — MANDATORY RELATIVE POSITIONING
    - ONLY position elements relative to screen edges or to other mobjects.
    - Screen edges: `.to_edge(UP)`, `.to_edge(DOWN)`, `.to_edge(LEFT)`, `.to_edge(RIGHT)`.
    - Stacking: `.next_to(other_mobject, DOWN, buff=0.5)` to place elements sequentially.
    - Centering: `.move_to(ORIGIN)` is allowed (ORIGIN is a named constant, not a raw array).
    - The Manim frame is {frame_desc}.
    - {layout_hint}

### RULE 4 — STANDARD GRID LAYOUT
    - Scene titles: ALWAYS `.to_edge(UP, buff=0.5)`.
    - Core visuals (graphs, arrays, diagrams): Anchor to `ORIGIN` or `.to_edge(LEFT, buff=1)`.
    - Explanatory text / bullet lists: Anchor `.to_edge(RIGHT, buff=1)` or stack below the title using `.next_to`.

### RULE 5 — BOUNDING BOX SAFETY (VGROUPS)
    - Whenever creating multiple lines of text or a complex diagram, wrap them in a `VGroup()`.
    - IMMEDIATELY after building the group, apply: `my_group.scale_to_fit_width(min(my_group.width, config.frame_width - 2))`
    - This prevents any element from bleeding off-screen.

## ANIMATION QUALITY
- Always FadeOut or Transform old elements before introducing new ones.
- Use smooth animations: FadeIn, FadeOut, Transform, Write.
- Add self.wait(1) to self.wait(3) between sections.
- Use run_time parameter for slower, clearer animations.

## CODE STRUCTURE
- Use Manim Community Edition syntax.
- Import: from manim import *
- Create a Scene class with construct method.
- Include numpy as np if needed.
- Use only standard Manim colors: RED, BLUE, GREEN, YELLOW, WHITE, ORANGE, PURPLE, PINK, TEAL, GOLD."""
    
    return VideoGenState(
        system_message=system_message or default_system,
        scene_title=scene_title,
        scene_prompt_description=scene_prompt_description,
        scene_length=scene_length,
        explanation_depth=explanation_depth,
        orientation=orientation,
        duration_mode=duration_mode,
        render_quality=render_quality,
        render_fps=render_fps,

        web_search_enabled=web_search_enabled,
        web_context="",
        web_sources=[],

        retrieved_examples=[],
        retrieved_context="",
        
        code="",
        scene_class_name="",
        
        transcript={},
        transcript_sections=[],
        
        error=None,
        error_count=0,
        max_retries=max_retries,
        
        temp_code_path=None,
        rendered_video_path=None,
        render_logs="",
        
        video_valid=False,
        validation_errors=[],
        
        checked_video_path=None,
        synced_video_path=None,
        
        audio_segments=[],

        temp_dirs=[],

        actual_duration=None,
        target_duration=scene_length * 60.0,
        duration_adjusted=False,
        duration_factor=None,
        
        final_output_path=None,

        pipeline_warnings=[],

        messages=[],
    )
