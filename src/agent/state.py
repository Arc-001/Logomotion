"""
State definitions for the LangGraph video generation workflow.

Based on the architecture diagram:
- video_code_gen.py with input/output structure
- Code generation with error correction loop
- Transcript processing for audio
- Video synchronization and merging
"""

import os
from typing import Optional, Annotated
from typing_extensions import TypedDict
from operator import add


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
    # Input fields
    system_message: str
    scene_title: str
    scene_prompt_description: str
    scene_length: float
    explanation_depth: str  # "basic", "detailed", or "comprehensive"
    
    # RAG context
    retrieved_examples: list[str]  # Code examples from Graph RAG
    retrieved_context: str  # Formatted context for LLM
    
    # Code generation
    code: str
    scene_class_name: str
    
    # Transcript
    transcript: dict  # {timestamp: text}
    transcript_sections: list[TranscriptSection]
    
    # Execution state
    error: Optional[str]
    error_count: Annotated[int, add]  # Tracks retry attempts
    max_retries: int
    
    # Rendering
    temp_code_path: Optional[str]
    rendered_video_path: Optional[str]
    render_logs: str
    
    # Validation
    video_valid: bool
    validation_errors: list[str]
    
    # Synchronization
    checked_video_path: Optional[str]
    synced_video_path: Optional[str]
    
    # Audio
    audio_segments: list[str]  # Paths to audio files
    
    # Final output
    final_output_path: Optional[str]
    
    # Metadata
    messages: Annotated[list[dict], add]  # Chat history for debugging


def create_initial_state(
    scene_title: str,
    scene_prompt_description: str,
    scene_length: float = 1.0,
    explanation_depth: str = "detailed",
    system_message: Optional[str] = None,
    max_retries: Optional[int] = None,
) -> VideoGenState:
    """Create initial state for the workflow."""
    # Read MAX_RETRIES from environment, default to 3
    if max_retries is None:
        max_retries = int(os.getenv("MAX_RETRIES", "3"))
    default_system = """You are an expert Manim animator creating educational math videos. Follow these CRITICAL guidelines:

## VIDEO LENGTH
- Target video duration is specified in minutes. Calculate total time needed.
- Use self.wait(seconds) generously between animations for pacing.
- Add at minimum 2-3 seconds wait after each major concept.
- Total animation time should match the target length.

## ELEMENT POSITIONING (CRITICAL - AVOID OVERLAPS)
- NEVER place elements at the same position. Always use explicit positioning.
- Use .to_edge(UP/DOWN/LEFT/RIGHT) to anchor elements to screen edges.
- Use .next_to(other_mobject, direction, buff=0.5) to position relative to others.
- Use .shift(direction * amount) to move elements.
- Group related elements with VGroup() and position the group.
- Clear previous elements with FadeOut() before showing new ones in the same area.
- Screen layout: UP for titles, LEFT for visuals, RIGHT for equations.

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
        # Input
        system_message=system_message or default_system,
        scene_title=scene_title,
        scene_prompt_description=scene_prompt_description,
        scene_length=scene_length,
        explanation_depth=explanation_depth,
        
        # RAG
        retrieved_examples=[],
        retrieved_context="",
        
        # Code
        code="",
        scene_class_name="",
        
        # Transcript
        transcript={},
        transcript_sections=[],
        
        # Execution
        error=None,
        error_count=0,
        max_retries=max_retries,
        
        # Rendering
        temp_code_path=None,
        rendered_video_path=None,
        render_logs="",
        
        # Validation
        video_valid=False,
        validation_errors=[],
        
        # Sync
        checked_video_path=None,
        synced_video_path=None,
        
        # Audio
        audio_segments=[],
        
        # Output
        final_output_path=None,
        
        # Debug
        messages=[],
    )
