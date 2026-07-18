"""
LangGraph StateGraph builder for the Manim video generation workflow.
"""

from langgraph.graph import StateGraph, END

from .state import VideoGenState, create_initial_state
from .nodes import (
    video_code_gen_node,
    code_executor_node,
    recorrector_node,
    transcript_processor_node,
    render_checker_node,
    video_duration_fixer_node,
    synchronizer_node,
    audio_video_merger_node,
    web_research_node,
    storyboard_node,
    should_retry_or_continue,
    should_fix_duration,
)


def build_video_gen_graph() -> StateGraph:
    """
    Build the complete LangGraph for video generation.
    
    Graph structure:
    
    START
      │
      ├──web_search_enabled?──► [web_research] ──┐
      │                                           │
      └───────────────────────────────────────────┤
                                                  ▼
                                         [video_code_gen] ─────────────────┐
                                           │                                │
                                           │ code, transcript               │
                                           ▼                                ▼
                                         [code_executor]       [transcript_processor]
                                           │                                │
                                           ├──error?──► [recorrector]    │
                                           │                               │
                                           ▼ rendered video               │
                                         [render_checker]                  │
                                           │                               │
                                           ├──duration off?──► [video_duration_fixer]
                                           ▼                               │
                                         [synchronizer] ◄──────────────────┘
                                           │
                                           ▼
                                         [audio_video_merger]
                                           │
                                           ▼
                                          END
    """
    builder = StateGraph(VideoGenState)

    # Register all nodes
    builder.add_node("web_research", web_research_node)
    builder.add_node("storyboard", storyboard_node)
    builder.add_node("video_code_gen", video_code_gen_node)
    builder.add_node("code_executor", code_executor_node)
    builder.add_node("recorrector", recorrector_node)
    builder.add_node("transcript_processor", transcript_processor_node)
    builder.add_node("render_checker", render_checker_node)
    builder.add_node("video_duration_fixer", video_duration_fixer_node)
    builder.add_node("synchronizer", synchronizer_node)
    builder.add_node("audio_video_merger", audio_video_merger_node)

    # Conditional entry: web research first when enabled, then storyboard
    # planning when enabled, then code generation.
    def _entry_router(state: VideoGenState) -> str:
        if state.get("web_search_enabled"):
            return "web_research"
        if state.get("storyboard_enabled"):
            return "storyboard"
        return "video_code_gen"

    builder.set_conditional_entry_point(
        _entry_router,
        {
            "web_research": "web_research",
            "storyboard": "storyboard",
            "video_code_gen": "video_code_gen",
        },
    )

    def _after_web_research(state: VideoGenState) -> str:
        return "storyboard" if state.get("storyboard_enabled") else "video_code_gen"

    builder.add_conditional_edges(
        "web_research",
        _after_web_research,
        {
            "storyboard": "storyboard",
            "video_code_gen": "video_code_gen",
        },
    )

    builder.add_edge("storyboard", "video_code_gen")

    # After code generation, run both executor and transcript processor in parallel
    builder.add_edge("video_code_gen", "code_executor")
    builder.add_edge("video_code_gen", "transcript_processor")
    
    builder.add_conditional_edges(
        "code_executor",
        should_retry_or_continue,
        {
            "recorrector": "recorrector",
            "render_checker": "render_checker",
        }
    )
    
    builder.add_edge("recorrector", "code_executor")
    
    # After render checking, decide if duration needs fixing
    builder.add_conditional_edges(
        "render_checker",
        should_fix_duration,
        {
            "video_duration_fixer": "video_duration_fixer",
            "synchronizer": "synchronizer",
        }
    )
    
    builder.add_edge("video_duration_fixer", "synchronizer")
    builder.add_edge("transcript_processor", "synchronizer")
    builder.add_edge("synchronizer", "audio_video_merger")
    builder.add_edge("audio_video_merger", END)
    
    return builder


def compile_graph():
    """Build and compile the graph."""
    builder = build_video_gen_graph()
    return builder.compile()


graph = None


def get_graph():
    """Get or create the compiled graph."""
    global graph
    if graph is None:
        graph = compile_graph()
    return graph


async def generate_video(
    scene_title: str,
    scene_prompt_description: str,
    scene_length: float = 1.0,
    explanation_depth: str = "detailed",
    orientation: str = "landscape",
    duration_mode: str = "guide",
    web_search_enabled: bool = False,
    system_message: str = None,
    render_quality: str = None,
    render_fps: int = None,
) -> dict:
    """
    Generate a Manim video from a description.

    Args:
        scene_title: Title of the scene
        scene_prompt_description: Natural language description
        scene_length: Target length in minutes
        explanation_depth: Level of detail (basic, detailed, comprehensive)
        orientation: Video orientation (landscape or portrait)
        duration_mode: 'guide' = soft hint (default), 'strict' = ffmpeg speed adjust
        web_search_enabled: Fetch latest web/Wikipedia data before generating
        system_message: Optional custom system prompt
        render_quality: Manim quality flag l/m/h/p/k (default from settings)
        render_fps: Frame rate override (default: manim's default for the quality)

    Returns:
        Final state dict with paths to generated files
    """
    initial_state = create_initial_state(
        scene_title=scene_title,
        scene_prompt_description=scene_prompt_description,
        scene_length=scene_length,
        explanation_depth=explanation_depth,
        orientation=orientation,
        duration_mode=duration_mode,
        web_search_enabled=web_search_enabled,
        system_message=system_message,
        render_quality=render_quality,
        render_fps=render_fps,
    )
    
    compiled_graph = get_graph()
    
    final_state = await compiled_graph.ainvoke(initial_state)
    
    return final_state


def generate_video_sync(
    scene_title: str,
    scene_prompt_description: str,
    scene_length: float = 1.0,
    explanation_depth: str = "detailed",
    orientation: str = "landscape",
    duration_mode: str = "guide",
    web_search_enabled: bool = False,
    system_message: str = None,
    render_quality: str = None,
    render_fps: int = None,
) -> dict:
    """Synchronous version of generate_video."""
    import asyncio

    return asyncio.run(generate_video(
        scene_title=scene_title,
        scene_prompt_description=scene_prompt_description,
        scene_length=scene_length,
        explanation_depth=explanation_depth,
        orientation=orientation,
        duration_mode=duration_mode,
        web_search_enabled=web_search_enabled,
        system_message=system_message,
        render_quality=render_quality,
        render_fps=render_fps,
    ))
