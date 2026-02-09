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
    synchronizer_node,
    audio_video_merger_node,
    should_retry_or_continue,
)


def build_video_gen_graph() -> StateGraph:
    """
    Build the complete LangGraph for video generation.
    
    Graph structure (matching the architecture diagram):
    
    START
      │
      ▼
    [video_code_gen] ─────────────────────────┐
      │                                        │
      │ code, transcript                       │
      ▼                                        ▼
    [code_executor]                [transcript_processor]
      │                                        │
      ├──error?──► [recorrector] ◄────────┐   │
      │            │                       │   │
      │            └───────────────────────┘   │
      │                                        │
      ▼ rendered video                         │
    [render_checker]                           │
      │                                        │
      ▼                                        │
    [synchronizer] ◄───────────────────────────┘
      │
      ▼
    [audio_video_merger]
      │
      ▼
     END
    """
    builder = StateGraph(VideoGenState)
    
    builder.add_node("video_code_gen", video_code_gen_node)
    builder.add_node("code_executor", code_executor_node)
    builder.add_node("recorrector", recorrector_node)
    builder.add_node("transcript_processor", transcript_processor_node)
    builder.add_node("render_checker", render_checker_node)
    builder.add_node("synchronizer", synchronizer_node)
    builder.add_node("audio_video_merger", audio_video_merger_node)
    
    builder.set_entry_point("video_code_gen")
    
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
    builder.add_edge("render_checker", "synchronizer")
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
    system_message: str = None,
) -> dict:
    """
    Generate a Manim video from a description.
    
    Args:
        scene_title: Title of the scene
        scene_prompt_description: Natural language description
        scene_length: Target length in minutes
        explanation_depth: Level of detail (basic, detailed, comprehensive)
        system_message: Optional custom system prompt
    
    Returns:
        Final state dict with paths to generated files
    """
    initial_state = create_initial_state(
        scene_title=scene_title,
        scene_prompt_description=scene_prompt_description,
        scene_length=scene_length,
        explanation_depth=explanation_depth,
        system_message=system_message,
    )
    
    compiled_graph = get_graph()
    
    final_state = await compiled_graph.ainvoke(initial_state)
    
    return final_state


def generate_video_sync(
    scene_title: str,
    scene_prompt_description: str,
    scene_length: float = 1.0,
    explanation_depth: str = "detailed",
    system_message: str = None,
) -> dict:
    """Synchronous version of generate_video."""
    import asyncio
    
    return asyncio.run(generate_video(
        scene_title=scene_title,
        scene_prompt_description=scene_prompt_description,
        scene_length=scene_length,
        explanation_depth=explanation_depth,
        system_message=system_message,
    ))
