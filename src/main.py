"""
Main entry point for the Manim Graph RAG Agent.

CLI commands:
- index: Index the JSONL dataset into Graph RAG
- generate: Generate a Manim video from a prompt
- serve: Start the API server
"""

import argparse
import shutil
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

from .config import get_settings


def cmd_index(args):
    """Index the Manim dataset."""
    from .graph_rag.indexer import ManimIndexer

    indexer = ManimIndexer()
    try:
        count = indexer.index_directory(args.data)
        print(f"\n✓ Successfully indexed {count} examples")
    finally:
        indexer.close()


def cmd_generate(args):
    """Generate a Manim video."""
    from .agent.graph import generate_video_sync

    settings = get_settings()

    print(f"Generating video for: {args.prompt}")
    print("-" * 50)

    from .config import normalize_render_quality

    scene_length = args.length if args.length != 1.0 else settings.video_length
    explanation_depth = args.depth or settings.explanation_depth
    orientation = args.orientation or settings.video_orientation
    duration_mode = args.duration_mode or settings.duration_mode
    render_quality = normalize_render_quality(args.quality) if args.quality else settings.render_quality
    render_fps = args.fps or settings.render_fps

    print(f"[CONFIG] Scene length: {scene_length} minutes ({scene_length * 60} seconds)")
    print(f"[CONFIG] Explanation depth: {explanation_depth}")
    print(f"[CONFIG] Orientation: {orientation}")
    print(f"[CONFIG] Duration mode: {duration_mode}")
    print(f"[CONFIG] Render quality: {render_quality}" + (f", fps: {render_fps}" if render_fps else ""))

    result = generate_video_sync(
        scene_title=args.title or "Generated Scene",
        scene_prompt_description=args.prompt,
        scene_length=scene_length,
        explanation_depth=explanation_depth,
        orientation=orientation,
        duration_mode=duration_mode,
        render_quality=render_quality,
        render_fps=render_fps,
    )

    if result.get("final_output_path"):
        print(f"\n✓ Video generated: {result['final_output_path']}")
    elif result.get("rendered_video_path"):
        print(f"\n✓ Video rendered: {result['rendered_video_path']}")
    else:
        print("\n✗ Video generation failed")
        if result.get("error"):
            print(f"Error: {result['error']}")

    warnings = result.get("pipeline_warnings")
    if warnings:
        print("\nWarnings:")
        for warning in warnings:
            print(f"  - {warning}")

    if args.output and result.get("code"):
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)

        code_path = output_dir / "scene.py"
        with open(code_path, "w") as f:
            f.write(result["code"])
        print(f"✓ Code saved: {code_path}")

        video_src = result.get("final_output_path") or result.get("rendered_video_path")
        if video_src:
            video_dst = output_dir / "output.mp4"
            shutil.copy2(video_src, video_dst)
            print(f"✓ Video saved: {video_dst}")


def cmd_serve(args):
    """Start the API server."""
    try:
        import uvicorn
        from .api import app

        print(f"Starting server on http://{args.host}:{args.port}")
        uvicorn.run(app, host=args.host, port=args.port)
    except ImportError:
        print("Error: uvicorn or fastapi not installed")
        print("Install with: pip install fastapi uvicorn")


def main():
    parser = argparse.ArgumentParser(
        prog="manim-agent",
        description="Manim Graph RAG Agent - Generate mathematical animations",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    index_parser = subparsers.add_parser("index", help="Index the JSONL dataset")
    index_parser.add_argument(
        "--data", "-d",
        required=True,
        help="Path to directory containing JSONL files",
    )

    gen_parser = subparsers.add_parser("generate", help="Generate a Manim video")
    gen_parser.add_argument(
        "--prompt", "-p",
        required=True,
        help="Description of the animation to create",
    )
    gen_parser.add_argument(
        "--title", "-t",
        help="Title for the scene",
    )
    gen_parser.add_argument(
        "--length", "-l",
        type=float,
        default=1.0,
        help="Target length in minutes (default: 1.0)",
    )
    gen_parser.add_argument(
        "--output", "-o",
        help="Output directory for generated files",
    )
    gen_parser.add_argument(
        "--depth",
        choices=["basic", "detailed", "comprehensive"],
        help="Explanation depth level (default: from .env or 'detailed')",
    )
    gen_parser.add_argument(
        "--orientation", "-O",
        choices=["landscape", "portrait"],
        default=None,
        help="Video orientation (default: from .env or 'landscape')",
    )
    gen_parser.add_argument(
        "--duration-mode",
        choices=["strict", "guide"],
        default=None,
        dest="duration_mode",
        help="Duration enforcement: 'guide' = soft hint (default), 'strict' = ffmpeg speed adjust",
    )
    gen_parser.add_argument(
        "--quality", "-q",
        choices=["low", "medium", "high", "l", "m", "h", "p", "k"],
        default=None,
        help="Render quality (default: from .env or 'medium')",
    )
    gen_parser.add_argument(
        "--fps",
        type=int,
        default=None,
        help="Frame rate override (default: manim's default for the quality)",
    )

    serve_parser = subparsers.add_parser("serve", help="Start the API server")
    serve_parser.add_argument(
        "--host",
        default="0.0.0.0",
        help="Host to bind to (default: 0.0.0.0)",
    )
    serve_parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to bind to (default: 8000)",
    )

    args = parser.parse_args()

    if args.command == "index":
        cmd_index(args)
    elif args.command == "generate":
        cmd_generate(args)
    elif args.command == "serve":
        cmd_serve(args)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
