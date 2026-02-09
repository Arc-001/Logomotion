"""
Main entry point for the Manim Graph RAG Agent.

CLI commands:
- index: Index the JSONL dataset into Graph RAG
- generate: Generate a Manim video from a prompt
"""

import os
import argparse
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


def cmd_index(args):
    """Index the Manim dataset."""
    from .graph_rag.indexer import ManimIndexer, IndexerConfig
    
    config = IndexerConfig(
        neo4j_uri=os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        neo4j_user=os.getenv("NEO4J_USER", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", "password"),
        chroma_persist_dir=os.getenv("CHROMA_PERSIST_DIRECTORY", "./chroma_db"),
    )
    
    indexer = ManimIndexer(config)
    try:
        count = indexer.index_directory(args.data)
        print(f"\n✓ Successfully indexed {count} examples")
    finally:
        indexer.close()


def cmd_generate(args):
    """Generate a Manim video."""
    from .agent.graph import generate_video_sync
    
    print(f"Generating video for: {args.prompt}")
    print("-" * 50)
    
    default_length = float(os.getenv("VIDEO_LENGTH", "1.0"))
    explanation_depth = args.depth or os.getenv("EXPLANATION_DEPTH", "detailed")
    scene_length = args.length if args.length != 1.0 else default_length
    
    print(f"[CONFIG] Scene length: {scene_length} minutes ({scene_length * 60} seconds)")
    print(f"[CONFIG] Explanation depth: {explanation_depth}")
    
    result = generate_video_sync(
        scene_title=args.title or "Generated Scene",
        scene_prompt_description=args.prompt,
        scene_length=scene_length,
        explanation_depth=explanation_depth,
    )
    
    if result.get("final_output_path"):
        print(f"\n✓ Video generated: {result['final_output_path']}")
    elif result.get("rendered_video_path"):
        print(f"\n✓ Video rendered: {result['rendered_video_path']}")
    else:
        print("\n✗ Video generation failed")
        if result.get("error"):
            print(f"Error: {result['error']}")
    
    if args.output and result.get("code"):
        output_dir = Path(args.output)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        code_path = output_dir / "scene.py"
        with open(code_path, "w") as f:
            f.write(result["code"])
        print(f"✓ Code saved: {code_path}")
        
        if result.get("final_output_path") or result.get("rendered_video_path"):
            import shutil
            video_src = result.get("final_output_path") or result["rendered_video_path"]
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
        "--depth", "-d",
        choices=["basic", "detailed", "comprehensive"],
        help="Explanation depth level (default: from .env or 'detailed')",
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
