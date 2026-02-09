"""
LangGraph nodes for the Manim video generation workflow.

Implements the following nodes from the architecture diagram:
1. video_code_gen_node - Main code generation with RAG
2. code_executor_node - Runs Manim and captures output/errors
3. recorrector_node - Fixes code based on error feedback
4. transcript_processor_node - Splits transcript into sections
5. render_checker_node - Validates rendered video
6. synchronizer_node - Aligns video with transcript
7. audio_video_merger_node - Final merge
"""

import re
import os
import tempfile
import subprocess
from pathlib import Path
from typing import Literal, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

from .state import VideoGenState, TranscriptSection
from ..graph_rag.retriever import ManimRetriever


# ============================================================================
# OpenRouter LLM Client
# ============================================================================

def get_llm_client() -> Optional["OpenAI"]:
    """Get OpenRouter client configured from environment."""
    if not OPENAI_AVAILABLE:
        return None
    
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    
    if not api_key:
        return None
    
    return OpenAI(base_url=base_url, api_key=api_key)


def llm_chat(messages: list[dict], temperature: float = 0.2) -> Optional[str]:
    """Call OpenRouter LLM with messages."""
    client = get_llm_client()
    if not client:
        return None
    
    model = os.getenv("OPENROUTER_MODEL", "google/gemini-3-flash-preview")
    
    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://manim-agent.local",
                "X-Title": "Manim Graph RAG Agent",
            },
            model=model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


# ============================================================================
# NODE 1: Video Code Generator (with Graph RAG)
# ============================================================================

def video_code_gen_node(state: VideoGenState) -> dict:
    """
    Generate Manim code using Graph RAG for context.
    
    Input: system_message, scene_title, scene_prompt_description
    Output: code, transcript, retrieved_examples, retrieved_context
    """
    # Initialize retriever
    retriever = ManimRetriever()
    
    try:
        # Hybrid search for relevant examples
        results = retriever.hybrid_search(
            query=state["scene_prompt_description"],
            limit=3,
        )
        
        # Format context from retrieved examples
        context_parts = []
        example_ids = []
        for result in results:
            example_ids.append(result.example_id)
            context_parts.append(f"""
### Example: {result.prompt[:100]}...
```python
{result.code[:1500]}...
```
Used classes: {', '.join(result.used_classes)}
Used animations: {', '.join(result.used_animations)}
""")
        
        retrieved_context = "\n---\n".join(context_parts) if context_parts else "No similar examples found."
    except Exception as e:
        example_ids = []
        retrieved_context = f"RAG retrieval failed: {e}"
    finally:
        retriever.close()
    
    # Calculate target duration in seconds
    target_seconds = int(state["scene_length"] * 60)
    
    # Configure based on explanation depth
    depth_configs = {
        "basic": {
            "detail_level": "high-level overview with key points only",
            "transcript_density": "entries every 10-15 seconds",
            "animation_style": "quick transitions, minimal pauses",
        },
        "detailed": {
            "detail_level": "step-by-step explanation with examples",
            "transcript_density": "entries every 5-10 seconds",
            "animation_style": "moderate pacing with clear transitions",
        },
        "comprehensive": {
            "detail_level": "thorough in-depth explanation with multiple examples and edge cases",
            "transcript_density": "entries every 3-5 seconds",
            "animation_style": "slow, educational pacing with extended pauses for understanding",
        },
    }
    depth = state.get("explanation_depth", "detailed")
    depth_config = depth_configs.get(depth, depth_configs["detailed"])
    
    # Generate code with LLM (OpenRouter)
    prompt = f"""Create a Manim animation based on this request:

**Title:** {state["scene_title"]}
**Description:** {state["scene_prompt_description"]}
**Target Duration:** {target_seconds} seconds ({state["scene_length"]} minutes)
**Explanation Level:** {depth} - {depth_config["detail_level"]}

## CRITICAL REQUIREMENTS (MUST FOLLOW STRICTLY):

### 1. SCREEN BOUNDS - NEVER GO OUT OF FRAME
- The Manim frame is 14.2 units wide (-7.1 to +7.1) and 8 units tall (-4 to +4)
- ALWAYS use .scale() to keep objects within bounds (typical scale: 0.5 to 0.8 for text)
- ALWAYS check positions: x should be in [-6, 6], y should be in [-3.5, 3.5] to leave margins
- For multiple elements, use VGroup and .arrange(DOWN/RIGHT, buff=0.3) then scale the group
- NEVER animate objects that start or end outside the visible frame

### 2. POSITIONING - PREVENT ALL OVERLAPS
- Title: ALWAYS use .to_edge(UP, buff=0.5) with .scale(0.7)
- Main content: Use .move_to(ORIGIN) or explicit coordinates
- Side elements: Use .to_edge(LEFT/RIGHT, buff=0.5)
- Labels: Use .next_to(target, direction, buff=0.3)
- ALWAYS FadeOut() previous elements before showing new ones in the same area
- Use .shift() with small values (max 2 units) for adjustments

### 3. SAFE LAYOUT PATTERN (FOLLOW THIS)
```python
# Title at top
title = Text("Title").scale(0.7).to_edge(UP, buff=0.5)
# Main content in center (scaled to fit)
content = VGroup(item1, item2, item3).arrange(DOWN, buff=0.3).scale(0.6).move_to(ORIGIN)
# Labels positioned relative to content
label = Text("Label").scale(0.4).next_to(content, RIGHT, buff=0.5)
```

### 4. DURATION
- Target: approximately {target_seconds} seconds
- Use self.wait(1) to self.wait(3) between sections
- Use run_time=2 for major animations
- Animation style: {depth_config["animation_style"]}

### 5. MANIM API (v0.18+ ONLY - CRITICAL)
Use ONLY these correct APIs. DO NOT use deprecated syntax:
- Arrow: `Arrow(start, end, tip_length=0.2)` - NOT `length=` or `tip_size=`
- Line with tip: `Line(start, end).add_tip(tip_length=0.2)` - NOT `length=`
- Text: `Text("string")` - NOT `TextMobject` or `TexMobject`
- MathTex: `MathTex(r"\\frac{{a}}{{b}}")` - use raw strings with double braces
- Colors: `RED, BLUE, GREEN, YELLOW, WHITE` - built-in constants
- Positioning: `.move_to(point)`, `.next_to(obj, direction, buff=0.3)`
- VGroup: `VGroup(obj1, obj2).arrange(DOWN, buff=0.3)`
- Avoid: `ShowCreation` (use `Create`), `FadeInFromDown` (use `FadeIn` with shift)

### 6. TRANSCRIPT FOR AUDIO
- Detail level: {depth_config["detail_level"]}
- Add transcript {depth_config["transcript_density"]}

Here are some reference examples:

{retrieved_context}

## RESPONSE FORMAT:

```python
# CODE_START
from manim import *
import numpy as np

class YourSceneName(Scene):
    def construct(self):
        # Your animation code here
        pass
# CODE_END
```

```python
# TRANSCRIPT_START
transcript = {{
    0: "Introduction text spoken at the start...",
    5: "Explanation of the first concept...",
    15: "Moving on to the next topic...",
    30: "Further explanation...",
    # Add entries {depth_config["transcript_density"]} covering the full {target_seconds} seconds
}}
# TRANSCRIPT_END
```
"""
    
    messages = [
        {"role": "system", "content": state["system_message"]},
        {"role": "user", "content": prompt},
    ]
    
    response_text = llm_chat(messages, temperature=0.2)
    
    if response_text:
        # Parse code
        code_match = re.search(r'# CODE_START\n(.*?)# CODE_END', response_text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else response_text
        
        # Parse transcript
        transcript = {}
        transcript_match = re.search(r'# TRANSCRIPT_START\n.*?transcript\s*=\s*(\{.*?\})\s*# TRANSCRIPT_END', response_text, re.DOTALL)
        if transcript_match:
            try:
                transcript = eval(transcript_match.group(1))
            except:
                pass
        
        # Extract scene class name
        scene_match = re.search(r'class\s+(\w+)\s*\([^)]*Scene[^)]*\)', code)
        scene_class_name = scene_match.group(1) if scene_match else "GeneratedScene"
    else:
        # Fallback if no LLM available
        code = f'''from manim import *

class GeneratedScene(Scene):
    def construct(self):
        # Generated for: {state["scene_title"]}
        title = Text("{state["scene_title"]}")
        self.play(Write(title))
        self.wait(2)
'''
        transcript = {0: f"Welcome to {state['scene_title']}"}
        scene_class_name = "GeneratedScene"
    
    return {
        "code": code,
        "scene_class_name": scene_class_name,
        "transcript": transcript,
        "retrieved_examples": example_ids,
        "retrieved_context": retrieved_context,
        "messages": [{"role": "assistant", "content": f"Generated code for {state['scene_title']}"}],
    }


# ============================================================================
# NODE 2: Code Executor
# ============================================================================

def code_executor_node(state: VideoGenState) -> dict:
    """
    Execute Manim code and capture rendered video or errors.
    
    Input: code, scene_class_name
    Output: rendered_video_path OR error
    """
    print(f"[EXECUTOR] Starting Manim render for scene: {state.get('scene_class_name')}")
    
    # Write code to temp file
    temp_dir = tempfile.mkdtemp(prefix="manim_")
    code_path = Path(temp_dir) / "scene.py"
    
    with open(code_path, "w") as f:
        f.write(state["code"])
    
    print(f"[EXECUTOR] Code saved to: {code_path}")
    
    # Run manim render
    output_dir = Path(temp_dir) / "media"
    cmd = [
        "manim", "render",
        str(code_path),
        state["scene_class_name"],
        "-ql",  # Low quality for faster testing
        "--media_dir", str(output_dir),
    ]
    
    print(f"[EXECUTOR] Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,  # 2 minute timeout
            cwd=temp_dir,
        )
        
        render_logs = f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}"
        
        if result.returncode != 0:
            # Extract error message
            error_msg = result.stderr or result.stdout or "Unknown render error"
            print(f"[EXECUTOR] Render FAILED: {error_msg[:200]}...")
            return {
                "error": error_msg,
                "error_count": 1,
                "render_logs": render_logs,
                "temp_code_path": str(code_path),
            }
        
        # Find rendered video
        video_files = list(output_dir.rglob("*.mp4"))
        print(f"[EXECUTOR] Found {len(video_files)} video files")
        
        if not video_files:
            print("[EXECUTOR] No video file produced!")
            return {
                "error": "No video file produced",
                "error_count": 1,
                "render_logs": render_logs,
                "temp_code_path": str(code_path),
            }
        
        print(f"[EXECUTOR] Render SUCCESS: {video_files[0]}")
        return {
            "rendered_video_path": str(video_files[0]),
            "error": None,
            "error_count": 0,  # Reset on success (add 0 keeps current value but marks success)
            "render_logs": render_logs,
            "temp_code_path": str(code_path),
        }
    
    except subprocess.TimeoutExpired:
        return {
            "error": "Render timeout exceeded (120s)",
            "error_count": 1,
            "temp_code_path": str(code_path),
        }
    except Exception as e:
        return {
            "error": str(e),
            "error_count": 1,
            "temp_code_path": str(code_path),
        }


# ============================================================================
# NODE 3: Code Re-corrector (OpenRouter)
# ============================================================================

def recorrector_node(state: VideoGenState) -> dict:
    """
    Fix code based on error feedback using OpenRouter LLM.
    
    Input: code, error, retrieved_context
    Output: corrected code
    """
    messages = [
        {"role": "system", "content": "You are an expert at debugging Manim code. Fix the error in the code below. Return only the corrected Python code, no explanations."},
        {"role": "user", "content": f"""
The following Manim code produced an error:

```python
{state["code"]}
```

Error message:
```
{state["error"][:2000]}
```

Please fix the code and return the complete corrected version.
Only return the Python code, nothing else.
"""}
    ]
    
    fixed_code = llm_chat(messages, temperature=0.1)
    
    if not fixed_code:
        # Fallback: add error as comment and return
        fixed_code = f"# Error was: {state['error'][:100]}\n{state['code']}"
    else:
        # Clean up markdown code blocks if present
        if "```python" in fixed_code:
            match = re.search(r'```python\n(.*?)```', fixed_code, re.DOTALL)
            if match:
                fixed_code = match.group(1)
        elif "```" in fixed_code:
            match = re.search(r'```\n?(.*?)```', fixed_code, re.DOTALL)
            if match:
                fixed_code = match.group(1)
    
    return {
        "code": fixed_code.strip(),
        "error": None,
        "messages": [{"role": "assistant", "content": "Fixed code based on error feedback"}],
    }


# ============================================================================
# NODE 4: Transcript Processor (with Kokoro TTS)
# ============================================================================

def transcript_processor_node(state: VideoGenState) -> dict:
    """
    Process transcript into timestamped sections with TTS audio.
    
    Uses Kokoro 82M local model for high-quality speech synthesis.
    
    Input: transcript dict
    Output: transcript_sections list, audio_segments list
    """
    sections = []
    audio_segments = []
    transcript = state.get("transcript", {})
    
    # Sort by timestamp
    sorted_items = sorted(transcript.items(), key=lambda x: float(x[0]))
    
    print(f"[TTS] Processing {len(sorted_items)} transcript segments...")
    
    # Try to use Kokoro TTS
    tts = None
    try:
        from ..tts import KokoroTTS, KOKORO_AVAILABLE
        if KOKORO_AVAILABLE:
            tts = KokoroTTS(voice="af_bella")  # American female voice
            print("[TTS] Kokoro TTS initialized successfully")
        else:
            print("[TTS] WARNING: Kokoro not available (kokoro-onnx not installed)")
    except ImportError as e:
        print(f"[TTS] WARNING: Could not import TTS module: {e}")
    
    for i, (timestamp, text) in enumerate(sorted_items):
        audio_path = None
        
        # Generate audio with Kokoro TTS if available
        if tts and text.strip():
            try:
                print(f"[TTS] Generating audio for segment {i+1}/{len(sorted_items)}: {text[:50]}...")
                result = tts.synthesize(text)
                if result.success:
                    audio_path = result.audio_path
                    audio_segments.append(audio_path)
                    print(f"[TTS] ✓ Segment {i+1} saved to: {audio_path}")
                else:
                    print(f"[TTS] ✗ Segment {i+1} failed: {result.error}")
            except Exception as e:
                print(f"[TTS] ✗ Segment {i+1} exception: {e}")
        
        section = TranscriptSection(
            timestamp=float(timestamp),
            text=str(text),
            audio_path=audio_path,
        )
        sections.append(section)
    
    print(f"[TTS] Complete: {len(audio_segments)} audio files generated")
    
    return {
        "transcript_sections": sections,
        "audio_segments": audio_segments,
    }


# ============================================================================
# NODE 5: Render Checker
# ============================================================================

def render_checker_node(state: VideoGenState) -> dict:
    """
    Validate the rendered video.
    
    Input: rendered_video_path
    Output: video_valid, validation_errors, checked_video_path
    """
    video_path = state.get("rendered_video_path")
    errors = []
    
    print(f"[RENDER CHECK] Input video path: {video_path}")
    
    if not video_path or not Path(video_path).exists():
        print(f"[RENDER CHECK] Video file does not exist!")
        return {
            "video_valid": False,
            "validation_errors": ["Video file does not exist"],
            "checked_video_path": None,
        }
    
    # Check file size
    file_size = Path(video_path).stat().st_size
    print(f"[RENDER CHECK] Video file size: {file_size} bytes")
    if file_size < 1000:  # Less than 1KB is suspicious
        errors.append(f"Video file too small: {file_size} bytes")
    
    # Check with ffprobe if available
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-show_format", "-show_streams", video_path],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode != 0:
            errors.append(f"ffprobe error: {result.stderr}")
    except FileNotFoundError:
        pass  # ffprobe not available, skip check
    except Exception as e:
        errors.append(f"Video validation error: {e}")
    
    checked_path = video_path if len(errors) == 0 else None
    print(f"[RENDER CHECK] Validation {'passed' if checked_path else 'failed'}: {errors}")
    
    return {
        "video_valid": len(errors) == 0,
        "validation_errors": errors,
        "checked_video_path": checked_path,
    }


# ============================================================================
# NODE 6: Synchronizer
# ============================================================================

def synchronizer_node(state: VideoGenState) -> dict:
    """
    Synchronize video with transcript timestamps.
    
    This node receives state from both the video rendering branch and
    the transcript processing branch. It passes through all relevant data.
    
    Input: checked_video_path, transcript_sections, audio_segments
    Output: synced_video_path (plus passthrough of audio data)
    """
    video_path = state.get("checked_video_path")
    audio_segments = state.get("audio_segments", [])
    transcript_sections = state.get("transcript_sections", [])
    
    print(f"[SYNC] Video path: {video_path}")
    print(f"[SYNC] Audio segments: {len(audio_segments)}")
    print(f"[SYNC] Transcript sections: {len(transcript_sections)}")
    
    # Pass through all data - don't drop audio_segments!
    return {
        "synced_video_path": video_path,
        "audio_segments": audio_segments,  # Preserve audio from transcript_processor
        "transcript_sections": transcript_sections,
    }


# ============================================================================
# NODE 7: Audio Video Merger (with ffmpeg)
# ============================================================================

def audio_video_merger_node(state: VideoGenState) -> dict:
    """
    Merge audio segments with video using ffmpeg.
    
    Concatenates all audio segments and mixes with video.
    Handles audio/video length mismatches by padding audio with silence.
    
    Input: synced_video_path, audio_segments, transcript_sections
    Output: final_output_path
    """
    video_path = state.get("synced_video_path")
    audio_segments = state.get("audio_segments", [])
    transcript_sections = state.get("transcript_sections", [])
    
    print(f"[AUDIO MERGE] Video path: {video_path}")
    print(f"[AUDIO MERGE] Audio segments: {len(audio_segments)}")
    
    if not video_path:
        print("[AUDIO MERGE] No video path provided")
        return {"final_output_path": None}
    
    # Filter to only existing audio files
    valid_audio_segments = [p for p in audio_segments if p and Path(p).exists()]
    print(f"[AUDIO MERGE] Valid audio files: {len(valid_audio_segments)}")
    
    # If no audio segments, just use video as final
    if not valid_audio_segments:
        print("[AUDIO MERGE] No valid audio segments, returning video only")
        return {"final_output_path": video_path}
    
    try:
        # Create temp directory for processing
        temp_dir = tempfile.mkdtemp(prefix="manim_merge_")
        
        # Create concat file for ffmpeg
        concat_file = Path(temp_dir) / "concat.txt"
        
        # Build audio concat file
        with open(concat_file, "w") as f:
            for audio_path in valid_audio_segments:
                f.write(f"file '{audio_path}'\n")
        
        print(f"[AUDIO MERGE] Created concat file: {concat_file}")
        
        # Concatenate all audio segments
        merged_audio = Path(temp_dir) / "merged_audio.wav"
        concat_cmd = [
            "ffmpeg", "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", str(concat_file),
            "-c", "copy",
            str(merged_audio),
        ]
        
        print(f"[AUDIO MERGE] Concatenating audio...")
        result = subprocess.run(
            concat_cmd,
            capture_output=True,
            text=True,
            timeout=60,
        )
        
        if result.returncode != 0 or not merged_audio.exists():
            print(f"[AUDIO MERGE] Audio concat failed: {result.stderr}")
            return {"final_output_path": video_path}
        
        print(f"[AUDIO MERGE] Audio concatenated: {merged_audio}")
        
        # Merge audio with video using filter_complex to pad audio
        # apad pads audio with silence if shorter than video
        final_output = Path(temp_dir) / "final_with_audio.mp4"
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", str(merged_audio),
            "-filter_complex", "[1:a]apad[a]",  # Pad audio with silence if needed
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "[a]",
            "-shortest",  # Use shortest (video length) since audio is padded
            str(final_output),
        ]
        
        print(f"[AUDIO MERGE] Merging audio with video...")
        result = subprocess.run(
            merge_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode != 0:
            print(f"[AUDIO MERGE] Merge command failed: {result.stderr}")
            # Try simpler merge without filter_complex as fallback
            print("[AUDIO MERGE] Trying simple merge fallback...")
            simple_merge_cmd = [
                "ffmpeg", "-y",
                "-i", video_path,
                "-i", str(merged_audio),
                "-c:v", "copy",
                "-c:a", "aac",
                "-map", "0:v:0",
                "-map", "1:a:0",
                str(final_output),
            ]
            result = subprocess.run(
                simple_merge_cmd,
                capture_output=True,
                text=True,
                timeout=120,
            )
        
        if not final_output.exists():
            print(f"[AUDIO MERGE] Final output not created")
            return {"final_output_path": video_path}
        
        print(f"[AUDIO MERGE] Success! Final output: {final_output}")
        return {"final_output_path": str(final_output)}
    
    except Exception as e:
        print(f"[AUDIO MERGE] Error: {e}")
        return {"final_output_path": video_path}


# ============================================================================
# Conditional Edge Functions
# ============================================================================

def should_retry_or_continue(state: VideoGenState) -> Literal["recorrector", "render_checker"]:
    """Decide whether to retry code correction or proceed to render checking."""
    if state.get("error") and state.get("error_count", 0) < state.get("max_retries", 3):
        return "recorrector"
    return "render_checker"


def should_rerender(state: VideoGenState) -> Literal["code_executor", "synchronizer"]:
    """Decide whether to re-render or proceed to sync."""
    if state.get("video_valid"):
        return "synchronizer"
    # If validation failed but we have error count space, try re-rendering
    if state.get("error_count", 0) < state.get("max_retries", 3):
        return "code_executor"
    return "synchronizer"  # Give up and proceed anyway
