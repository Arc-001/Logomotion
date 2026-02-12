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
    retriever = ManimRetriever()
    
    try:
        # Hybrid search for relevant examples
        results = retriever.hybrid_search(
            query=state["scene_prompt_description"],
            limit=3,
        )
        
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
    
    target_seconds = int(state["scene_length"] * 60)
    print(f"[CODE GEN] Target duration: {target_seconds} seconds")
    
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
- Code: DO NOT use `Code()` class - instead use `Text()` with monospace font or `Paragraph()`
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
        code_match = re.search(r'# CODE_START\n(.*?)# CODE_END', response_text, re.DOTALL)
        code = code_match.group(1).strip() if code_match else response_text
        
        transcript = {}
        transcript_match = re.search(r'# TRANSCRIPT_START\n.*?transcript\s*=\s*(\{.*?\})\s*# TRANSCRIPT_END', response_text, re.DOTALL)
        if transcript_match:
            try:
                transcript = eval(transcript_match.group(1))
            except:
                pass
        
        scene_match = re.search(r'class\s+(\w+)\s*\([^)]*Scene[^)]*\)', code)
        scene_class_name = scene_match.group(1) if scene_match else "GeneratedScene"
    else:
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
    
    temp_dir = tempfile.mkdtemp(prefix="manim_")
    code_path = Path(temp_dir) / "scene.py"
    
    with open(code_path, "w") as f:
        f.write(state["code"])
    
    print(f"[EXECUTOR] Code saved to: {code_path}")
    
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
            error_msg = result.stderr or result.stdout or "Unknown render error"
            print(f"[EXECUTOR] Render FAILED: {error_msg[:200]}...")
            print(f"[EXECUTOR] Returning error state for recorrection")
            return {
                "error": error_msg,
                "error_count": 1,
                "render_logs": render_logs,
                "temp_code_path": str(code_path),
            }
        
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
    error_count = state.get("error_count", 0)
    print(f"[RECORRECTOR] Attempt {error_count + 1} to fix error")
    print(f"[RECORRECTOR] Error: {state.get('error', 'unknown')[:200]}...")
    
    messages = [
        {"role": "system", "content": """You are an expert at debugging Manim code for Manim v0.18+.

CRITICAL API RULES:
- Arrow: use tip_length=0.2, NOT length=
- Line.add_tip(): use tip_length=0.2, NOT length=
- DO NOT use Code() class - use Text() with monospace styling instead
- Use Text() not TextMobject or TexMobject
- Use Create() not ShowCreation()
- Use MathTex(r"...") with raw strings

Fix the error and return ONLY the corrected Python code."""},
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
        fixed_code = f"# Error was: {state['error'][:100]}\n{state['code']}"
    else:
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
    
    sorted_items = sorted(transcript.items(), key=lambda x: float(x[0]))
    
    print(f"[TTS] Processing {len(sorted_items)} transcript segments...")
    
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
    Validate the rendered video and measure its actual duration.
    
    Compares actual duration against target_duration (scene_length * 60).
    Flags duration mismatch if outside ±20% tolerance.
    
    Input: rendered_video_path, scene_length
    Output: video_valid, validation_errors, checked_video_path, actual_duration
    """
    video_path = state.get("rendered_video_path")
    target_duration = state.get("target_duration", state.get("scene_length", 1.0) * 60)
    errors = []
    actual_duration = None
    
    print(f"[RENDER CHECK] Input video path: {video_path}")
    print(f"[RENDER CHECK] Target duration: {target_duration}s")
    
    if not video_path or not Path(video_path).exists():
        print(f"[RENDER CHECK] Video file does not exist!")
        return {
            "video_valid": False,
            "validation_errors": ["Video file does not exist"],
            "checked_video_path": None,
            "actual_duration": None,
        }
    
    file_size = Path(video_path).stat().st_size
    print(f"[RENDER CHECK] Video file size: {file_size} bytes")
    if file_size < 1000:
        errors.append(f"Video file too small: {file_size} bytes")
    
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                video_path,
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if result.returncode == 0 and result.stdout.strip():
            actual_duration = float(result.stdout.strip())
            print(f"[RENDER CHECK] Actual duration: {actual_duration:.1f}s (target: {target_duration:.1f}s)")
            
            duration_ratio = actual_duration / target_duration if target_duration > 0 else 1.0
            tolerance = 0.20  # ±20%
            if abs(duration_ratio - 1.0) > tolerance:
                deviation_pct = (duration_ratio - 1.0) * 100
                direction = "longer" if deviation_pct > 0 else "shorter"
                errors.append(
                    f"Duration mismatch: video is {abs(deviation_pct):.0f}% {direction} "
                    f"than target ({actual_duration:.1f}s vs {target_duration:.1f}s)"
                )
                print(f"[RENDER CHECK] Duration mismatch: {deviation_pct:+.0f}%")
            else:
                print(f"[RENDER CHECK] Duration within tolerance ({duration_ratio:.2f}x)")
        else:
            print(f"[RENDER CHECK] Could not determine video duration")
            if result.stderr:
                errors.append(f"ffprobe error: {result.stderr}")
    except FileNotFoundError:
        print("[RENDER CHECK] ffprobe not available, skipping duration check")
    except Exception as e:
        errors.append(f"Video validation error: {e}")
    
    duration_errors = [e for e in errors if "Duration mismatch" in e]
    non_duration_errors = [e for e in errors if "Duration mismatch" not in e]
    
    video_valid = len(non_duration_errors) == 0
    checked_path = video_path if video_valid else None
    
    print(f"[RENDER CHECK] Validation {'passed' if checked_path else 'failed'}: {errors}")
    
    return {
        "video_valid": video_valid,
        "validation_errors": errors,
        "checked_video_path": checked_path,
        "actual_duration": actual_duration,
    }


# ============================================================================
# NODE 5b: Video Duration Fixer
# ============================================================================

def video_duration_fixer_node(state: VideoGenState) -> dict:
    """
    Adjust video playback speed to match target duration using ffmpeg.
    
    Uses PTS (Presentation Time Stamp) manipulation to speed up or slow
    down the video. Caps adjustment at 2x to avoid extreme distortion.
    
    Input: checked_video_path, actual_duration, target_duration
    Output: checked_video_path (adjusted), duration_adjusted, duration_factor
    """
    video_path = state.get("checked_video_path")
    actual_duration = state.get("actual_duration")
    target_duration = state.get("target_duration", state.get("scene_length", 1.0) * 60)
    
    print(f"[DURATION FIX] Video: {video_path}")
    print(f"[DURATION FIX] Actual: {actual_duration}s -> Target: {target_duration}s")
    
    if not video_path or not actual_duration or actual_duration <= 0:
        print("[DURATION FIX] Cannot fix — missing video or duration info")
        return {
            "duration_adjusted": False,
            "duration_factor": None,
        }
    
    if target_duration <= 0:
        print("[DURATION FIX] Invalid target duration, skipping")
        return {
            "duration_adjusted": False,
            "duration_factor": None,
        }
    
    factor = actual_duration / target_duration
    tolerance = 0.20
    
    if abs(factor - 1.0) <= tolerance:
        print(f"[DURATION FIX] Duration within tolerance ({factor:.2f}x), no adjustment needed")
        return {
            "duration_adjusted": False,
            "duration_factor": None,
        }
    
    # setpts multiplier: >1 slows down (stretches), <1 speeds up (compresses)
    # If video is too long (factor > 1), we need pts < 1 to speed up
    # If video is too short (factor < 1), we need pts > 1 to slow down
    pts_factor = target_duration / actual_duration  # inverse of factor
    
    max_pts = 2.0  # Cap at 2x slowdown or 2x speedup
    min_pts = 1.0 / max_pts
    clamped_pts = max(min_pts, min(max_pts, pts_factor))
    
    if clamped_pts != pts_factor:
        print(f"[DURATION FIX] PTS factor clamped from {pts_factor:.4f} to {clamped_pts:.4f} (range {min_pts}-{max_pts})")
        pts_factor = clamped_pts
    
    if factor > 1.0:
        print(f"[DURATION FIX] Video is too long ({actual_duration:.1f}s > {target_duration:.1f}s) — speeding up (setpts={pts_factor:.4f}*PTS)")
    else:
        print(f"[DURATION FIX] Video is too short ({actual_duration:.1f}s < {target_duration:.1f}s) — slowing down (setpts={pts_factor:.4f}*PTS)")
    
    try:
        temp_dir = tempfile.mkdtemp(prefix="manim_durfix_")
        adjusted_path = Path(temp_dir) / "duration_adjusted.mp4"
        
        cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-filter:v", f"setpts={pts_factor}*PTS",
            "-an",  # Strip audio since we merge later
            str(adjusted_path),
        ]
        
        print(f"[DURATION FIX] Running: {' '.join(cmd)}")
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode != 0:
            print(f"[DURATION FIX] ffmpeg failed: {result.stderr[:300]}")
            return {
                "duration_adjusted": False,
                "duration_factor": None,
            }
        
        if not adjusted_path.exists():
            print("[DURATION FIX] Output file not created")
            return {
                "duration_adjusted": False,
                "duration_factor": None,
            }
        
        verify_result = subprocess.run(
            [
                "ffprobe", "-v", "error",
                "-show_entries", "format=duration",
                "-of", "csv=p=0",
                str(adjusted_path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        
        new_duration = None
        if verify_result.returncode == 0 and verify_result.stdout.strip():
            new_duration = float(verify_result.stdout.strip())
            print(f"[DURATION FIX] Adjusted duration: {new_duration:.1f}s (target: {target_duration:.1f}s)")
        
        print(f"[DURATION FIX] Success! Adjusted video: {adjusted_path}")
        return {
            "checked_video_path": str(adjusted_path),
            "duration_adjusted": True,
            "duration_factor": factor,
            "actual_duration": new_duration or target_duration,
        }
    
    except subprocess.TimeoutExpired:
        print("[DURATION FIX] ffmpeg timeout")
        return {"duration_adjusted": False, "duration_factor": None}
    except Exception as e:
        print(f"[DURATION FIX] Error: {e}")
        return {"duration_adjusted": False, "duration_factor": None}


# ============================================================================
# NODE 6: Synchronizer
# ============================================================================

def synchronizer_node(state: VideoGenState) -> dict:
    """
    Synchronize video with transcript timestamps.
    
    If the video duration was adjusted, scales all transcript timestamps
    proportionally so audio cues stay aligned with the visual content.
    
    Input: checked_video_path, transcript_sections, audio_segments, duration_factor
    Output: synced_video_path, transcript_sections (adjusted), audio_segments
    """
    video_path = state.get("checked_video_path")
    audio_segments = state.get("audio_segments", [])
    transcript_sections = state.get("transcript_sections", [])
    duration_adjusted = state.get("duration_adjusted", False)
    duration_factor = state.get("duration_factor")
    
    print(f"[SYNC] Video path: {video_path}")
    print(f"[SYNC] Audio segments: {len(audio_segments)}")
    print(f"[SYNC] Transcript sections: {len(transcript_sections)}")
    print(f"[SYNC] Duration adjusted: {duration_adjusted} (factor: {duration_factor})")
    
    synced_sections = list(transcript_sections)
    
    if duration_adjusted and duration_factor and duration_factor != 1.0:
        print(f"[SYNC] Scaling transcript timestamps by factor {duration_factor:.4f}")
        synced_sections = []
        for section in transcript_sections:
            adjusted_section = TranscriptSection(
                timestamp=section["timestamp"] / duration_factor,
                text=section["text"],
                audio_path=section.get("audio_path"),
            )
            synced_sections.append(adjusted_section)
        print(f"[SYNC] Adjusted {len(synced_sections)} transcript timestamps")
    
    return {
        "synced_video_path": video_path,
        "audio_segments": audio_segments,
        "transcript_sections": synced_sections,
    }


# ============================================================================
# NODE 7: Audio Video Merger (with ffmpeg)
# ============================================================================

def audio_video_merger_node(state: VideoGenState) -> dict:
    """
    Merge audio segments with video using ffmpeg.
    
    Places each audio segment at its transcript timestamp position,
    creating a properly timed narration track that aligns with the video.
    Handles audio/video length mismatches by padding audio with silence.
    
    Input: synced_video_path, audio_segments, transcript_sections
    Output: final_output_path
    """
    video_path = state.get("synced_video_path")
    audio_segments = state.get("audio_segments", [])
    transcript_sections = state.get("transcript_sections", [])
    
    print(f"[AUDIO MERGE] Video path: {video_path}")
    print(f"[AUDIO MERGE] Audio segments: {len(audio_segments)}")
    print(f"[AUDIO MERGE] Transcript sections: {len(transcript_sections)}")
    
    if not video_path:
        print("[AUDIO MERGE] No video path provided")
        return {"final_output_path": None}
    
    valid_audio_segments = [p for p in audio_segments if p and Path(p).exists()]
    print(f"[AUDIO MERGE] Valid audio files: {len(valid_audio_segments)}")
    
    if not valid_audio_segments:
        print("[AUDIO MERGE] No valid audio segments, returning video only")
        return {"final_output_path": video_path}
    
    try:
        temp_dir = tempfile.mkdtemp(prefix="manim_merge_")
        
        # Get video duration for padding calculations
        video_duration = None
        try:
            dur_result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    video_path,
                ],
                capture_output=True, text=True, timeout=30,
            )
            if dur_result.returncode == 0 and dur_result.stdout.strip():
                video_duration = float(dur_result.stdout.strip())
                print(f"[AUDIO MERGE] Video duration: {video_duration:.1f}s")
        except Exception as e:
            print(f"[AUDIO MERGE] Could not get video duration: {e}")
        
        # Build timestamp-aligned audio using transcript sections
        # Match audio segments to transcript sections that have audio_path
        sections_with_audio = []
        for section in transcript_sections:
            audio_path = section.get("audio_path")
            if audio_path and Path(audio_path).exists():
                sections_with_audio.append(section)
        
        print(f"[AUDIO MERGE] Sections with valid audio: {len(sections_with_audio)}")
        
        if not sections_with_audio:
            # Fallback: no timestamp info, use valid_audio_segments directly
            sections_with_audio = [
                {"timestamp": 0.0, "audio_path": p}
                for p in valid_audio_segments
            ]
        
        merged_audio = Path(temp_dir) / "merged_audio.wav"
        
        if len(sections_with_audio) == 1:
            # Single segment: just use adelay for its timestamp
            section = sections_with_audio[0]
            delay_ms = int(section["timestamp"] * 1000)
            
            cmd = [
                "ffmpeg", "-y",
                "-i", section["audio_path"],
                "-af", f"adelay={delay_ms}|{delay_ms}",
                str(merged_audio),
            ]
            print(f"[AUDIO MERGE] Single segment at {section['timestamp']:.1f}s")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0:
                print(f"[AUDIO MERGE] Single adelay failed: {result.stderr[:200]}")
                # Fallback: just copy
                import shutil
                shutil.copy2(section["audio_path"], str(merged_audio))
        else:
            # Multiple segments: create silence-padded individual files, then concat
            # Strategy: for each segment, create a file that has silence from
            # the end of the previous segment to this segment's timestamp,
            # then the actual audio. Concatenate all these in order.
            concat_parts = []
            
            for i, section in enumerate(sections_with_audio):
                timestamp = section["timestamp"]
                audio_path = section["audio_path"]
                
                # Get this audio segment's duration
                seg_duration = None
                try:
                    seg_result = subprocess.run(
                        [
                            "ffprobe", "-v", "error",
                            "-show_entries", "format=duration",
                            "-of", "csv=p=0",
                            audio_path,
                        ],
                        capture_output=True, text=True, timeout=10,
                    )
                    if seg_result.returncode == 0 and seg_result.stdout.strip():
                        seg_duration = float(seg_result.stdout.strip())
                except Exception:
                    pass
                
                # Calculate how much silence to insert before this segment
                if i == 0:
                    prev_end = 0.0
                else:
                    prev_section = sections_with_audio[i - 1]
                    prev_timestamp = prev_section["timestamp"]
                    # Estimate previous segment's end time
                    prev_dur_result = subprocess.run(
                        [
                            "ffprobe", "-v", "error",
                            "-show_entries", "format=duration",
                            "-of", "csv=p=0",
                            prev_section["audio_path"],
                        ],
                        capture_output=True, text=True, timeout=10,
                    )
                    prev_audio_dur = 0.0
                    if prev_dur_result.returncode == 0 and prev_dur_result.stdout.strip():
                        prev_audio_dur = float(prev_dur_result.stdout.strip())
                    prev_end = prev_timestamp + prev_audio_dur
                
                silence_duration = max(0.0, timestamp - prev_end)
                
                if silence_duration > 0.05:  # More than 50ms of silence
                    # Create a silence WAV file
                    silence_path = Path(temp_dir) / f"silence_{i:03d}.wav"
                    silence_cmd = [
                        "ffmpeg", "-y",
                        "-f", "lavfi", "-i",
                        f"anullsrc=r=24000:cl=mono:d={silence_duration}",
                        str(silence_path),
                    ]
                    result = subprocess.run(
                        silence_cmd, capture_output=True, text=True, timeout=30,
                    )
                    if result.returncode == 0 and silence_path.exists():
                        concat_parts.append(str(silence_path))
                        print(f"[AUDIO MERGE]   Segment {i}: {silence_duration:.1f}s silence, then audio at {timestamp:.1f}s")
                    else:
                        print(f"[AUDIO MERGE]   Segment {i}: silence generation failed, skipping gap")
                else:
                    print(f"[AUDIO MERGE]   Segment {i}: no gap needed at {timestamp:.1f}s")
                
                concat_parts.append(audio_path)
            
            if not concat_parts:
                print("[AUDIO MERGE] No concat parts produced")
                return {"final_output_path": video_path}
            
            # First, normalize all audio files to the same format (PCM s16le, 24kHz, mono)
            normalized_parts = []
            for j, part in enumerate(concat_parts):
                norm_path = Path(temp_dir) / f"norm_{j:03d}.wav"
                norm_cmd = [
                    "ffmpeg", "-y",
                    "-i", part,
                    "-ar", "24000", "-ac", "1", "-c:a", "pcm_s16le",
                    str(norm_path),
                ]
                result = subprocess.run(
                    norm_cmd, capture_output=True, text=True, timeout=30,
                )
                if result.returncode == 0 and norm_path.exists():
                    normalized_parts.append(str(norm_path))
                else:
                    print(f"[AUDIO MERGE] Failed to normalize part {j}: {result.stderr[:100]}")
            
            # Concatenate all normalized parts
            concat_file = Path(temp_dir) / "concat.txt"
            with open(concat_file, "w") as f:
                for part_path in normalized_parts:
                    f.write(f"file '{part_path}'\n")
            
            print(f"[AUDIO MERGE] Concatenating {len(normalized_parts)} parts (audio + silence gaps)...")
            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat",
                "-safe", "0",
                "-i", str(concat_file),
                "-c", "copy",
                str(merged_audio),
            ]
            result = subprocess.run(
                concat_cmd, capture_output=True, text=True, timeout=60,
            )
            
            if result.returncode != 0 or not merged_audio.exists():
                print(f"[AUDIO MERGE] Concat failed: {result.stderr[:200]}")
                return {"final_output_path": video_path}
        
        # Log merged audio duration
        try:
            ma_result = subprocess.run(
                [
                    "ffprobe", "-v", "error",
                    "-show_entries", "format=duration",
                    "-of", "csv=p=0",
                    str(merged_audio),
                ],
                capture_output=True, text=True, timeout=10,
            )
            if ma_result.returncode == 0 and ma_result.stdout.strip():
                audio_dur = float(ma_result.stdout.strip())
                print(f"[AUDIO MERGE] Merged audio duration: {audio_dur:.1f}s")
        except Exception:
            pass
        
        print(f"[AUDIO MERGE] Timestamp-aligned audio created: {merged_audio}")
        
        # Merge audio with video
        # apad pads audio with silence if shorter than video
        final_output = Path(temp_dir) / "final_with_audio.mp4"
        merge_cmd = [
            "ffmpeg", "-y",
            "-i", video_path,
            "-i", str(merged_audio),
            "-filter_complex", "[1:a]apad[a]",
            "-c:v", "copy",
            "-c:a", "aac",
            "-b:a", "128k",
            "-map", "0:v:0",
            "-map", "[a]",
            "-shortest",
            str(final_output),
        ]
        
        print(f"[AUDIO MERGE] Merging timestamp-aligned audio with video...")
        result = subprocess.run(
            merge_cmd,
            capture_output=True,
            text=True,
            timeout=120,
        )
        
        if result.returncode != 0:
            print(f"[AUDIO MERGE] Merge command failed: {result.stderr[:200]}")
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
        
        # Copy to the project output directory for easy access
        project_output = Path("output")
        project_output.mkdir(parents=True, exist_ok=True)
        output_copy = project_output / "output.mp4"
        import shutil
        shutil.copy2(str(final_output), str(output_copy))
        print(f"[AUDIO MERGE] Copied to: {output_copy.resolve()}")
        
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
    error = state.get("error")
    error_count = state.get("error_count", 0)
    max_retries = state.get("max_retries", 3)
    
    if error and error_count < max_retries:
        print(f"[RETRY] Error detected, attempt {error_count + 1}/{max_retries} - sending to recorrector")
        return "recorrector"
    
    if error:
        print(f"[RETRY] Max retries ({max_retries}) reached, proceeding to render_checker")
    else:
        print("[RETRY] No error, proceeding to render_checker")
    return "render_checker"


def should_fix_duration(state: VideoGenState) -> Literal["video_duration_fixer", "synchronizer"]:
    """Decide whether video duration needs fixing before sync."""
    actual_duration = state.get("actual_duration")
    target_duration = state.get("target_duration", state.get("scene_length", 1.0) * 60)
    
    if not actual_duration or not target_duration or target_duration <= 0:
        print("[DURATION CHECK] Missing duration info, skipping fix")
        return "synchronizer"
    
    factor = actual_duration / target_duration
    tolerance = 0.20
    
    if abs(factor - 1.0) > tolerance:
        print(f"[DURATION CHECK] Duration off by {(factor - 1.0) * 100:+.0f}% — routing to fixer")
        return "video_duration_fixer"
    
    print(f"[DURATION CHECK] Duration OK ({factor:.2f}x) — skipping fix")
    return "synchronizer"


def should_rerender(state: VideoGenState) -> Literal["code_executor", "synchronizer"]:
    """Decide whether to re-render or proceed to sync."""
    if state.get("video_valid"):
        return "synchronizer"
    # If validation failed but we have error count space, try re-rendering
    if state.get("error_count", 0) < state.get("max_retries", 3):
        return "code_executor"
    return "synchronizer"  # Give up and proceed anyway
