"""
LangGraph nodes for the Manim video generation workflow.

Implements the following nodes from the architecture diagram:
1. video_code_gen_node - Main code generation with RAG
2. code_executor_node - Runs Manim and captures output/errors
3. recorrector_node - Fixes code based on error feedback
4. transcript_processor_node - Splits transcript into sections
5. render_checker_node - Validates rendered video
6. video_duration_fixer_node - Adjusts video duration with ffmpeg
7. synchronizer_node - Aligns video with transcript
8. audio_video_merger_node - Final merge
"""

import asyncio
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal, Optional

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OpenAI = None
    OPENAI_AVAILABLE = False

from .state import VideoGenState, TranscriptSection
from ..config import get_settings
from ..graph_rag.retriever import ManimRetriever
from ..manim_runner.executor import ManimExecutor
from ..manim_runner.validator import get_video_duration


# ============================================================================
# OpenRouter LLM Client
# ============================================================================

def get_llm_client() -> Optional["OpenAI"]:
    """Get OpenRouter client configured from environment."""
    if not OPENAI_AVAILABLE:
        return None

    settings = get_settings()
    if not settings.openrouter_api_key:
        return None

    return OpenAI(base_url=settings.openrouter_base_url, api_key=settings.openrouter_api_key)


def llm_chat(messages: list[dict], temperature: float = 0.2) -> Optional[str]:
    """Call OpenRouter LLM with messages."""
    client = get_llm_client()
    if not client:
        return None

    settings = get_settings()

    try:
        completion = client.chat.completions.create(
            extra_headers={
                "HTTP-Referer": "https://manim-agent.local",
                "X-Title": "Manim Graph RAG Agent",
            },
            model=settings.openrouter_model,
            messages=messages,
            temperature=temperature,
        )
        return completion.choices[0].message.content
    except Exception as e:
        print(f"LLM call failed: {e}")
        return None


# ============================================================================
# Shared helpers
# ============================================================================

def _extract_code_block(text: str) -> str:
    """Extract Python code from an LLM response containing markdown fences.

    Tries CODE_START/CODE_END markers first, then ```python, then bare ```.
    Falls back to the raw text if no fences are found.
    """
    # 1. Explicit markers
    marker_match = re.search(r'# CODE_START\n(.*?)# CODE_END', text, re.DOTALL)
    if marker_match:
        return marker_match.group(1).strip()

    # 2. ```python block
    py_match = re.search(r'```python\n(.*?)```', text, re.DOTALL)
    if py_match:
        return py_match.group(1).strip()

    # 3. Generic ``` block
    generic_match = re.search(r'```\n?(.*?)```', text, re.DOTALL)
    if generic_match:
        return generic_match.group(1).strip()

    return text.strip()


# ============================================================================
# DEPTH CONFIG — used by code gen prompt
# ============================================================================

DEPTH_CONFIGS = {
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


# ============================================================================
# NODE 0: Web Research (optional — runs only when web_search_enabled=True)
# ============================================================================

def web_research_node(state: VideoGenState) -> dict:
    """
    Gather the latest real-world information about the animation topic.

    Uses an LLM to craft search queries, then runs DuckDuckGo + Wikipedia
    searches in parallel, scrapes the top results, and synthesises a compact
    research brief that is injected into the code-gen prompt.

    Input:  scene_prompt_description, web_search_enabled
    Output: web_context (str), web_sources (list[dict])
    """
    topic = state.get("scene_prompt_description", "")
    print(f"[WebResearch] Node triggered for topic: {topic!r}")

    try:
        from ..search.web import research_topic  # import here to keep cold-start fast

        # research_topic is async — run it in the current event loop (or a new one)
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                # We are inside an async context (e.g. FastAPI background task)
                # Use asyncio.run_coroutine_threadsafe or nest_asyncio approach.
                # The simplest portable solution: create a new loop in a thread.
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor(max_workers=1) as ex:
                    future = ex.submit(asyncio.run, research_topic(topic, llm_chat))
                    brief = future.result(timeout=120)
            else:
                brief = loop.run_until_complete(research_topic(topic, llm_chat))
        except RuntimeError:
            brief = asyncio.run(research_topic(topic, llm_chat))

        web_context = brief.to_prompt_block()
        web_sources = [
            {
                "url": s.url,
                "title": s.title,
                "snippet": s.snippet,
                "source_type": s.source_type,
            }
            for s in brief.sources
        ]
        print(f"[WebResearch] Brief ready: {len(brief.key_facts)} facts, "
              f"{len(brief.sources)} sources")

    except Exception as exc:
        print(f"[WebResearch] Node failed with: {exc}")
        web_context = f"[Web research unavailable: {exc}]"
        web_sources = []

    return {
        "web_context": web_context,
        "web_sources": web_sources,
    }


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
    orientation = state.get("orientation", "landscape")
    print(f"[CODE GEN] Target duration: {target_seconds} seconds, orientation: {orientation}")

    if orientation == "portrait":
        frame_width = "8 units wide (-4 to +4)"
        frame_height = "14.2 units tall (-7.1 to +7.1)"
        x_bounds = "[-3.5, 3.5]"
        y_bounds = "[-6, 6]"
        layout_note = "Content is tall and narrow (9:16 portrait). Stack elements vertically. Use .arrange(DOWN) for groups."
    else:
        frame_width = "14.2 units wide (-7.1 to +7.1)"
        frame_height = "8 units tall (-4 to +4)"
        x_bounds = "[-6, 6]"
        y_bounds = "[-3.5, 3.5]"
        layout_note = "Content is wide (16:9 landscape). Use full horizontal space. Side-by-side layouts work well."

    depth = state.get("explanation_depth", "detailed")
    depth_config = DEPTH_CONFIGS.get(depth, DEPTH_CONFIGS["detailed"])

    duration_mode = state.get("duration_mode", "guide")
    if duration_mode == "guide":
        duration_instruction = f"""- Target: approximately {target_seconds} seconds (this is a SOFT GUIDE, not exact)
- The video does NOT need to be exactly this length — aim for roughly this range
- Use self.wait(1) to self.wait(3) between sections for natural pacing
- Use run_time=2 for major animations"""
    else:
        duration_instruction = f"""- Target: EXACTLY {target_seconds} seconds (STRICT requirement)
- Calculate total animation + wait time to match this precisely
- Use self.wait(1) to self.wait(3) between sections
- Use run_time=2 for major animations"""

    # Optionally inject web research context gathered by web_research_node
    web_context_block = state.get("web_context", "")
    if web_context_block and not web_context_block.startswith("[Web research"):
        web_section = f"""
## Real-World Research Context (USE THIS DATA IN YOUR ANIMATION)
The following up-to-date information was retrieved from the web specifically for this topic.
Use concrete facts, numbers, and recent developments from this research to make the animation
accurate, current, and informative:

{web_context_block}
"""
    else:
        web_section = ""

    prompt = f"""Create a Manim animation based on this request:

**Title:** {state["scene_title"]}
**Description:** {state["scene_prompt_description"]}
**Target Duration:** {target_seconds} seconds ({state["scene_length"]} minutes)
**Explanation Level:** {depth} - {depth_config["detail_level"]}
**Orientation:** {orientation} ({frame_width} × {frame_height})
{web_section}
## CRITICAL REQUIREMENTS (MUST FOLLOW STRICTLY):

### 1. SCREEN BOUNDS — NEVER GO OUT OF FRAME
- The Manim frame is {frame_width} wide and {frame_height} tall.
- Keep objects within bounds (typical text scale: 0.5 to 0.8).
- x must stay within {x_bounds}, y within {y_bounds} (leave margins).
- {layout_note}
- **BOUNDING BOX SAFETY:** Every `VGroup` with multiple text lines or diagram pieces MUST be scaled immediately after construction:
  `my_group.scale_to_fit_width(min(my_group.width, config.frame_width - 2))`

### 2. POSITIONING — FORBIDDEN AND REQUIRED PATTERNS
**FORBIDDEN (NEVER USE):**
- `move_to([2, -1, 0])` — raw coordinate arrays are BANNED.
- `move_to(np.array([x, y, 0]))` — BANNED.
- Any hard-coded spatial number pair used to separate or offset elements.

**REQUIRED (ALWAYS USE):**
- Screen edges: `.to_edge(UP)`, `.to_edge(DOWN)`, `.to_edge(LEFT)`, `.to_edge(RIGHT)`.
- Sequential stacking: `.next_to(other_mobject, DOWN, buff=0.5)`.
- Centering: `.move_to(ORIGIN)` is the ONLY allowed positional constant (it is not a raw array).
- Title: ALWAYS `.to_edge(UP, buff=0.5)`.
- Side elements: `.to_edge(LEFT, buff=1)` or `.to_edge(RIGHT, buff=1)`.
- Labels: `.next_to(target, direction, buff=0.3)`.

### 3. MANDATORY STANDARD GRID LAYOUT
- **Scene title**: `.to_edge(UP, buff=0.5)` — always, no exceptions.
- **Core visualization** (graph, array, diagram): `ORIGIN` or `.to_edge(LEFT, buff=1)`.
- **Explanatory text / bullets**: `.to_edge(RIGHT, buff=1)` or `.next_to(title, DOWN, buff=0.5)`.

### 3a. THE "CLEAR DESK" RULE — CRITICAL
Before a new major topic or any new full-screen text list, ERASE all current elements:
```python
self.play(FadeOut(VGroup(*self.mobjects)))  # clear desk before new topic
```
Or use `self.clear()`. NEVER render new content on top of still-visible old elements.

### 3b. SAFE LAYOUT PATTERN (FOLLOW THIS EXACTLY)
```python
# --- New topic: clear the desk first ---
self.play(FadeOut(VGroup(*self.mobjects)))

# Title always at top edge
title = Text("Title").scale(0.7).to_edge(UP, buff=0.5)
self.play(Write(title))

# Multi-line content: VGroup → scale to fit → position relative to title
bullets = VGroup(
    Text("Point 1").scale(0.5),
    Text("Point 2").scale(0.5),
    Text("Point 3").scale(0.5),
).arrange(DOWN, buff=0.3)
bullets.scale_to_fit_width(min(bullets.width, config.frame_width - 2))
bullets.next_to(title, DOWN, buff=0.5)
self.play(FadeIn(bullets))

# Side label: always relative to another object, never absolute coords
label = Text("Note").scale(0.4).next_to(bullets, RIGHT, buff=0.5)
```

### 4. DURATION
{duration_instruction}
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

### 6. TRANSCRIPT / NARRATION (CRITICAL — NO SILENCE)
- **The narration MUST cover the ENTIRE video duration with NO silent gaps longer than 3 seconds.**
- Detail level: {depth_config["detail_level"]}
- Add transcript entries {depth_config["transcript_density"]} — this is the MINIMUM density
- Every visual change, animation, or concept must have accompanying narration
- Start narration at timestamp 0 and continue until the very end of the video
- If a section has a self.wait(), there MUST be narration during or right before it
- Think of this as a voiceover for a YouTube educational video — continuous speaking

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
    3: "Continuing the explanation without gaps...",
    8: "Every few seconds, add more narration...",
    12: "Keep talking through the entire video...",
    18: "No long silences — every section needs voice...",
    25: "Wrap up with a concluding sentence...",
    # MUST have entries {depth_config["transcript_density"]} covering the FULL {target_seconds} seconds
    # NO gaps longer than 3 seconds between entries
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
        code = _extract_code_block(response_text)

        transcript = {}
        transcript_match = re.search(
            r'# TRANSCRIPT_START\n.*?transcript\s*=\s*(\{.*?\})\s*# TRANSCRIPT_END',
            response_text,
            re.DOTALL,
        )
        if transcript_match:
            try:
                transcript = eval(transcript_match.group(1))
            except (SyntaxError, ValueError, NameError):
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
# NODE 2: Code Executor (delegates to ManimExecutor)
# ============================================================================

def code_executor_node(state: VideoGenState) -> dict:
    """
    Execute Manim code and capture rendered video or errors.

    Delegates to ManimExecutor for actual rendering.

    Input: code, scene_class_name
    Output: rendered_video_path OR error
    """
    scene_class = state.get("scene_class_name", "GeneratedScene")
    orientation = state.get("orientation", "landscape")

    print(f"[EXECUTOR] Starting Manim render for scene: {scene_class}")

    settings = get_settings()
    executor = ManimExecutor(quality="l", timeout=settings.render_timeout)
    result = executor.execute(
        code=state["code"],
        scene_class_name=scene_class,
        orientation=orientation,
    )

    if result.success:
        print(f"[EXECUTOR] Render SUCCESS: {result.video_path}")
        return {
            "rendered_video_path": result.video_path,
            "error": None,
            "render_logs": f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
            "temp_code_path": result.code_path,
        }

    print(f"[EXECUTOR] Render FAILED: {(result.error or '')[:200]}...")
    return {
        "error": result.error,
        "error_count": 1,
        "render_logs": f"STDOUT:\n{result.stdout}\n\nSTDERR:\n{result.stderr}",
        "temp_code_path": result.code_path,
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

CRITICAL LAYOUT RULES (enforce these while fixing — do NOT introduce violations):

1. THE "CLEAR DESK" RULE
   - Before every new major topic or full-screen text list, ALL previous elements must be cleared.
   - Use: self.play(FadeOut(VGroup(*self.mobjects)))  OR  self.clear()
   - NEVER place new content on top of still-visible old elements.

2. FORBIDDEN SYNTAX — ABSOLUTE COORDINATES
   - NEVER use raw coordinate arrays: move_to([2, -1, 0]) or move_to(np.array([x, y, 0])) are BANNED.
   - Replace any such calls with relative positioning (see rule 3).

3. MANDATORY RELATIVE POSITIONING
   - Anchor to screen edges: .to_edge(UP), .to_edge(DOWN), .to_edge(LEFT), .to_edge(RIGHT)
   - Stack elements: .next_to(other_mobject, DOWN, buff=0.5)
   - Center: .move_to(ORIGIN) is allowed (named constant, not a raw array)

4. STANDARD GRID LAYOUT
   - Titles: always .to_edge(UP, buff=0.5)
   - Core visuals (graphs, arrays): ORIGIN or .to_edge(LEFT, buff=1)
   - Explanatory text / bullets: .to_edge(RIGHT, buff=1) or .next_to(title, DOWN, buff=0.5)

5. BOUNDING BOX SAFETY
   - Every VGroup with multiple items must be scaled after construction:
     my_group.scale_to_fit_width(min(my_group.width, config.frame_width - 2))

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
        fixed_code = _extract_code_block(fixed_code)

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
            tts = KokoroTTS(voice="af_bella")
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

_DURATION_TOLERANCE = 0.20  # ±20%


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
        print("[RENDER CHECK] Video file does not exist!")
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

    actual_duration = get_video_duration(video_path)
    if actual_duration is not None:
        print(f"[RENDER CHECK] Actual duration: {actual_duration:.1f}s (target: {target_duration:.1f}s)")

        duration_ratio = actual_duration / target_duration if target_duration > 0 else 1.0
        if abs(duration_ratio - 1.0) > _DURATION_TOLERANCE:
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
        print("[RENDER CHECK] Could not determine video duration")

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

_MAX_PTS_FACTOR = 2.0  # Cap at 2x slowdown or 2x speedup


def video_duration_fixer_node(state: VideoGenState) -> dict:
    """
    Adjust video playback speed to match target duration using ffmpeg.

    Uses PTS (Presentation Time Stamp) manipulation to speed up or slow
    down the video. Caps adjustment at 2x to avoid extreme distortion.

    Stores the *actual* speed ratio (adjusted_duration / original_duration)
    so the synchronizer can accurately rescale transcript timestamps,
    even when PTS is clamped.

    Input: checked_video_path, actual_duration, target_duration
    Output: checked_video_path (adjusted), duration_adjusted, duration_factor
    """
    video_path = state.get("checked_video_path")
    original_duration = state.get("actual_duration")
    target_duration = state.get("target_duration", state.get("scene_length", 1.0) * 60)

    print(f"[DURATION FIX] Video: {video_path}")
    print(f"[DURATION FIX] Original: {original_duration}s -> Target: {target_duration}s")

    if not video_path or not original_duration or original_duration <= 0:
        print("[DURATION FIX] Cannot fix — missing video or duration info")
        return {"duration_adjusted": False, "duration_factor": None}

    if target_duration <= 0:
        print("[DURATION FIX] Invalid target duration, skipping")
        return {"duration_adjusted": False, "duration_factor": None}

    ratio = original_duration / target_duration

    if abs(ratio - 1.0) <= _DURATION_TOLERANCE:
        print(f"[DURATION FIX] Duration within tolerance ({ratio:.2f}x), no adjustment needed")
        return {"duration_adjusted": False, "duration_factor": None}

    # setpts multiplier: >1 slows down, <1 speeds up
    pts_factor = target_duration / original_duration  # inverse of ratio
    min_pts = 1.0 / _MAX_PTS_FACTOR
    clamped_pts = max(min_pts, min(_MAX_PTS_FACTOR, pts_factor))

    if clamped_pts != pts_factor:
        print(f"[DURATION FIX] PTS factor clamped from {pts_factor:.4f} to {clamped_pts:.4f}")
        pts_factor = clamped_pts

    direction = "too long" if ratio > 1.0 else "too short"
    action = "speeding up" if ratio > 1.0 else "slowing down"
    print(f"[DURATION FIX] Video is {direction} ({original_duration:.1f}s vs {target_duration:.1f}s) — {action} (setpts={pts_factor:.4f}*PTS)")

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
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"[DURATION FIX] ffmpeg failed: {result.stderr[:300]}")
            return {"duration_adjusted": False, "duration_factor": None}

        if not adjusted_path.exists():
            print("[DURATION FIX] Output file not created")
            return {"duration_adjusted": False, "duration_factor": None}

        new_duration = get_video_duration(str(adjusted_path))
        if new_duration is None:
            # Estimate from PTS if ffprobe fails
            new_duration = original_duration * pts_factor

        # Actual speed ratio: how much the video was stretched/compressed
        # >1 means video got longer, <1 means video got shorter
        actual_speed_ratio = new_duration / original_duration

        print(f"[DURATION FIX] Adjusted duration: {new_duration:.1f}s (original: {original_duration:.1f}s, speed_ratio: {actual_speed_ratio:.4f})")
        print(f"[DURATION FIX] Success! Adjusted video: {adjusted_path}")

        return {
            "checked_video_path": str(adjusted_path),
            "duration_adjusted": True,
            "duration_factor": actual_speed_ratio,
            "actual_duration": new_duration,
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

    ``duration_factor`` is the actual speed ratio (adjusted_dur / original_dur)
    returned by ``video_duration_fixer_node``.  A value of 2.0 means the video
    is now twice as long, so every timestamp must be multiplied by 2.0.

    Input: checked_video_path, transcript_sections, audio_segments, duration_factor
    Output: synced_video_path, transcript_sections (adjusted), audio_segments
    """
    video_path = state.get("checked_video_path")
    audio_segments = state.get("audio_segments", [])
    transcript_sections = state.get("transcript_sections", [])
    duration_adjusted = state.get("duration_adjusted", False)
    speed_ratio = state.get("duration_factor")  # adjusted_dur / original_dur

    print(f"[SYNC] Video path: {video_path}")
    print(f"[SYNC] Audio segments: {len(audio_segments)}")
    print(f"[SYNC] Transcript sections: {len(transcript_sections)}")
    print(f"[SYNC] Duration adjusted: {duration_adjusted} (speed_ratio: {speed_ratio})")

    synced_sections = list(transcript_sections)

    if duration_adjusted and speed_ratio and speed_ratio != 1.0:
        # Multiply timestamps by speed_ratio:
        #   video slowed 2x → speed_ratio=2.0 → timestamps double
        #   video sped up 2x → speed_ratio=0.5 → timestamps halve
        print(f"[SYNC] Scaling transcript timestamps × {speed_ratio:.4f}")
        synced_sections = [
            TranscriptSection(
                timestamp=section["timestamp"] * speed_ratio,
                text=section["text"],
                audio_path=section.get("audio_path"),
            )
            for section in transcript_sections
        ]
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

    Builds a single narration track by concatenating silence gaps and
    (optionally sped-up) audio segments so each one lands at its transcript
    timestamp with zero overlap.

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

        video_duration = get_video_duration(video_path)
        if video_duration is not None:
            print(f"[AUDIO MERGE] Video duration: {video_duration:.1f}s")

        # Collect sections that have valid audio files
        sections_with_audio = [
            section for section in transcript_sections
            if section.get("audio_path") and Path(section["audio_path"]).exists()
        ]
        print(f"[AUDIO MERGE] Sections with valid audio: {len(sections_with_audio)}")

        if not sections_with_audio:
            sections_with_audio = [
                {"timestamp": 0.0, "audio_path": p}
                for p in valid_audio_segments
            ]

        # Drop segments that start past video end
        if video_duration is not None:
            before = len(sections_with_audio)
            sections_with_audio = [
                s for s in sections_with_audio
                if s["timestamp"] < video_duration
            ]
            dropped = before - len(sections_with_audio)
            if dropped > 0:
                print(f"[AUDIO MERGE] Dropped {dropped} segment(s) past video end ({video_duration:.1f}s)")

        if not sections_with_audio:
            print("[AUDIO MERGE] No audio segments within video bounds")
            return {"final_output_path": video_path}

        # Sort by timestamp
        sections_with_audio.sort(key=lambda s: s["timestamp"])

        # --- Build the audio track by concatenating silence + segments ---
        # For each segment:
        #   1. Calculate available window (until next segment or video end)
        #   2. Normalize to consistent format (24kHz mono s16le)
        #   3. If audio is longer than window, speed it up with atempo
        #   4. Generate silence to fill the gap before this segment
        #   5. Collect all pieces for final concat

        SAMPLE_RATE = 24000
        concat_pieces = []  # list of wav file paths in order
        cursor = 0.0  # current position in the timeline

        for i, section in enumerate(sections_with_audio):
            ts = section["timestamp"]

            # Calculate available window for this segment
            if i + 1 < len(sections_with_audio):
                window = sections_with_audio[i + 1]["timestamp"] - ts
            elif video_duration is not None:
                window = video_duration - ts
            else:
                window = 999.0  # no limit for last segment if no video duration

            window = max(window, 0.1)  # safety floor

            # 1. Normalize audio to consistent format
            norm_path = Path(temp_dir) / f"norm_{i:03d}.wav"
            norm_cmd = [
                "ffmpeg", "-y",
                "-i", section["audio_path"],
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                str(norm_path),
            ]
            result = subprocess.run(norm_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0 or not norm_path.exists():
                print(f"[AUDIO MERGE]   Segment {i} ({ts:.1f}s): normalize failed, skipping")
                continue

            # 2. Get actual duration of this audio clip
            clip_dur = get_video_duration(str(norm_path), timeout=10)
            if clip_dur is None or clip_dur <= 0:
                print(f"[AUDIO MERGE]   Segment {i} ({ts:.1f}s): could not measure duration, skipping")
                continue

            # 3. If audio overflows the window, speed it up
            processed_path = norm_path
            if clip_dur > window + 0.1:
                speed_factor = clip_dur / window
                # atempo only supports 0.5 to 100.0; chain for extreme values
                processed_path = Path(temp_dir) / f"fast_{i:03d}.wav"
                atempo_filters = _build_atempo_chain(speed_factor)
                speed_cmd = [
                    "ffmpeg", "-y",
                    "-i", str(norm_path),
                    "-af", atempo_filters,
                    "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                    str(processed_path),
                ]
                result = subprocess.run(speed_cmd, capture_output=True, text=True, timeout=30)
                if result.returncode != 0 or not processed_path.exists():
                    print(f"[AUDIO MERGE]   Segment {i} ({ts:.1f}s): speed-up failed, using original (may overlap)")
                    processed_path = norm_path
                else:
                    new_dur = get_video_duration(str(processed_path), timeout=10)
                    print(f"[AUDIO MERGE]   Segment {i} ({ts:.1f}s): sped up {speed_factor:.2f}x ({clip_dur:.1f}s → {new_dur:.1f}s to fit {window:.1f}s window)")
                    clip_dur = new_dur or (clip_dur / speed_factor)
            else:
                print(f"[AUDIO MERGE]   Segment {i} ({ts:.1f}s): {clip_dur:.1f}s audio fits in {window:.1f}s window")

            # 4. If there's a gap before this segment, insert silence
            gap = ts - cursor
            if gap > 0.01:
                silence_path = Path(temp_dir) / f"silence_{i:03d}.wav"
                silence_cmd = [
                    "ffmpeg", "-y",
                    "-f", "lavfi", "-i", f"anullsrc=r={SAMPLE_RATE}:cl=mono",
                    "-t", f"{gap:.4f}",
                    "-c:a", "pcm_s16le",
                    str(silence_path),
                ]
                result = subprocess.run(silence_cmd, capture_output=True, text=True, timeout=10)
                if result.returncode == 0 and silence_path.exists():
                    concat_pieces.append(str(silence_path))

            concat_pieces.append(str(processed_path))
            cursor = ts + clip_dur

        if not concat_pieces:
            print("[AUDIO MERGE] No audio pieces produced")
            return {"final_output_path": video_path}

        # 5. Concatenate all pieces into one audio file
        merged_audio = Path(temp_dir) / "merged_audio.wav"

        if len(concat_pieces) == 1:
            shutil.copy2(concat_pieces[0], str(merged_audio))
        else:
            concat_list = Path(temp_dir) / "concat.txt"
            with open(concat_list, "w") as f:
                for piece in concat_pieces:
                    f.write(f"file '{piece}'\n")

            concat_cmd = [
                "ffmpeg", "-y",
                "-f", "concat", "-safe", "0",
                "-i", str(concat_list),
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                str(merged_audio),
            ]
            print(f"[AUDIO MERGE] Concatenating {len(concat_pieces)} audio pieces...")
            result = subprocess.run(concat_cmd, capture_output=True, text=True, timeout=60)
            if result.returncode != 0 or not merged_audio.exists():
                print(f"[AUDIO MERGE] Concat failed: {result.stderr[:200]}")
                return {"final_output_path": video_path}

        # Log merged audio duration
        audio_dur = get_video_duration(str(merged_audio), timeout=10)
        if audio_dur is not None:
            print(f"[AUDIO MERGE] Merged audio duration: {audio_dur:.1f}s")

        # Trim if audio still extends past video
        if video_duration is not None and audio_dur is not None and audio_dur > video_duration + 0.5:
            trimmed_audio = Path(temp_dir) / "trimmed_audio.wav"
            trim_cmd = [
                "ffmpeg", "-y",
                "-i", str(merged_audio),
                "-af", f"atrim=0:{video_duration},asetpts=PTS-STARTPTS",
                "-ar", str(SAMPLE_RATE), "-ac", "1", "-c:a", "pcm_s16le",
                str(trimmed_audio),
            ]
            trim_result = subprocess.run(trim_cmd, capture_output=True, text=True, timeout=30)
            if trim_result.returncode == 0 and trimmed_audio.exists():
                merged_audio = trimmed_audio
                print(f"[AUDIO MERGE] Trimmed audio to {video_duration:.1f}s to match video")

        print(f"[AUDIO MERGE] Final audio track: {merged_audio}")

        # Merge audio with video
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

        print("[AUDIO MERGE] Merging audio track with video...")
        result = subprocess.run(merge_cmd, capture_output=True, text=True, timeout=120)

        if result.returncode != 0:
            print(f"[AUDIO MERGE] Merge failed: {result.stderr[:200]}")
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
            subprocess.run(simple_merge_cmd, capture_output=True, text=True, timeout=120)

        if not final_output.exists():
            print("[AUDIO MERGE] Final output not created")
            return {"final_output_path": video_path}

        # Copy to the project output directory for easy access
        project_output = Path("output")
        project_output.mkdir(parents=True, exist_ok=True)
        output_copy = project_output / "output.mp4"
        shutil.copy2(str(final_output), str(output_copy))
        print(f"[AUDIO MERGE] Copied to: {output_copy.resolve()}")

        print(f"[AUDIO MERGE] Success! Final output: {final_output}")
        return {"final_output_path": str(final_output)}

    except Exception as e:
        print(f"[AUDIO MERGE] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"final_output_path": video_path}


def _build_atempo_chain(factor: float) -> str:
    """Build an atempo filter chain for ffmpeg.

    atempo only supports values between 0.5 and 100.0, so extreme
    speed-ups are chained (e.g., 4x = atempo=2.0,atempo=2.0).
    """
    if factor <= 0:
        return "atempo=1.0"
    parts = []
    remaining = factor
    while remaining > 100.0:
        parts.append("atempo=100.0")
        remaining /= 100.0
    while remaining < 0.5:
        parts.append("atempo=0.5")
        remaining /= 0.5
    parts.append(f"atempo={remaining:.4f}")
    return ",".join(parts)


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
    duration_mode = state.get("duration_mode", "guide")

    if duration_mode == "guide":
        print("[DURATION CHECK] Mode is 'guide' — skipping ffmpeg speed adjustment")
        return "synchronizer"

    actual_duration = state.get("actual_duration")
    target_duration = state.get("target_duration", state.get("scene_length", 1.0) * 60)

    if not actual_duration or not target_duration or target_duration <= 0:
        print("[DURATION CHECK] Missing duration info, skipping fix")
        return "synchronizer"

    factor = actual_duration / target_duration

    if abs(factor - 1.0) > _DURATION_TOLERANCE:
        print(f"[DURATION CHECK] Duration off by {(factor - 1.0) * 100:+.0f}% — routing to fixer")
        return "video_duration_fixer"

    print(f"[DURATION CHECK] Duration OK ({factor:.2f}x) — skipping fix")
    return "synchronizer"
