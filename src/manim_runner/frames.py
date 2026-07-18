"""
Frame extraction for visual QA.

Samples evenly spaced frames from a rendered video with ffmpeg so a
multimodal LLM can review the actual layout of the output.
"""

import subprocess
import tempfile
from pathlib import Path
from typing import Optional

from .validator import get_video_duration


def extract_frames(
    video_path: str,
    count: int = 6,
    out_dir: Optional[str] = None,
) -> list[dict]:
    """
    Extract ``count`` evenly spaced JPEG frames from a video.

    Skips a small margin at both ends so title fade-ins and final fade-outs
    are not sampled mid-transition.

    Returns a list of {"path": str, "timestamp": float} entries, empty when
    the video cannot be read.
    """
    duration = get_video_duration(video_path)
    if not duration or duration <= 0 or count < 1:
        return []

    if out_dir is None:
        out_dir = tempfile.mkdtemp(prefix="manim_frames_")

    margin = min(0.25, duration / 4)
    span = duration - 2 * margin

    frames = []
    for i in range(count):
        timestamp = margin + span * (i / max(count - 1, 1))
        frame_path = Path(out_dir) / f"frame_{i:02d}.jpg"
        cmd = [
            "ffmpeg", "-y",
            "-ss", f"{timestamp:.3f}",
            "-i", video_path,
            "-frames:v", "1",
            "-q:v", "4",
            str(frame_path),
        ]
        try:
            result = subprocess.run(cmd, capture_output=True, timeout=30)
        except (subprocess.TimeoutExpired, OSError):
            continue
        if result.returncode == 0 and frame_path.exists():
            frames.append({"path": str(frame_path), "timestamp": round(timestamp, 2)})

    return frames
