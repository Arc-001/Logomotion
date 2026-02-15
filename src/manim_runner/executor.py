"""
Manim Code Executor.

Handles writing code to files, executing manim render,
and capturing output/errors.
"""

import os
import sys
import tempfile
import subprocess
import shutil
from pathlib import Path
from typing import Optional
from dataclasses import dataclass


@dataclass
class ExecutionResult:
    """Result from executing Manim code."""
    success: bool
    video_path: Optional[str]
    error: Optional[str]
    stdout: str
    stderr: str
    code_path: str
    output_dir: str


class ManimExecutor:
    """Executes Manim code in a sandboxed environment."""
    
    def __init__(
        self,
        base_output_dir: Optional[str] = None,
        quality: str = "l",  # l=low, m=medium, h=high
        timeout: int = 120,
    ):
        self.base_output_dir = base_output_dir or tempfile.gettempdir()
        self.quality = quality
        self.timeout = timeout
    
    @staticmethod
    def _is_manim_runnable(path: str) -> bool:
        """Check if a manim binary is actually runnable (valid shebang, etc.)."""
        try:
            result = subprocess.run(
                [path, "--version"],
                capture_output=True,
                text=True,
                timeout=10,
            )
            return result.returncode == 0
        except (FileNotFoundError, subprocess.TimeoutExpired, OSError):
            return False

    @classmethod
    def _resolve_manim_cmd(cls) -> str:
        """Auto-discover a working manim binary.

        Search order:
          1. Same bin/ directory as the running Python interpreter (venv)
          2. ~/.local/bin/manim
          3. Fall back to bare 'manim' (relies on PATH)

        Each candidate is tested with ``manim --version`` to guard
        against stale shebangs or broken installs.
        """
        candidates = [
            Path(sys.executable).parent / "manim",
            Path.home() / ".local" / "bin" / "manim",
        ]

        for candidate in candidates:
            if candidate.is_file() and cls._is_manim_runnable(str(candidate)):
                return str(candidate)

        # Fallback — let the OS resolve it via PATH
        return "manim"
    
    @staticmethod
    def _build_env(cwd: str) -> dict:
        """Build subprocess environment with venv/bin and ~/.local/bin on PATH."""
        env = os.environ.copy()
        extra_paths = []
        
        venv_bin = str(Path(sys.executable).parent)
        if venv_bin not in env.get("PATH", ""):
            extra_paths.append(venv_bin)
        
        local_bin = str(Path.home() / ".local" / "bin")
        if local_bin not in env.get("PATH", ""):
            extra_paths.append(local_bin)
        
        if extra_paths:
            env["PATH"] = os.pathsep.join(extra_paths) + os.pathsep + env.get("PATH", "")
        
        env["PYTHONPATH"] = cwd
        return env
    
    def execute(
        self,
        code: str,
        scene_class_name: str,
        output_name: Optional[str] = None,
        orientation: str = "landscape",
    ) -> ExecutionResult:
        """
        Execute Manim code and return the result.
        
        Args:
            code: Python code containing a Manim Scene
            scene_class_name: Name of the Scene class to render
            output_name: Optional name for output files
        
        Returns:
            ExecutionResult with video path or error
        """
        temp_dir = tempfile.mkdtemp(prefix="manim_exec_")
        code_path = Path(temp_dir) / "scene.py"
        output_dir = Path(temp_dir) / "media"
        
        manim_cmd = self._resolve_manim_cmd()
        
        try:
            with open(code_path, "w", encoding="utf-8") as f:
                f.write(code)
            
            cmd = [
                manim_cmd, "render",
                str(code_path),
                scene_class_name,
                f"-q{self.quality}",
                "--media_dir", str(output_dir),
            ]

            if orientation == "portrait":
                cmd.extend(["--resolution", "1080,1920"])
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=self.timeout,
                cwd=temp_dir,
                env=self._build_env(temp_dir),
            )
            
            if result.returncode != 0:
                return ExecutionResult(
                    success=False,
                    video_path=None,
                    error=self._parse_error(result.stderr or result.stdout),
                    stdout=result.stdout,
                    stderr=result.stderr,
                    code_path=str(code_path),
                    output_dir=str(output_dir),
                )
            
            video_path = self._find_video(output_dir, scene_class_name)
            
            if not video_path:
                return ExecutionResult(
                    success=False,
                    video_path=None,
                    error="No video file was produced",
                    stdout=result.stdout,
                    stderr=result.stderr,
                    code_path=str(code_path),
                    output_dir=str(output_dir),
                )
            
            if output_name:
                final_dir = Path(self.base_output_dir) / output_name
                final_dir.mkdir(parents=True, exist_ok=True)
                final_video = final_dir / f"{scene_class_name}.mp4"
                shutil.copy2(video_path, final_video)
                video_path = str(final_video)
            
            return ExecutionResult(
                success=True,
                video_path=video_path,
                error=None,
                stdout=result.stdout,
                stderr=result.stderr,
                code_path=str(code_path),
                output_dir=str(output_dir),
            )
        
        except subprocess.TimeoutExpired:
            return ExecutionResult(
                success=False,
                video_path=None,
                error=f"Render timeout exceeded ({self.timeout}s)",
                stdout="",
                stderr="",
                code_path=str(code_path),
                output_dir=str(output_dir),
            )
        except Exception as e:
            return ExecutionResult(
                success=False,
                video_path=None,
                error=str(e),
                stdout="",
                stderr="",
                code_path=str(code_path),
                output_dir=str(output_dir),
            )
    
    def _find_video(self, output_dir: Path, scene_name: str) -> Optional[str]:
        """Find the rendered video file."""
        if not output_dir.exists():
            return None
        
        video_files = list(output_dir.rglob("*.mp4"))
        
        if not video_files:
            return None
        
        # Prefer file matching scene name
        for vf in video_files:
            if scene_name.lower() in vf.stem.lower():
                return str(vf)
        
        return str(video_files[0])
    
    def _parse_error(self, error_text: str) -> str:
        """Extract meaningful error messages from output."""
        lines = error_text.split("\n")
        
        error_lines = []
        in_traceback = False
        
        for line in lines:
            if "Traceback" in line:
                in_traceback = True
            if in_traceback:
                error_lines.append(line)
            elif "Error:" in line or "Exception:" in line:
                error_lines.append(line)
        
        if error_lines:
            return "\n".join(error_lines[-20:])  # Last 20 lines of error
        
        return error_text[:1000]  # First 1000 chars as fallback
    
    def cleanup(self, result: ExecutionResult):
        """Clean up temporary files from an execution."""
        try:
            temp_dir = Path(result.code_path).parent
            if temp_dir.exists() and "manim_exec_" in str(temp_dir):
                shutil.rmtree(temp_dir)
        except Exception:
            pass  # Ignore cleanup errors


def execute_manim(
    code: str,
    scene_class_name: str,
    output_dir: Optional[str] = None,
    quality: str = "l",
    timeout: int = 120,
    orientation: str = "landscape",
) -> ExecutionResult:
    """
    Convenience function to execute Manim code.

    Args:
        code: Python code with Manim Scene
        scene_class_name: Name of the Scene class
        output_dir: Optional output directory
        quality: Render quality (l/m/h)
        timeout: Execution timeout in seconds
        orientation: Video orientation (landscape/portrait)

    Returns:
        ExecutionResult with success status and paths
    """
    executor = ManimExecutor(
        base_output_dir=output_dir,
        quality=quality,
        timeout=timeout,
    )
    return executor.execute(code, scene_class_name, orientation=orientation)
