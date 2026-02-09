"""
Kokoro TTS module using the local Kokoro 82M model.

Uses kokoro-onnx for efficient local text-to-speech synthesis.
Model files are downloaded automatically on first use.
"""

import os
import tempfile
from pathlib import Path
from typing import Optional
from dataclasses import dataclass
import urllib.request

try:
    from kokoro_onnx import Kokoro
    KOKORO_AVAILABLE = True
except ImportError:
    Kokoro = None
    KOKORO_AVAILABLE = False

try:
    import soundfile as sf
    SOUNDFILE_AVAILABLE = True
except ImportError:
    sf = None
    SOUNDFILE_AVAILABLE = False


MODEL_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/kokoro-v1.0.onnx"
VOICES_URL = "https://github.com/thewh1teagle/kokoro-onnx/releases/download/model-files-v1.0/voices-v1.0.bin"


@dataclass
class TTSResult:
    """Result from TTS synthesis."""
    success: bool
    audio_path: Optional[str]
    error: Optional[str]
    duration: Optional[float]


class KokoroTTS:
    """
    Text-to-Speech using Kokoro 82M local model.
    
    Uses kokoro-onnx for ONNX Runtime inference.
    Fast performance, even on CPU.
    """
    
    # Available voice options (see: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md)
    VOICES = {
        # American English Female
        "af_bella": "af_bella",
        "af_nicole": "af_nicole",
        "af_sarah": "af_sarah",
        "af_sky": "af_sky",
        # American English Male
        "am_adam": "am_adam",
        "am_michael": "am_michael",
        # British English Female
        "bf_emma": "bf_emma",
        "bf_isabella": "bf_isabella",
        # British English Male
        "bm_george": "bm_george",
        "bm_lewis": "bm_lewis",
    }
    
    DEFAULT_VOICE = "af_bella"
    
    def __init__(
        self,
        model_dir: Optional[str] = None,
        voice: str = DEFAULT_VOICE,
        speed: float = 1.0,
    ):
        """
        Initialize Kokoro TTS.
        
        Args:
            model_dir: Directory for model files. Defaults to ~/.kokoro
            voice: Voice ID to use (see VOICES dict)
            speed: Speech speed multiplier (1.0 = normal)
        """
        self.model_dir = Path(model_dir or os.path.expanduser("~/.kokoro"))
        self.voice = voice
        self.speed = speed
        self._kokoro = None
        
        if not KOKORO_AVAILABLE:
            raise ImportError(
                "kokoro-onnx is not installed. Install with: pip install kokoro-onnx"
            )
        if not SOUNDFILE_AVAILABLE:
            raise ImportError(
                "soundfile is not installed. Install with: pip install soundfile"
            )
    
    def _download_file(self, url: str, dest: Path) -> bool:
        """Download a file if it doesn't exist."""
        if dest.exists():
            return True
        
        print(f"Downloading {dest.name}...")
        dest.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            urllib.request.urlretrieve(url, dest)
            print(f"Downloaded {dest.name}")
            return True
        except Exception as e:
            print(f"Failed to download {dest.name}: {e}")
            return False
    
    def _ensure_models(self) -> bool:
        """Ensure model files are downloaded."""
        model_path = self.model_dir / "kokoro-v1.0.onnx"
        voices_path = self.model_dir / "voices-v1.0.bin"
        
        if not self._download_file(MODEL_URL, model_path):
            return False
        if not self._download_file(VOICES_URL, voices_path):
            return False
        
        return True
    
    @property
    def kokoro(self):
        """Lazy-load Kokoro model."""
        if self._kokoro is None:
            if not self._ensure_models():
                raise RuntimeError("Failed to download Kokoro model files")
            
            model_path = self.model_dir / "kokoro-v1.0.onnx"
            voices_path = self.model_dir / "voices-v1.0.bin"
            
            self._kokoro = Kokoro(str(model_path), str(voices_path))
        
        return self._kokoro
    
    def synthesize(
        self,
        text: str,
        output_path: Optional[str] = None,
        voice: Optional[str] = None,
        speed: Optional[float] = None,
    ) -> TTSResult:
        """
        Synthesize speech from text.
        
        Args:
            text: Text to speak
            output_path: Path for output WAV file. Auto-generated if None.
            voice: Voice ID (overrides default)
            speed: Speed multiplier (overrides default)
        
        Returns:
            TTSResult with audio path or error
        """
        try:
            voice = voice or self.voice
            speed = speed or self.speed
            
            if output_path is None:
                fd, output_path = tempfile.mkstemp(suffix=".wav", prefix="kokoro_")
                os.close(fd)
            
            samples, sample_rate = self.kokoro.create(
                text,
                voice=voice,
                speed=speed,
            )
            
            sf.write(output_path, samples, sample_rate)
            
            duration = len(samples) / sample_rate
            
            return TTSResult(
                success=True,
                audio_path=output_path,
                error=None,
                duration=duration,
            )
        
        except Exception as e:
            return TTSResult(
                success=False,
                audio_path=None,
                error=str(e),
                duration=None,
            )
    
    def synthesize_segments(
        self,
        segments: list[dict],
        output_dir: Optional[str] = None,
        voice: Optional[str] = None,
    ) -> list[TTSResult]:
        """
        Synthesize multiple text segments.
        
        Args:
            segments: List of {"timestamp": float, "text": str}
            output_dir: Directory for output files
            voice: Voice ID to use
        
        Returns:
            List of TTSResult for each segment
        """
        if output_dir is None:
            output_dir = tempfile.mkdtemp(prefix="kokoro_segments_")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        results = []
        for i, segment in enumerate(segments):
            text = segment.get("text", "")
            if not text.strip():
                results.append(TTSResult(
                    success=True,
                    audio_path=None,
                    error=None,
                    duration=0.0,
                ))
                continue
            
            audio_path = output_path / f"segment_{i:03d}.wav"
            result = self.synthesize(
                text=text,
                output_path=str(audio_path),
                voice=voice,
            )
            results.append(result)
        
        return results


_tts_instance: Optional[KokoroTTS] = None


def get_tts(voice: str = KokoroTTS.DEFAULT_VOICE) -> KokoroTTS:
    """Get or create a KokoroTTS instance."""
    global _tts_instance
    if _tts_instance is None:
        _tts_instance = KokoroTTS(voice=voice)
    return _tts_instance


def synthesize_text(
    text: str,
    output_path: Optional[str] = None,
    voice: str = KokoroTTS.DEFAULT_VOICE,
) -> TTSResult:
    """
    Convenience function to synthesize text to speech.
    
    Args:
        text: Text to speak
        output_path: Output WAV path (auto-generated if None)
        voice: Voice ID
    
    Returns:
        TTSResult with audio path
    """
    tts = get_tts(voice)
    return tts.synthesize(text, output_path)
