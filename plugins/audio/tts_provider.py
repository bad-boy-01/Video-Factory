import logging
import wave

logger = logging.getLogger(__name__)


class TTSProvider:
    def generate_voice(self, text: str, voice_id: str, output_path: str) -> float:
        raise NotImplementedError


class KokoroTTSProvider(TTSProvider):
    """
    Real TTS via the Kokoro-82M model (https://github.com/hexgrad/kokoro) when the
    'kokoro' + 'soundfile' packages are installed.

    Falls back to writing a technically valid (silent) WAV file sized to a rough
    reading-speed estimate when they are not available, or if synthesis fails for
    any reason. This matters because downstream code (ffmpeg_renderer.py) calls
    wave.open() on these files to measure duration and then feeds them to ffmpeg
    for muxing - a non-WAV placeholder (e.g. a plain text file) would silently
    fall back to a wrong default duration, and would either be rejected or
    produce broken/garbage output when ffmpeg tries to decode it as audio.
    """

    SAMPLE_RATE = 24000  # Kokoro's native output rate

    def __init__(self, lang_code: str = "a", default_voice: str = "af_heart"):
        self.lang_code = lang_code
        self.default_voice = default_voice
        self._pipeline = None

    def _get_pipeline(self):
        if self._pipeline is not None:
            return self._pipeline
        from kokoro import KPipeline  # raises ImportError if the package isn't installed
        self._pipeline = KPipeline(lang_code=self.lang_code)
        return self._pipeline

    def generate_voice(self, text: str, voice_id: str, output_path: str) -> float:
        text = (text or "").strip() or "..."
        voice = voice_id if (voice_id and voice_id != "default") else self.default_voice

        try:
            return self._generate_real(text, voice, output_path)
        except ImportError:
            logger.warning(
                "[TTS] 'kokoro'/'soundfile' not installed - writing a silent "
                "placeholder WAV instead. Install with: pip install kokoro soundfile"
            )
            return self._write_silent_wav(text, output_path)
        except Exception as e:
            logger.error(f"[TTS] Kokoro synthesis failed ({e}); writing silent placeholder WAV.")
            return self._write_silent_wav(text, output_path)

    def _generate_real(self, text: str, voice: str, output_path: str) -> float:
        import numpy as np
        import soundfile as sf

        pipeline = self._get_pipeline()
        chunks = [audio for _, _, audio in pipeline(text, voice=voice)]
        audio = np.concatenate(chunks) if chunks else np.zeros(self.SAMPLE_RATE, dtype=np.float32)

        sf.write(output_path, audio, self.SAMPLE_RATE)
        return len(audio) / self.SAMPLE_RATE

    def _write_silent_wav(self, text: str, output_path: str) -> float:
        """Writes a real, valid, silent WAV file sized to a rough reading-speed estimate."""
        duration = max(1.0, len(text) * 0.06)  # ~16-17 chars/sec, a rough average speech rate
        n_frames = int(duration * self.SAMPLE_RATE)

        with wave.open(output_path, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)  # 16-bit PCM
            w.setframerate(self.SAMPLE_RATE)
            w.writeframes(b"\x00\x00" * n_frames)

        return duration
