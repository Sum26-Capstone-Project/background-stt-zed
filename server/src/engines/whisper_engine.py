import gc
import ctypes
import torch
import numpy as np
from typing import List
from faster_whisper import WhisperModel

from src.engines.base import STTEngine, TranscriptionSegment, EngineInfo


def _cuda_runtime_available() -> bool:
    if not torch.cuda.is_available():
        return False

    try:
        ctypes.CDLL("libcublas.so.12")
    except OSError:
        print("Falling back to CPU for Whisper.")
        return False

    print("Running model on GPU")
    return True


class WhisperEngine(STTEngine):
    engine_name = "whisper"
    # Rough resident-memory estimates per faster-whisper model size, used only
    # for the informational /status endpoint (not enforced anywhere).
    _VRAM_ESTIMATE_MB = {
        "tiny": 1000,
        "base": 1200,
        "small": 2000,
        "medium": 5000,
        "large-v3": 10000,
    }

    def __init__(self):
        self.model = None
        self.model_name = "base"
        self._language = "en"
        self._loaded = False
        self._device = "cpu"
        self._compute_type = "int8"
        self._previous_text = ""  # context carryover across segments

    def load(self, language: str) -> None:
        from src.config import settings

        if settings.default_language != "auto":
            self._language = settings.default_language
        else:
            self._language = language

        self.model_name = settings.whisper_model_size

        if _cuda_runtime_available():
            self._device = "cuda"
            self._compute_type = "float16"
        else:
            self._device = "cpu"
            self._compute_type = "int8"

        print(f"Loading {self.engine_name} ({self.model_name}) on {self._device} ({self._compute_type})...")
        self.model = WhisperModel(self.model_name, device=self._device, compute_type=self._compute_type)
        self._loaded = True

    # Whisper hallucination: when audio has no speech, the model tends to echo
    # the initial_prompt verbatim. We detect this via two signals:
    #   1. no_speech_prob: per-segment probability that there is no speech.
    #   2. prompt_echo: the segment text is a substring of the initial_prompt
    #      (or vice-versa), meaning Whisper is regurgitating the prompt.
    _NO_SPEECH_PROB_THRESHOLD = 0.6

    def _is_hallucination(self, segment_text: str, no_speech_prob: float, prompt: str) -> bool:
        if no_speech_prob >= self._NO_SPEECH_PROB_THRESHOLD:
            return True
        # Check if the text is just echoing the prompt (prompt-echo hallucination).
        # Normalise both strings so casing/whitespace differences don't matter.
        text_norm = segment_text.strip().lower()
        prompt_norm = prompt.strip().lower()
        if text_norm and prompt_norm:
            if text_norm in prompt_norm or prompt_norm.startswith(text_norm):
                return True
        return False

    @staticmethod
    def _preprocess(audio: np.ndarray) -> np.ndarray:
        """Remove DC offset and peak-normalise to avoid quiet/clipped input."""
        audio = audio - np.mean(audio)
        peak = np.max(np.abs(audio))
        if peak > 0.01:  # don't amplify pure silence
            audio = audio * (0.9 / peak)
        return audio

    def transcribe(self, audio: np.ndarray, is_final: bool = False) -> List[TranscriptionSegment]:
        if not self._loaded or self.model is None:
            raise RuntimeError("Whisper model is not loaded.")

        from src.config import settings

        if settings.default_language != "auto":
            transcribe_language = settings.default_language
        elif self._language != "auto":
            transcribe_language = self._language
        else:
            transcribe_language = None

        # Build prompt: static domain hint + last segment for cross-segment consistency
        prompt_parts = [settings.whisper_initial_prompt] if settings.whisper_initial_prompt else []
        if self._previous_text:
            prompt_parts.append(self._previous_text[-200:])  # last ~200 chars as context
        prompt = " ".join(prompt_parts)

        # Preprocess audio: remove DC offset, normalise volume
        audio = self._preprocess(audio)

        # Pad 200 ms of silence on each side so Whisper doesn't clip the first/last phoneme
        pad_samples = int(0.2 * 16000)
        audio = np.pad(audio, (pad_samples, pad_samples), mode="constant", constant_values=0.0)

        # Beam search for finals (accuracy), greedy for partials (low latency)
        beam_size = 5 if is_final else 1

        segments, info = self.model.transcribe(
            audio,
            language=transcribe_language,
            beam_size=beam_size,
            vad_filter=False,  # we handle VAD externally
            without_timestamps=False,
            initial_prompt=prompt or None,
            multilingual=False,
        )

        if settings.default_language != "auto":
            resolved_language = settings.default_language
        elif self._language != "auto":
            resolved_language = self._language
        else:
            resolved_language = info.language

        results = []
        for segment in segments:
            if self._is_hallucination(segment.text, segment.no_speech_prob, prompt):
                continue
            results.append(TranscriptionSegment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                is_final=is_final,
                language=resolved_language,
            ))

        # Carry the transcribed text forward for next segment's context
        if is_final and results:
            self._previous_text = " ".join(r.text for r in results)

        return results

    def unload(self) -> None:
        if self.model is not None:
            del self.model
            self.model = None
        self._loaded = False
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def get_info(self) -> EngineInfo:
        return EngineInfo(
            name=self.engine_name,
            loaded=self._loaded,
            vram_estimate_mb=self._VRAM_ESTIMATE_MB.get(self.model_name, 1200),
            supported_languages=["en"],
        )
