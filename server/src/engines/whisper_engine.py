import gc
import ctypes
import re
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
    engine_name = "whisper_turbo"
    model_name = "large-v3-turbo"
    vram_estimate_mb = 6000
    transcribe_language = "en"

    def __init__(self):
        self.model = None
        self._language = self.transcribe_language
        self._loaded = False
        self._device = "cpu"
        self._compute_type = "int8"

    def load(self, _language: str) -> None:
        self._language = self.transcribe_language

        if _cuda_runtime_available():
            self._device = "cuda"
            self._compute_type = "float16"
        else:
            self._device = "cpu"
            self._compute_type = "int8"

        print(f"Loading {self.engine_name} ({self.model_name}) on {self._device} ({self._compute_type})...")
        self.model = WhisperModel(self.model_name, device=self._device, compute_type=self._compute_type)
        self._loaded = True

    _NO_SPEECH_PROB_THRESHOLD = 0.5
    _MIN_AUDIO_SAMPLES = 3200  # 0.2s at 16kHz — reject extremely short clips

    def _is_hallucination(self, segment_text: str, no_speech_prob: float,
                          prompt: str, avg_logprob: float) -> bool:
        """Detect hallucinated segments using multiple signals."""
        if no_speech_prob >= self._NO_SPEECH_PROB_THRESHOLD:
            return True

        # Low-confidence output is likely hallucination
        if avg_logprob < -1.0:
            return True

        text_norm = segment_text.strip().lower()

        # Repetition pattern: "no, no, no" or "I don't know, I don't know"
        # Split into words and check if the same short sequence repeats
        words = re.findall(r"[a-z']+", text_norm)
        if len(words) >= 4:
            # Check for single-word repetition (e.g. "no no no no")
            unique_words = set(words)
            if len(unique_words) <= 2 and len(words) >= 4:
                return True
            # Check for repeated bigrams/trigrams covering most of the text
            for ngram_size in (2, 3):
                if len(words) >= ngram_size * 2:
                    ngrams = [tuple(words[i:i+ngram_size]) for i in range(len(words) - ngram_size + 1)]
                    most_common_count = max(ngrams.count(ng) for ng in set(ngrams))
                    if most_common_count >= 3:
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

    def transcribe(self, audio: np.ndarray, is_final: bool = False, prompt_phrases: list[str] | None = None) -> List[TranscriptionSegment]:
        if not self._loaded or self.model is None:
            raise RuntimeError("Whisper model is not loaded.")

        from src.config import settings

        transcribe_language = self.transcribe_language

        # Build prompt dynamically from provided phrases, falling back to config
        if prompt_phrases:
            prompt = ", ".join(prompt_phrases)
        else:
            prompt = settings.whisper_initial_prompt or None

        # Preprocess audio: remove DC offset, normalise volume
        audio = self._preprocess(audio)

        # Reject audio that is too short or too quiet
        if len(audio) < self._MIN_AUDIO_SAMPLES:
            return []
        rms = np.sqrt(np.mean(audio ** 2))
        if rms < 0.02:
            return []

        # Pad 200 ms of silence on each side so Whisper doesn't clip the first/last phoneme
        pad_samples = int(0.2 * 16000)
        audio = np.pad(audio, (pad_samples, pad_samples), mode="constant", constant_values=0.0)

        # Beam search for finals (accuracy), greedy for partials (low latency)
        beam_size = 5 if is_final else 1

        segments, _info = self.model.transcribe(
            audio,
            language=transcribe_language,
            beam_size=beam_size,
            vad_filter=False,  # we handle VAD externally
            without_timestamps=False,
            initial_prompt=prompt,
            multilingual=False,
            # Anti-hallucination parameters
            condition_on_previous_text=False,
            repetition_penalty=1.2,
            no_repeat_ngram_size=3,
            compression_ratio_threshold=2.0,
            log_prob_threshold=-0.8,
            no_speech_threshold=0.5,
            temperature=0.0,
        )

        resolved_language = self.transcribe_language

        results = []
        for segment in segments:
            if self._is_hallucination(segment.text, segment.no_speech_prob,
                                      prompt or "", segment.avg_logprob):
                continue
            results.append(TranscriptionSegment(
                text=segment.text,
                start=segment.start,
                end=segment.end,
                is_final=is_final,
                language=resolved_language,
            ))

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
            vram_estimate_mb=self.vram_estimate_mb,
            supported_languages=["en"],
        )


class WhisperTinyEngine(WhisperEngine):
    engine_name = "whisper_tiny"
    model_name = "tiny.en"
    vram_estimate_mb = 1000


class WhisperBaseEngine(WhisperEngine):
    """Lightweight engine using base.en — good balance of speed and accuracy
    for short phrase/keyword detection. Much less hallucination-prone than
    tiny.en while still being very fast on CPU."""
    engine_name = "whisper_base"
    model_name = "base.en"
    vram_estimate_mb = 1500
