from abc import ABC, abstractmethod
from typing import List
import numpy as np
from dataclasses import dataclass


@dataclass
class TranscriptionSegment:
    text: str
    start: float
    end: float
    is_final: bool
    language: str


@dataclass
class EngineInfo:
    name: str
    loaded: bool
    vram_estimate_mb: int
    supported_languages: List[str]


class STTEngine(ABC):
    @abstractmethod
    def load(self, language: str) -> None:
        pass

    @abstractmethod
    def transcribe(self, audio: np.ndarray, is_final: bool = False, prompt_phrases: list[str] | None = None) -> List[TranscriptionSegment]:
        pass

    @abstractmethod
    def unload(self) -> None:
        pass

    @abstractmethod
    def get_info(self) -> EngineInfo:
        pass
