import os
from pathlib import Path
import yaml
from pydantic import BaseModel, ConfigDict
from typing import Literal


class Settings(BaseModel):
    model_config = ConfigDict(use_attribute_docstrings=True, extra="forbid")

    host: str = "127.0.0.1"
    port: int = 8764

    default_model: Literal["whisper"] = "whisper"
    default_language: str = "en"

    # faster-whisper model size: tiny/base/small/medium/large-v3.
    # tiny was too weak in practice; base is the new default.
    whisper_model_size: str = "base"

    whisper_initial_prompt: str = ""

    vad_threshold: float = 0.5
    audio_sample_rate: int = 16000
    audio_chunk_ms: int = 100
    partial_interval_s: float = 1.0
    max_buffer_s: float = 30.0

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        if not path.exists():
            return cls()
        with open(path, encoding="utf-8") as f:
            yaml_config: dict = yaml.safe_load(f) or {}
            yaml_config.pop("$schema", None)
        return cls.model_validate(yaml_config)


settings_path = os.getenv("SETTINGS_PATH", "settings.yaml")
settings = Settings.from_yaml(Path(settings_path))
