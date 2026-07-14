from src.engines.whisper_engine import WhisperEngine, WhisperTinyEngine, WhisperBaseEngine

ENGINE_REGISTRY = {
    "whisper_turbo": WhisperEngine,
    "whisper_tiny": WhisperTinyEngine,
    "whisper_base": WhisperBaseEngine,
}
