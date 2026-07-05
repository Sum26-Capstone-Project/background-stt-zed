# A simple test to verify our imports and config load correctly
def test_config_loads():
    from src.config import Settings
    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8764
    assert settings.default_model == "whisper"
    assert settings.whisper_model_size == "base"


def test_whisper_registered():
    from src.engines import ENGINE_REGISTRY
    from src.engines.whisper_engine import WhisperEngine

    assert ENGINE_REGISTRY["whisper"] is WhisperEngine
    assert WhisperEngine().get_info().name == "whisper"
