# A simple test to verify our imports and config load correctly
def test_config_loads():
    from src.config import Settings
    settings = Settings()
    assert settings.host == "127.0.0.1"
    assert settings.port == 8764
    assert settings.default_model == "whisper_base"
    assert settings.default_language == "en"


def test_whisper_registered():
    from src.engines import ENGINE_REGISTRY
    from src.engines.whisper_engine import WhisperBaseEngine

    assert ENGINE_REGISTRY["whisper_base"] is WhisperBaseEngine
    assert WhisperBaseEngine().get_info().name == "whisper_base"


def test_config_migrates_legacy_whisper_model(tmp_path):
    from src.config import Settings

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text(
        'default_model: "whisper"\nwhisper_model_size: "base"\n',
        encoding="utf-8",
    )
    settings = Settings.from_yaml(settings_path)
    assert settings.default_model == "whisper_base"
    assert settings.default_language == "en"


def test_config_migrates_non_english_language_to_en(tmp_path):
    from src.config import Settings

    settings_path = tmp_path / "settings.yaml"
    settings_path.write_text('default_language: "ru"\n', encoding="utf-8")
    settings = Settings.from_yaml(settings_path)
    assert settings.default_language == "en"
