# background-realtime-stt-server

Local FastAPI + WebSocket real-time STT server, streaming 16kHz mono PCM16 audio through Silero VAD segmentation into a single `whisper` (faster-whisper) engine. Adapted from [ZolotarevAlexandr/realtime_stt](https://github.com/ZolotarevAlexandr/realtime_stt), trimmed to one configurable Whisper engine instead of the turbo/small/qwen3 lineup.

Model size is set via `whisper_model_size` in `settings.yaml` (`tiny`/`base`/`small`/`medium`/`large-v3`) — default is `base`; `tiny` turned out too weak in practice.

## Run standalone (without Zed)

```bash
./run-local.sh
```

Creates `.venv`, installs deps from `pyproject.toml`, copies `settings.example.yaml` to `settings.yaml` if missing, starts the server on `127.0.0.1:8764`.

## Test

```bash
.venv/bin/python -m tests.tester --file test.wav --language en
# or from the mic (needs pyaudio + portaudio):
.venv/bin/python -m tests.tester --language ru
```

## API

- `GET /health` — liveness check.
- `GET /status` — currently loaded model.
- `POST /model/select` — `{"model_name": "whisper", "language": "ru"}`.
- `WS /ws/stream` — send raw 16kHz mono PCM16 chunks, receive `{"type": "partial"|"final", "text": ...}` events.
