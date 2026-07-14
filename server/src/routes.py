import asyncio
import difflib
import json
import logging
import re
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, Query
from pydantic import BaseModel
from starlette.websockets import WebSocketState

from src.model_manager import model_manager
from src.audio import AudioProcessor
from src.config import settings

router = APIRouter()
logger = logging.getLogger(__name__)


class ModelSelectRequest(BaseModel):
    model_name: str
    language: str


@router.get("/status")
async def get_status():
    return model_manager.get_status()


@router.post("/model/select")
async def select_model(req: ModelSelectRequest):
    try:
        # Note: In a real async app, loading should be offloaded to a thread
        # to avoid blocking the event loop. For MVP, this is acceptable.
        model_manager.load_model(req.model_name, req.language)
        return {"status": "success", "message": f"Loaded {req.model_name}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/health")
async def health_check():
    return {"status": "ok"}


def _tokenize(s: str) -> list[str]:
    # lowercase, keep only alphanumeric runs -> drops punctuation/casing noise
    return re.findall(r"[a-z0-9]+", s.lower())


def phrase_in_text(phrase: str, text: str, word_threshold: float = 0.75) -> bool:
    """Fuzzy word-sequence match.

    Returns True if the phrase's words appear consecutively in the text,
    tolerating punctuation, casing and small transcription errors per word
    (e.g. "stop zed" matching "Stop, Zed." or "stop zedd").
    """
    phrase_tokens = _tokenize(phrase)
    text_tokens = _tokenize(text)

    if not phrase_tokens:
        return False
    if len(text_tokens) < len(phrase_tokens):
        return False

    n = len(phrase_tokens)
    for i in range(len(text_tokens) - n + 1):
        window = text_tokens[i:i + n]
        if all(
            difflib.SequenceMatcher(None, pw, tw).ratio() >= word_threshold
            for pw, tw in zip(phrase_tokens, window)
        ):
            return True
    return False


async def send_phrase_match(
    websocket: WebSocket,
    event_type: str,
    audio,
    segment_id: int,
    phrases: list[str],
    is_final: bool = False,
) -> bool:
    try:
        segments = await asyncio.to_thread(model_manager.active_engine.transcribe, audio, is_final, phrases)
    except Exception as e:
        await websocket.send_text(json.dumps({
            "type": "error",
            "message": f"Transcription failed: {e}",
        }))
        return False

    text = " ".join(s.text for s in segments)
    found = [phrase_in_text(p, text) for p in phrases]

    if not text.strip() and event_type == "partial":
        return True

    logger.info(
        "transcription %s segment=%s text=%r phrases=%s found=%s",
        event_type,
        segment_id,
        text,
        phrases,
        found,
    )

    payload = {
        "type": event_type,
        "text": text,
        "found": found,
        "segment_id": segment_id,
    }
    if event_type == "final":
        payload["is_final"] = True

    await websocket.send_text(json.dumps(payload))
    return True


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket, phrase: list[str] = Query(default=[])):
    await websocket.accept()

    if model_manager.active_engine is None:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "No model loaded. Please load a model first."}))
        await websocket.close()
        return

    phrases = [p for p in phrase if p.strip()]
    if not phrases:
        await websocket.send_text(
            json.dumps({"type": "error", "message": "No phrases provided. Pass one or more 'phrase' query parameters."}))
        await websocket.close()
        return

    logger.info("websocket connected phrases=%s", phrases)

    await websocket.send_text(json.dumps({"type": "status", "text": "ready", "phrases": phrases}))

    audio_processor = AudioProcessor(
        sample_rate=settings.audio_sample_rate,
        vad_threshold=settings.vad_threshold,
        partial_interval_s=settings.partial_interval_s,
        max_buffer_s=settings.max_buffer_s
    )

    segment_id = 0
    transcription_failed = False

    try:
        while True:
            message = await websocket.receive()

            if "bytes" in message:
                raw_audio = message["bytes"]
                events = audio_processor.feed(raw_audio)

                for event in events:
                    if event.type == "partial_ready":
                        # For partials, only transcribe the last 2 seconds (32000 samples)
                        # to keep inference time constant and fast.
                        partial_audio = event.audio[-32000:] if len(event.audio) > 32000 else event.audio
                        if not await send_phrase_match(websocket, "partial", partial_audio, segment_id, phrases, is_final=False):
                            transcription_failed = True
                            return
                    elif event.type == "speech_end":
                        if not await send_phrase_match(websocket, "final", event.audio, segment_id, phrases, is_final=True):
                            transcription_failed = True
                            return
                        segment_id += 1

            elif "text" in message:
                try:
                    data = json.loads(message["text"])
                    if data.get("command") == "stop":
                        break
                    elif data.get("command") == "set_phrases":
                        new_phrases = [p for p in data.get("phrases", []) if p.strip()]
                        if new_phrases:
                            phrases = new_phrases
                            await websocket.send_text(json.dumps({"type": "status", "text": "phrases_updated", "phrases": phrases}))
                except json.JSONDecodeError:
                    pass

    except WebSocketDisconnect:
        pass
    finally:
        if (
            not transcription_failed
            and websocket.client_state == WebSocketState.CONNECTED
            and len(audio_processor._buffer) > 0
        ):
            try:
                segments = await asyncio.to_thread(
                    model_manager.active_engine.transcribe, audio_processor._buffer, True
                )
                text = " ".join(s.text for s in segments)
                if text.strip():
                    found = [phrase_in_text(p, text) for p in phrases]
                    logger.info(
                        "transcription final segment=%s text=%r phrases=%s found=%s (disconnect flush)",
                        segment_id,
                        text,
                        phrases,
                        found,
                    )
                    await websocket.send_text(json.dumps({
                        "type": "final",
                        "text": text,
                        "found": found,
                        "segment_id": segment_id,
                        "is_final": True,
                    }))
            except Exception as e:
                logger.exception("Error finalizing transcription: %s", e)

        if websocket.client_state == WebSocketState.CONNECTED:
            try:
                await websocket.close()
            except RuntimeError:
                pass
