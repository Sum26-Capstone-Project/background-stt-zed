import asyncio
import json
import argparse
import sys
import wave
import websockets
from websockets.exceptions import ConnectionClosed


def clear_line():
    sys.stdout.write('\r\033[K')
    sys.stdout.flush()


async def audio_stream_task(websocket, file_path=None, chunk_ms=100, sample_rate=16000):
    if file_path:
        with wave.open(file_path, 'rb') as wf:
            if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getframerate() != sample_rate:
                print(f"Error: WAV file must be mono, 16-bit, {sample_rate}Hz")
                return

            chunk_size = int(sample_rate * (chunk_ms / 1000.0))
            print(f"Streaming from {file_path}...")

            while True:
                data = wf.readframes(chunk_size)
                if not data:
                    break
                await websocket.send(data)
                await asyncio.sleep(chunk_ms / 1000.0)

            # Send stop command
            await websocket.send(json.dumps({"command": "stop"}))
    else:
        try:
            import pyaudio
        except ImportError:
            print("Error: pyaudio is not installed. Please install it to use live microphone.")
            return

        p = pyaudio.PyAudio()
        chunk_size = int(sample_rate * (chunk_ms / 1000.0))
        stream = p.open(format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        input=True,
                        frames_per_buffer=chunk_size)

        print("Listening... Press Ctrl+C to stop.")
        try:
            while True:
                data = stream.read(chunk_size, exception_on_overflow=False)
                await websocket.send(data)
                # yield to event loop
                await asyncio.sleep(0.001)
        except (asyncio.CancelledError, ConnectionClosed):
            pass
        finally:
            stream.stop_stream()
            stream.close()
            p.terminate()
            try:
                await websocket.send(json.dumps({"command": "stop"}))
            except ConnectionClosed:
                pass


async def receive_task(websocket):
    try:
        while True:
            message = await websocket.recv()
            data = json.loads(message)

            if data["type"] == "partial":
                clear_line()
                text = data["text"]
                sys.stdout.write(f"\033[90mPartial:\033[0m {text}")
                sys.stdout.flush()
            elif data["type"] == "final":
                clear_line()
                text = data["text"]
                print(f"\033[92mFinal:\033[0m {text}")
            elif data["type"] == "status":
                pass  # Ignore ready status here
            elif data["type"] == "error":
                print(f"\nError from server: {data['message']}")
                break
    except ConnectionClosed as e:
        reason = f" ({e.reason})" if e.reason else ""
        print(f"\nConnection closed: {e.code}{reason}")


async def main(args):
    import httpx

    # Optional: Select model first
    if args.model:
        print(f"Selecting model {args.model} ({args.language})...")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"http://{args.host}:{args.port}/model/select",
                json={"model_name": args.model, "language": args.language},
                timeout=60.0  # Model load can take a while
            )
            if resp.status_code != 200:
                print(f"Failed to select model: {resp.text}")
                return
            print("Model selected successfully.")

    uri = f"ws://{args.host}:{args.port}/ws/stream"
    print(f"Connecting to {uri}...")

    try:
        async with websockets.connect(uri) as websocket:
            # Wait for ready signal
            message = await websocket.recv()
            data = json.loads(message)
            if data["type"] != "status" or data["text"] != "ready":
                print("Failed to initialize stream.")
                return

            stream_task = asyncio.create_task(audio_stream_task(websocket, args.file))
            recv_task = asyncio.create_task(receive_task(websocket))

            done, pending = await asyncio.wait(
                {stream_task, recv_task},
                return_when=asyncio.FIRST_COMPLETED,
            )
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)
            for task in done:
                task.result()
    except Exception as e:
        print(f"Error: {e}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Terminal Test Client for Background Realtime STT")
    parser.add_argument("--host", default="127.0.0.1", help="Server host")
    parser.add_argument("--port", default=8764, type=int, help="Server port")
    parser.add_argument("--model", type=str, default="whisper", help="Model to select before streaming")
    parser.add_argument("--language", default="en", help="Language code")
    parser.add_argument("--file", type=str, help="Path to a 16kHz mono WAV file to stream instead of mic")

    args = parser.parse_args()

    try:
        asyncio.run(main(args))
    except KeyboardInterrupt:
        print("\nExiting...")
