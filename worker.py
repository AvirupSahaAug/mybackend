import os
import json
import asyncio
import websockets
import requests

# --- Configuration (edit these or set as env vars) ---
RENDER_WS_URL = os.environ.get("RENDER_WS_URL", "wss://your-app.onrender.com/ws/worker")
WORKER_TOKEN = os.environ.get("WORKER_TOKEN", "change-me")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://localhost:11434/api/generate")
DEFAULT_MODEL = os.environ.get("OLLAMA_MODEL", "gemma3:4b")  # <-- update this to your model name

RECONNECT_DELAY_SECONDS = 5


def run_ollama(prompt: str, model: str) -> str:
    """Blocking call to local Ollama."""
    response = requests.post(
        OLLAMA_URL,
        json={"model": model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    data = response.json()
    return data.get("response", "")


async def handle_job(websocket, job: dict):
    job_id = job["job_id"]
    prompt = job["prompt"]
    model = job.get("model") or DEFAULT_MODEL

    print(f"[job {job_id}] prompt: {prompt[:80]}...")

    try:
        loop = asyncio.get_event_loop()
        result_text = await loop.run_in_executor(None, run_ollama, prompt, model)
        reply = {"job_id": job_id, "response": result_text}
    except Exception as e:
        reply = {"job_id": job_id, "error": str(e)}

    await websocket.send(json.dumps(reply))
    print(f"[job {job_id}] done.")


async def worker_loop():
    while True:
        try:
            print(f"Connecting to {RENDER_WS_URL} ...")
            async with websockets.connect(RENDER_WS_URL) as websocket:
                # Authenticate first
                await websocket.send(json.dumps({"token": WORKER_TOKEN}))
                print("Connected and authenticated. Waiting for jobs...")

                async for raw_message in websocket:
                    job = json.loads(raw_message)
                    asyncio.create_task(handle_job(websocket, job))

        except (websockets.exceptions.ConnectionClosed, OSError) as e:
            print(f"Connection lost ({e}). Reconnecting in {RECONNECT_DELAY_SECONDS}s...")
            await asyncio.sleep(RECONNECT_DELAY_SECONDS)


if __name__ == "__main__":
    asyncio.run(worker_loop())
