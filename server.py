import os
import uuid
import asyncio
import json
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

app = FastAPI(title="Ollama Relay Server")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Shared secret so random people can't connect a fake "worker" to your relay.
# Set this in Render's environment variables and in your laptop worker's env too.
WORKER_TOKEN ="secret"

# Only supporting a single connected laptop worker for now.
worker_socket: WebSocket | None = None
worker_lock = asyncio.Lock()

# job_id -> Future that resolves when the worker sends back a result
pending_jobs: dict[str, asyncio.Future] = {}

JOB_TIMEOUT_SECONDS = 120


class AskRequest(BaseModel):
    prompt: str
    model: str | None = None  # optional override, else worker uses its default


@app.get("/health")
async def health():
    return {"status": "ok", "worker_connected": worker_socket is not None}


@app.websocket("/ws/worker")
async def worker_endpoint(websocket: WebSocket):
    global worker_socket

    await websocket.accept()

    # Simple auth: first message from the worker must be the shared token
    try:
        auth_msg = await asyncio.wait_for(websocket.receive_text(), timeout=10)
        auth_data = json.loads(auth_msg)
        if auth_data.get("token") != WORKER_TOKEN:
            await websocket.close(code=4001)
            return
    except Exception:
        await websocket.close(code=4001)
        return

    async with worker_lock:
        worker_socket = websocket
    print("Laptop worker connected.")

    try:
        while True:
            raw = await websocket.receive_text()
            data = json.loads(raw)
            job_id = data.get("job_id")
            future = pending_jobs.pop(job_id, None)
            if future and not future.done():
                future.set_result(data)
    except WebSocketDisconnect:
        print("Laptop worker disconnected.")
    finally:
        async with worker_lock:
            if worker_socket is websocket:
                worker_socket = None


@app.post("/ask")
async def ask_endpoint(request: AskRequest):
    if worker_socket is None:
        raise HTTPException(status_code=503, detail="No laptop worker is currently connected.")

    job_id = str(uuid.uuid4())
    future = asyncio.get_event_loop().create_future()
    pending_jobs[job_id] = future

    job_payload = {
        "job_id": job_id,
        "prompt": request.prompt,
        "model": request.model,
    }

    try:
        await worker_socket.send_text(json.dumps(job_payload))
    except Exception as e:
        pending_jobs.pop(job_id, None)
        raise HTTPException(status_code=503, detail=f"Failed to reach worker: {str(e)}")

    try:
        result = await asyncio.wait_for(future, timeout=JOB_TIMEOUT_SECONDS)
    except asyncio.TimeoutError:
        pending_jobs.pop(job_id, None)
        raise HTTPException(status_code=504, detail="Worker did not respond in time.")

    if result.get("error"):
        raise HTTPException(status_code=500, detail=result["error"])

    return {"response": result.get("response")}


if __name__ == "__main__":
    import uvicorn

    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("server:app", host="0.0.0.0", port=port)
