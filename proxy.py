"""
Load-balancing proxy with request queue for LTX-2 workers.

Routes requests to available workers, queues when all busy.
"""

import asyncio
import time
import os
from typing import Optional
from dataclasses import dataclass, field
from collections import deque

import httpx
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field


# Configuration
WORKER_URLS = os.environ.get("WORKER_URLS", "http://worker-1:8000,http://worker-2:8000,http://worker-3:8000").split(",")
MAX_QUEUE_SIZE = int(os.environ.get("MAX_QUEUE_SIZE", "100"))
REQUEST_TIMEOUT = int(os.environ.get("REQUEST_TIMEOUT", "600"))  # 10 minutes max
HEALTH_CHECK_INTERVAL = int(os.environ.get("HEALTH_CHECK_INTERVAL", "10"))


@dataclass
class Worker:
    url: str
    healthy: bool = True
    busy: bool = False
    current_request_id: Optional[str] = None
    requests_completed: int = 0
    last_health_check: float = 0


@dataclass
class QueuedRequest:
    id: str
    data: dict
    created_at: float = field(default_factory=time.time)
    future: asyncio.Future = field(default_factory=lambda: asyncio.get_event_loop().create_future())


app = FastAPI(
    title="LTX-2 Video Generation Proxy",
    description="Load-balancing proxy for LTX-2 video generation workers",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# State
workers: list[Worker] = []
request_queue: deque[QueuedRequest] = deque(maxlen=MAX_QUEUE_SIZE)
request_counter = 0
queue_processor_task: Optional[asyncio.Task] = None


@app.on_event("startup")
async def startup():
    global workers, queue_processor_task

    # Initialize workers
    workers = [Worker(url=url.strip()) for url in WORKER_URLS if url.strip()]
    print(f"Initialized {len(workers)} workers: {[w.url for w in workers]}")

    # Start background tasks
    queue_processor_task = asyncio.create_task(process_queue())
    asyncio.create_task(health_check_loop())


@app.on_event("shutdown")
async def shutdown():
    if queue_processor_task:
        queue_processor_task.cancel()


async def health_check_loop():
    """Periodically check worker health and busy status."""
    async with httpx.AsyncClient(timeout=5.0) as client:
        while True:
            await asyncio.sleep(HEALTH_CHECK_INTERVAL)
            for worker in workers:
                try:
                    # Check health
                    resp = await client.get(f"{worker.url}/health")
                    worker.healthy = resp.status_code == 200
                    worker.last_health_check = time.time()

                    # Check if busy (sync local state with actual worker state)
                    if worker.healthy:
                        busy_resp = await client.get(f"{worker.url}/busy")
                        if busy_resp.status_code == 200:
                            actual_busy = busy_resp.json().get("busy", False)
                            # Only update if our state says free but worker says busy
                            # (handles cases where proxy state got out of sync)
                            if not worker.busy and actual_busy:
                                worker.busy = True
                            elif worker.busy and not actual_busy:
                                # Worker finished but we didn't get the response
                                worker.busy = False
                except Exception:
                    worker.healthy = False


def get_available_worker() -> Optional[Worker]:
    """Get a healthy, non-busy worker."""
    for worker in workers:
        if worker.healthy and not worker.busy:
            return worker
    return None


async def process_queue():
    """Background task to process queued requests."""
    while True:
        # Check if we have queued requests and available workers
        if request_queue:
            worker = get_available_worker()
            if worker:
                queued = request_queue.popleft()
                # Mark as busy BEFORE starting task to prevent race conditions
                worker.busy = True
                worker.current_request_id = queued.id
                asyncio.create_task(execute_request(worker, queued))

        await asyncio.sleep(0.1)  # Check every 100ms


async def execute_request(worker: Worker, queued: QueuedRequest):
    """Execute a request on a worker. Worker should already be marked busy."""
    try:
        async with httpx.AsyncClient(timeout=REQUEST_TIMEOUT) as client:
            print(f"[{queued.id}] Sending to {worker.url}")
            start = time.time()

            resp = await client.post(
                f"{worker.url}/generate",
                json=queued.data,
                timeout=REQUEST_TIMEOUT,
            )

            elapsed = time.time() - start
            print(f"[{queued.id}] Completed in {elapsed:.1f}s on {worker.url}")

            worker.requests_completed += 1

            if resp.status_code == 200:
                queued.future.set_result(resp.json())
            else:
                queued.future.set_exception(
                    HTTPException(status_code=resp.status_code, detail=resp.text)
                )
    except asyncio.CancelledError:
        queued.future.set_exception(HTTPException(status_code=503, detail="Request cancelled"))
    except Exception as e:
        print(f"[{queued.id}] Error: {e}")
        queued.future.set_exception(HTTPException(status_code=500, detail=str(e)))
    finally:
        worker.busy = False
        worker.current_request_id = None


@app.get("/health")
async def health():
    """Health check - reports proxy and worker status."""
    healthy_workers = sum(1 for w in workers if w.healthy)
    available_workers = sum(1 for w in workers if w.healthy and not w.busy)

    return {
        "status": "healthy" if healthy_workers > 0 else "degraded",
        "workers": {
            "total": len(workers),
            "healthy": healthy_workers,
            "available": available_workers,
            "busy": healthy_workers - available_workers,
        },
        "queue": {
            "size": len(request_queue),
            "max_size": MAX_QUEUE_SIZE,
        },
    }


@app.get("/")
async def root():
    """API info."""
    return {
        "name": "LTX-2 Video Generation Proxy",
        "workers": len(workers),
        "queue_size": len(request_queue),
        "endpoints": {
            "POST /generate": "Generate video (queued if workers busy)",
            "GET /health": "Health and status check",
            "GET /status": "Detailed worker status",
        },
    }


@app.get("/status")
async def status():
    """Detailed status of all workers."""
    return {
        "workers": [
            {
                "url": w.url,
                "healthy": w.healthy,
                "busy": w.busy,
                "current_request": w.current_request_id,
                "requests_completed": w.requests_completed,
            }
            for w in workers
        ],
        "queue": {
            "size": len(request_queue),
            "requests": [
                {
                    "id": r.id,
                    "waiting_seconds": round(time.time() - r.created_at, 1),
                }
                for r in request_queue
            ],
        },
    }


@app.post("/generate")
async def generate(request: Request):
    """
    Generate video - routes to available worker or queues if all busy.

    Returns immediately if worker available, otherwise queues and waits.
    """
    global request_counter

    # Parse request body
    try:
        data = await request.json()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid JSON body")

    if "prompt" not in data:
        raise HTTPException(status_code=400, detail="Missing required field: prompt")

    # Check queue capacity
    if len(request_queue) >= MAX_QUEUE_SIZE:
        raise HTTPException(
            status_code=503,
            detail=f"Queue full ({MAX_QUEUE_SIZE} requests). Try again later."
        )

    # Create queued request
    request_counter += 1
    request_id = f"req-{request_counter:06d}"
    queued = QueuedRequest(id=request_id, data=data)

    # Try to get an available worker immediately
    worker = get_available_worker()
    if worker:
        # Mark as busy BEFORE starting task to prevent race conditions
        worker.busy = True
        worker.current_request_id = request_id
        # Execute immediately
        asyncio.create_task(execute_request(worker, queued))
        print(f"[{request_id}] Assigned immediately to {worker.url}")
    else:
        # Queue the request
        request_queue.append(queued)
        queue_pos = len(request_queue)
        print(f"[{request_id}] Queued at position {queue_pos}")

    # Wait for result
    try:
        result = await asyncio.wait_for(queued.future, timeout=REQUEST_TIMEOUT)
        return JSONResponse(content=result)
    except asyncio.TimeoutError:
        raise HTTPException(status_code=504, detail="Request timed out")


if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", "8080"))
    uvicorn.run(app, host="0.0.0.0", port=port)
