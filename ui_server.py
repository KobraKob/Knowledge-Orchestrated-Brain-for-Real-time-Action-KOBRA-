"""
ui_server.py — Local web UI server for KOBRA.

Serves the sci-fi dashboard on http://127.0.0.1:7474
and bridges voice-loop events to the browser via WebSocket.

Thread-safe: post_event() can be called from any thread.
The browser can send text commands back via WebSocket.
"""

import asyncio
import json
import logging
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

# ── Module-level state ─────────────────────────────────────────────────────────
_clients: set = set()
_event_loop: asyncio.AbstractEventLoop | None = None
_event_queue: asyncio.Queue | None = None
_server_started = threading.Event()

UI_HOST = "127.0.0.1"
UI_PORT = 7474
UI_URL  = f"http://{UI_HOST}:{UI_PORT}"


# ── Public API (callable from any thread) ──────────────────────────────────────

def post_event(event_type: str, **kwargs) -> None:
    """
    Broadcast a named event to all connected browser clients.
    Thread-safe — call from the voice loop, agents, anywhere.
    """
    global _event_loop, _event_queue
    if _event_loop is None or _event_loop.is_closed() or _event_queue is None:
        return
    payload = json.dumps({"type": event_type, **kwargs})
    try:
        _event_loop.call_soon_threadsafe(_event_queue.put_nowait, payload)
    except Exception:
        pass


# ── FastAPI app ────────────────────────────────────────────────────────────────

def _create_app():
    from fastapi import FastAPI, WebSocket, WebSocketDisconnect
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="KOBRA UI")

    @app.get("/")
    async def index():
        html_path = Path(__file__).parent / "static" / "index.html"
        return HTMLResponse(html_path.read_text(encoding="utf-8"))

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        await ws.accept()
        _clients.add(ws)
        logger.info("[UI] Browser connected — %d client(s)", len(_clients))
        try:
            while True:
                raw = await ws.receive_text()
                msg = json.loads(raw)
                if msg.get("type") == "command":
                    text = msg.get("text", "").strip()
                    if text:
                        from kobra_events import ui_command_queue
                        ui_command_queue.put(text)
        except WebSocketDisconnect:
            pass
        except Exception as exc:
            logger.debug("[UI] WS error: %s", exc)
        finally:
            _clients.discard(ws)
            logger.info("[UI] Browser disconnected — %d client(s)", len(_clients))

    return app


# ── Internal async tasks ───────────────────────────────────────────────────────

async def _broadcaster():
    """Drain the event queue and send to all connected clients."""
    while True:
        payload = await _event_queue.get()
        dead = []
        for client in list(_clients):
            try:
                await client.send_text(payload)
            except Exception:
                dead.append(client)
        for d in dead:
            _clients.discard(d)


async def _stats_ticker():
    """Emit system stats to the UI every 3 seconds."""
    import psutil
    from datetime import datetime
    while True:
        await asyncio.sleep(3)
        try:
            cpu  = psutil.cpu_percent(interval=None)
            ram  = psutil.virtual_memory()
            bat  = psutil.sensors_battery()
            post_event(
                "system_stats",
                cpu=f"{cpu:.0f}",
                ram=f"{ram.percent:.0f}",
                ram_used=f"{ram.used / 1e9:.1f}",
                ram_total=f"{ram.total / 1e9:.1f}",
                battery=f"{bat.percent:.0f}" if bat else None,
                charging=bat.power_plugged if bat else False,
                clock=datetime.now().strftime("%H:%M:%S"),
            )
        except Exception:
            pass


# ── Server thread entry point ──────────────────────────────────────────────────

def run(host: str = UI_HOST, port: int = UI_PORT) -> None:
    """
    Start the FastAPI server in the calling thread.
    Designed to run inside a daemon thread from main.py.
    """
    global _event_loop, _event_queue
    import uvicorn

    _event_loop = asyncio.new_event_loop()
    asyncio.set_event_loop(_event_loop)
    _event_queue = asyncio.Queue()

    app = _create_app()

    _event_loop.create_task(_broadcaster())
    _event_loop.create_task(_stats_ticker())

    uvicorn_config = uvicorn.Config(
        app,
        host=host,
        port=port,
        loop="none",
        log_level="error",
        access_log=False,
    )
    server = uvicorn.Server(uvicorn_config)

    logger.info("[UI] Dashboard → %s", UI_URL)
    _server_started.set()
    _event_loop.run_until_complete(server.serve())
