"""Serveur neuro-lens : sert la page web et analyse le code envoyé par l'éditeur.

Lancement : python -m server.app   puis ouvrir http://127.0.0.1:8000
"""

from __future__ import annotations

import multiprocessing as mp
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
TIMEOUT_S = 10


def _worker_loop(conn) -> None:
    from server.analyzer import analyze

    conn.send("ready")  # importer PyTorch peut prendre plusieurs secondes : hors du délai
    while True:
        code, pixels = conn.recv()
        try:
            conn.send(analyze(code, pixels))
        except Exception as exc:  # noqa: BLE001
            conn.send({"ok": False, "error": str(exc), "line": None})


class Worker:
    """Exécute le code dans un processus séparé : une boucle infinie ne bloque pas le serveur."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._start()

    def _start(self) -> None:
        ctx = mp.get_context("spawn")
        self._conn, child = ctx.Pipe()
        self._proc = ctx.Process(target=_worker_loop, args=(child,), daemon=True)
        self._proc.start()
        self._ready = False

    def run(self, code: str, pixels) -> dict:
        with self._lock:
            if not self._proc.is_alive():
                self._start()
            if not self._ready:
                self._conn.recv()
                self._ready = True
            self._conn.send((code, pixels))
            if self._conn.poll(TIMEOUT_S):
                return self._conn.recv()
            self._proc.kill()
            self._proc.join()
            self._start()
            return {"ok": False, "error": f"Le code a mis plus de {TIMEOUT_S} s : boucle infinie ?", "line": None}


class AnalyzeRequest(BaseModel):
    code: str
    pixels: list[list[float]] | None = None


worker: Worker | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global worker
    worker = Worker()
    yield


app = FastAPI(title="neuro-lens", lifespan=lifespan)


@app.post("/api/analyze")
def analyze_endpoint(req: AnalyzeRequest) -> dict:
    return worker.run(req.code, req.pixels)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
