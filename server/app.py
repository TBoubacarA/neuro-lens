"""Serveur neuro-lens : sert la page web et analyse le code envoyé par l'éditeur.

Lancement : python -m server.app   puis ouvrir http://127.0.0.1:8000
"""

from __future__ import annotations

import multiprocessing as mp
import os
import threading
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
TIMEOUT_S = 10
LIVE_TIMEOUT_S = 120  # un agent piloté par un vrai modèle de langage attend ses réponses


def _worker_loop(conn) -> None:
    from server import trainer
    from server import mlops
    from server.analyzer import analyze, attack, deploy, dream, landscape, train

    conn.send("ready")  # importer PyTorch peut prendre plusieurs secondes : hors du délai
    trainer.preload()
    while True:
        op, code, pixels, text, off, epoch, override = conn.recv()
        try:
            if op == "train":
                conn.send(train(code))
            elif op == "dream":
                conn.send(dream(code))
            elif op == "attack":
                conn.send(attack(code, pixels, epoch))  # epoch porte ici le chiffre visé
            elif op == "landscape":
                conn.send(landscape(code))
            elif op == "deploy":
                conn.send(deploy(code, override))
            elif op == "predict":  # `override` porte ici la demande
                conn.send(mlops.predict(override["rows"]))
            elif op == "simulate":
                conn.send(mlops.simulate(override["n"], override["drift"]))
            elif op == "live":  # le code peut appeler un vrai modèle de langage (voir l'exemple agent)
                os.environ["NEURO_LENS_LIVE"] = "1"
                try:
                    conn.send(analyze(code, pixels, text, off, epoch, override))
                finally:
                    os.environ.pop("NEURO_LENS_LIVE", None)
            elif op == "reset":
                trainer.reset(code)
                conn.send({"ok": True})
            else:
                conn.send(analyze(code, pixels, text, off, epoch, override))
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

    def run(self, code: str, pixels, text=None, op: str = "analyze", off=None, epoch=None, override=None) -> dict:
        with self._lock:
            if not self._proc.is_alive():
                self._start()
            if not self._ready:
                self._conn.recv()
                self._ready = True
            self._conn.send((op, code, pixels, text, off, epoch, override))
            limit = LIVE_TIMEOUT_S if op == "live" else TIMEOUT_S
            if self._conn.poll(limit):
                return self._conn.recv()
            self._proc.kill()
            self._proc.join()
            self._start()
            return {"ok": False, "error": f"Le code a mis plus de {limit} s : boucle infinie ?", "line": None}


class AnalyzeRequest(BaseModel):
    code: str
    pixels: list[list[float]] | None = None
    text: str | None = None
    off: list[list[int]] | None = None  # [couche, unité] éteintes à la main (« et si ? »)
    live: bool = False  # True : le code a le droit d'appeler un vrai modèle (payant)
    epoch: int | None = None  # montrer le réseau tel qu'il était après N époques
    override: dict[str, float | int | None] | None = None  # réglage scikit-learn choisi au curseur


worker: Worker | None = None


@asynccontextmanager
async def lifespan(_app: FastAPI):
    global worker
    worker = Worker()
    yield


app = FastAPI(title="neuro-lens", lifespan=lifespan)


@app.post("/api/analyze")
def analyze_endpoint(req: AnalyzeRequest) -> dict:
    return worker.run(req.code, req.pixels, req.text, op="live" if req.live else "analyze",
                      off=req.off, epoch=req.epoch, override=req.override)


class CodeRequest(BaseModel):
    code: str


@app.post("/api/train")
def train_endpoint(req: CodeRequest) -> dict:
    """Une époque d'entraînement ; les analyses suivantes de ce code montrent le réseau entraîné."""
    return worker.run(req.code, None, op="train")


@app.post("/api/dream")
def dream_endpoint(req: CodeRequest) -> dict:
    """Pour chaque chiffre, l'image que le réseau trouve la plus convaincante."""
    return worker.run(req.code, None, op="dream")


class AttackRequest(BaseModel):
    code: str
    pixels: list[list[float]] | None = None
    target: int


@app.post("/api/attack")
def attack_endpoint(req: AttackRequest) -> dict:
    """Retouche à peine le dessin pour que le réseau y lise le chiffre visé."""
    return worker.run(req.code, req.pixels, op="attack", epoch=req.target)


@app.post("/api/landscape")
def landscape_endpoint(req: CodeRequest) -> dict:
    """Le paysage de l'erreur autour des poids actuels, avec le chemin de l'entraînement."""
    return worker.run(req.code, None, op="landscape")


class DeployRequest(BaseModel):
    code: str
    override: dict[str, float | int | None] | None = None


@app.post("/api/deploy")
def deploy_endpoint(req: DeployRequest) -> dict:
    """Fige le modèle scikit-learn du code dans une nouvelle version et la met en production."""
    return worker.run(req.code, None, op="deploy", override=req.override)


class PredictRequest(BaseModel):
    rows: list[list[float]]


@app.post("/api/predict")
def predict_endpoint(req: PredictRequest) -> dict:
    """Le service de prédiction : répond avec la version en production."""
    return worker.run("", None, op="predict", override={"rows": req.rows})


class SimulateRequest(BaseModel):
    n: int = 200
    drift: float = 0.0


@app.post("/api/simulate")
def simulate_endpoint(req: SimulateRequest) -> dict:
    """Envoie de fausses requêtes de production, plus ou moins éloignées des données d'entraînement."""
    return worker.run("", None, op="simulate", override={"n": req.n, "drift": req.drift})


@app.post("/api/infra/build")
def infra_build_endpoint() -> dict:
    """Construit l'image Docker de la version en production et lance son conteneur."""
    from server import infra

    return infra.build()


@app.get("/api/infra/status")
def infra_status_endpoint() -> dict:
    from server import infra

    return infra.status()


@app.post("/api/infra/stop")
def infra_stop_endpoint() -> dict:
    from server import infra

    return infra.stop()


@app.post("/api/infra/predict")
def infra_predict_endpoint(req: PredictRequest) -> dict:
    """La même demande de prédiction, mais envoyée au conteneur."""
    from server import infra

    return infra.predict(req.rows)


@app.post("/api/workspace/test")
def workspace_test_endpoint() -> dict:
    """Lance pytest dans le projet écrit autour du code de l'éditeur."""
    from server import workspace

    return workspace.run_tests()


@app.post("/api/workspace/open")
def workspace_open_endpoint() -> dict:
    """Ouvre ce projet dans VS Code."""
    from server import workspace

    return workspace.open_in_editor()


@app.post("/api/reset")
def reset_endpoint(req: CodeRequest) -> dict:
    return worker.run(req.code, None, op="reset")


@app.get("/api/project")
def project_endpoint(root: str = "app") -> dict:
    """La vue Projet : l'arborescence des fichiers avec leur rôle, et le graphe des modules du serveur."""
    from server import project

    return project.describe(root)


@app.get("/api/project/file")
def project_file_endpoint(root: str = "app", path: str = "") -> dict:
    """Un fichier du projet, en lecture seule, avec son rôle, ses imports et son plan."""
    from server import project

    return project.read(root, path)


@app.get("/")
def index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


app.mount("/static", StaticFiles(directory=WEB_DIR), name="static")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)
