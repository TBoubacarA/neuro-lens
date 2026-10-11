"""L'infrastructure pour de bon : construire l'image Docker d'une version et lancer son conteneur.

Tout part du dossier écrit par le déploiement (.neuro-lens/models/vN). Ici on ne lance aucun
code de l'éditeur : seulement `docker build` et `docker run` sur ce dossier. Le conteneur
n'écoute que sur 127.0.0.1.
"""

from __future__ import annotations

import json
import shutil
import socket
import subprocess
import threading
import time
import urllib.error
import urllib.request

from server import mlops

CONTAINER = "neuro-lens-model"
MAX_LOG = 120

_lock = threading.Lock()
_job: dict = {"phase": "idle", "log": []}  # idle | building | starting | running | failed | stopped


def _docker(*args: str, timeout: int = 20) -> subprocess.CompletedProcess:
    return subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout)


def available() -> str | None:
    """None si Docker répond ; sinon ce qu'il faut faire."""
    if shutil.which("docker") is None:
        return "Docker n'est pas installé sur cette machine. Installe Docker Desktop pour construire l'image."
    try:
        if _docker("info", "--format", "{{.ServerVersion}}").returncode != 0:
            return "Docker est installé mais ne tourne pas. Lance Docker Desktop, puis réessaie."
    except (OSError, subprocess.TimeoutExpired):
        return "Docker ne répond pas. Lance Docker Desktop, puis réessaie."
    return None


def _say(line: str) -> None:
    with _lock:
        _job["log"].append(line.rstrip())
        del _job["log"][:-MAX_LOG]


def _free_port() -> int:
    """Un port libre de la machine, à relier au port 8080 du conteneur."""
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _health() -> bool:
    port = _job.get("port")
    if not port:
        return False
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/health", timeout=2) as r:
            return json.load(r).get("status") == "ok"
    except (OSError, ValueError):
        return False


def _build_and_run(version: int, folder: str, image: str) -> None:
    start = time.monotonic()
    try:
        _say(f"$ docker build -t {image} {folder}")
        proc = subprocess.Popen(
            ["docker", "build", "--progress=plain", "-t", image, folder],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        )
        for line in proc.stdout:
            _say(line)
        if proc.wait() != 0:
            raise RuntimeError("La construction de l'image a échoué (voir le journal).")
        size = _docker("image", "inspect", image, "--format", "{{.Size}}").stdout.strip()
        with _lock:
            _job.update(phase="starting", size_mb=round(int(size) / 1e6, 1) if size.isdigit() else None,
                        build_s=round(time.monotonic() - start, 1))
        _docker("rm", "-f", CONTAINER)
        port = _free_port()
        with _lock:
            _job["port"] = port
        _say(f"$ docker run -d --name {CONTAINER} -p 127.0.0.1:{port}:8080 {image}")
        run = _docker("run", "-d", "--name", CONTAINER, "-p", f"127.0.0.1:{port}:8080", image)
        if run.returncode != 0:
            _docker("rm", "-f", CONTAINER)  # ne pas laisser un conteneur à moitié créé
            raise RuntimeError(run.stderr.strip().splitlines()[0] if run.stderr.strip() else "Le conteneur n'a pas démarré.")
        for _ in range(40):  # le service met quelques secondes à charger le modèle
            if _health():
                break
            time.sleep(0.5)
        else:
            raise RuntimeError("Le conteneur a démarré mais /health ne répond pas.")
        _say(f"Le conteneur répond sur http://127.0.0.1:{port}")
        with _lock:
            _job.update(phase="running", container=run.stdout.strip()[:12])
    except Exception as exc:  # noqa: BLE001 - tout échec est montré dans la page
        _say(str(exc))
        with _lock:
            _job.update(phase="failed", error=str(exc))


def build() -> dict:
    """Construit l'image de la version en production et lance son conteneur (en arrière-plan)."""
    reason = available()
    if reason:
        return {"ok": False, "error": reason}
    pointer = mlops.home() / "production.json"
    if not pointer.exists():
        return {"ok": False, "error": "Aucune version en production : déploie d'abord un modèle."}
    version = json.loads(pointer.read_text())["version"]
    folder = mlops.home() / "models" / f"v{version}"
    with _lock:
        if _job["phase"] in ("building", "starting"):
            return {"ok": True, "infra": dict(_job)}
        _job.clear()
        _job.update(phase="building", log=[], version=version, image=f"{CONTAINER}:v{version}")
    threading.Thread(target=_build_and_run, args=(version, str(folder), f"{CONTAINER}:v{version}"), daemon=True).start()
    return status()


def stop() -> dict:
    if available() is None:
        _docker("rm", "-f", CONTAINER)
    with _lock:
        if _job["phase"] == "running":
            _job["phase"] = "stopped"
    _say(f"$ docker rm -f {CONTAINER}")
    return status()


def status() -> dict:
    with _lock:
        job = dict(_job, log=list(_job["log"]))
    if job["phase"] == "running" and not _health():  # arrêté depuis l'extérieur
        job["phase"] = "stopped"
        with _lock:
            _job["phase"] = "stopped"
    return {"ok": True, "infra": job}


def predict(rows: list[list[float]]) -> dict:
    """Envoie une demande au conteneur, par le réseau, comme le ferait n'importe quelle application."""
    body = json.dumps({"rows": rows}).encode()
    port = _job.get("port")
    if not port:
        return {"ok": False, "error": "Le conteneur ne répond pas : construis et lance l'image d'abord."}
    request = urllib.request.Request(f"http://127.0.0.1:{port}/predict", body, {"Content-Type": "application/json"})
    start = time.perf_counter()
    try:
        with urllib.request.urlopen(request, timeout=5) as r:
            answer = json.load(r)
    except urllib.error.HTTPError as exc:
        return {"ok": False, "error": f"Le conteneur a refusé la demande ({exc.code})."}
    except (OSError, ValueError):
        return {"ok": False, "error": "Le conteneur ne répond pas : construis et lance l'image d'abord."}
    return {"ok": True, **answer, "round_trip_ms": round((time.perf_counter() - start) * 1000, 2)}
