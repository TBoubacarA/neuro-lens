"""La suite de l'histoire après l'entraînement : essais, versions, mise en production, surveillance.

Pour un modèle scikit-learn, ce module tient quatre choses, toutes réelles et locales :

- le suivi des essais : chaque combinaison modèle + réglages + données essayée, avec ses scores ;
- le registre : chaque déploiement crée une version numérotée dans un dossier prêt à être servi
  (modèle, fiche du modèle, serve.py, Dockerfile, requirements.txt) ;
- le service : la version « en production » répond aux demandes de prédiction ;
- la surveillance : latence, qualité et dérive des données reçues par rapport aux données d'entraînement.

Tout est écrit sous `.neuro-lens/` à la racine du projet (ou `NEURO_LENS_HOME`).
"""

from __future__ import annotations

import hashlib
import json
import os
import platform
import time
from datetime import datetime
from pathlib import Path

MAX_RUNS = 40
MAX_EVENTS = 3000
WINDOW = 50  # requêtes par point de la courbe de qualité
BINS = 12
MIN_EVENTS = 50  # en dessous, trop peu de requêtes pour parler de dérive
DRIFT_WATCH, DRIFT_ALERT = 0.1, 0.25  # seuils usuels de l'indice de stabilité (PSI)

_runs: list[dict] = []
_prod: dict | None = None  # {"version", "model", "card", "X", "y"}
_events: list[dict] = []  # une ligne par requête reçue en production


def home() -> Path:
    return Path(os.environ.get("NEURO_LENS_HOME") or Path(__file__).resolve().parent.parent / ".neuro-lens")


def fingerprint(X, y) -> str:
    """Empreinte courte des données : si elle change, ce ne sont plus les mêmes données."""
    h = hashlib.sha1(X.tobytes())
    if y is not None:
        h.update(y.tobytes())
    return h.hexdigest()[:8]


# ---------- Suivi des essais ----------

def record_run(result: dict) -> None:
    """Note l'essai décrit par `result` (une analyse scikit-learn réussie), une seule fois par combinaison."""
    knob = result.get("knob") or {}
    held_out = None
    if knob.get("test") and knob.get("value") in knob.get("values", []):
        held_out = knob["test"][knob["values"].index(knob["value"])]
    key = json.dumps([result["estimator"], result["params"], result["data_id"]], sort_keys=True, default=str)
    result["run_key"] = hashlib.sha1(key.encode()).hexdigest()[:8]
    if any(r["key"] == result["run_key"] for r in _runs):
        return
    _runs.append({
        "id": (_runs[-1]["id"] + 1) if _runs else 1,
        "key": result["run_key"],
        "at": datetime.now().strftime("%H:%M:%S"),
        "estimator": result["estimator"],
        "setting": f"{knob['name']}={knob['value']}" if knob.get("name") else "",
        "data_id": result["data_id"],
        "train": result.get("score"),
        "held_out": held_out,
    })
    del _runs[:-MAX_RUNS]


# ---------- Registre et déploiement ----------

SERVE_PY = '''"""Service de prédiction — généré par neuro-lens pour la version {version} ({estimator}).

Lancer :  uvicorn serve:app --host 0.0.0.0 --port 8080
Essayer : curl -X POST localhost:8080/predict -H "Content-Type: application/json" -d '{{"rows": [{example}]}}'
"""

import time
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

HERE = Path(__file__).resolve().parent
bundle = joblib.load(HERE / "model.joblib")  # chargé une fois, au démarrage
model, FEATURES = bundle["model"], bundle["features"]

app = FastAPI(title="{estimator} v{version}")


class Rows(BaseModel):
    rows: list[list[float]]  # une ligne par exemple, dans l'ordre : {features}


@app.get("/health")
def health() -> dict:
    """Pour l'orchestrateur (Docker, Kubernetes) : le service est-il vivant ?"""
    return {{"status": "ok", "version": {version}}}


@app.post("/predict")
def predict(body: Rows) -> dict:
    if any(len(row) != len(FEATURES) for row in body.rows):
        raise HTTPException(422, f"Chaque ligne doit avoir {{len(FEATURES)}} valeurs : {{FEATURES}}")
    start = time.perf_counter()
    predictions = model.predict(body.rows).tolist()
    return {{"predictions": predictions, "version": {version}, "latency_ms": round((time.perf_counter() - start) * 1000, 2)}}
'''

DOCKERFILE = '''# Image du service de prédiction — générée par neuro-lens (version {version}).
# Construire : docker build -t {image}:{version} .
# Lancer :     docker run -p 8080:8080 {image}:{version}
FROM python:{python}-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Le modèle est figé dans l'image : une image = une version, on peut revenir en arrière.
COPY model.joblib model_card.json serve.py ./

EXPOSE 8080
HEALTHCHECK CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/health')"
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8080"]
'''


def _versions() -> list[dict]:
    cards = []
    for card in sorted((home() / "models").glob("v*/model_card.json")):
        try:
            cards.append(json.loads(card.read_text()))
        except (OSError, ValueError):
            continue
    return sorted(cards, key=lambda c: c["version"])


def deploy(fitted: dict, result: dict) -> dict:
    """Fige le modèle entraîné dans une nouvelle version, écrit son dossier de déploiement et la met en production."""
    import joblib
    import numpy as np
    import sklearn

    global _prod
    versions = _versions()
    version = (versions[-1]["version"] + 1) if versions else 1
    folder = home() / "models" / f"v{version}"
    folder.mkdir(parents=True, exist_ok=True)
    X, y = fitted["X"], fitted["y"]
    run = next((r for r in _runs if r["key"] == result.get("run_key")), {})
    card = {
        "version": version,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "estimator": result["estimator"],
        "task": result["task"],
        "params": result["params"],
        "features": result["features"],
        "classes": result.get("classes", []),
        "n_samples": result["n_samples"],
        "data_id": result["data_id"],
        "run_key": result.get("run_key"),
        "score_train": result.get("score"),
        "score_held_out": run.get("held_out"),
        "example": [round(float(v), 4) for v in X[0]],
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
    }
    joblib.dump({"model": fitted["model"], "features": result["features"]}, folder / "model.joblib")
    # Un échantillon de référence, à part du modèle : c'est à lui qu'on comparera les données reçues.
    labels = np.zeros(0) if y is None else (y.astype(str) if y.dtype == object else y)
    np.savez_compressed(folder / "reference.npz", X=X, y=labels)
    card["size_kb"] = round((folder / "model.joblib").stat().st_size / 1024, 1)
    (folder / "model_card.json").write_text(json.dumps(card, indent=2, ensure_ascii=False, default=str))
    (folder / "serve.py").write_text(SERVE_PY.format(
        version=version, estimator=card["estimator"], features=", ".join(card["features"]),
        example=json.dumps(card["example"]),
    ))
    (folder / "Dockerfile").write_text(DOCKERFILE.format(
        version=version, image=card["estimator"].lower(), python=".".join(platform.python_version_tuple()[:2]),
    ))
    (folder / "requirements.txt").write_text(
        f"scikit-learn=={sklearn.__version__}\nnumpy=={np.__version__}\njoblib=={joblib.__version__}\nfastapi\nuvicorn\n"
    )
    (home() / "production.json").write_text(json.dumps({"version": version}))
    _prod = {"version": version, "model": fitted["model"], "card": card, "X": X, "y": y}
    _events.clear()
    return {"ok": True, "mlops": state(), "deployed": version}


def _production() -> dict | None:
    """La version en production, rechargée depuis le disque si le processus a redémarré."""
    global _prod
    if _prod is not None:
        return _prod
    pointer = home() / "production.json"
    if not pointer.exists():
        return None
    import joblib

    version = json.loads(pointer.read_text())["version"]
    folder = home() / "models" / f"v{version}"
    import numpy as np

    try:
        bundle = joblib.load(folder / "model.joblib")
        card = json.loads((folder / "model_card.json").read_text())
        ref = np.load(folder / "reference.npz")
    except (OSError, ValueError):
        return None
    _prod = {"version": version, "model": bundle["model"], "card": card, "X": ref["X"], "y": ref["y"] if len(ref["y"]) else None}
    return _prod


def _files(version: int) -> dict:
    folder = home() / "models" / f"v{version}"
    texts = {}
    for name in ("serve.py", "Dockerfile", "requirements.txt", "model_card.json"):
        try:
            texts[name] = (folder / name).read_text()
        except OSError:
            pass
    return {"folder": str(folder), "texts": texts}


# ---------- Service ----------

def predict(rows: list[list[float]], truth: list | None = None) -> dict:
    """Répond à une demande de prédiction avec la version en production, et la note pour la surveillance."""
    import numpy as np

    prod = _production()
    if prod is None:
        return {"ok": False, "error": "Aucun modèle en production : déploie d'abord une version.", "line": None}
    n = len(prod["card"]["features"])
    if not rows or any(len(r) != n for r in rows):
        return {"ok": False, "error": f"Chaque ligne doit avoir {n} valeurs : {', '.join(prod['card']['features'])}.", "line": None}
    data = np.asarray(rows, dtype=float)
    start = time.perf_counter()
    out = prod["model"].predict(data)
    latency = (time.perf_counter() - start) * 1000
    answer: dict = {
        "ok": True,
        "version": prod["version"],
        "predictions": [p.item() if hasattr(p, "item") else p for p in out],
        "latency_ms": round(latency, 3),
    }
    if hasattr(prod["model"], "predict_proba"):
        answer["probabilities"] = [[round(float(v), 3) for v in row] for row in prod["model"].predict_proba(data)]
    for i, row in enumerate(data):
        _events.append({
            "x": row, "pred": out[i], "latency": latency / len(data),
            "truth": None if truth is None else truth[i],
        })
    del _events[:-MAX_EVENTS]
    return answer


def simulate(n: int = 200, drift: float = 0.0) -> dict:
    """Envoie `n` requêtes qui ressemblent aux données d'entraînement, décalées de `drift` écarts-types.

    drift = 0 : le monde n'a pas changé. Plus on l'augmente, plus les données reçues s'éloignent
    de celles que le modèle a apprises — c'est la dérive, la première cause de panne silencieuse.
    """
    import numpy as np

    prod = _production()
    if prod is None:
        return {"ok": False, "error": "Aucun modèle en production : déploie d'abord une version.", "line": None}
    X, y = prod["X"], prod["y"]
    n = max(0, min(int(n), 500))  # 0 : seulement relire l'état, sans trafic
    rng = np.random.default_rng(len(_events))
    pick = rng.integers(0, len(X), size=n)
    spread = X.std(axis=0) + 1e-9
    rows = X[pick] + rng.normal(0, 0.05, size=(n, X.shape[1])) * spread + float(drift) * spread
    for i in range(n):  # une requête à la fois, comme en vrai : la latence mesurée est celle d'un appel
        predict([rows[i].tolist()], truth=None if y is None else [y[pick[i]]])
    return {"ok": True, "mlops": state()}


# ---------- Surveillance ----------

def _psi(reference, received) -> float:
    """Indice de stabilité de population : 0 = mêmes données ; au-delà de 0,25, elles ont nettement changé."""
    import numpy as np

    edges = np.unique(np.quantile(reference, np.linspace(0, 1, 11)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    a = np.histogram(reference, edges)[0] / len(reference) + 1e-4
    b = np.histogram(received, edges)[0] / len(received) + 1e-4
    return round(float(((b - a) * np.log(b / a)).sum()), 3)


def _monitoring(prod: dict) -> dict | None:
    import numpy as np

    if not _events:
        return None
    card, X = prod["card"], prod["X"]
    received = np.array([e["x"] for e in _events])
    latencies = np.array([e["latency"] for e in _events])
    features = []
    for j, name in enumerate(card["features"]):
        lo, hi = min(X[:, j].min(), received[:, j].min()), max(X[:, j].max(), received[:, j].max())
        edges = np.linspace(lo, hi if hi > lo else lo + 1, BINS + 1)
        features.append({
            "name": name,
            "psi": _psi(X[:, j], received[:, j]),
            "range": [round(float(lo), 3), round(float(hi), 3)],
            "reference": [round(float(v), 4) for v in np.histogram(X[:, j], edges)[0] / len(X)],
            "received": [round(float(v), 4) for v in np.histogram(received[:, j], edges)[0] / len(received)],
        })
    worst = max(f["psi"] for f in features)
    summary: dict = {
        "requests": len(_events),
        "latency_p50": round(float(np.percentile(latencies, 50)), 3),
        "latency_p95": round(float(np.percentile(latencies, 95)), 3),
        "features": features,
        "drift": worst,
        # Avec une poignée de requêtes, tout écart est dû au hasard : on ne juge pas encore.
        "status": "few" if len(_events) < MIN_EVENTS
        else "alert" if worst >= DRIFT_ALERT else "watch" if worst >= DRIFT_WATCH else "ok",
    }
    labelled = [e for e in _events if e["truth"] is not None]
    if labelled and card["task"] == "classification":
        hits = [float(e["pred"] == e["truth"]) for e in labelled]
        summary["quality"] = round(sum(hits) / len(hits), 4)
        summary["quality_curve"] = [  # par tranche complète : une tranche de 2 requêtes ne dit rien
            round(sum(hits[i : i + WINDOW]) / WINDOW, 4) for i in range(0, len(hits) - WINDOW + 1, WINDOW)
        ]
        summary["quality_at_deploy"] = card.get("score_held_out") or card.get("score_train")
    return summary


def state() -> dict:
    """Tout ce que la page affiche de la chaîne : essais, versions, production, surveillance."""
    best = max((r["held_out"] for r in _runs if r["held_out"] is not None), default=None)
    prod = _production()
    return {
        "runs": [{**r, "best": best is not None and r["held_out"] == best} for r in _runs[-12:]],
        "versions": [
            {k: c.get(k) for k in ("version", "created", "estimator", "score_train", "score_held_out", "size_kb", "run_key")}
            for c in _versions()
        ],
        "production": None if prod is None else {**prod["card"], **_files(prod["version"])},
        "monitoring": None if prod is None else _monitoring(prod),
    }
