"""Entraînement en direct d'un réseau d'images sur de vrais chiffres manuscrits.

Les données viennent de scikit-learn (load_digits : 1 797 chiffres 8×8, livrés avec la
bibliothèque, rien à télécharger), agrandis en 28×28 comme MNIST. Chaque appel à `train`
fait une époque ; les poids appris sont gardés en mémoire pour ce code précis, et
l'analyseur les recharge pour que toutes les vues montrent le réseau entraîné.
"""

from __future__ import annotations

import threading
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

BATCH = 64
LR = 1e-3
TIME_BUDGET_S = 7.0  # une époque doit tenir dans la limite de 10 s du serveur

_data: dict | None = None
_lock = threading.Lock()
_trained: dict = {}  # code -> {"state", "opt", "losses", "epochs": [{"epoch", "train_acc", "test_acc"}]}


def preload() -> None:
    """Charge les chiffres en arrière-plan dès le démarrage, pour que le 1er clic soit rapide."""
    threading.Thread(target=_load, daemon=True).start()


def _load() -> dict:
    global _data
    with _lock:
        if _data is not None:
            return _data
        from sklearn.datasets import load_digits

        d = load_digits()
        x = torch.tensor(d.images, dtype=torch.float32).unsqueeze(1) / 16.0  # (n, 1, 8, 8)
        x = F.interpolate(x, size=20, mode="bilinear", align_corners=False)
        x = F.pad(x, (4, 4, 4, 4)).clamp(0, 1)  # chiffre de 20×20 centré dans 28×28, comme MNIST
        y = torch.tensor(d.target)
        g = torch.Generator().manual_seed(0)
        perm = torch.randperm(len(y), generator=g)
        cut = int(len(y) * 0.8)
        _data = {"x": x[perm[:cut]], "y": y[perm[:cut]], "x_test": x[perm[cut:]], "y_test": y[perm[cut:]]}
    return _data


def can_train(model: nn.Module) -> str | None:
    """None si le modèle prend une image 28×28 et sort 10 scores ; sinon la raison."""
    try:
        with torch.no_grad():
            out = model.eval()(torch.zeros(2, 1, 28, 28))
    except Exception:  # noqa: BLE001
        return "L'entraînement en direct marche pour les modèles qui lisent une image 28×28."
    if not isinstance(out, torch.Tensor) or tuple(out.shape) != (2, 10):
        return "L'entraînement en direct attend 10 scores en sortie (un par chiffre)."
    return None


def apply(code: str, model: nn.Module) -> dict | None:
    """Recharge les poids appris pour ce code, s'il y en a. Renvoie le résumé de l'entraînement."""
    entry = _trained.get(code)
    if entry is None:
        return None
    model.load_state_dict(entry["state"])
    return summary(entry)


def summary(entry: dict) -> dict:
    return {"epochs": entry["epochs"], "losses": entry["losses"]}


def reset(code: str) -> None:
    _trained.pop(code, None)


def _accuracy(model: nn.Module, x: torch.Tensor, y: torch.Tensor) -> float:
    model.eval()
    with torch.no_grad():
        return round(float((model(x).argmax(dim=1) == y).float().mean()), 4)


def train(code: str, model: nn.Module) -> dict:
    reason = can_train(model)
    if reason:
        return {"ok": False, "error": reason, "line": None}
    data = _load()
    entry = _trained.get(code)
    opt = torch.optim.Adam(model.parameters(), lr=LR)
    if entry:
        model.load_state_dict(entry["state"])
        opt.load_state_dict(entry["opt"])
    else:
        entry = {"losses": [], "epochs": []}

    model.train()
    g = torch.Generator().manual_seed(len(entry["epochs"]))
    perm = torch.randperm(len(data["y"]), generator=g)
    start, seen = time.monotonic(), 0
    for i in range(0, len(perm), BATCH):
        idx = perm[i : i + BATCH]
        loss = F.cross_entropy(model(data["x"][idx]), data["y"][idx])
        opt.zero_grad()
        loss.backward()
        opt.step()
        entry["losses"].append(round(float(loss.detach()), 4))
        seen += len(idx)
        if time.monotonic() - start > TIME_BUDGET_S:
            break

    entry["epochs"].append({
        "epoch": len(entry["epochs"]) + 1,
        "seen": seen,
        "train_acc": _accuracy(model, data["x"], data["y"]),
        "test_acc": _accuracy(model, data["x_test"], data["y_test"]),
    })
    entry["state"] = {k: v.detach().clone() for k, v in model.state_dict().items()}
    entry["opt"] = opt.state_dict()
    _trained.clear()  # on ne garde que le dernier modèle entraîné
    _trained[code] = entry
    return {"ok": True, "training": summary(entry), "n_train": len(data["y"]), "n_test": len(data["y_test"])}


DREAM_STEPS = 120


def dream(code: str, model: nn.Module) -> dict:
    """Ce que le réseau « imagine » pour chaque chiffre.

    On part d'une image vide et on la modifie pas à pas pour que le score d'un chiffre monte le
    plus possible. Le résultat montre ce que le réseau a retenu de chaque chiffre : du bruit s'il
    n'a rien appris, des formes reconnaissables après l'entraînement.
    """
    reason = can_train(model)
    if reason:
        return {"ok": False, "error": reason, "line": None}
    entry = _trained.get(code)
    if entry:
        model.load_state_dict(entry["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    torch.manual_seed(0)
    raw = (torch.randn(10, 1, 28, 28) * 0.1 - 2.0).requires_grad_()  # presque noir au départ
    opt = torch.optim.Adam([raw], lr=0.15)
    target = torch.arange(10)
    for _ in range(DREAM_STEPS):
        img = torch.sigmoid(raw)
        # Le score du bon chiffre doit monter ; l'image doit rester sobre et lisse.
        rough = (img[..., 1:, :] - img[..., :-1, :]).abs().mean() + (img[..., :, 1:] - img[..., :, :-1]).abs().mean()
        loss = F.cross_entropy(model(img), target) + 0.6 * img.mean() + 0.8 * rough
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        img = torch.sigmoid(raw)
        probs = torch.softmax(model(img), dim=1)
    return {
        "ok": True,
        "images": [[[round(v, 2) for v in row] for row in im[0]] for im in img.tolist()],
        "confidence": [round(float(probs[i, i]), 3) for i in range(10)],
        "epochs": len(entry["epochs"]) if entry else 0,
    }
