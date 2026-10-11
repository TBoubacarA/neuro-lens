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
_trained: dict = {}  # code -> {"state", "states", "opt", "losses", "epochs": [...], "eval": {...}}
MAX_STATES = 40  # poids gardés époque par époque, pour revoir l'apprentissage
MAX_MISTAKES = 12


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


def apply(code: str, model: nn.Module, epoch: int | None = None) -> dict | None:
    """Recharge les poids appris pour ce code, s'il y en a : les derniers, ou ceux après `epoch` époques."""
    entry = _trained.get(code)
    if entry is None:
        return None
    shown = len(entry["epochs"])
    if epoch is not None and 0 <= epoch < len(entry["states"]):
        shown = epoch
        model.load_state_dict(entry["states"][epoch])
    else:
        model.load_state_dict(entry["state"])
    return {**summary(entry), "shown": shown}


def summary(entry: dict) -> dict:
    return {"epochs": entry["epochs"], "losses": entry["losses"], "eval": entry.get("eval"), "kept": len(entry["states"])}


def _snapshot(model: nn.Module) -> dict:
    return {k: v.detach().clone() for k, v in model.state_dict().items()}


def _evaluate(model: nn.Module, data: dict) -> dict:
    """Matrice de confusion sur les chiffres jamais vus, et quelques-uns de ceux qu'il rate."""
    model.eval()
    with torch.no_grad():
        pred = model(data["x_test"]).argmax(dim=1)
    y = data["y_test"]
    confusion = torch.zeros(10, 10, dtype=torch.long)
    for t, p in zip(y.tolist(), pred.tolist()):
        confusion[t, p] += 1
    wrong = (pred != y).nonzero().flatten()[:MAX_MISTAKES]
    mistakes = [
        {"true": int(y[i]), "pred": int(pred[i]), "image": [[round(v, 2) for v in row] for row in data["x_test"][i, 0].tolist()]}
        for i in wrong.tolist()
    ]
    return {"confusion": confusion.tolist(), "mistakes": mistakes, "n_wrong": int((pred != y).sum()), "n_test": len(y)}


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
        entry = {"losses": [], "epochs": [], "states": [_snapshot(model)]}  # états[0] : avant d'apprendre

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
    entry["state"] = _snapshot(model)
    if len(entry["states"]) < MAX_STATES:
        entry["states"].append(entry["state"])
    entry["eval"] = _evaluate(model, data)
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


ATTACK_STEPS = 40
ATTACK_EPS = 0.25  # chaque pixel bouge d'au plus 25 % : presque invisible à l'œil


def attack(code: str, model: nn.Module, pixels, target: int) -> dict:
    """Tromper le réseau : retoucher à peine le dessin pour qu'il y lise le chiffre `target`."""
    reason = can_train(model)
    if reason:
        return {"ok": False, "error": reason, "line": None}
    entry = _trained.get(code)
    if entry:
        model.load_state_dict(entry["state"])
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    x0 = torch.tensor(pixels, dtype=torch.float32).view(1, 1, 28, 28)
    with torch.no_grad():
        before = torch.softmax(model(x0)[0], dim=0)
    x = x0.clone()
    y = torch.tensor([target])
    for _ in range(ATTACK_STEPS):  # descente de gradient projetée (PGD) sur l'image, pas sur les poids
        x.requires_grad_(True)
        loss = F.cross_entropy(model(x), y)
        (grad,) = torch.autograd.grad(loss, x)
        with torch.no_grad():
            x = x - 0.02 * grad.sign()
            x = (x0 + (x - x0).clamp(-ATTACK_EPS, ATTACK_EPS)).clamp(0, 1)
        with torch.no_grad():
            if int(model(x).argmax()) == target and _ > 5:
                break
    with torch.no_grad():
        after = torch.softmax(model(x)[0], dim=0)
    delta = (x - x0)[0, 0]
    return {
        "ok": True,
        "original": [[round(v, 3) for v in row] for row in x0[0, 0].tolist()],
        "image": [[round(v, 3) for v in row] for row in x[0, 0].tolist()],
        "delta": [[round(v, 3) for v in row] for row in (delta / ATTACK_EPS).tolist()],  # entre -1 et 1
        "before": {"probs": [round(v, 4) for v in before.tolist()], "top": int(before.argmax())},
        "after": {"probs": [round(v, 4) for v in after.tolist()], "top": int(after.argmax())},
        "max_change": round(float(delta.abs().max()), 3),
        "fooled": int(after.argmax()) == target,
    }


LANDSCAPE_N = 17  # points par côté de la carte
LANDSCAPE_SAMPLES = 256


def _directions(model: nn.Module) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Deux directions aléatoires dans l'espace des poids, mises à l'échelle de chaque filtre/neurone."""
    g = torch.Generator().manual_seed(1)
    out = []
    for p in model.parameters():
        d1, d2 = torch.randn(p.shape, generator=g), torch.randn(p.shape, generator=g)
        if p.dim() > 1:
            flat = p.detach().flatten(1).norm(dim=1).view(-1, *[1] * (p.dim() - 1))
            d1 = d1 / d1.flatten(1).norm(dim=1).view_as(flat).clamp_min(1e-9) * flat
            d2 = d2 / d2.flatten(1).norm(dim=1).view_as(flat).clamp_min(1e-9) * flat
        else:
            d1, d2 = torch.zeros_like(p), torch.zeros_like(p)  # on ne bouge pas les biais
        out.append((d1, d2))
    return out


def landscape(code: str, model: nn.Module) -> dict:
    """Le paysage de l'erreur autour des poids actuels, et le chemin suivi par l'entraînement.

    Les deux axes sont les directions principales (ACP) du chemin parcouru par les poids pendant
    l'entraînement : le chemin est donc dessiné dans le plan où il bouge vraiment. Avant
    l'entraînement, on prend deux directions aléatoires mises à l'échelle de chaque filtre.
    """
    reason = can_train(model)
    if reason:
        return {"ok": False, "error": reason, "line": None}
    data = _load()
    entry = _trained.get(code)
    if entry:
        model.load_state_dict(entry["state"])
    model.eval()
    params = list(model.parameters())
    names = [n for n, _ in model.named_parameters()]
    center = [p.detach().clone() for p in params]
    x, y = data["x"][:LANDSCAPE_SAMPLES], data["y"][:LANDSCAPE_SAMPLES]

    path = []
    states = entry["states"] if entry else []
    if len(states) >= 3:
        deltas = torch.stack([torch.cat([(st[n] - c).flatten() for n, c in zip(names, center)]) for st in states])
        _, _, vt = torch.linalg.svd(deltas, full_matrices=False)
        axes = vt[:2]
        if axes.shape[0] < 2:  # chemin en ligne droite : on ajoute un axe perpendiculaire quelconque
            axes = torch.cat([axes, torch.randn(1, axes.shape[1], generator=torch.Generator().manual_seed(1))])
        coords = deltas @ axes.T
        path = [[round(float(u), 5), round(float(v), 5), k] for k, (u, v) in enumerate(coords.tolist())]
        dirs, start = [], 0
        for c in center:
            n = c.numel()
            dirs.append((axes[0, start : start + n].view_as(c), axes[1, start : start + n].view_as(c)))
            start += n
        radius = 1.3 * max(max(abs(a), abs(b)) for a, b, _ in path) or 1.0
    else:
        dirs = _directions(model)
        radius = 1.0

    ticks = torch.linspace(-radius, radius, LANDSCAPE_N)
    grid = []
    with torch.no_grad():
        for b in ticks:
            row = []
            for a in ticks:
                for p, c, (d1, d2) in zip(params, center, dirs):
                    p.copy_(c + a * d1 + b * d2)
                row.append(round(float(F.cross_entropy(model(x), y)), 4))
            grid.append(row)
        for p, c in zip(params, center):
            p.copy_(c)
    return {"ok": True, "grid": grid, "radius": radius, "path": path, "n_samples": len(y), "pca": len(states) >= 3}
