"""Exécute le code du modèle écrit dans l'éditeur et décrit ce qui se passe dans chaque couche.

Le code doit définir une variable `model` (un torch.nn.Module). On fait passer une image
d'entrée dans le modèle, on capture la sortie de chaque couche avec des hooks, et on renvoie
une description JSON que la page web dessine.
"""

from __future__ import annotations

import re
import traceback

import torch
import torch.nn as nn

FILENAME = "<editeur>"
MAX_PARAMS = 20_000_000
MAX_MAPS = 8  # cartes d'activation renvoyées par couche
MAX_MAP_SIZE = 32  # côté maximal d'une carte renvoyée
MAX_VECTOR = 64  # valeurs renvoyées pour une sortie à plat


def default_input() -> torch.Tensor:
    """Un « 7 » dessiné à la main, au format MNIST (1, 1, 28, 28)."""
    img = torch.zeros(28, 28)
    img[6:9, 7:21] = 1.0
    for i in range(16):
        r, c = 8 + i, 20 - i // 2
        img[r, c - 1 : c + 2] = 1.0
    return img.view(1, 1, 28, 28)


def _layer_params(m: nn.Module) -> dict:
    """Hyperparamètres lisibles d'une couche (kernel, stride, nombre de neurones...)."""
    keys = [
        "in_channels", "out_channels", "kernel_size", "stride", "padding",
        "in_features", "out_features", "p", "num_features", "negative_slope", "dim",
    ]
    out = {}
    for k in keys:
        if hasattr(m, k):
            v = getattr(m, k)
            if isinstance(v, (int, float, tuple, str)) and v is not None:
                out[k] = list(v) if isinstance(v, tuple) else v
    return out


def _normalize(t: torch.Tensor) -> list:
    t = t.float()
    lo, hi = t.min(), t.max()
    t = (t - lo) / (hi - lo) if hi > lo else torch.zeros_like(t)
    return [[round(x, 3) for x in row] for row in t.tolist()]


def _preview(out: torch.Tensor) -> dict:
    """Résumé visuel d'une sortie : cartes 2D pour les convolutions, vecteur sinon."""
    x = out.detach()[0]
    stats = {
        "min": round(float(x.min()), 4),
        "max": round(float(x.max()), 4),
        "mean": round(float(x.float().mean()), 4),
    }
    if x.dim() == 3:  # (C, H, W)
        maps = x[:MAX_MAPS]
        if max(maps.shape[1:]) > MAX_MAP_SIZE:
            maps = nn.functional.adaptive_avg_pool2d(maps, MAX_MAP_SIZE)
        return {"kind": "maps", "maps": [_normalize(m) for m in maps], "channels": x.shape[0], **stats}
    flat = x.flatten()
    return {
        "kind": "vector",
        "values": [round(v, 4) for v in flat[:MAX_VECTOR].tolist()],
        "length": flat.numel(),
        **stats,
    }


def _find_lines(code: str, layers: list[dict]) -> None:
    """Associe chaque couche à la ligne du code qui la crée (meilleur effort)."""
    lines = code.splitlines()
    used: set[int] = set()
    cursor = 0
    for layer in layers:
        found = None
        attr = layer["name"].split(".")[-1]
        if not attr.isdigit():  # self.conv1 = nn.Conv2d(...)
            pat = re.compile(rf"\b{re.escape(attr)}\s*=")
            for i, line in enumerate(lines):
                if pat.search(line) and i not in used:
                    found = i
                    break
        if found is None:  # couches dans nn.Sequential : on suit l'ordre des « Type( »
            pat = re.compile(rf"\b{re.escape(layer['type'])}\s*\(")
            for i in range(cursor, len(lines)):
                if pat.search(lines[i]) and i not in used:
                    found = i
                    break
        if found is not None:
            used.add(found)
            cursor = found + 1
            layer["line"] = found + 1


def _error_line(exc: BaseException) -> int | None:
    for frame in reversed(traceback.extract_tb(exc.__traceback__)):
        if frame.filename == FILENAME:
            return frame.lineno
    if isinstance(exc, SyntaxError) and exc.filename == FILENAME:
        return exc.lineno
    return None


def _explain(exc: BaseException) -> str:
    """Traduit les erreurs les plus fréquentes en une phrase compréhensible."""
    msg = str(exc)
    m = re.search(r"mat1 and mat2 shapes cannot be multiplied \((\d+)x(\d+) and (\d+)x(\d+)\)", msg)
    if m:
        return (
            f"La couche Linear attend {m.group(3)} valeurs en entrée, mais la couche précédente "
            f"en donne {m.group(2)}. Mets in_features={m.group(2)}."
        )
    m = re.search(r"expected input\[.*?\] to have (\d+) channels, but got (\d+) channels", msg)
    if m:
        return (
            f"Cette convolution attend {m.group(1)} canaux en entrée, mais en reçoit {m.group(2)}. "
            f"Mets in_channels={m.group(2)}."
        )
    if "Output size is too small" in msg or "Kernel size can't be greater" in msg:
        return "L'image est devenue trop petite pour ce noyau : retire un MaxPool ou réduis kernel_size."
    return msg


def analyze(code: str, pixels: list[list[float]] | None = None) -> dict:
    """Point d'entrée : code Python -> description des couches et de leurs sorties."""
    namespace: dict = {"torch": torch, "nn": nn, "F": nn.functional, "__name__": "__neuro_lens__"}
    try:
        exec(compile(code, FILENAME, "exec"), namespace)
    except Exception as exc:  # noqa: BLE001 - toute erreur du code utilisateur est renvoyée
        return {"ok": False, "error": _explain(exc), "line": _error_line(exc)}

    model = namespace.get("model")
    if not isinstance(model, nn.Module):
        return {"ok": False, "error": "Définis une variable `model` qui est un nn.Module.", "line": None}

    n_params = sum(p.numel() for p in model.parameters())
    if n_params > MAX_PARAMS:
        return {"ok": False, "error": f"Modèle trop gros pour l'aperçu ({n_params:,} paramètres).", "line": None}

    if pixels is not None:
        x = torch.tensor(pixels, dtype=torch.float32).view(1, 1, 28, 28)
    else:
        x = namespace.get("example_input")
        if not isinstance(x, torch.Tensor):
            x = default_input()

    layers: list[dict] = []
    hooks = []

    def make_hook(name: str):
        def hook(module, _inp, out):
            if not isinstance(out, torch.Tensor):
                return
            layers.append({
                "name": name,
                "type": type(module).__name__,
                "params": _layer_params(module),
                "n_params": sum(p.numel() for p in module.parameters(recurse=False)),
                "shape": list(out.shape),
                "preview": _preview(out),
            })
        return hook

    for name, module in model.named_modules():
        if name and not list(module.children()):
            hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        with torch.no_grad():
            out = model(x)
    except Exception as exc:  # noqa: BLE001
        line = _error_line(exc)
        if line is None:  # l'erreur vient de PyTorch : on désigne la première couche qui n'a pas abouti
            leaves = [
                {"name": n, "type": type(m).__name__}
                for n, m in model.named_modules()
                if n and not list(m.children())
            ]
            _find_lines(code, leaves)
            if len(layers) < len(leaves):
                line = leaves[len(layers)].get("line")
        _find_lines(code, layers)
        return {"ok": False, "error": _explain(exc), "line": line, "layers": layers}
    finally:
        for h in hooks:
            h.remove()

    _find_lines(code, layers)
    result = {
        "ok": True,
        "input": {"shape": list(x.shape), "preview": _preview(x)},
        "layers": layers,
        "n_params": n_params,
    }
    if isinstance(out, torch.Tensor) and out.dim() == 2 and out.shape[1] <= 20:
        probs = torch.softmax(out[0], dim=0)
        result["prediction"] = {
            "probs": [round(p, 4) for p in probs.tolist()],
            "top": int(probs.argmax()),
        }
    return result
