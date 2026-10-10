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
MAX_TOKENS = 20  # tokens renvoyés pour une séquence
MAX_DIMS = 8  # dimensions affichées par vecteur de token
MAX_HEADS = 8  # têtes d'attention renvoyées
MAX_GRAPH_NODES = 5000  # opérations parcourues pour retrouver d'où vient une valeur
INPUT = -1  # numéro de « l'entrée du modèle » dans les liens entre couches
DEFAULT_TEXT = "le chat dort sur le canapé du salon"


def default_input() -> torch.Tensor:
    """Un « 7 » dessiné à la main, au format MNIST (1, 1, 28, 28)."""
    img = torch.zeros(28, 28)
    img[6:9, 7:21] = 1.0
    for i in range(16):
        r, c = 8 + i, 20 - i // 2
        img[r, c - 1 : c + 2] = 1.0
    return img.view(1, 1, 28, 28)


def tokenize(text: str, vocab_size: int) -> tuple[list[str], torch.Tensor]:
    """Découpe le texte en mots ; un même mot reçoit toujours le même numéro."""
    words = re.findall(r"\w+|[^\w\s]", text.lower())[:MAX_TOKENS] or ["?"]
    ids: dict[str, int] = {}
    for w in words:
        ids.setdefault(w, len(ids) % vocab_size)
    return words, torch.tensor([[ids[w] for w in words]], dtype=torch.long)


def _is_leaf(m: nn.Module) -> bool:
    """Une couche qu'on montre telle quelle : sans sous-couches, ou un bloc d'attention."""
    return isinstance(m, nn.MultiheadAttention) or not list(m.children())


def _leaves(model: nn.Module) -> list[tuple[str, nn.Module]]:
    out: list[tuple[str, nn.Module]] = []
    for name, m in model.named_modules():
        if not name or not _is_leaf(m):
            continue
        if any(name.startswith(n + ".") and isinstance(p, nn.MultiheadAttention) for n, p in out):
            continue  # l'intérieur d'un bloc d'attention reste caché
        out.append((name, m))
    return out


def _layer_params(m: nn.Module) -> dict:
    """Hyperparamètres lisibles d'une couche (kernel, stride, nombre de neurones...)."""
    keys = [
        "in_channels", "out_channels", "kernel_size", "stride", "padding",
        "in_features", "out_features", "p", "num_features", "negative_slope", "dim",
        "num_embeddings", "embedding_dim", "embed_dim", "num_heads", "normalized_shape",
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
    if x.dim() == 2:  # (tokens, dimensions) : un vecteur par token
        rows = x[:MAX_TOKENS].float()
        probs = x.shape[1] <= 16 and bool((x >= 0).all()) and bool(((x.float().sum(dim=1) - 1).abs() < 1e-3).all())
        return {
            "kind": "tokens",
            "probs": probs,  # chaque ligne est une répartition qui somme à 1 (softmax, routeur...)
            "rows": [[round(v, 3) for v in r] for r in rows[:, :MAX_DIMS].tolist()],
            "dim": x.shape[1],
            "length": x.shape[0],
            "values": [round(v, 4) for v in x.float().abs().mean(dim=0)[:MAX_VECTOR].tolist()],
            **stats,
        }
    flat = x.flatten()
    return {
        "kind": "vector",
        "values": [round(v, 4) for v in flat[:MAX_VECTOR].tolist()],
        "length": flat.numel(),
        **stats,
    }


def _points(x: torch.Tensor) -> list | None:
    """Projette les vecteurs des tokens sur leurs 2 axes principaux (ACP) pour les placer sur un plan."""
    rows = x.detach()[0][:MAX_TOKENS].float()
    if rows.dim() != 2 or rows.shape[0] < 3 or rows.shape[1] < 2:
        return None
    centered = rows - rows.mean(dim=0)
    u, s, _ = torch.linalg.svd(centered, full_matrices=False)
    pts = u[:, :2] * s[:2]
    scale = pts.abs().max()
    if scale > 0:
        pts = pts / scale
    return [[round(v, 3) for v in p] for p in pts.tolist()]


def _sources(tensors, producers: dict, x: torch.Tensor) -> list[int]:
    """Remonte le fil des calculs pour trouver quelles couches ont produit ces valeurs.

    On suit le graphe que PyTorch construit pour calculer les gradients : il contient aussi les
    opérations écrites à la main dans `forward` (x + y, torch.cat...), donc les raccourcis
    (connexions résiduelles) et les branches parallèles sont retrouvés tels qu'ils sont.
    """
    found: set[int] = set()
    stack = []
    for t in tensors:
        if t is x:
            found.add(INPUT)
        elif isinstance(t, torch.Tensor) and t.grad_fn is not None:
            stack.append(t.grad_fn)
    seen = set()
    while stack and len(seen) < MAX_GRAPH_NODES:
        fn = stack.pop()
        if fn is None or fn in seen:
            continue
        seen.add(fn)
        if fn in producers:
            found.add(producers[fn])
            continue
        stack.extend(parent for parent, _ in fn.next_functions)
    return sorted(found)


def _find_lines(code: str, layers: list[dict]) -> None:
    """Associe chaque couche à la ligne du code qui la crée (meilleur effort)."""
    lines = code.splitlines()
    used: set[int] = set()
    pending = []
    for layer in layers:  # d'abord les couches nommées : self.conv1 = nn.Conv2d(...)
        attr = layer["name"].split(".")[-1]
        found = None
        if not attr.isdigit():
            pat = re.compile(rf"\b{re.escape(attr)}\s*=")
            found = next((i for i, line in enumerate(lines) if pat.search(line) and i not in used), None)
        if found is not None:
            used.add(found)
            layer["line"] = found + 1
        elif attr.isdigit() or "." not in layer["name"]:
            pending.append(layer)
        # sinon : couche nommée à l'intérieur d'un bloc tout fait, voir _group_lines

    cursor = 0
    last: dict[str, int] = {}
    for layer in pending:  # puis les couches d'une liste : on suit l'ordre des « Type( »
        pat = re.compile(rf"\b{re.escape(layer['type'])}\s*\(")
        found = next((i for i in range(cursor, len(lines)) if pat.search(lines[i]) and i not in used), None)
        if found is not None:
            used.add(found)
            cursor = found + 1
            last[layer["type"]] = found
        elif layer["type"] in last:  # créées en boucle : plusieurs couches pour une seule ligne
            found = last[layer["type"]]
        if found is not None:
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


def _group_lines(code: str, model: nn.Module, layers: list[dict]) -> None:
    """Une couche créée par un bloc tout fait (TransformerEncoderLayer...) hérite de la ligne du bloc."""
    modules = dict(model.named_modules())
    groups: dict[str, dict] = {}
    for layer in layers:
        parent = layer["name"].rpartition(".")[0]
        if not parent:
            continue
        layer["group"] = {"name": parent, "type": type(modules[parent]).__name__}
        if "line" not in layer:
            groups.setdefault(parent, {"name": parent, "type": type(modules[parent]).__name__})
    _find_lines(code, list(groups.values()))
    for layer in layers:
        line = groups.get(layer["name"].rpartition(".")[0], {}).get("line")
        if "line" not in layer and line:
            layer["line"] = line


def analyze(code: str, pixels: list[list[float]] | None = None, text: str | None = None) -> dict:
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

    leaves = _leaves(model)
    tokens = None
    if leaves and isinstance(leaves[0][1], nn.Embedding):  # le modèle lit du texte
        if not text:
            text = namespace.get("example_text")
        if not isinstance(text, str) or not text.strip():
            text = DEFAULT_TEXT
        tokens, x = tokenize(text, leaves[0][1].num_embeddings)
    elif pixels is not None:
        x = torch.tensor(pixels, dtype=torch.float32).view(1, 1, 28, 28)
    else:
        x = namespace.get("example_input")
        if not isinstance(x, torch.Tensor):
            x = default_input()

    layers: list[dict] = []
    hooks = []
    producers: dict = {}  # opération PyTorch -> numéro de la couche qui l'a faite
    if x.is_floating_point():
        x = x.clone().requires_grad_().clone()  # pour pouvoir suivre le trajet de l'entrée
        producers[x.grad_fn] = INPUT

    def make_hook(name: str):
        def hook(module, inp, out):
            weights = None
            if isinstance(module, nn.MultiheadAttention):
                out, weights = out
                if not module.batch_first:  # (tokens, lot, dimensions) -> (lot, tokens, dimensions)
                    out = out.transpose(0, 1)
            if not isinstance(out, torch.Tensor):
                return
            preview = _preview(out)
            sources = _sources(inp, producers, x)
            if out.grad_fn is not None:
                producers[out.grad_fn] = len(layers)
            k, stride = getattr(module, "kernel_size", None), getattr(module, "stride", None)
            if isinstance(module, nn.Conv2d) and k == stride and k[0] > 1 and sources == [INPUT]:
                preview["patches"] = {"size": list(k), "grid": list(out.shape[2:])}
            if weights is not None:  # (lot, têtes, token qui regarde, token regardé)
                w = weights.detach()[0][:MAX_HEADS, :MAX_TOKENS, :MAX_TOKENS]
                preview["attention"] = [[[round(v, 3) for v in row] for row in h] for h in w.tolist()]
                preview["heads"] = weights.shape[1]
            if isinstance(module, nn.Embedding):
                preview["points"] = _points(out)
            layers.append({
                "name": name,
                "type": type(module).__name__,
                "params": _layer_params(module),
                "n_params": sum(p.numel() for p in module.parameters()),
                "shape": list(out.shape),
                "preview": preview,
                "inputs": sources,
            })
        return hook

    def want_weights(_module, args, kwargs):
        return args, {**kwargs, "need_weights": True, "average_attn_weights": False}

    for name, module in leaves:
        if isinstance(module, nn.MultiheadAttention):
            hooks.append(module.register_forward_pre_hook(want_weights, with_kwargs=True))
        hooks.append(module.register_forward_hook(make_hook(name)))

    model.eval()
    try:
        out = model(x)  # sans torch.no_grad() : _sources a besoin du graphe des calculs
    except Exception as exc:  # noqa: BLE001
        line = _error_line(exc)
        if line is None:  # l'erreur vient de PyTorch : on désigne la première couche qui n'a pas abouti
            pending = [{"name": n, "type": type(m).__name__} for n, m in leaves]
            _find_lines(code, pending)
            _group_lines(code, model, pending)
            if len(layers) < len(pending):
                line = pending[len(layers)].get("line")
        _find_lines(code, layers)
        _group_lines(code, model, layers)
        return {"ok": False, "error": _explain(exc), "line": line, "layers": layers}
    finally:
        for h in hooks:
            h.remove()

    _find_lines(code, layers)
    _group_lines(code, model, layers)
    if tokens is None:
        described = {"shape": list(x.shape), "preview": _preview(x)}
    else:
        described = {
            "shape": list(x.shape),
            "preview": {"kind": "text", "ids": x[0].tolist()},
            "tokens": tokens,
            "text": text,
        }
    result = {
        "ok": True,
        "input": described,
        "layers": layers,
        "n_params": n_params,
        "output_from": _sources([out], producers, x),
    }
    if isinstance(out, torch.Tensor) and out.dim() == 2 and out.shape[1] <= 20:
        probs = torch.softmax(out[0], dim=0)
        result["prediction"] = {
            "probs": [round(p, 4) for p in probs.tolist()],
            "top": int(probs.argmax()),
        }
    return result
