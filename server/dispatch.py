"""Choisit la vue selon ce que le code définit, puis délègue à l'analyseur correspondant.

- `model` PyTorch (nn.Module) : server.analyzer (couches, activations, Transformers...)
- `model` scikit-learn : server.sk_analyzer (arbre, frontière de décision, clusters...)
- `graph` LangGraph : server.agent_analyzer (graphe de l'agent et chemin suivi)
"""

from __future__ import annotations

import torch
import torch.nn as nn

from server import analyzer
from server.analyzer import FILENAME, _error_line, _explain


def _is_sklearn(obj) -> bool:
    return type(obj).__module__.startswith("sklearn") or (hasattr(obj, "get_params") and hasattr(obj, "fit"))


def _is_langgraph(obj) -> bool:
    return hasattr(obj, "get_graph") and hasattr(obj, "stream")


def analyze(code: str, pixels=None) -> dict:
    namespace: dict = {"torch": torch, "nn": nn, "F": nn.functional, "__name__": "__neuro_lens__"}
    try:
        exec(compile(code, FILENAME, "exec"), namespace)
    except Exception as exc:  # noqa: BLE001 - toute erreur du code utilisateur est renvoyée
        return {"ok": False, "error": _explain(exc), "line": _error_line(exc)}

    model = namespace.get("model")
    graph = namespace.get("graph", model)
    try:
        if _is_langgraph(graph):
            from server.agent_analyzer import analyze_agent

            return analyze_agent(code, namespace, graph)
        if model is not None and not isinstance(model, nn.Module) and _is_sklearn(model):
            from server.sk_analyzer import analyze_sklearn

            return analyze_sklearn(code, namespace, model)
    except ImportError as exc:
        return {"ok": False, "error": f"Bibliothèque manquante : {exc.name}. Lance pip install -r requirements.txt.", "line": None}
    except Exception as exc:  # noqa: BLE001
        return {"ok": False, "error": _explain(exc), "line": _error_line(exc)}
    # PyTorch, ou rien de reconnu : l'analyseur PyTorch donne le bon message d'erreur.
    return analyzer.analyze(code, pixels)
