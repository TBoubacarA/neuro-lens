"""Décrit un agent écrit avec LangGraph : ses étapes, ses embranchements, et un déroulé réel.

Le code de l'éditeur définit `graph` (un StateGraph, compilé ou non). S'il définit aussi `state`
(l'état de départ), on fait tourner l'agent et on note ce que chaque étape a changé.
"""

from __future__ import annotations

import re

MAX_STEPS = 40
MAX_TEXT = 160  # longueur maximale d'une valeur affichée


def is_langgraph(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "langgraph"


def _short(value) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1] + "…"


def _node_line(code: str, name: str) -> int | None:
    """Ligne où l'étape est ajoutée au graphe, sinon celle de la fonction du même nom."""
    quoted = re.escape(name)
    for pattern in (rf"add_node\(\s*[\"']{quoted}[\"']", rf"def\s+{quoted}\s*\("):
        for i, line in enumerate(code.splitlines()):
            if re.search(pattern, line):
                return i + 1
    return None


def describe(graph, namespace: dict, code: str) -> dict:
    if hasattr(graph, "compile"):  # un StateGraph pas encore compilé
        graph = graph.compile()
    drawn = graph.get_graph()

    nodes = []
    for key, node in drawn.nodes.items():
        kind = "start" if key == "__start__" else "end" if key == "__end__" else "step"
        nodes.append({"id": key, "kind": kind, "line": _node_line(code, node.name) if kind == "step" else None})
    edges = [
        {"source": e.source, "target": e.target, "conditional": bool(e.conditional), "label": None if e.data is None else str(e.data)}
        for e in drawn.edges
    ]
    result: dict = {"ok": True, "kind": "agent", "nodes": nodes, "edges": edges, "steps": []}

    state = namespace.get("state")
    if not isinstance(state, dict):
        result["hint"] = "Définis `state` (l'état de départ, un dictionnaire) pour voir l'agent tourner étape par étape."
        return result

    result["start"] = {k: _short(v) for k, v in state.items()}
    current = dict(state)
    try:
        for chunk in graph.stream(state, stream_mode="updates", config={"recursion_limit": MAX_STEPS}):
            for name, update in chunk.items():
                changes = update if isinstance(update, dict) else {}
                current.update(changes)
                result["steps"].append({"node": name, "changes": {k: _short(v) for k, v in changes.items()}})
    except Exception as exc:  # noqa: BLE001 - on montre le déroulé jusqu'à l'erreur
        too_long = type(exc).__name__ == "GraphRecursionError"
        result["stopped"] = (
            f"L'agent tourne en rond : arrêté après {MAX_STEPS} étapes." if too_long else f"{type(exc).__name__} : {exc}"
        )
    result["final"] = {k: _short(v) for k, v in current.items()}
    return result
