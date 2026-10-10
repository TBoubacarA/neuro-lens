"""Vue « agents IA » : dessine le graphe d'un agent LangGraph et le chemin qu'il suit.

Le code définit `graph` (le résultat de `StateGraph(...).compile()`) et, pour le faire
tourner, `inputs` (l'état de départ). On renvoie les nœuds, les arêtes (conditionnelles
ou non) et la suite des nœuds réellement exécutés avec ce que chacun a modifié.
"""

from __future__ import annotations

import re

MAX_STEPS = 40
MAX_TEXT = 300


def _short(v) -> str:
    s = repr(v) if not isinstance(v, str) else v
    return s if len(s) <= MAX_TEXT else s[: MAX_TEXT - 1] + "…"


def _node_lines(code: str) -> dict[str, int]:
    """Ligne où chaque nœud est ajouté : builder.add_node("nom", ...)."""
    out = {}
    pat = re.compile(r"add_node\(\s*['\"]([^'\"]+)['\"]")
    for i, line in enumerate(code.splitlines(), 1):
        m = pat.search(line)
        if m and m.group(1) not in out:
            out[m.group(1)] = i
    return out


def _def_lines(code: str) -> dict[str, int]:
    return {m.group(1): i for i, line in enumerate(code.splitlines(), 1) if (m := re.match(r"\s*def\s+(\w+)\s*\(", line))}


def _kind(node_id: str) -> str:
    if node_id == "__start__":
        return "start"
    if node_id == "__end__":
        return "end"
    if re.search(r"tool|outil|calcul|search|recherche|api", node_id, re.I):
        return "tool"
    return "agent"


def analyze_agent(code: str, namespace: dict, graph) -> dict:
    g = graph.get_graph()
    lines = _node_lines(code)
    defs = _def_lines(code)
    nodes = [
        {"id": nid, "label": {"__start__": "Début", "__end__": "Fin"}.get(nid, nid),
         "kind": _kind(nid), "line": lines.get(nid), "def_line": defs.get(nid)}
        for nid in g.nodes
    ]
    edges = [
        {"source": e.source, "target": e.target, "conditional": bool(e.conditional),
         "label": str(e.data) if e.data not in (None, e.target) else None}
        for e in g.edges
    ]

    result = {"ok": True, "kind": "agent", "nodes": nodes, "edges": edges, "trace": []}
    inputs = namespace.get("inputs")
    if inputs is None:
        result["note"] = "Définis `inputs` (l'état de départ) pour voir le chemin suivi par l'agent."
        return result

    result["inputs"] = _short(inputs)
    state = dict(inputs) if isinstance(inputs, dict) else {}
    try:
        for chunk in graph.stream(inputs, {"recursion_limit": MAX_STEPS}, stream_mode="updates"):
            for node, update in chunk.items():
                result["trace"].append({
                    "node": node,
                    "line": defs.get(node, lines.get(node)),
                    "update": {k: _short(v) for k, v in update.items()} if isinstance(update, dict) else _short(update),
                })
                if isinstance(update, dict):
                    state.update(update)
    except Exception as exc:  # noqa: BLE001 - on montre le chemin jusqu'à l'erreur
        from server.analyzer import _error_line, _explain

        result.update({"ok": False, "error": _explain(exc), "line": _error_line(exc)})
        if "recursion" in str(exc).lower():
            result["error"] = f"L'agent tourne en rond : plus de {MAX_STEPS} étapes sans atteindre la fin."
    result["final_state"] = {k: _short(v) for k, v in state.items()}
    return result
