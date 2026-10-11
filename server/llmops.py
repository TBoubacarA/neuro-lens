"""LLMOps et AgentOps : ce qui se passe quand un agent tourne, appel par appel.

Pendant que l'agent (server/agents.py) s'exécute, on note une « trace » : chaque étape du graphe avec
sa durée, et chaque appel à un modèle de langage avec son modèle, ses tokens, son coût, ce qu'il a
décidé. C'est le rôle d'outils comme Langfuse ou LangSmith en production.

Trois réglages facultatifs dans le code de l'éditeur :
  PRIX    = {"claude-opus-5-5": {"entree": 5.0, "sortie": 25.0}}  dollars par million de tokens
  LIMITES = {"appels": 8, "tokens": 20000}                         garde-fou : l'agent s'arrête au-delà
  cas     = [{"question": "...", "attendu": "..."}, ...]            jeu d'évaluation : l'agent est noté dessus
  variantes = {"court": {"SYSTEM": "..."}, "détaillé": {"SYSTEM": "..."}}  versions à comparer : chaque
            variante remplace ces variables globales le temps de l'évaluation (prompt, outils, modèle...)
"""

from __future__ import annotations

import time
from contextlib import contextmanager

MAX_TEXT = 160
DEFAULT_LIMITS = {"appels": 12, "tokens": 200_000}  # même sans LIMITES : on protège le porte-monnaie
MAX_CASES = 8
MAX_VARIANTS = 4
EVAL_STEPS = 40


class BudgetExceeded(RuntimeError):
    """Levée dans l'appel au modèle qui dépasserait le budget : l'agent s'arrête proprement."""


class Trace:
    def __init__(self, limits: dict) -> None:
        self.t0 = time.perf_counter()
        self.spans: list[dict] = []
        self.limits = limits
        self.calls = 0
        self.tokens = 0
        self.stopped: str | None = None

    def now(self) -> float:
        return round((time.perf_counter() - self.t0) * 1000, 1)

    def add(self, kind: str, name: str, start: float, **extra) -> dict:
        span = {"kind": kind, "name": name, "start": start, "end": self.now(), **extra}
        self.spans.append(span)
        return span

    def check(self) -> None:
        """Avant un appel au modèle : a-t-on encore le droit ?"""
        if self.calls >= self.limits.get("appels", DEFAULT_LIMITS["appels"]):
            self.stopped = f"Garde-fou : {self.calls} appels au modèle, la limite LIMITES['appels'] est atteinte."
        elif self.tokens >= self.limits.get("tokens", DEFAULT_LIMITS["tokens"]):
            self.stopped = f"Garde-fou : {self.tokens} tokens consommés, la limite LIMITES['tokens'] est atteinte."
        if self.stopped:
            raise BudgetExceeded(self.stopped)


_current: Trace | None = None


def _short(value) -> str:
    text = value if isinstance(value, str) else repr(value)
    return text if len(text) <= MAX_TEXT else text[: MAX_TEXT - 1] + "…"


def _usage(response, field: str) -> int:
    usage = getattr(response, "usage", None)
    return int(getattr(usage, field, 0) or 0) if usage is not None else 0


def _record_llm(trace: Trace, start: float, kwargs: dict, response) -> None:
    content = getattr(response, "content", None) or []
    tools = [getattr(b, "name", "?") for b in content if getattr(b, "type", None) == "tool_use"]
    text = "".join(getattr(b, "text", "") for b in content if getattr(b, "type", None) == "text")
    tokens_in = _usage(response, "input_tokens")
    tokens_out = _usage(response, "output_tokens")
    cached = _usage(response, "cache_read_input_tokens")
    messages = kwargs.get("messages") or []
    trace.calls += 1
    trace.tokens += tokens_in + tokens_out
    trace.add(
        "llm", str(getattr(response, "model", None) or kwargs.get("model") or "modèle"), start,
        model=str(kwargs.get("model") or getattr(response, "model", "?")),
        tokens_in=tokens_in, tokens_out=tokens_out, cached=cached,
        messages=len(messages), tools=tools,
        stop=str(getattr(response, "stop_reason", "") or ""),
        text=_short(text) if text else None,
    )


def _wrap_create(original):
    def create(self, *args, **kwargs):
        trace = _current
        if trace is None:
            return original(self, *args, **kwargs)
        trace.check()
        start = trace.now()
        response = original(self, *args, **kwargs)
        _record_llm(trace, start, kwargs, response)
        return response

    return create


def _wrap_stream(original):
    def stream(self, *args, **kwargs):
        trace = _current
        if trace is None:
            yield from original(self, *args, **kwargs)
            return
        start = trace.now()
        for chunk in original(self, *args, **kwargs):
            if isinstance(chunk, dict):
                for name, update in chunk.items():
                    keys = list(update) if isinstance(update, dict) else []
                    trace.add("step", str(name), start, changed=keys)
            start = trace.now()
            yield chunk

    return stream


@contextmanager
def trace(namespace: dict):
    """Pendant le bloc, les étapes du graphe et les appels au modèle sont notés dans la trace rendue."""
    global _current
    limits = namespace.get("LIMITES") if isinstance(namespace.get("LIMITES"), dict) else {}
    tr = Trace({**DEFAULT_LIMITS, **limits})
    patched = []
    try:
        from langgraph.pregel import Pregel

        patched.append((Pregel, "stream", Pregel.stream))
        Pregel.stream = _wrap_stream(Pregel.stream)
    except ImportError:
        pass
    try:
        from anthropic.resources.beta.messages.messages import Messages as BetaMessages
        from anthropic.resources.messages.messages import Messages

        for cls in (BetaMessages, Messages):
            patched.append((cls, "create", cls.create))
            cls.create = _wrap_create(cls.create)
    except ImportError:
        pass
    previous, _current = _current, tr
    try:
        yield tr
    finally:
        _current = previous
        for cls, attr, fn in patched:
            setattr(cls, attr, fn)


def _prices(namespace: dict) -> dict:
    prices = namespace.get("PRIX")
    return prices if isinstance(prices, dict) else {}


def _cost(spans: list[dict], prices: dict) -> float | None:
    """Coût en dollars, si le code donne le tarif de chaque modèle utilisé ; sinon None."""
    llm = [s for s in spans if s["kind"] == "llm"]
    if not llm:
        return 0.0
    total = 0.0
    for s in llm:
        rate = prices.get(s["model"]) or next((v for k, v in prices.items() if s["model"].startswith(k)), None)
        if not isinstance(rate, dict):
            return None
        total += (s["tokens_in"] * float(rate.get("entree", 0)) + s["tokens_out"] * float(rate.get("sortie", 0))) / 1e6
    return round(total, 6)


def summary(tr: Trace, namespace: dict) -> dict:
    """La trace, résumée pour la page : les spans, les totaux, le coût."""
    llm = [s for s in tr.spans if s["kind"] == "llm"]
    steps = [s for s in tr.spans if s["kind"] == "step"]
    tools: dict[str, int] = {}
    for s in llm:
        for t in s["tools"]:
            tools[t] = tools.get(t, 0) + 1
    return {
        "spans": tr.spans,
        "duration": tr.now(),
        "steps": len(steps),
        "calls": len(llm),
        "tokens_in": sum(s["tokens_in"] for s in llm),
        "tokens_out": sum(s["tokens_out"] for s in llm),
        "cached": sum(s["cached"] for s in llm),
        "cost": _cost(tr.spans, _prices(namespace)),
        "priced": bool(_prices(namespace)),
        "tools": tools,
        "limits": tr.limits,
        "stopped": tr.stopped,
    }


def _normalize(text) -> str:
    return "".join(ch for ch in str(text).lower() if ch.isalnum())


def _answer(final: dict):
    """La réponse de l'agent : la clé `reponse` (ou `answer`), sinon la dernière valeur texte."""
    for key in ("reponse", "réponse", "answer", "output"):
        if key in final:
            return final[key]
    texts = [v for v in final.values() if isinstance(v, str)]
    return texts[-1] if texts else ""


def _run_cases(graph, namespace: dict, cases: list, base: dict) -> list[dict]:
    prices = _prices(namespace)
    rows = []
    for case in cases[:MAX_CASES]:
        if not isinstance(case, dict):
            continue
        expected = case.get("attendu", "")
        state = {**base, **{k: v for k, v in case.items() if k != "attendu"}}
        current = dict(state)
        stopped = None
        with trace(namespace) as tr:
            try:
                for chunk in graph.stream(state, stream_mode="updates", config={"recursion_limit": EVAL_STEPS}):
                    for update in chunk.values():
                        if isinstance(update, dict):
                            current.update(update)
            except BudgetExceeded as exc:
                stopped = str(exc)
            except Exception as exc:  # noqa: BLE001 - un cas qui plante est un cas raté
                stopped = ("L'agent tourne en rond." if type(exc).__name__ == "GraphRecursionError"
                           else f"{type(exc).__name__} : {exc}")
        got = _answer(current)
        llm = [s for s in tr.spans if s["kind"] == "llm"]
        rows.append({
            "input": _short(next((v for k, v in case.items() if k != "attendu"), "")),
            "expected": _short(expected),
            "got": _short(got),
            "ok": bool(_normalize(expected)) and _normalize(expected) in _normalize(got),
            "steps": sum(1 for s in tr.spans if s["kind"] == "step"),
            "calls": len(llm),
            "tokens": sum(s["tokens_in"] + s["tokens_out"] for s in llm),
            "cost": _cost(tr.spans, prices),
            "duration": tr.now(),
            "stopped": stopped,
        })
    return rows


_MISSING = object()


def evaluate(graph, namespace: dict) -> dict | None:
    """Fait tourner l'agent sur chaque cas de `cas` et le note : la réponse attendue doit s'y trouver.

    Avec `variantes`, chaque variante est évaluée à son tour, ses variables globales remplaçant celles du
    code le temps de ses cas : c'est ainsi qu'on compare deux prompts ou deux modèles sur le même jeu.
    """
    cases = namespace.get("cas")
    base = namespace.get("state")
    if not isinstance(cases, list) or not isinstance(base, dict):
        return None
    if hasattr(graph, "compile"):
        graph = graph.compile()
    variants = namespace.get("variantes")
    if not isinstance(variants, dict) or not all(isinstance(v, dict) for v in variants.values()):
        variants = {"actuel": {}}
    results = []
    for name, overrides in list(variants.items())[:MAX_VARIANTS]:
        saved = {k: namespace.get(k, _MISSING) for k in overrides}
        namespace.update(overrides)
        try:
            rows = _run_cases(graph, namespace, cases, base)
        finally:
            for k, v in saved.items():
                if v is _MISSING:
                    namespace.pop(k, None)
                else:
                    namespace[k] = v
        costs = [r["cost"] for r in rows]
        results.append({
            "name": str(name), "rows": rows, "passed": sum(r["ok"] for r in rows), "total": len(rows),
            "tokens": sum(r["tokens"] for r in rows), "calls": sum(r["calls"] for r in rows),
            "cost": None if any(c is None for c in costs) else round(sum(costs), 6),
            "duration": round(sum(r["duration"] for r in rows), 1),
            "changes": sorted(str(k) for k in overrides),
        })
    first = results[0]
    return {"variants": results, "rows": first["rows"], "passed": first["passed"], "total": first["total"],
            "skipped": max(0, len(cases) - MAX_CASES), "skipped_variants": max(0, len(variants) - MAX_VARIANTS)}
