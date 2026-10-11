"""Décrit un modèle scikit-learn : ses données, ce qu'il a appris, la frontière qu'il trace.

Le code de l'éditeur définit `model` (un estimateur scikit-learn) et ses données `X` (et `y`
pour l'apprentissage supervisé). Si le modèle n'est pas encore entraîné, on l'entraîne ici.
"""

from __future__ import annotations

MAX_POINTS = 300  # points renvoyés pour le nuage
GRID = 44  # côté de la grille qui dessine la frontière de décision
MAX_TREE_DEPTH = 4
MAX_FEATURES = 12  # coefficients / importances renvoyés


def is_sklearn(obj) -> bool:
    return type(obj).__module__.split(".")[0] == "sklearn"


def _task(model) -> str:
    from sklearn.base import is_classifier, is_regressor

    if is_classifier(model):
        return "classification"
    if is_regressor(model):
        return "regression"
    return "clustering"


def _final(model):
    """Dans un Pipeline, le modèle qui décide est la dernière étape."""
    return model.steps[-1][1] if hasattr(model, "steps") else model


def _tree(estimator, names: list[str], classes: list[str]) -> dict | None:
    """Arbre de décision sous forme de nœuds imbriqués (coupé à MAX_TREE_DEPTH niveaux)."""
    t = getattr(estimator, "tree_", None)
    if t is None:
        return None

    def node(i: int, depth: int) -> dict:
        value = t.value[i][0]
        out = {"samples": int(t.n_node_samples[i])}
        if classes:
            out["label"] = classes[int(value.argmax())]
            out["cls"] = int(value.argmax())
            out["purity"] = round(float(value.max() / value.sum()), 3)
        else:
            out["label"] = f"{float(value[0]):.3g}"
        if t.children_left[i] != -1:
            if depth >= MAX_TREE_DEPTH:
                out["cut"] = True  # l'arbre continue, mais on ne dessine pas plus bas
            else:
                out["feature"] = names[t.feature[i]]
                out["threshold"] = round(float(t.threshold[i]), 3)
                out["children"] = [node(int(t.children_left[i]), depth + 1), node(int(t.children_right[i]), depth + 1)]
        return out

    return {"root": node(0, 0), "depth": int(t.max_depth), "leaves": int(t.n_leaves)}


def _bars(values, names: list[str]) -> list[dict]:
    pairs = sorted(zip(names, (float(v) for v in values)), key=lambda p: -abs(p[1]))
    return [{"name": n, "value": round(v, 4)} for n, v in pairs[:MAX_FEATURES]]


def _fitted(model) -> bool:
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False


LOGS = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30, 100]
# Le réglage le plus parlant de chaque modèle, et les valeurs à essayer.
KNOBS = {
    "DecisionTreeClassifier": ("max_depth", list(range(1, 13))),
    "DecisionTreeRegressor": ("max_depth", list(range(1, 13))),
    "RandomForestClassifier": ("max_depth", list(range(1, 11))),
    "RandomForestRegressor": ("max_depth", list(range(1, 11))),
    "GradientBoostingClassifier": ("n_estimators", [1, 3, 10, 30, 100]),
    "KNeighborsClassifier": ("n_neighbors", [1, 3, 5, 7, 11, 15, 21, 31]),
    "KNeighborsRegressor": ("n_neighbors", [1, 3, 5, 7, 11, 15, 21, 31]),
    "SVC": ("C", LOGS),
    "LogisticRegression": ("C", LOGS),
    "Ridge": ("alpha", LOGS),
    "KMeans": ("n_clusters", list(range(2, 9))),
}
HELD_OUT = 0.3  # part des exemples mise de côté pour mesurer sans tricher


def _knob(model, X, y, task: str) -> dict | None:
    """Essaie plusieurs valeurs du réglage principal : score sur les exemples appris et sur des exemples mis de côté.

    C'est ce qui fait voir le sur-apprentissage : la première courbe monte toujours, la seconde finit par redescendre.
    """
    import numpy as np
    from sklearn.base import clone

    core = _final(model)
    if type(core).__name__ not in KNOBS:
        return None
    name, values = KNOBS[type(core).__name__]
    path = f"{model.steps[-1][0]}__{name}" if hasattr(model, "steps") else name
    current = core.get_params(deep=False).get(name)
    knob: dict = {"name": name, "values": values, "value": current}
    if task == "clustering":
        knob["inertia"] = [round(float(clone(model).set_params(**{path: v}).fit(X).inertia_), 3) for v in values]
        return knob
    if y is None or len(X) < 20:
        return knob
    order = np.random.default_rng(0).permutation(len(X))
    cut = int(len(X) * (1 - HELD_OUT))
    a, b = order[:cut], order[cut:]
    knob["train"], knob["test"] = [], []
    for v in values:
        if name == "n_neighbors" and v > len(a):
            knob["values"] = values[: len(knob["train"])]
            break
        trial = clone(model).set_params(**{path: v}).fit(X[a], y[a])
        knob["train"].append(round(float(trial.score(X[a], y[a])), 4))
        knob["test"].append(round(float(trial.score(X[b], y[b])), 4))
    knob["held_out"] = len(b)
    return knob


def describe(model, namespace: dict, override: dict | None = None) -> dict:
    import numpy as np

    X = namespace.get("X")
    if X is None:
        return {"ok": False, "error": "Définis les données dans une variable `X` (et `y` pour les réponses attendues).", "line": None}
    names = [str(c) for c in X.columns] if hasattr(X, "columns") else None
    X = np.asarray(X, dtype=float)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    names = names or [f"x{i + 1}" for i in range(X.shape[1])]
    y = namespace.get("y")
    y = None if y is None else np.asarray(y)
    task = _task(model)

    if override:  # réglage choisi au curseur : on remplace la valeur du code et on réentraîne
        name, values = KNOBS.get(type(_final(model)).__name__, (None, []))
        override = {k: v for k, v in override.items() if k == name and v in values}
    if override:
        from sklearn.base import clone

        path = f"{model.steps[-1][0]}__" if hasattr(model, "steps") else ""
        model = clone(model).set_params(**{path + k: v for k, v in override.items()})
    trained_here = not _fitted(model)
    if trained_here:
        model.fit(X) if y is None or task == "clustering" else model.fit(X, y)

    core = _final(model)
    result: dict = {
        "ok": True,
        "kind": "ml",
        "estimator": type(core).__name__,
        "steps": [type(s).__name__ for _, s in model.steps] if hasattr(model, "steps") else [],
        "task": task,
        "trained_here": trained_here,
        "n_samples": int(X.shape[0]),
        "features": names,
        "params": {k: v for k, v in core.get_params(deep=False).items() if isinstance(v, (int, float, str, bool))},
    }

    # Ce que le modèle répond pour chaque exemple.
    if hasattr(model, "predict"):
        pred = np.asarray(model.predict(X))
    else:  # regroupements sans `predict` (DBSCAN...)
        pred = np.asarray(core.labels_)

    classes: list[str] = []
    if task == "regression":
        shown = pred.astype(float)
    else:
        raw = [str(c) for c in getattr(core, "classes_", np.unique(pred))]
        classes = raw if task == "classification" else [f"groupe {c}" for c in raw]
        index = {c: i for i, c in enumerate(raw)}
        shown = np.array([index[str(p)] for p in pred])
        result["classes"] = classes
    if y is not None and task != "clustering":
        result["score"] = round(float(model.score(X, y)), 4)
    knob = _knob(model, X, y, task)
    if knob:
        result["knob"] = knob

    # Le nuage de points : les 2 colonnes si elles suffisent, sinon une projection ACP.
    rng = np.random.default_rng(0)
    keep = np.sort(rng.choice(len(X), MAX_POINTS, replace=False)) if len(X) > MAX_POINTS else np.arange(len(X))
    if X.shape[1] == 1 and task == "regression" and y is not None:
        xs = np.linspace(X.min(), X.max(), 80)
        result["curve"] = {
            "x": [round(float(v), 4) for v in xs],
            "y": [round(float(v), 4) for v in model.predict(xs.reshape(-1, 1))],
            "points": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(X[keep, 0], y[keep].astype(float))],
            "axes": [names[0], "y"],
        }
    elif task == "regression" and y is not None:
        result["fit"] = {"points": [[round(float(a), 4), round(float(b), 4)] for a, b in zip(y[keep].astype(float), shown[keep])]}
    else:
        if X.shape[1] == 1:
            plane, axes, projected = np.column_stack([X[:, 0], np.zeros(len(X))]), [names[0], ""], False
        elif X.shape[1] == 2:
            plane, axes, projected = X, names, False
        else:
            centered = X - X.mean(axis=0)
            _, _, vt = np.linalg.svd(centered, full_matrices=False)
            plane, axes, projected = centered @ vt[:2].T, ["axe 1", "axe 2"], True
        scatter = {
            "points": [[round(float(a), 4), round(float(b), 4)] for a, b in plane[keep]],
            "labels": [int(v) for v in shown[keep]],
            "axes": axes,
            "projected": projected,
        }
        if y is not None and task == "classification":
            scatter["wrong"] = [bool(a != b) for a, b in zip(pred[keep], y[keep])]
        if X.shape[1] == 2 and hasattr(model, "predict"):  # la frontière : on interroge le modèle partout
            lo, hi = X.min(axis=0), X.max(axis=0)
            margin = (hi - lo) * 0.08 + 1e-9
            gx = np.linspace(lo[0] - margin[0], hi[0] + margin[0], GRID)
            gy = np.linspace(lo[1] - margin[1], hi[1] + margin[1], GRID)
            cells = np.array([[a, b] for b in gy for a in gx])
            answers = [index[str(p)] for p in model.predict(cells)]
            scatter["regions"] = {
                "cells": [answers[r * GRID : (r + 1) * GRID] for r in range(GRID)],
                "x": [float(gx[0]), float(gx[-1])],
                "y": [float(gy[0]), float(gy[-1])],
            }
        result["scatter"] = scatter

    tree = _tree(core, names, classes if task == "classification" else [])
    forest = getattr(core, "estimators_", None)
    if tree is None and forest is not None and hasattr(forest[0], "tree_"):  # une forêt : on montre son premier arbre
        tree = _tree(forest[0], names, classes if task == "classification" else [])
        result["n_trees"] = len(forest)
    if tree:
        result["tree"] = tree
    if hasattr(core, "feature_importances_") and len(core.feature_importances_) == len(names):
        result["importances"] = _bars(core.feature_importances_, names)
    coef = getattr(core, "coef_", None)
    if coef is not None and np.asarray(coef).shape[-1] == len(names):
        coef = np.asarray(coef, dtype=float)
        result["coefficients"] = _bars(coef if coef.ndim == 1 else coef[0], names)
        if coef.ndim == 2 and coef.shape[0] > 1:
            result["coefficients_of"] = classes[0] if classes else None
    return result
