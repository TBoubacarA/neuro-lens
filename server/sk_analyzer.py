"""Vue « ML classique » : décrit un modèle scikit-learn pour que la page le dessine.

Le code définit `model` (un estimateur ou un Pipeline scikit-learn) et les données `X`
(et `y` pour l'apprentissage supervisé). Si le modèle n'est pas encore entraîné, on
l'entraîne sur X, y. On renvoie selon le modèle : l'arbre de décision, la frontière de
décision sur le plan (ou une projection 2D des données), les coefficients, les clusters.
"""

from __future__ import annotations

import re

import numpy as np

GRID = 60  # résolution de la frontière de décision
MAX_POINTS = 300
MAX_DEPTH = 4  # profondeur d'arbre dessinée

EXPLAIN = {
    "DecisionTreeClassifier": "Pose une suite de questions « caractéristique ≤ seuil ? » : chaque feuille donne une classe.",
    "DecisionTreeRegressor": "Pose une suite de questions « caractéristique ≤ seuil ? » : chaque feuille donne une valeur moyenne.",
    "RandomForestClassifier": "Fait voter de nombreux arbres entraînés sur des morceaux différents des données.",
    "RandomForestRegressor": "Fait la moyenne de nombreux arbres entraînés sur des morceaux différents des données.",
    "GradientBoostingClassifier": "Ajoute des petits arbres l'un après l'autre, chacun corrigeant les erreurs des précédents.",
    "LogisticRegression": "Trace une frontière droite et donne la probabilité d'être de chaque côté.",
    "LinearRegression": "Cherche la droite (ou le plan) qui passe au plus près de tous les points.",
    "Ridge": "Régression linéaire qui garde des coefficients petits pour éviter le sur-apprentissage.",
    "Lasso": "Régression linéaire qui pousse les coefficients inutiles à zéro.",
    "SVC": "Cherche la frontière qui laisse la plus grande marge entre les classes.",
    "KNeighborsClassifier": "Donne à chaque point la classe majoritaire de ses k plus proches voisins.",
    "KMeans": "Place k centres et rattache chaque point au centre le plus proche, sans connaître les classes.",
    "DBSCAN": "Regroupe les points situés dans des zones denses ; les points isolés sont du bruit.",
    "GaussianNB": "Applique la règle de Bayes en supposant les caractéristiques indépendantes.",
    "MLPClassifier": "Un petit réseau de neurones entièrement connecté.",
    "PCA": "Trouve les axes qui gardent le plus de variation des données.",
}


def _arr(v):
    if v is None:
        return None
    if hasattr(v, "to_numpy"):
        v = v.to_numpy()
    return np.asarray(v)


def _r(v, nd=4):
    return round(float(v), nd)


def _line_of(code: str, pattern: str) -> int | None:
    pat = re.compile(pattern)
    for i, line in enumerate(code.splitlines(), 1):
        if pat.search(line):
            return i
    return None


def _is_fitted(est) -> bool:
    from sklearn.exceptions import NotFittedError
    from sklearn.utils.validation import check_is_fitted

    try:
        check_is_fitted(est)
        return True
    except NotFittedError:
        return False
    except TypeError:
        return True


def _tree_view(tree, feature_names, class_names, is_classifier) -> dict:
    t = tree.tree_
    nodes = []

    def walk(i: int, depth: int) -> int:
        idx = len(nodes)
        value = t.value[i][0]
        node = {
            "samples": int(t.n_node_samples[i]),
            "impurity": _r(t.impurity[i], 3),
            "leaf": bool(t.children_left[i] == -1),
            "depth": depth,
        }
        if is_classifier:
            dist = value / value.sum() if value.sum() else value
            k = int(np.argmax(dist))
            node["label"] = class_names[k] if k < len(class_names) else str(k)
            node["class"] = k
            node["dist"] = [_r(v, 3) for v in dist]
        else:
            node["label"] = f"{value[0]:.3g}"
        nodes.append(node)
        if not node["leaf"]:
            f = t.feature[i]
            node["question"] = f"{feature_names[f]} ≤ {t.threshold[i]:.3g}"
            if depth + 1 < MAX_DEPTH:
                node["children"] = [walk(t.children_left[i], depth + 1), walk(t.children_right[i], depth + 1)]
            else:
                node["truncated"] = True
        return idx

    walk(0, 0)
    return {"kind": "tree", "nodes": nodes, "depth": int(tree.get_depth()), "leaves": int(tree.get_n_leaves())}


def _labels(model, final, X, y):
    """Étiquettes des points : la vérité si on l'a, sinon ce que le modèle a trouvé."""
    if y is not None:
        return y
    if hasattr(final, "labels_"):
        return final.labels_
    if hasattr(model, "predict"):
        return model.predict(X)
    return None


def _plane_view(model, final, X, y, feature_names, is_classifier, is_cluster) -> dict:
    labels = _labels(model, final, X, y)
    sel = np.arange(len(X))
    if len(X) > MAX_POINTS:
        sel = np.random.default_rng(0).choice(len(X), MAX_POINTS, replace=False)
    view = {"kind": "plane", "axes": feature_names[:2]}
    if X.shape[1] == 2:
        P = X.astype(float)
        if hasattr(model, "predict"):
            (x0, x1), (y0, y1) = [(P[:, j].min(), P[:, j].max()) for j in (0, 1)]
            px, py = (x1 - x0) * 0.08 or 1, (y1 - y0) * 0.08 or 1
            xs = np.linspace(x0 - px, x1 + px, GRID)
            ys = np.linspace(y0 - py, y1 + py, GRID)
            gx, gy = np.meshgrid(xs, ys)
            pred = model.predict(np.c_[gx.ravel(), gy.ravel()]).reshape(GRID, GRID)
            view["grid"] = {
                "x": [_r(xs[0]), _r(xs[-1])],
                "y": [_r(ys[0]), _r(ys[-1])],
                "values": pred.tolist() if (is_classifier or is_cluster) else [[_r(v) for v in row] for row in pred],
                "discrete": bool(is_classifier or is_cluster),
            }
        if hasattr(final, "cluster_centers_"):
            view["centers"] = [[_r(a), _r(b)] for a, b in final.cluster_centers_[:, :2]]
    else:
        # Plus de 2 caractéristiques : on projette les points en 2D (ACP) pour les voir.
        from sklearn.decomposition import PCA

        P = PCA(n_components=2).fit_transform(X.astype(float))
        view["axes"] = ["axe principal 1", "axe principal 2"]
        view["projected"] = True
    view["points"] = [
        {"x": _r(P[i, 0]), "y": _r(P[i, 1]), "label": (labels[i].item() if hasattr(labels[i], "item") else labels[i]) if labels is not None else None}
        for i in sel
    ]
    return view


def analyze_sklearn(code: str, namespace: dict, model) -> dict:
    from sklearn.base import is_classifier as sk_is_classifier
    from sklearn.pipeline import Pipeline

    X = _arr(namespace.get("X"))
    y = _arr(namespace.get("y"))
    if X is None:
        return {"ok": False, "error": "Définis tes données dans `X` (et les réponses attendues dans `y`).", "line": None}
    if X.ndim == 1:
        X = X.reshape(-1, 1)

    names = namespace.get("feature_names")
    if names is None and hasattr(namespace.get("X"), "columns"):
        names = list(namespace["X"].columns)
    feature_names = [str(n) for n in names] if names is not None else [f"x{i}" for i in range(X.shape[1])]

    final = model.steps[-1][1] if isinstance(model, Pipeline) else model
    is_cluster = hasattr(final, "n_clusters") or type(final).__name__ in ("DBSCAN", "AgglomerativeClustering")
    is_classifier = sk_is_classifier(model)

    fitted_here = False
    if not _is_fitted(final):
        model.fit(X, None if is_cluster else y)
        fitted_here = True

    cls = namespace.get("class_names")
    if cls is None and hasattr(final, "classes_"):
        cls = final.classes_
    class_names = [str(c) for c in cls] if cls is not None else []

    name = type(final).__name__
    result = {
        "ok": True,
        "kind": "sklearn",
        "estimator": name,
        "explain": EXPLAIN.get(name, ""),
        "line": _line_of(code, r"^\s*model\s*="),
        "fit_line": _line_of(code, r"\.fit\("),
        "fitted_here": fitted_here,
        "n_samples": int(X.shape[0]),
        "features": feature_names,
        "classes": class_names,
        "task": "classification" if is_classifier else "clustering" if is_cluster else "régression",
        "params": {k: (v if isinstance(v, (int, float, str, bool)) or v is None else str(v))
                   for k, v in final.get_params(deep=False).items()},
        "views": [],
    }
    if isinstance(model, Pipeline):
        result["views"].append({"kind": "pipeline", "steps": [[n, type(s).__name__] for n, s in model.steps]})

    if hasattr(final, "tree_"):
        result["views"].append(_tree_view(final, feature_names, class_names, is_classifier))
    elif hasattr(final, "estimators_") and len(getattr(final, "estimators_", [])) and hasattr(np.ravel(final.estimators_)[0], "tree_"):
        first = np.ravel(final.estimators_)[0]
        view = _tree_view(first, feature_names, class_names, is_classifier)
        view["note"] = f"Premier des {len(np.ravel(final.estimators_))} arbres."
        result["views"].append(view)

    result["views"].append(_plane_view(model, final, X, y, feature_names, is_classifier, is_cluster))

    if hasattr(final, "coef_"):
        coef = np.atleast_2d(final.coef_)[0]
        if is_classifier and len(class_names) == 2:
            note = f"Un coefficient positif pousse vers « {class_names[1]} », un négatif vers « {class_names[0]} »."
        elif is_classifier:
            note = f"Coefficients de la classe « {class_names[0]} » contre les autres."
        else:
            note = "Un coefficient positif fait monter la prédiction, un négatif la fait baisser."
        result["views"].append({
            "kind": "bars", "title": "Coefficients", "labels": feature_names[: len(coef)],
            "values": [_r(v) for v in coef], "note": note,
        })
    if hasattr(final, "feature_importances_"):
        result["views"].append({
            "kind": "bars", "title": "Importance des caractéristiques", "labels": feature_names,
            "values": [_r(v) for v in final.feature_importances_],
            "note": "Part de chaque caractéristique dans les décisions du modèle.",
        })

    if y is not None and hasattr(model, "score") and not is_cluster:
        result["score"] = {"name": "précision" if is_classifier else "R²", "value": _r(model.score(X, y))}
    elif hasattr(final, "inertia_"):
        result["score"] = {"name": "inertie", "value": _r(final.inertia_, 2)}
    return result
