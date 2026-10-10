from pathlib import Path

from server.dispatch import analyze

TREE = """from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier
iris = load_iris()
X, y = iris.data[:, 2:], iris.target
class_names = list(iris.target_names)
model = DecisionTreeClassifier(max_depth=2)
model.fit(X, y)
"""

KMEANS = """from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
X, _ = make_blobs(n_samples=100, centers=3, random_state=0)
model = KMeans(n_clusters=3, n_init=3, random_state=0)
"""

AGENT = (Path(__file__).parent.parent / "web" / "examples-extra.js").read_text()


def _agent_code():
    # L'exemple LangGraph du menu, extrait du fichier JS (chaîne JSON).
    import json
    import re

    body = re.search(r"Object\.assign\(window\.NEURO_EXAMPLES, (\{.*\})\);", AGENT, re.S).group(1)
    return json.loads(body)["Agent IA · LangGraph (outil calculatrice)"]


def test_decision_tree():
    res = analyze(TREE)
    assert res["ok"] and res["kind"] == "sklearn" and res["task"] == "classification"
    kinds = [v["kind"] for v in res["views"]]
    assert kinds[:2] == ["tree", "plane"]
    root = res["views"][0]["nodes"][0]
    assert root["samples"] == 150 and "≤" in root["question"]
    assert res["line"] == 6 and not res["fitted_here"]
    assert len(res["views"][1]["grid"]["values"]) == 60


def test_kmeans_is_fitted_and_has_centers():
    res = analyze(KMEANS)
    assert res["ok"] and res["task"] == "clustering" and res["fitted_here"]
    plane = next(v for v in res["views"] if v["kind"] == "plane")
    assert len(plane["centers"]) == 3


def test_more_than_two_features_are_projected():
    res = analyze(TREE.replace("iris.data[:, 2:]", "iris.data"))
    plane = next(v for v in res["views"] if v["kind"] == "plane")
    assert plane["projected"] and "grid" not in plane


def test_sklearn_without_data():
    res = analyze("from sklearn.linear_model import LinearRegression\nmodel = LinearRegression()\n")
    assert not res["ok"] and "X" in res["error"]


def test_agent_graph_and_trace():
    res = analyze(_agent_code())
    assert res["ok"] and res["kind"] == "agent"
    assert {n["id"] for n in res["nodes"]} == {"__start__", "agent", "calculatrice", "__end__"}
    assert any(e["conditional"] and e["target"] == "calculatrice" for e in res["edges"])
    assert [s["node"] for s in res["trace"]] == ["agent", "calculatrice", "agent"]
    assert res["final_state"]["reponse"] == "12 * 7 = 84"


def test_agent_error_points_at_node_code():
    code = _agent_code().replace('eval(state["calcul"])', "1 / 0")
    res = analyze(code)
    assert not res["ok"] and [s["node"] for s in res["trace"]] == ["agent"]
    assert "1 / 0" in code.splitlines()[res["line"] - 1]


def test_torch_still_goes_to_torch_analyzer():
    res = analyze("import torch.nn as nn\nmodel = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))\n")
    assert res["ok"] and len(res["layers"]) == 2
