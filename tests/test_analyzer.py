from fastapi.testclient import TestClient

from server.analyzer import analyze

CNN = """import torch.nn as nn
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(8 * 14 * 14, 10),
)
"""

CLASS_MODEL = """import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 4, 3)
        self.fc = nn.Linear(4 * 26 * 26, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        return self.fc(x.flatten(1))

model = Net()
"""


def test_layers_shapes_and_lines():
    res = analyze(CNN)
    assert res["ok"]
    got = [(l["type"], l["shape"], l["line"]) for l in res["layers"]]
    assert got == [
        ("Conv2d", [1, 8, 28, 28], 3),
        ("ReLU", [1, 8, 28, 28], 4),
        ("MaxPool2d", [1, 8, 14, 14], 5),
        ("Flatten", [1, 1568], 6),
        ("Linear", [1, 10], 7),
    ]
    assert res["layers"][0]["preview"]["kind"] == "maps"
    assert len(res["layers"][0]["preview"]["maps"]) == 8
    assert len(res["prediction"]["probs"]) == 10


def test_class_model_lines_follow_attributes():
    res = analyze(CLASS_MODEL)
    assert res["ok"]
    assert [(l["name"], l["line"]) for l in res["layers"]] == [("conv1", 7), ("fc", 8)]


def test_shape_mismatch_points_at_failing_layer():
    res = analyze(CNN.replace("8 * 14 * 14", "100"))
    assert not res["ok"]
    assert res["line"] == 7
    assert "in_features=1568" in res["error"]
    assert len(res["layers"]) == 4  # les couches avant l'erreur restent visibles


def test_syntax_error_has_line():
    res = analyze("import torch.nn as nn\nmodel = nn.Sequential(\n")
    assert not res["ok"] and res["line"] is not None


def test_missing_model():
    res = analyze("x = 1")
    assert not res["ok"] and "model" in res["error"]


def test_drawn_pixels_are_used():
    blank = [[0.0] * 28 for _ in range(28)]
    res = analyze(CNN, blank)
    assert res["ok"] and res["input"]["preview"]["max"] == 0.0


def test_api_and_timeout(monkeypatch):
    import server.app as app_module

    monkeypatch.setattr(app_module, "TIMEOUT_S", 3)
    with TestClient(app_module.app) as client:
        assert client.get("/").status_code == 200
        r = client.post("/api/analyze", json={"code": CNN})
        assert r.json()["ok"]
        r = client.post("/api/analyze", json={"code": "while True:\n    pass\n"})
        assert not r.json()["ok"] and "boucle" in r.json()["error"]
        r = client.post("/api/analyze", json={"code": CNN})  # le processus a redémarré
        assert r.json()["ok"]


TRANSFORMER = """import torch.nn as nn

class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(100, 16)
        self.block = nn.TransformerEncoderLayer(16, nhead=4, dim_feedforward=32, batch_first=True)
        self.head = nn.Linear(16, 3)

    def forward(self, ids):
        x = self.block(self.embed(ids))
        return self.head(x.mean(dim=1))

model = MiniTransformer()
"""


def test_text_model_shows_embeddings_and_attention():
    res = analyze(TRANSFORMER, text="le chat et le chien")
    assert res["ok"]
    assert res["input"]["tokens"] == ["le", "chat", "et", "le", "chien"]
    ids = res["input"]["preview"]["ids"]
    assert ids[0] == ids[3]  # même mot, même numéro

    embed = res["layers"][0]
    assert (embed["type"], embed["shape"], embed["line"]) == ("Embedding", [1, 5, 16], 6)
    assert embed["preview"]["kind"] == "tokens"
    assert embed["preview"]["rows"][0] == embed["preview"]["rows"][3]
    assert len(embed["preview"]["points"]) == 5

    attn = next(l for l in res["layers"] if l["type"] == "MultiheadAttention")
    assert attn["line"] == 7 and attn["group"]["type"] == "TransformerEncoderLayer"
    heads = attn["preview"]["attention"]
    assert len(heads) == 4 and len(heads[0]) == 5
    assert abs(sum(heads[0][0]) - 1) < 0.01  # chaque token répartit 100 % de son attention
    assert not any(l["name"].startswith("block.self_attn.") for l in res["layers"])
    assert len(res["prediction"]["probs"]) == 3
    lines = {l["name"]: l.get("line") for l in res["layers"]}
    assert lines["block.linear1"] == 7 and lines["head"] == 8  # le bloc ne vole pas la ligne de `head`


def test_data_flow_finds_shortcuts():
    res = analyze(TRANSFORMER, text="un deux trois")
    flow = {l["name"]: [res["layers"][i]["name"] if i >= 0 else "entrée" for i in l["inputs"]] for l in res["layers"]}
    assert flow["embed"] == ["entrée"]
    assert flow["block.self_attn"] == ["embed"]
    # Add + Normalize : la normalisation reçoit l'attention ET le raccourci venu de l'embedding
    assert flow["block.norm1"] == ["embed", "block.dropout1"]
    assert flow["block.norm2"] == ["block.norm1", "block.dropout2"]
    assert [res["layers"][i]["name"] for i in res["output_from"]] == ["head"]
    assert [l["inputs"] for l in analyze(CNN)["layers"]] == [[-1], [0], [1], [2], [3]]


VIT = """import torch
import torch.nn as nn

class MiniViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patches = nn.Conv2d(1, 16, kernel_size=7, stride=7)
        self.cls = nn.Parameter(torch.zeros(1, 1, 16))
        self.head = nn.Linear(16, 10)

    def forward(self, image):
        x = self.patches(image).flatten(2).transpose(1, 2)
        x = torch.cat([self.cls, x], dim=1)
        return self.head(x[:, 0])

model = MiniViT()
"""


def test_patches_and_probability_rows():
    res = analyze(VIT)
    assert res["ok"]
    assert res["layers"][0]["preview"]["patches"] == {"size": [7, 7], "grid": [4, 4]}
    soft = analyze(TRANSFORMER.replace("self.head = nn.Linear(16, 3)", "self.head = nn.Softmax(dim=-1)")
                   .replace("self.head(x.mean(dim=1))", "self.head(x)"))
    assert soft["layers"][-1]["preview"]["probs"] is True
    assert soft["layers"][0]["preview"]["probs"] is False


TREE = """from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

X, y = make_moons(n_samples=120, noise=0.2, random_state=0)
model = DecisionTreeClassifier(max_depth=3)
"""


def test_sklearn_tree_is_trained_and_described():
    res = analyze(TREE)
    assert res["ok"] and res["kind"] == "ml" and res["line"] == 5
    assert res["estimator"] == "DecisionTreeClassifier" and res["trained_here"]
    assert res["classes"] == ["0", "1"] and 0.8 < res["score"] <= 1
    root = res["tree"]["root"]
    assert root["samples"] == 120 and root["feature"] in ("x1", "x2") and len(root["children"]) == 2
    regions = res["scatter"]["regions"]["cells"]
    assert len(regions) == len(regions[0]) and {c for row in regions for c in row} == {0, 1}
    assert len(res["scatter"]["points"]) == 120 and len(res["importances"]) == 2


def test_sklearn_regression_and_clustering():
    reg = analyze("import numpy as np\nfrom sklearn.linear_model import LinearRegression\n"
                  "X = np.arange(20).reshape(-1, 1)\ny = 3 * X[:, 0] + 1\nmodel = LinearRegression()\n")
    assert reg["ok"] and reg["task"] == "regression" and reg["score"] > 0.99
    assert abs(reg["coefficients"][0]["value"] - 3) < 1e-6 and len(reg["curve"]["x"]) == 80
    km = analyze("from sklearn.datasets import load_iris\nfrom sklearn.cluster import KMeans\n"
                 "X = load_iris().data\nmodel = KMeans(n_clusters=3, n_init=5, random_state=0)\n")
    assert km["ok"] and km["task"] == "clustering" and km["scatter"]["projected"]
    assert set(km["scatter"]["labels"]) == {0, 1, 2}
    assert not analyze("from sklearn.svm import SVC\nmodel = SVC()\n")["ok"]  # pas de données


AGENT = """from typing import TypedDict
from langgraph.graph import StateGraph, START, END

class State(TypedDict):
    notes: list
    answer: str

def think(state):
    return {"notes": state["notes"] + ["?"]}

def route(state):
    return "tool" if len(state["notes"]) < 2 else "done"

def tool(state):
    return {"notes": state["notes"] + ["42"]}

def reply(state):
    return {"answer": state["notes"][-1]}

graph = StateGraph(State)
graph.add_node("think", think)
graph.add_node("tool", tool)
graph.add_node("reply", reply)
graph.add_edge(START, "think")
graph.add_conditional_edges("think", route, {"tool": "tool", "done": "reply"})
graph.add_edge("tool", "think")
graph.add_edge("reply", END)

state = {"notes": [], "answer": ""}
"""


def test_agent_graph_and_run():
    res = analyze(AGENT)
    assert res["ok"] and res["kind"] == "agent"
    assert [n["id"] for n in res["nodes"]] == ["__start__", "think", "tool", "reply", "__end__"]
    assert next(n for n in res["nodes"] if n["id"] == "tool")["line"] == 22
    assert {"source": "think", "target": "reply", "conditional": True, "label": "done"} in res["edges"]
    assert [s["node"] for s in res["steps"]] == ["think", "tool", "think", "reply"]
    assert res["final"]["answer"] == "?"
    loop = analyze(AGENT.replace("< 2", "< 999"))
    assert "tourne en rond" in loop["stopped"] and len(loop["steps"]) > 10
    assert "state" in analyze(AGENT.replace("state = {", "start = {"))["hint"]


def test_training_improves_and_is_reused_by_analysis():
    from server import trainer
    from server.analyzer import train

    code = CNN.replace("model =", "torch.manual_seed(0)\nmodel =")
    first = train(code)
    assert first["ok"] and first["training"]["epochs"][0]["seen"] == first["n_train"]
    for _ in range(2):
        last = train(code)
    accs = [e["test_acc"] for e in last["training"]["epochs"]]
    assert len(accs) == 3 and accs[-1] > 0.5 and accs[-1] > accs[0]
    res = analyze(code)
    assert res["trainable"] and len(res["training"]["epochs"]) == 3
    trainer.reset(code)
    assert analyze(code)["training"] is None
    assert not train(TRANSFORMER)["ok"]  # texte : pas d'entraînement sur les chiffres
