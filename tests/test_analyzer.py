from fastapi.testclient import TestClient

from server.analyzer import analyze, train

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
    assert len(embed["preview"]["wide"]) == 5 and len(embed["preview"]["wide"][0]) == 16  # vue 3D

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
    assert res["trainable"] and len(res["training"]["epochs"]) == 3 and res["training"]["shown"] == 3
    ev = res["training"]["eval"]  # matrice de confusion sur les chiffres jamais vus, et ses erreurs
    assert sum(ev["confusion"][i][i] for i in range(10)) + ev["n_wrong"] == ev["n_test"] == first["n_test"]
    assert all(m["true"] != m["pred"] and len(m["image"]) == 28 for m in ev["mistakes"])
    # Revoir l'apprentissage : les poids après 1 époque donnent la réussite notée à l'époque 1
    before = analyze(code, epoch=0)
    assert before["training"]["shown"] == 0 and before["training"]["kept"] == 4
    assert analyze(code, epoch=1)["training"]["shown"] == 1
    assert analyze(code, epoch=1)["prediction"] != analyze(code, epoch=3)["prediction"]
    trainer.reset(code)
    assert analyze(code)["training"] is None
    assert not train(TRANSFORMER)["ok"]  # texte : pas d'entraînement sur les chiffres


def _example(name):
    import json
    import re
    from pathlib import Path

    src = (Path(__file__).parent.parent / "web" / "examples.js").read_text()
    start = src.index(json.dumps(name) + ": `") + len(name) + 5
    return re.sub(r"\\`", "`", src[start : src.index("`,", start)])


def test_claude_agent_without_key_uses_the_rule(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    res = analyze(_example("Agent IA avec Claude (LangGraph)"))
    assert res["ok"] and res["llm"] == {"key": False, "live": False}
    assert [s["node"] for s in res["steps"]] == ["reflechir", "chercher", "reflechir", "calculer", "reflechir", "repondre"]
    assert res["final"]["reponse"] == "La tour Eiffel mesure 33 000 cm."


def test_claude_agent_live_follows_the_model(monkeypatch):
    """Un faux client Claude : vérifie qu'on suit ses appels d'outils et qu'on lui rend les résultats."""
    import types

    import anthropic

    calls = []

    def block(**kw):
        return types.SimpleNamespace(**kw)

    def create(**kw):
        calls.append(kw)
        script = [
            [block(type="tool_use", id="t1", name="chercher", input={"sujet": "Tour Eiffel"})],
            [block(type="tool_use", id="t2", name="calculer", input={"operation": "330 * 100"})],
            [block(type="text", text="Elle mesure 33 000 cm.")],
        ]
        return types.SimpleNamespace(content=script[len(calls) - 1], stop_reason="end_turn")

    fake = types.SimpleNamespace(beta=types.SimpleNamespace(messages=types.SimpleNamespace(create=create)))
    monkeypatch.setattr(anthropic, "Anthropic", lambda: fake)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("NEURO_LENS_LIVE", "1")
    res = analyze(_example("Agent IA avec Claude (LangGraph)"))
    assert res["ok"] and res["llm"] == {"key": True, "live": True}
    assert [s["node"] for s in res["steps"]] == ["reflechir", "chercher", "reflechir", "calculer", "reflechir", "repondre"]
    assert res["final"]["reponse"] == "Elle mesure 33 000 cm."
    assert calls[0]["model"] == "claude-opus-5-5" and calls[0]["fallbacks"] == "default"
    results = [m["content"][0] for m in calls[2]["messages"] if m["role"] == "user" and isinstance(m["content"], list)]
    assert [(r["tool_use_id"], r["content"]) for r in results] == [("t1", "La tour Eiffel mesure 330 m."), ("t2", "33000")]


def test_switching_units_off_changes_what_follows():
    base = analyze(CNN)
    cut = analyze(CNN, off=[[1, 0], [1, 3], [1, 99]])  # canaux 0 et 3 après le ReLU ; 99 n'existe pas
    assert cut["ok"] and cut["layers"][1]["off"] == [0, 3] and cut["layers"][0]["off"] == []
    relu = cut["layers"][1]["preview"]["maps"]
    assert max(map(max, relu[0])) == 0 and max(map(max, relu[3])) == 0
    assert cut["layers"][0]["preview"] == base["layers"][0]["preview"]  # avant la coupure : rien ne change
    assert cut["layers"][-1]["preview"]["values"] != base["layers"][-1]["preview"]["values"]
    assert [l["inputs"] for l in cut["layers"]] == [l["inputs"] for l in base["layers"]]


def test_saliency_and_kernels():
    res = analyze(CNN)
    sal = res["input"]["saliency"]
    assert len(sal) == 28 and len(sal[0]) == 28 and max(map(max, sal)) == 1.0
    kernels = res["layers"][0]["preview"]["kernels"]
    assert len(kernels) == 8 and len(kernels[0]) == 3 and max(abs(v) for row in kernels[0] for v in row) == 1.0
    assert "saliency" not in analyze(TRANSFORMER)["input"]


def test_dream_returns_one_image_per_digit():
    from server.analyzer import dream

    res = dream(CNN)
    assert res["ok"] and len(res["images"]) == 10 and len(res["images"][0]) == 28
    assert len(res["confidence"]) == 10 and res["epochs"] == 0
    assert not dream(TRANSFORMER)["ok"]


def test_attack_and_landscape():
    from server import trainer
    from server.analyzer import attack, landscape

    code = CNN.replace("model =", "torch.manual_seed(0)\nmodel =")
    flat = landscape(code)
    assert flat["ok"] and not flat["pca"] and flat["path"] == [] and len(flat["grid"]) == 17
    for _ in range(3):
        train(code)
    before = analyze(code)["prediction"]["top"]
    target = (before + 1) % 10
    hit = attack(code, None, target)
    assert hit["ok"] and hit["before"]["top"] == before and hit["max_change"] <= trainer.ATTACK_EPS + 1e-6
    assert all(0 <= v <= 1 for row in hit["image"] for v in row) and len(hit["original"]) == 28
    land = landscape(code)
    assert land["ok"] and land["pca"] and [p[2] for p in land["path"]] == [0, 1, 2, 3]
    assert land["path"][-1][:2] == [0.0, 0.0]  # le point actuel est au centre de la carte
    assert all(abs(u) <= land["radius"] and abs(v) <= land["radius"] for u, v, _ in land["path"])
    trainer.reset(code)


def test_knob_shows_overfitting_and_can_be_overridden():
    res = analyze(TREE)
    knob = res["knob"]
    assert knob["name"] == "max_depth" and knob["value"] == 3 and len(knob["train"]) == len(knob["values"]) == 12
    assert knob["train"][-1] > knob["train"][0] and knob["train"][-1] >= knob["test"][-1]  # l'arbre profond colle aux données
    deep = analyze(TREE, override={"max_depth": 8})
    assert deep["knob"]["value"] == 8 and deep["tree"]["depth"] > res["tree"]["depth"]
    assert analyze(TREE, override={"max_depth": 999, "criterion": "x"})["knob"]["value"] == 3  # valeurs hors liste ignorées
    km = analyze("from sklearn.datasets import load_iris\nfrom sklearn.cluster import KMeans\n"
                 "X = load_iris().data\nmodel = KMeans(n_clusters=3, n_init=3, random_state=0)\n")
    assert len(km["knob"]["inertia"]) == 7 and km["knob"]["inertia"][0] > km["knob"]["inertia"][-1]
