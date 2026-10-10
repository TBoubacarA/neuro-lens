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
