"""Le projet autour du code de l'éditeur : les fichiers qu'aurait un vrai dépôt, prêt pour VS Code.

L'éditeur ne montre qu'un fichier, mais un modèle qui part en production vit dans un projet :
son code, ses tests, ses dépendances, sa recette d'image, son intégration continue. `build`
décrit ce projet à partir de l'analyse en cours (chaque fichier avec son contenu réel et une
phrase qui dit à quoi il sert) ; `sync` le tient à jour sur le disque, dans `.neuro-lens/projet`,
où la vue « Projet » (server/project.py) le montre ; `run_tests` y lance pytest et
`open_in_editor` l'ouvre dans VS Code.
"""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import sys
from pathlib import Path

from server import mlops

TEST_TIMEOUT_S = 120

GITIGNORE = """# Ce que git ne doit pas suivre : environnement, caches, gros fichiers produits par le code
.venv/
__pycache__/
.pytest_cache/
*.pyc
# Réglages VS Code propres à cette machine (chemin de l'environnement Python)
.vscode/
"""

CI = """# Intégration continue : à chaque push et à chaque pull request, GitHub installe le projet
# sur une machine neuve et lance les tests. Si un test casse, on le sait avant la mise en production.
name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  pytest:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "{python}"
      - run: pip install -r requirements-dev.txt
      - run: pytest -q
"""

TEST_ML = '''"""Tests du modèle : lancés par `pytest`, sur ta machine et sur GitHub à chaque push."""

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "src" / "train.py"


def load() -> dict:
    """Exécute src/train.py et renvoie ce qu'il définit (model, X, y...)."""
    return runpy.run_path(str(SCRIPT))


def test_le_script_definit_un_modele_et_ses_donnees():
    project = load()
    assert "model" in project and "X" in project
    assert len(project["X"]) > 0
{quality}'''

TEST_ML_QUALITY = '''

def test_le_modele_fait_mieux_que_le_seuil():
    """Garde-fou : si un changement fait chuter la qualité sous {threshold:.0%}, le test échoue."""
    from sklearn.model_selection import cross_val_score

    project = load()
    scores = cross_val_score(project["model"], project["X"], project["y"], cv=3)
    assert scores.mean() > {threshold}
'''

TEST_TORCH = '''"""Tests du réseau : lancés par `pytest`, sur ta machine et sur GitHub à chaque push."""

import runpy
from pathlib import Path

import torch

SCRIPT = Path(__file__).resolve().parent.parent / "src" / "model.py"


def test_la_sortie_a_la_bonne_forme():
    """Une entrée de forme {input_shape} doit donner une sortie de forme {output_shape}."""
    model = runpy.run_path(str(SCRIPT))["model"]
    model.eval()
    x = torch.zeros({input_shape}{dtype})
    with torch.no_grad():
        out = model(x)
    assert list(out.shape) == {output_shape}


def test_le_reseau_a_le_nombre_de_parametres_attendu():
    model = runpy.run_path(str(SCRIPT))["model"]
    assert sum(p.numel() for p in model.parameters()) == {n_params}
'''

TEST_AGENT = '''"""Tests de l'agent : lancés par `pytest`, sur ta machine et sur GitHub à chaque push."""

import runpy
from pathlib import Path

SCRIPT = Path(__file__).resolve().parent.parent / "src" / "agent.py"


def test_l_agent_va_au_bout_sans_tourner_en_rond():
    project = runpy.run_path(str(SCRIPT))
    graph = project["graph"]
    if hasattr(graph, "compile"):
        graph = graph.compile()
    final = graph.invoke(project["state"], config={"recursion_limit": 40})
    assert isinstance(final, dict)
'''

README = """# {title}

{summary}

## Structure

```
{tree}
```

{roles}

## Commandes

### Sur cette machine : rien à installer

Ouvre le dossier dans VS Code, puis un **nouveau terminal** (menu Terminal > New Terminal).
Les réglages de `.vscode/settings.json` y branchent l'environnement Python de neuro-lens, qui
contient déjà toutes les bibliothèques, dans les bonnes versions :

```bash
pytest                      # lancer les tests
{run}
```

Les tests apparaissent aussi dans l'onglet « Testing » de VS Code (l'icône en forme de fiole).

### Sur une autre machine

Il faut Python {python} ou plus récent : les versions fixées dans `requirements.txt` n'existent pas
pour un Python plus ancien (c'est l'erreur « No matching distribution found »).

```bash
python{python} -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
pytest
```
"""


def _file(path: str, role: str, content: str, **flags) -> dict:
    return {"path": path, "role": role, "content": content, "generated": True, **flags}


def _tree(paths: list[str]) -> str:
    lines, seen = [], set()
    for path in sorted(paths, key=lambda p: (p.count("/") == 0, p)):
        parts = path.split("/")
        for depth in range(1, len(parts)):
            folder = "/".join(parts[:depth])
            if folder not in seen:
                seen.add(folder)
                lines.append("  " * (depth - 1) + parts[depth - 1] + "/")
        lines.append("  " * (len(parts) - 1) + parts[-1])
    return "\n".join(lines)


def _readme(title: str, summary: str, files: list[dict], run: str) -> dict:
    roles = "\n".join(f"- `{f['path']}` : {f['role']}" for f in files)
    text = README.format(title=title, summary=summary, tree=_tree([f["path"] for f in files] + ["README.md"]), roles=roles, run=run, python=_python())
    return _file("README.md", "La porte d'entrée du projet : ce qu'il fait, comment il est rangé, comment le lancer.", text)


def _python() -> str:
    return ".".join(platform.python_version_tuple()[:2])


def _vscode() -> str:
    """Réglages VS Code : l'interpréteur et le terminal utilisent l'environnement qui fait tourner neuro-lens."""
    bin_dir = Path(sys.executable).parent
    path = f"{bin_dir}:${{env:PATH}}"
    return json.dumps({
        "python.defaultInterpreterPath": sys.executable,
        "terminal.integrated.env.osx": {"PATH": path},
        "terminal.integrated.env.linux": {"PATH": path},
        "python.testing.pytestEnabled": True,
        "python.testing.unittestEnabled": False,
        "python.testing.pytestArgs": ["tests"],
    }, indent=2) + "\n"


def _common(requirements: str) -> list[dict]:
    needs = f"# Python {_python()} ou plus récent : ces versions n'existent pas pour un Python plus ancien.\n"
    return [
        _file("requirements.txt", "Les bibliothèques dont le projet a besoin, avec leur version : même installation partout.", needs + requirements),
        _file("requirements-dev.txt", "Les outils en plus pour développer et tester (pytest), inutiles en production.", "-r requirements.txt\npytest\n"),
        _file(".vscode/settings.json", "Les réglages VS Code du projet : quel Python utiliser, et où sont les tests. Dans un nouveau terminal, `pip` et `pytest` marchent tout de suite.", _vscode()),
        _file(".github/workflows/tests.yml", "L'intégration continue : GitHub lance les tests à chaque push et chaque pull request.", CI.format(python=_python())),
        _file(".gitignore", "Ce que git ne doit pas suivre : environnement virtuel, caches.", GITIGNORE),
    ]


def _ml(code: str, result: dict) -> list[dict]:
    import numpy as np
    import sklearn

    prod = (result.get("mlops") or {}).get("production")
    deployed = bool(prod and prod.get("run_key") == result.get("run_key"))
    files = [{
        "path": "src/train.py", "main": True, "generated": False, "content": code,
        "role": "Le code que tu écris : il prépare les données et définit le modèle. C'est le seul fichier que tu édites ici.",
    }]
    quality = ""
    if result["task"] == "classification":
        knob = result.get("knob") or {}
        held = knob["test"][knob["values"].index(knob["value"])] if knob.get("test") and knob.get("value") in knob.get("values", []) else None
        chance = 1 / max(2, len(result.get("classes", [])))
        threshold = round(max(chance + 0.1, (held or result.get("score") or 0.6) - 0.15), 2)
        quality = TEST_ML_QUALITY.format(threshold=threshold)
    files.append(_file("tests/test_train.py", "Les tests : ils vérifient que le script tourne et que le modèle reste au-dessus d'un seuil de qualité.", TEST_ML.format(quality=quality)))
    requirements = f"scikit-learn=={sklearn.__version__}\nnumpy=={np.__version__}\n"
    if prod:  # le dossier du service est l'unité qu'on déploie : tout ce qu'il faut pour répondre, rien d'autre
        version = prod["version"]
        note = "" if deployed else " (le modèle affiché a changé depuis ce déploiement)"
        files += [
            _file("service/serve.py", "Le service : il charge le modèle une fois, puis répond aux demandes de prédiction.", prod["texts"].get("serve.py", "")),
            _file("service/model.joblib", f"Le modèle entraîné, figé dans la version {version}{note}. Fichier binaire : il ne se lit pas, il se charge.",
                  f"Fichier binaire de {prod.get('size_kb', '?')} ko : le modèle entraîné (version {version}), enregistré avec joblib.\nIl se charge avec joblib.load(\"model.joblib\").",
                  binary=True, source=f"models/v{version}/model.joblib"),
            _file("service/model_card.json", "La fiche du modèle : d'où il vient, ses réglages, ses scores, l'empreinte de ses données.", prod["texts"].get("model_card.json", "")),
            _file("service/Dockerfile", "La recette de l'image : Python, les bibliothèques et le modèle dans une boîte identique partout.", prod["texts"].get("Dockerfile", "")),
            _file("service/requirements.txt", "Les bibliothèques du service seulement, aux versions exactes de l'entraînement.", prod["texts"].get("requirements.txt", "")),
        ]
    else:
        files.append(_file("service/À_VENIR.md", "Rien n'est encore déployé : le service, le modèle figé, sa fiche et le Dockerfile apparaîtront ici.",
                           "# Pas encore de service\n\nClique sur « Déployer ce modèle » dans la chaîne MLOps, à droite :\n`serve.py`, `model.joblib`, `model_card.json` et le `Dockerfile` s'ajouteront dans ce dossier.\n"))
    files += _common(requirements)
    run = "python src/train.py         # exécuter le script"
    if prod:
        run += "\ncd service && uvicorn serve:app --port 8080   # lancer le service\ndocker build -t mon-modele service/           # construire son image"
    files.append(_readme(f"Projet {result['estimator']}", f"Un modèle scikit-learn ({result['task']}) entraîné sur {result['n_samples']} exemples, de l'entraînement à la mise en production.", files, run))
    return files


def _torch(code: str, result: dict) -> list[dict]:
    import torch

    text = bool(result["input"].get("tokens"))
    files = [{
        "path": "src/model.py", "main": True, "generated": False, "content": code,
        "role": "Le code que tu écris : l'architecture du réseau. C'est le seul fichier que tu édites ici.",
    }]
    shape = result["layers"][-1]["shape"] if result["layers"] else []
    files.append(_file("tests/test_model.py", "Les tests : la forme de la sortie et le nombre de paramètres ne doivent pas changer par accident.", TEST_TORCH.format(
        input_shape=result["input"]["shape"], output_shape=shape, n_params=result["n_params"], dtype=", dtype=torch.long" if text else "",
    )))
    files += _common(f"torch=={torch.__version__.split('+')[0]}\n")
    files.append(_readme("Projet réseau de neurones", f"Un réseau PyTorch de {len(result['layers'])} couches et {result['n_params']:,} paramètres.".replace(",", " "), files, "python src/model.py         # construire le réseau"))
    return files


def _agent(code: str, result: dict) -> list[dict]:
    files = [{
        "path": "src/agent.py", "main": True, "generated": False, "content": code,
        "role": "Le code que tu écris : les étapes de l'agent, ses outils et les chemins entre eux. C'est le seul fichier que tu édites ici.",
    }]
    if result.get("start"):
        files.append(_file("tests/test_agent.py", "Les tests : l'agent doit aller au bout de sa tâche sans tourner en rond.", TEST_AGENT))
    from importlib.metadata import version

    files += _common(f"langgraph=={version('langgraph')}\n")
    steps = sum(1 for n in result["nodes"] if n["kind"] == "step")
    files.append(_readme("Projet agent", f"Un agent LangGraph de {steps} étapes.", files, "python src/agent.py         # construire le graphe de l'agent"))
    return files


def build(code: str, result: dict) -> list[dict]:
    """La liste des fichiers du projet pour cette analyse réussie."""
    kind = result.get("kind")
    files = _ml(code, result) if kind == "ml" else _agent(code, result) if kind == "agent" else _torch(code, result)
    return sorted(files, key=lambda f: (not f.get("main"), f["path"].count("/") == 0, f["path"]))


# ---------- Sur le disque ----------

def folder() -> Path:
    return mlops.home() / "projet"


def roles_file() -> Path:
    return mlops.home() / "projet.roles.json"


def roles() -> dict:
    """Le rôle de chaque fichier du projet, tel que la dernière analyse l'a écrit."""
    try:
        return json.loads(roles_file().read_text())
    except (OSError, ValueError):
        return {}


def sync(files: list[dict]) -> dict:
    """Met le dossier du projet au niveau de `files` : n'écrit que ce qui a changé, retire ce qui n'existe plus."""
    root = folder()
    root.mkdir(parents=True, exist_ok=True)
    wanted: dict[Path, dict] = {root / f["path"]: f for f in files}
    for package in ("src", "tests"):
        if any(f["path"].startswith(package + "/") for f in files):
            wanted[root / package / "__init__.py"] = {"content": "", "role": "Fait du dossier un paquet Python importable."}
    for target, f in wanted.items():
        target.parent.mkdir(parents=True, exist_ok=True)
        if f.get("binary"):  # le vrai modèle, copié depuis le registre des versions
            source = mlops.home() / f["source"]
            if source.exists() and (not target.exists() or target.stat().st_size != source.stat().st_size):
                shutil.copyfile(source, target)
        elif not target.exists() or target.read_text() != f["content"]:
            target.write_text(f["content"])
    for old in sorted(root.rglob("*"), reverse=True):  # les plus profonds d'abord
        if any(part in (".venv", ".git", "__pycache__", ".pytest_cache") for part in old.relative_to(root).parts):
            continue
        if old.is_file() and old not in wanted:
            old.unlink()
        elif old.is_dir() and not any(old.iterdir()):
            old.rmdir()
    main = next((f["path"] for f in files if f.get("main")), None)
    roles_file().write_text(json.dumps({"main": main, "roles": {f["path"]: f["role"] for f in files}}, ensure_ascii=False))
    return {"ok": True, "folder": str(root), "written": len(files), "main": main}


def run_tests() -> dict:
    """Lance pytest dans le dossier du projet, avec le Python du serveur."""
    root = folder()
    if not (root / "tests").is_dir():
        return {"ok": False, "error": "Le projet n'a pas encore été écrit sur le disque."}
    try:
        run = subprocess.run([sys.executable, "-m", "pytest", "-q", "--no-header", "-p", "no:cacheprovider"],
                             cwd=root, capture_output=True, text=True, timeout=TEST_TIMEOUT_S)
    except subprocess.TimeoutExpired:
        return {"ok": True, "passed": False, "output": f"Les tests ont mis plus de {TEST_TIMEOUT_S} s."}
    lines = [line for line in (run.stdout + run.stderr).splitlines() if line.strip() and "Warning" not in line]
    return {"ok": True, "passed": run.returncode == 0, "output": "\n".join(lines[-25:]), "folder": str(root)}


def open_in_editor() -> dict:
    """Ouvre le dossier du projet dans VS Code, si on le trouve sur cette machine."""
    root = folder()
    if not root.is_dir():
        return {"ok": False, "error": "Le projet n'a pas encore été écrit sur le disque."}
    commands = []
    if shutil.which("code"):
        commands.append(["code", str(root)])
    if sys.platform == "darwin":
        commands.append(["open", "-a", "Visual Studio Code", str(root)])
    for command in commands:
        try:
            if subprocess.run(command, capture_output=True, timeout=15).returncode == 0:
                return {"ok": True, "folder": str(root)}
        except (OSError, subprocess.TimeoutExpired):
            continue
    return {"ok": False, "error": f"VS Code introuvable sur cette machine. Ouvre ce dossier à la main : {root}", "folder": str(root)}
