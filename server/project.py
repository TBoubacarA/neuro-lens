"""La vue « Projet » : les fichiers d'un vrai projet, façon explorateur de VS Code, avec le rôle de chacun.

Deux racines : neuro-lens lui-même (le dépôt), et le dossier d'une version mise en production
(`.neuro-lens/models/vN`, écrit par server/mlops.py). Pour chaque fichier Python du serveur on relève
ses imports internes : le graphe de dépendances montre comment les modules se tiennent.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SKIP = {".git", "__pycache__", ".pytest_cache", "node_modules", ".neuro-lens", ".ruff_cache", ".venv", "venv"}
MAX_DEPTH = 4
MAX_FILE = 200_000
TEXT_EXT = {".py", ".js", ".html", ".css", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".cfg", ".ini", ".sh", ""}

# Le rôle d'un fichier ou d'un dossier, en une phrase, d'après son nom (le code réel est lu ensuite).
ROLES = [
    (r"^README\.md$", "La porte d'entrée : ce que fait le projet, comment le lancer, comment il est organisé."),
    (r"^requirements\.txt$", "Les bibliothèques dont le projet a besoin pour tourner, avec leurs versions : `pip install -r`."),
    (r"^requirements-dev\.txt$", "Les outils en plus pour développer et tester (pytest…), inutiles en production."),
    (r"^Dockerfile$", "La recette de l'image : système de base, dépendances, code, commande de démarrage."),
    (r"^\.github/workflows/.*\.ya?ml$", "Intégration continue : GitHub lance ces étapes (tests) à chaque push et pull request."),
    (r"^\.github/?$", "Ce que GitHub exécute tout seul : les workflows d'intégration continue."),
    (r"^\.gitignore$", "Ce que git ne doit pas suivre : caches, environnements, fichiers générés."),
    (r"^tests/?$", "Les tests automatiques : chaque fonctionnalité a un test qui la protège des régressions."),
    (r"^tests/test_.*\.py$", "Un fichier de tests pytest : des fonctions `test_…` qui vérifient le comportement."),
    (r"^tests/__init__\.py$", "Fait du dossier un paquet Python, pour que pytest importe `server` proprement."),
    (r"^server/?$", "Le côté serveur (Python) : l'API, et tout ce qui exécute ou analyse du code."),
    (r"^server/__init__\.py$", "Fait du dossier un paquet Python importable (`from server import …`)."),
    (r"^server/app\.py$", "Le serveur web (FastAPI) : les routes de l'API et le processus isolé qui exécute le code."),
    (r"^web/?$", "Le côté navigateur : la page, ses styles et ses scripts, sans framework ni étape de build."),
    (r"^web/index\.html$", "La page unique : l'éditeur à gauche, les cartes à droite, et l'ordre de chargement des scripts."),
    (r"^web/style\.css$", "Tous les styles de la page : couleurs, cartes, graphiques."),
    (r"^web/app\.js$", "Le chef d'orchestre côté navigateur : éditeur, appels à l'API, choix de la vue à dessiner."),
    (r"^web/examples\.js$", "Les exemples du menu : un bloc de code Python par entrée."),
    (r"^web/.*\.js$", "Un module de la page : dessine une famille de cartes à partir du JSON du serveur."),
    (r"^docs/?$", "Documentation et images."),
    (r"^serve\.py$", "Le service de prédiction : une API FastAPI qui charge le modèle et répond à POST /predict."),
    (r"^model\.joblib$", "Le modèle entraîné, figé sur disque (joblib), tel qu'il a été déployé."),
    (r"^model_card\.json$", "La fiche du modèle : version, réglage, scores, empreinte des données, date."),
    (r"^requirements\.txt$", "Les bibliothèques dont le service a besoin."),
    (r"^docker-compose\.ya?ml$", "Les services à lancer ensemble (API, base, supervision) et leurs réglages."),
    (r"^Makefile$", "Les commandes du projet (test, lancer, construire) sous un nom court."),
    (r"^pyproject\.toml$", "La carte d'identité du paquet Python : nom, dépendances, outils de qualité."),
]

FOLDER_WHY = {
    "server": "Séparer le serveur du navigateur permet de tester la logique sans page web, et de la servir ailleurs.",
    "web": "La page ne fait que dessiner ce que l'API renvoie : aucune logique métier n'y vit.",
    "tests": "Un test par fonctionnalité : c'est ce qui autorise à modifier le code sans peur.",
}


def role_of(rel: str, path: Path) -> str:
    key = rel + ("/" if path.is_dir() else "")
    for pattern, role in ROLES:
        if re.match(pattern, key) or re.match(pattern, path.name + ("/" if path.is_dir() else "")):
            return role
    if path.is_dir():
        return "Un dossier."
    if path.suffix == ".py":
        return _docstring(path) or "Un module Python."
    return {".md": "Un document.", ".json": "Des données au format JSON.", ".yml": "Un fichier de configuration.",
            ".yaml": "Un fichier de configuration.", ".txt": "Un fichier texte."}.get(path.suffix, "Un fichier.")


def _docstring(path: Path) -> str | None:
    try:
        doc = ast.get_docstring(ast.parse(path.read_text(encoding="utf-8")))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return None
    return doc.strip().splitlines()[0] if doc else None


def _count_lines(path: Path) -> int | None:
    if path.suffix not in TEXT_EXT or path.stat().st_size > MAX_FILE:
        return None
    try:
        return path.read_text(encoding="utf-8").count("\n") + 1
    except (UnicodeDecodeError, OSError):
        return None


def tree(base: Path, rel: str = "", depth: int = 0) -> list[dict]:
    items = []
    for path in sorted(base.iterdir(), key=lambda p: (p.is_file(), p.name.lower())):
        if path.name in SKIP or (path.name.startswith(".") and path.name not in {".github", ".gitignore", ".vscode"}):
            continue
        r = f"{rel}/{path.name}" if rel else path.name
        node = {"name": path.name, "path": r, "kind": "dir" if path.is_dir() else "file", "role": role_of(r, path)}
        if path.is_dir():
            node["why"] = FOLDER_WHY.get(path.name)
            node["children"] = tree(path, r, depth + 1) if depth < MAX_DEPTH else []
        else:
            node["size"] = path.stat().st_size
            node["lines"] = _count_lines(path)
        items.append(node)
    return items


def imports_of(path: Path, package: str) -> list[str]:
    """Les modules du même paquet que ce fichier importe (`from server import x`, `import server.x`)."""
    try:
        mod = ast.parse(path.read_text(encoding="utf-8"))
    except (SyntaxError, UnicodeDecodeError, OSError):
        return []
    found = set()
    for node in ast.walk(mod):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.split(".")[0] == package:
            parts = node.module.split(".")
            if len(parts) > 1:
                found.add(parts[1])
            else:
                found.update(a.name for a in node.names)
        elif isinstance(node, ast.Import):
            for a in node.names:
                parts = a.name.split(".")
                if parts[0] == package and len(parts) > 1:
                    found.add(parts[1])
    return sorted(found)


def graph(base: Path, package: str = "server") -> dict:
    """Le graphe des modules du paquet : un nœud par fichier, une flèche par import interne."""
    folder = base / package
    if not folder.is_dir():
        return {"nodes": [], "edges": []}
    files = {p.stem: p for p in sorted(folder.glob("*.py")) if p.stem != "__init__"}
    nodes = [{"id": name, "path": f"{package}/{p.name}", "lines": _count_lines(p) or 0, "role": role_of(f"{package}/{p.name}", p)}
             for name, p in files.items()]
    edges = [{"source": name, "target": dep} for name, p in files.items() for dep in imports_of(p, package) if dep in files]
    # Les tests aussi : qui teste quoi.
    for p in sorted((base / "tests").glob("test_*.py")) if (base / "tests").is_dir() else []:
        nodes.append({"id": f"tests/{p.stem}", "path": f"tests/{p.name}", "lines": _count_lines(p) or 0, "role": role_of(f"tests/{p.name}", p), "test": True})
        edges += [{"source": f"tests/{p.stem}", "target": dep} for dep in imports_of(p, package) if dep in files]
    return {"nodes": nodes, "edges": edges}


CODE_FOLDERS = {
    ".vscode": "Les réglages de VS Code pour ce projet seulement.",
    "src": "Le code du projet : ce que tu écris.",
    "tests": "Les tests automatiques : ils protègent le code des régressions, sur ta machine et sur GitHub.",
    "service": "L'unité qu'on déploie : le service, le modèle figé, sa fiche, la recette de l'image. Rien d'autre ne part en production.",
    ".github": "Ce que GitHub exécute tout seul : l'intégration continue.",
    ".github/workflows": "Un fichier par automatisme ; ici, lancer les tests à chaque push.",
}


def roots() -> list[dict]:
    """Les projets qu'on peut ouvrir : neuro-lens, puis chaque version mise en production."""
    out = [{"id": "app", "name": "neuro-lens (ce projet)", "path": str(ROOT)}]
    try:
        from server import mlops, workspace

        if workspace.folder().is_dir():  # le projet autour du code de l'éditeur, tenu à jour à chaque analyse
            out.insert(0, {"id": "code", "name": "ton projet (autour de ton code)", "path": str(workspace.folder())})

        for folder in sorted((mlops.home() / "models").glob("v*"), key=lambda p: int(p.name[1:]) if p.name[1:].isdigit() else 0):
            if folder.is_dir():
                out.append({"id": folder.name, "name": f"modèle déployé {folder.name}", "path": str(folder)})
    except Exception:  # noqa: BLE001 - pas de registre : seule la racine du projet
        pass
    return out


def _base(root_id: str) -> Path | None:
    return next((Path(r["path"]) for r in roots() if r["id"] == root_id), None)


def describe(root_id: str = "app") -> dict:
    base = _base(root_id)
    if base is None:
        return {"ok": False, "error": f"Projet inconnu : {root_id}"}
    out = {"ok": True, "root": root_id, "roots": roots(), "tree": tree(base), "graph": graph(base) if root_id == "app" else {"nodes": [], "edges": []}}
    if root_id == "code":  # ici chaque fichier a un rôle précis, écrit par l'analyse (server/workspace.py)
        from server import workspace

        known = workspace.roles()
        out["main"] = known.get("main")

        def label(nodes: list[dict]) -> None:
            for node in nodes:
                node["role"] = known.get("roles", {}).get(node["path"], CODE_FOLDERS.get(node["path"], node["role"]))
                label(node.get("children", []))

        label(out["tree"])
    return out


def read(root_id: str, rel: str) -> dict:
    base = _base(root_id)
    if base is None:
        return {"ok": False, "error": f"Projet inconnu : {root_id}"}
    path = (base / rel).resolve()
    if base.resolve() not in path.parents or not path.is_file():
        return {"ok": False, "error": "Fichier hors du projet, ou introuvable."}
    if path.suffix not in TEXT_EXT or path.stat().st_size > MAX_FILE:
        return {"ok": True, "path": rel, "binary": True, "size": path.stat().st_size, "role": role_of(rel, path)}
    try:
        text = path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        return {"ok": True, "path": rel, "binary": True, "size": path.stat().st_size, "role": role_of(rel, path)}
    role = role_of(rel, path)
    if root_id == "code":
        from server import workspace

        role = workspace.roles().get("roles", {}).get(rel, role)
    out = {"ok": True, "path": rel, "text": text, "role": role, "lines": text.count("\n") + 1}
    if path.suffix == ".py":
        out["imports"] = imports_of(path, "server")
        out["symbols"] = _symbols(text)
    return out


def _symbols(text: str) -> list[dict]:
    """Les fonctions et classes du fichier, avec leur ligne et leur première ligne de doc : le plan du fichier."""
    try:
        mod = ast.parse(text)
    except SyntaxError:
        return []
    out = []
    for node in mod.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            doc = ast.get_docstring(node)
            out.append({"name": node.name, "kind": "classe" if isinstance(node, ast.ClassDef) else "fonction",
                        "line": node.lineno, "doc": doc.strip().splitlines()[0] if doc else None})
    return out
