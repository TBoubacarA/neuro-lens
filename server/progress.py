"""Le parcours expert : huit niveaux, et pour chacun des preuves datées plutôt qu'une case cochée.

Une preuve est un fait que l'outil a constaté lui-même (un défi réussi, une version déployée,
une alerte de dérive levée). Elles sont gardées dans `.neuro-lens/parcours.json`. Voir
docs/recherche-experts-ia.md pour l'origine des niveaux et des règles (réussite différée,
vérification automatique).
"""

from __future__ import annotations

import json
from datetime import date, datetime
from pathlib import Path

from server import mlops

MAX_PER_KEY = 30

# Chaque niveau : ce qu'on y apprend, et les vérifications. `need` = nombre de preuves distinctes
# demandées ; `days` = sur combien de jours différents (une réussite à chaud ne suffit pas).
# Une vérification sans `key` n'est pas encore mesurable par l'outil : elle est affichée comme telle.
LEVELS = [
    {"id": 1, "name": "Lire un modèle", "goal": "Prédire la forme de sortie et le nombre de paramètres de chaque couche avant de les voir.", "checks": [
        {"key": "defi.formes", "label": "Défis « prédis avant de voir » réussis sans faute, sur des réseaux différents", "need": 3, "days": 2,
         "how": "Active le mode défi, puis choisis un exemple PyTorch ou écris ton réseau."},
    ]},
    {"id": 2, "name": "Entraîner proprement", "goal": "Ligne de base, découpage sans fuite, suivi des essais.", "checks": [
        {"key": "essais.suivis", "label": "Essais suivis avec leurs scores (modèle, réglage, données)", "need": 3, "days": 1,
         "how": "Exemple « MLOps » : change le réglage ou le modèle, chaque combinaison est notée."},
        {"label": "Découpage sans fuite de données, détecté par un test caché"},
        {"label": "Même résultat en relançant avec la même graine"},
    ]},
    {"id": 3, "name": "Tester et évaluer", "goal": "Tests du code, des données et du modèle ; évaluation par segments.", "checks": [
        {"key": "tests.projet", "label": "Tests du projet lancés et verts", "need": 1, "days": 1,
         "how": "Explorateur, « Lancer les tests du projet »."},
        {"label": "Défauts injectés (schéma changé, valeurs manquantes) attrapés par tes tests"},
    ]},
    {"id": 4, "name": "Servir", "goal": "Registre de versions, service, image, API de prédiction.", "checks": [
        {"key": "service.version", "label": "Modèle figé dans une version et mis en production", "need": 1, "days": 1,
         "how": "Chaîne MLOps, « Déployer ce modèle »."},
        {"key": "service.conteneur", "label": "Conteneur construit, lancé, et qui répond à une demande", "need": 1, "days": 1,
         "how": "Carte Infrastructure, « Construire l'image et lancer le conteneur », puis « Interroger le conteneur »."},
        {"label": "Retour arrière vers une version précédente"},
    ]},
    {"id": 5, "name": "Surveiller", "goal": "Détecter la dérive des données et décider quand réentraîner.", "checks": [
        {"key": "derive.sain", "label": "Trafic sain observé sans fausse alerte", "need": 1, "days": 1,
         "how": "Carte Surveillance, « Simuler 200 requêtes » sans dérive."},
        {"key": "derive.alerte", "label": "Alerte de dérive levée sur un trafic qui a changé", "need": 1, "days": 1,
         "how": "Même carte, curseur de dérive vers 2, puis simuler."},
        {"label": "Dérive injectée à l'aveugle, détectée dans un délai donné"},
    ]},
    {"id": 6, "name": "Application LLM évaluée", "goal": "Prompt versionné, RAG, 20 à 50 cas d'évaluation, juge calibré.", "checks": [
        {"label": "Jeu d'évaluation de 20 cas ou plus, variés"},
        {"label": "Accord mesuré entre le juge automatique et tes propres étiquettes"},
    ]},
    {"id": 7, "name": "Agent sûr et observable", "goal": "Agent avec outils, trace, budget et garde-fous.", "checks": [
        {"label": "Agent qui résiste à une injection de prompt et à un outil qui ment"},
        {"label": "Plafond de coût respecté"},
    ]},
    {"id": 8, "name": "Système complet et incident", "goal": "Projet de bout en bout, puis un projet cassé inconnu à réparer.", "checks": [
        {"label": "Panne réparée en temps borné, avec son post-mortem"},
        {"label": "Rappel des niveaux 1 à 7 réussi un mois plus tard"},
    ]},
]


def _file() -> Path:
    return mlops.home() / "parcours.json"


def _load() -> dict:
    try:
        return json.loads(_file().read_text())
    except (OSError, ValueError):
        return {}


def record(key: str, what: str, unique: str | None = None) -> bool:
    """Note une preuve. `unique` évite de compter deux fois la même chose (le même réseau, la même version)."""
    data = _load()
    items = data.setdefault(key, [])
    if unique is not None and any(e.get("unique") == unique for e in items):
        return False
    items.append({"at": datetime.now().strftime("%Y-%m-%d %H:%M"), "what": what, "unique": unique})
    del items[:-MAX_PER_KEY]
    _file().parent.mkdir(parents=True, exist_ok=True)
    _file().write_text(json.dumps(data, ensure_ascii=False, indent=1))
    return True


def reset() -> None:
    _file().unlink(missing_ok=True)


def state() -> dict:
    """Les niveaux avec, pour chaque vérification, où on en est et sur quelles preuves."""
    data = _load()
    levels = []
    for level in LEVELS:
        checks = []
        for check in level["checks"]:
            key = check.get("key")
            if key is None:
                checks.append({"label": check["label"], "status": "untracked"})
                continue
            proofs = data.get(key, [])
            days = len({p["at"][:10] for p in proofs})
            enough = len(proofs) >= check["need"]
            status = "done" if enough and days >= check["days"] else "confirm" if enough else "doing" if proofs else "todo"
            checks.append({
                "label": check["label"], "how": check["how"], "status": status,
                "count": len(proofs), "need": check["need"], "days": days, "need_days": check["days"],
                "proofs": proofs[-3:],
            })
        tracked = [c for c in checks if c["status"] != "untracked"]
        done = sum(c["status"] == "done" for c in tracked)
        levels.append({
            "id": level["id"], "name": level["name"], "goal": level["goal"], "checks": checks,
            "tracked": len(tracked), "done": done,
            # un niveau n'est jamais « validé » tant qu'une de ses vérifications n'est pas mesurable
            "status": "empty" if not tracked else "validated" if done == len(tracked) == len(checks)
            else "measured" if done == len(tracked) else "started" if any(c["status"] != "todo" for c in tracked) else "todo",
        })
    return {"ok": True, "today": date.today().isoformat(), "levels": levels}
