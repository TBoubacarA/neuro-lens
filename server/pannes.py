"""Mode panne : une faute est glissée dans le code, l'apprenant doit la reconnaître aux symptômes.

C'est l'exercice de l'expert : un réseau qui n'apprend pas, un score trop beau, un agent qui tourne en
rond ne s'annoncent jamais ; on les reconnaît. Chaque panne de la banque dit sur quel code elle
s'applique (un morceau de texte à remplacer), ce qu'on devrait voir à droite, la cause, et la famille
de causes que l'apprenant doit nommer parmi plusieurs.

    start(code)              -> {"id", "code": code cassé, "famille", "choix": [familles possibles], "symptome"}
    guess(id, choix)         -> {"ok", ...}  et, si c'est juste, la cause et la ligne
    reveal(id, code)         -> la cause, la ligne fautive dans le code cassé, l'explication
"""

from __future__ import annotations

import random

# Les familles de causes que l'on propose au choix : les mêmes mots que ceux de l'expert en revue.
FAMILLES = {
    "reseau": ["Architecture trop étroite", "Régularisation trop forte", "Couche de sortie fausse", "Activation manquante"],
    "ml": ["Fuite d'information", "Étiquettes fausses", "Sur-apprentissage", "Classes déséquilibrées", "Données trop bruitées"],
    "agent": ["Outil qui renvoie faux", "Boucle sans fin", "Règle de sortie trop stricte", "Mauvais ordre des outils"],
}

# Chaque panne : ce qu'on remplace (toutes les occurrences), dans quel ordre, et comment on l'explique.
PANNES: list[dict] = [
    # ---- réseaux PyTorch ----
    {
        "id": "goulot",
        "famille": "reseau",
        "tous": True,
        "cause": "Architecture trop étroite",
        "remplace": [("nn.Linear(16 * 7 * 7, 64)", "nn.Linear(16 * 7 * 7, 2)"), ("nn.Linear(64, 10)", "nn.Linear(2, 10)")],
        "symptome": "L'entraînement plafonne bien en dessous de ce que ce réseau devrait atteindre.",
        "explication": "Toute l'information passe par une couche de 2 neurones avant les 10 scores : un goulot. "
        "Dans la vue réseau, l'avant-dernière colonne n'a que deux ronds. Un expert regarde d'abord la forme du réseau.",
    },
    {
        "id": "un-seul-filtre",
        "famille": "reseau",
        "tous": True,
        "cause": "Architecture trop étroite",
        "remplace": [("nn.Conv2d(1, 8, kernel_size=3, padding=1)", "nn.Conv2d(1, 1, kernel_size=3, padding=1)"),
                     ("nn.Conv2d(8, 16, kernel_size=3, padding=1)", "nn.Conv2d(1, 16, kernel_size=3, padding=1)")],
        "symptome": "Les cartes de la première couche se ressemblent toutes et le réseau apprend lentement.",
        "explication": "La première convolution n'a plus qu'un filtre : elle ne peut détecter qu'un seul motif. "
        "Dans la vue réseau, la première colonne n'a qu'une vignette.",
    },
    {
        "id": "dropout-massif",
        "famille": "reseau",
        "cause": "Régularisation trop forte",
        "remplace": [("nn.Dropout(0.2)", "nn.Dropout(0.9)")],
        "symptome": "La perte d'entraînement descend mal et la réussite sur le jeu d'entraînement reste basse.",
        "explication": "Dropout à 0,9 éteint 90 % des neurones à chaque pas : le réseau n'arrive plus à apprendre. "
        "Quand l'entraînement lui-même échoue (pas seulement le test), on cherche une régularisation trop forte.",
    },
    {
        "id": "sortie-9",
        "famille": "reseau",
        "cause": "Couche de sortie fausse",
        "remplace": [("nn.Linear(64, 10)", "nn.Linear(64, 9)"), ("nn.Linear(84, 10)", "nn.Linear(84, 9)")],
        "symptome": "Un chiffre ne peut jamais être prédit, et l'entraînement s'arrête sur une erreur d'indice.",
        "explication": "Il y a 10 chiffres mais 9 scores en sortie : la dernière classe n'existe pas pour le réseau. "
        "Toujours compter les sorties contre le nombre de classes.",
    },
    {
        "id": "sans-activation",
        "famille": "reseau",
        "cause": "Activation manquante",
        "remplace": [("    nn.ReLU(),\n", "")],
        "symptome": "Le réseau est plus plat que prévu et n'apprend que des frontières droites.",
        "explication": "Sans ReLU, une pile de couches linéaires reste une seule transformation linéaire : "
        "la profondeur ne sert à rien. Dans la vue réseau, les colonnes d'activation ont disparu.",
    },
    # ---- scikit-learn ----
    {
        "id": "fuite-cible",
        "famille": "ml",
        "tous": True,
        "cause": "Fuite d'information",
        "remplace": [("X = np.column_stack([temperature, vibration])", "X = np.column_stack([temperature, vibration, risque])"),
                     ('feature_names = ["température", "vibration"]', 'feature_names = ["température", "vibration", "risque"]')],
        "symptome": "Le score est presque parfait, trop beau pour deux capteurs bruités.",
        "explication": "La variable « risque » sert à fabriquer l'étiquette : la donner en entrée, c'est donner la réponse. "
        "En production, elle n'existera pas. Un score trop beau est le premier symptôme d'une fuite.",
    },
    {
        "id": "desequilibre",
        "famille": "ml",
        "cause": "Classes déséquilibrées",
        "remplace": [('y = np.where(risque > 0.8, "panne", "ok")', 'y = np.where(risque > 3.5, "panne", "ok")')],
        "symptome": "Le score est excellent, mais le modèle ne prédit presque jamais « panne ».",
        "explication": "Le seuil est si haut que presque toutes les machines sont « ok » : répondre « ok » partout donne un score élevé "
        "et ne sert à rien. On regarde la répartition des classes avant le score.",
    },
    {
        "id": "arbre-sans-fond",
        "famille": "ml",
        "cause": "Sur-apprentissage",
        "remplace": [("max_depth=3", "max_depth=None"), ("max_depth=4", "max_depth=None")],
        "symptome": "Score parfait à l'entraînement, bien moins bon sur les données gardées de côté ; frontière découpée.",
        "explication": "Sans limite de profondeur, l'arbre apprend chaque point par cœur, bruit compris. "
        "L'écart entre score d'entraînement et score de test est la signature du sur-apprentissage.",
    },
    {
        "id": "etiquettes-melangees",
        "famille": "ml",
        "cause": "Étiquettes fausses",
        "remplace": [("X, y = make_moons(n_samples=200, noise=0.25, random_state=0)",
                      "X, y = make_moons(n_samples=200, noise=0.25, random_state=0)\n"
                      "y = y[np.random.default_rng(1).permutation(len(y))]")],
        "prefixe": "import numpy as np\n",
        "symptome": "La frontière n'a aucun sens et le score reste proche du hasard, quel que soit le modèle.",
        "explication": "Les étiquettes ont été mélangées : chaque point porte la couleur d'un autre. "
        "Aucun modèle ne peut apprendre un lien qui n'existe pas : quand tous les modèles échouent, on soupçonne les données.",
    },
    {
        "id": "bruit",
        "famille": "ml",
        "cause": "Données trop bruitées",
        "remplace": [("noise=0.25", "noise=0.9")],
        "symptome": "Les deux croissants ne se distinguent plus ; la frontière est hésitante.",
        "explication": "Le bruit des données dépasse le signal. Avant de changer de modèle, on regarde les points eux-mêmes.",
    },
    # ---- agents ----
    {
        "id": "outil-menteur",
        "famille": "agent",
        "cause": "Outil qui renvoie faux",
        "remplace": [('"La tour Eiffel mesure 330 m."', '"La tour Eiffel mesure 33 m."')],
        "symptome": "Le déroulé est correct, les outils sont appelés dans le bon ordre, mais la réponse est fausse.",
        "explication": "L'outil de recherche renvoie une donnée fausse ; l'agent lui fait confiance. "
        "Une boucle propre ne garantit pas une bonne réponse : on vérifie ce que renvoient les outils.",
    },
    {
        "id": "boucle",
        "famille": "agent",
        "cause": "Boucle sans fin",
        "tous": True,
        "remplace": [("        if vus < len(plan):", "        if True:"),
                     ("            nom, entree = plan[vus]", "            nom, entree = plan[vus % len(plan)]")],
        "symptome": "L'agent appelle ses outils encore et encore, puis s'arrête sur le garde-fou sans répondre.",
        "explication": "La condition d'arrêt a disparu : l'agent ne décide jamais de répondre. "
        "Le garde-fou de budget l'arrête, c'est son rôle, mais la boucle est la cause.",
    },
    {
        "id": "boucle-simple",
        "famille": "agent",
        "cause": "Boucle sans fin",
        "remplace": [('    if not any(n.startswith("calcul") for n in notes):', '    if True:')],
        "symptome": "L'agent refait le même calcul encore et encore, puis s'arrête sur le garde-fou sans répondre.",
        "explication": "La condition qui mène à « répondre » est devenue toujours vraie pour « calculer » : "
        "l'agent ne sort jamais de la boucle. Le nombre de passages, à droite, le dit tout de suite.",
    },
    {
        "id": "regle-stricte",
        "famille": "agent",
        "cause": "Règle de sortie trop stricte",
        "remplace": [('"max_mots": 25', '"max_mots": 2')],
        "symptome": "L'agent répond juste, mais chaque réponse est refusée par les garde-fous de sortie.",
        "explication": "Deux mots maximum : aucune phrase ne passe. Un garde-fou mal réglé bloque un agent qui marche.",
    },
]


def applicables(code: str) -> list[dict]:
    """Les pannes dont les morceaux à remplacer existent dans ce code (tous, quand la panne l'exige)."""
    actives = [l for l in code.splitlines() if not l.strip().startswith("#")]
    out = []
    for p in PANNES:
        present = [any(avant.splitlines()[0] in l for l in actives) for avant, _ in p["remplace"]]
        if (all(present) if p.get("tous") else any(present)):
            out.append(p)
    return out


def inject(code: str, panne: dict) -> str:
    out = code
    for avant, apres in panne["remplace"]:
        if avant in out:
            out = out.replace(avant, apres)
    prefixe = panne.get("prefixe", "")
    if prefixe and prefixe not in out:
        out = prefixe + out
    return out


def ligne(code_casse: str, panne: dict) -> int | None:
    """La première ligne du code cassé qui porte la faute."""
    for _, apres in panne["remplace"]:
        morceau = apres.splitlines()[-1] if apres.strip() else ""
        if not morceau:
            continue
        for i, line in enumerate(code_casse.splitlines(), 1):
            if morceau.strip() in line:
                return i
    return None


def _find(panne_id: str) -> dict | None:
    return next((p for p in PANNES if p["id"] == panne_id), None)


def start(code: str, exclude: list[str] | None = None, seed: int | None = None) -> dict:
    choix = applicables(code)
    if not choix:
        return {"ok": False, "error": "Aucune panne connue ne s'applique à ce code : reviens sur un exemple du menu."}
    rest = [p for p in choix if p["id"] not in (exclude or [])] or choix
    panne = random.Random(seed).choice(rest)
    return {
        "ok": True,
        "id": panne["id"],
        "famille": panne["famille"],
        "choix": FAMILLES[panne["famille"]],
        "symptome": panne["symptome"],
        "code": inject(code, panne),
        "total": len(choix),
    }


def guess(panne_id: str, cause: str, code: str = "") -> dict:
    panne = _find(panne_id)
    if panne is None:
        return {"ok": False, "error": "Panne inconnue."}
    if cause != panne["cause"]:
        return {"ok": True, "juste": False}
    return {"ok": True, "juste": True, **reveal(panne_id, code)}


def reveal(panne_id: str, code: str = "") -> dict:
    panne = _find(panne_id)
    if panne is None:
        return {"ok": False, "error": "Panne inconnue."}
    return {"ok": True, "cause": panne["cause"], "explication": panne["explication"],
            "line": ligne(code, panne) if code else None}


def count() -> dict:
    return {"total": len(PANNES), "familles": {k: sum(1 for p in PANNES if p["famille"] == k) for k in FAMILLES}}
