# neuro-lens

Chaque ligne de code prend forme sous tes yeux. Neuro-lens ouvre une fenêtre sur l'intérieur des réseaux de neurones : tu écris, le réseau se dessine, les filtres s'allument. Le deep learning sans boîte noire. Tu tapes une couche, elle apparaît. Tu l'entraînes, elle s'anime.

![Aperçu](docs/apercu.png)

## Ce que ça fait

- **À gauche**, un éditeur Python. **À droite**, une page qui se met à jour pendant que tu tapes.
- **Lien code ↔ visuel** : le curseur sur une ligne allume l'élément correspondant ; survoler un élément surligne sa ligne.
- **Erreurs expliquées** : une erreur de taille est traduite en français (« La couche Linear attend 1568 valeurs… Mets in_features=1352 ») et la ligne fautive est soulignée.

Ce que la page montre dépend de ce que le code définit :

| Le code définit | La page montre |
| --- | --- |
| `model` : un réseau PyTorch (`nn.Sequential` ou classe `nn.Module`) | l'entrée, la vue réseau, la vue blocs, une carte par couche, la prédiction |
| `model` : un modèle scikit-learn, avec ses données `X` (et `y`) | le nuage de points, la frontière de décision, l'arbre, les poids appris |
| `graph` : un agent LangGraph, avec son état de départ `state` | le schéma des étapes, le chemin suivi, ce que chaque étape a changé |

### Réseaux PyTorch

- **Entrée** : une zone de dessin (un chiffre 28×28), ou une phrase si le modèle commence par `nn.Embedding`.
- **Vue réseau** : des neurones en colonnes reliés par des flèches, plus lumineux quand ils sont actifs.
- **Vue blocs** : le trajet réel des données, avec les raccourcis (connexions résiduelles) et les branches parallèles.
- **Vue 3D** (modèles de texte et ViT) : la sortie de chaque couche est une plaque, une ligne par mot ; les plaques s'empilent comme les couches de BERT et on les fait tourner à la souris.
- **Une carte par couche**, avec une phrase d'explication, la forme de la sortie et le nombre de paramètres :
  - cartes d'activation des convolutions et du MaxPool, valeurs des couches denses ;
  - découpage de l'image en patchs (Vision Transformer) ;
  - vecteur de chaque token, et projection en 2D des embeddings ;
  - une grille par tête d'attention : qui regarde qui ;
  - choix du routeur dans un mélange d'experts (MoE).
- **Prédiction** finale sous forme de barres.
- **Et si ?** : clique sur un nœud de la vue réseau pour l'éteindre ; tout ce qui suit est recalculé sans lui.
- **Ce que le réseau regarde** : à côté du dessin, les pixels qui pèsent le plus dans la réponse.
- **Les filtres** : sous chaque carte d'activation, les poids du filtre qui l'a produite.
- **Ce que le réseau imagine** : pour chaque chiffre, l'image qui le convainc le plus (à refaire après l'entraînement).
- **Entraînement en direct** : le bouton « Entraîner » fait apprendre le réseau sur 1 437 vrais chiffres manuscrits (load_digits de scikit-learn, rien à télécharger). Une courbe montre l'erreur baisser, le taux de réussite sur des chiffres jamais vus s'affiche après chaque époque, et toutes les vues passent aux poids appris.
- **Revoir l'apprentissage** : un curseur remet le réseau dans l'état où il était après chaque époque (0 = avant d'apprendre) ; toutes les vues suivent. Une matrice de confusion montre quel chiffre il confond avec quel autre, et un clic sur un des chiffres qu'il rate l'envoie dans la zone d'entrée pour voir pourquoi.
- **Tromper le réseau** : une attaque adverse. On retouche le dessin de façon presque invisible (chaque pixel bouge d'au plus 25 %) jusqu'à ce que le réseau y lise le chiffre choisi ; on voit la retouche, la réponse avant et après.
- **Paysage de l'erreur** : une carte de l'erreur autour des poids actuels, avec le chemin suivi par l'entraînement époque par époque (axes : les deux directions principales du chemin).
- **Arcs d'attention** (modèles de texte) : survole un mot de la phrase pour voir à quels mots il prête attention.
- **Partager** : le bouton en haut copie un lien qui contient le code de l'éditeur.

Tant qu'on n'a pas cliqué sur « Entraîner », les poids sont aléatoires, mais les formes, les filtres et le trajet des données sont réels.

### Modèles scikit-learn

Le modèle est entraîné sur `X`, `y` s'il ne l'est pas déjà. Le score affiché est mesuré sur ces mêmes données.

Un curseur change le réglage principal du modèle (profondeur d'un arbre, nombre de voisins, `C`…) sans toucher au code, et une courbe compare le score sur les exemples appris et sur 30 % d'exemples mis de côté : on y voit où commence le sur-apprentissage.

#### De l'essai à la production (chaîne MLOps)

Sous le modèle, six étapes réelles, qui tournent sur ta machine :

1. **Données** : une empreinte courte ; si elle change, ce ne sont plus les mêmes données.
2. **Entraînement** : le modèle et son réglage.
3. **Suivi des essais** : chaque combinaison modèle + réglage + données essayée est notée avec ses scores.
4. **Versions** : « Déployer » fige le modèle dans une version numérotée et écrit son dossier sous `.neuro-lens/models/vN/` : `model.joblib`, `model_card.json`, `serve.py` (service FastAPI), `Dockerfile`, `requirements.txt`.
5. **API de prédiction** : `POST /api/predict` répond avec la version en production ; la commande `curl` équivalente est affichée.
6. **Surveillance** : latence, bonnes réponses et dérive des données reçues (indice PSI par colonne). Un curseur simule un monde qui change, jusqu'à l'alerte.

Une carte **Infrastructure** construit pour de bon l'image Docker de la version en production (`docker build`), lance son conteneur sur un port libre de la machine (`127.0.0.1` seulement) et l'interroge par le réseau. Il faut Docker lancé sur la machine.

Le dossier d'une version se lance seul : `cd .neuro-lens/models/v1 && uvicorn serve:app --port 8080`.

### Agents LangGraph

L'agent est exécuté pour de bon à partir de `state`. Un lecteur rejoue le déroulé pas à pas. Le premier exemple remplace le modèle de langage par une fonction simple, pour tourner sans clé d'API.

#### Trace, coût, évaluation (LLMOps et AgentOps)

Sous le déroulé, ce qu'une équipe qui met des agents en production regarde tous les jours :

- **Trace** : une ligne de temps avec chaque étape de l'agent et, dedans, chaque appel au modèle de langage : modèle, tokens en entrée et en sortie, outils demandés, raison d'arrêt. Survoler une barre renvoie à la ligne du code.
- **Coût** : si le code donne `PRIX = {"claude-opus-5-5": {"entree": 5.0, "sortie": 25.0}}` (dollars par million de tokens), chaque appel et chaque exécution sont chiffrés.
- **Garde-fou** : `LIMITES = {"appels": 8, "tokens": 20000}` arrête l'agent avant qu'il ne dépasse le budget (12 appels par défaut).
- **Évaluation** : `cas = [{"question": "…", "attendu": "…"}, …]` fait tourner l'agent sur chaque cas et le note ; la réponse attendue doit se trouver dans sa réponse. On change le prompt ou le modèle, et on regarde la note bouger.
- **Variantes** : `variantes = {"court": {"SYSTEM": "…"}, "détaillé": {"SYSTEM": "…"}}` évalue chaque version sur les mêmes cas (chaque variante remplace les variables nommées : prompt, outils, modèle…) et les compare : réussite, tokens, coût. On garde la meilleure.

L'exemple « Agent IA avec Claude » est piloté par un vrai modèle : Claude choisit lui-même ses outils (recherche, calculatrice) et on voit chaque appel dans le déroulé. Lance le serveur avec ta clé, puis clique sur « Lancer avec Claude » :

```bash
ANTHROPIC_API_KEY=sk-ant-... python -m server.app
```

Le modèle n'est appelé que sur ce clic (jamais pendant la frappe), car chaque passage est payant. Sans clé, une règle fixe joue son rôle.

## Lancer

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m server.app
```

Puis ouvrir http://127.0.0.1:8000.

L'éditeur (Monaco, celui de VS Code) est chargé depuis un CDN ; sans connexion, une zone de texte simple prend le relais.

## Tests

Les tests tournent aussi sur GitHub à chaque push et à chaque pull request (`.github/workflows/tests.yml`).

```bash
pip install -r requirements-dev.txt
pytest
```

### Vue Projet : les fichiers, comme dans VS Code

Le bouton « Projet » ouvre un explorateur de fichiers à gauche du code. Un clic ouvre le fichier dans l'éditeur (en lecture, avec un onglet pour revenir à l'exemple) ; à droite, trois cartes : le rôle du fichier et de son dossier en une phrase, le plan du fichier (fonctions et classes, un clic y va), et le graphe des modules du serveur (A → B : A importe B ; les briques en bas, ceux qui assemblent en haut). Le menu du haut permet d'ouvrir aussi le dossier d'une version mise en production.

## Comment ça marche

```
web/        page HTML/JS : éditeur, vues et cartes
server/
  app.py       serveur FastAPI ; exécute le code dans un processus séparé (limite de 10 s, 120 s pour un agent piloté par Claude)
  analyzer.py  réseaux PyTorch : capture la sortie de chaque couche et le trajet des données
  ml.py        modèles scikit-learn
  mlops.py     suivi des essais, versions, service de prédiction, surveillance
  infra.py     construit l'image Docker d'une version et lance son conteneur
  agents.py    agents LangGraph
  project.py   vue Projet : arborescence, rôle des fichiers, plan, graphe des modules
  llmops.py    trace des agents, tokens et coût, garde-fou, évaluation sur un jeu de cas
tests/
```

Le serveur exécute le code tapé dans l'éditeur : il est fait pour tourner sur ta machine (il n'écoute que sur 127.0.0.1), pas pour être exposé sur Internet.

## Pistes

- Afficher aussi les opérations fonctionnelles (`F.relu`) appelées dans `forward`.
- Un vrai tokenizer et des embeddings pré-entraînés, pour que les mots proches se rapprochent.
- Volet MLOps : suivi des expériences avec MLflow.
