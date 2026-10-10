# neuro-lens

Chaque ligne de code prend forme sous tes yeux. Neuro-lens ouvre une fenêtre sur l'intérieur des réseaux de neurones : tu écris, le réseau se dessine, les filtres s'allument. Le deep learning sans boîte noire. Tu tapes une couche, elle apparaît. Tu l'entraînes, elle s'anime.

![Aperçu](docs/apercu.png)

## Ce que ça fait

- **À gauche**, un éditeur Python : tu écris un modèle PyTorch (`nn.Sequential` ou une classe `nn.Module`) dans une variable `model`.
- **À droite**, une page HTML qui se met à jour pendant que tu tapes :
  - une carte par couche (Conv2d, ReLU, MaxPool2d, Flatten, Linear…) avec une phrase qui explique ce qu'elle fait ;
  - la forme de sa sortie (`8 × 14 × 14`) et son nombre de paramètres ;
  - les cartes d'activation des convolutions et du MaxPool, et les valeurs des couches denses ;
  - la prédiction finale (10 classes) sous forme de barres.
- **Une zone de dessin** : dessine un chiffre, tout le réseau se recalcule.
- **Lien code ↔ visuel** : le curseur sur une ligne allume la couche correspondante ; survoler une couche surligne sa ligne.
- **Erreurs expliquées** : une erreur de taille est traduite en français (« La couche Linear attend 1568 valeurs… Mets in_features=1352 ») et la ligne fautive est soulignée.

Le modèle n'est pas encore entraîné : les poids sont aléatoires, mais les formes, les filtres et le trajet des données sont réels.

## Lancer

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m server.app
```

Puis ouvrir http://127.0.0.1:8000.

L'éditeur (Monaco, celui de VS Code) est chargé depuis un CDN ; sans connexion, une zone de texte simple prend le relais.

## Tests

```bash
pip install -r requirements-dev.txt
pytest
```

## Comment ça marche

```
web/        page HTML/JS : éditeur, cartes des couches, zone de dessin
server/
  app.py       serveur FastAPI ; exécute le code dans un processus séparé (limite de 10 s)
  analyzer.py  exécute le code, capture la sortie de chaque couche avec des hooks PyTorch
tests/
```

Le serveur exécute le code tapé dans l'éditeur : il est fait pour tourner sur ta machine (il n'écoute que sur 127.0.0.1), pas pour être exposé sur Internet.

## Pistes

- Bouton « Entraîner » sur MNIST, avec la courbe de loss et les filtres qui évoluent en direct.
- Afficher aussi les opérations fonctionnelles (`F.relu`) appelées dans `forward`.
- Volet MLOps : suivi des expériences avec MLflow.
