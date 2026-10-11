// Modèles d'exemple proposés dans le menu de la page.
window.NEURO_EXAMPLES = {

  "CNN simple (MNIST)": `import torch.nn as nn

# Une image 28x28 en niveaux de gris entre à gauche...
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3, padding=1),   # 8 filtres 3x3
    nn.ReLU(),
    nn.MaxPool2d(2),                             # 28x28 -> 14x14
    nn.Conv2d(8, 16, kernel_size=3, padding=1),  # 16 filtres
    nn.ReLU(),
    nn.MaxPool2d(2),                             # 14x14 -> 7x7
    nn.Flatten(),
    nn.Linear(16 * 7 * 7, 64),
    nn.ReLU(),
    nn.Linear(64, 10),                           # ...10 scores sortent : un par chiffre
)
`,

  "LeNet-5 (classe)": `import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 6, kernel_size=5, padding=2)
        self.pool1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(6, 16, kernel_size=5)
        self.pool2 = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


model = LeNet()
`,

  "Perceptron multicouche": `import torch.nn as nn

model = nn.Sequential(
    nn.Flatten(),            # 28x28 -> 784 valeurs
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Dropout(0.2),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, 10),
)
`,

  "Erreur à corriger": `import torch.nn as nn

# Ce modèle contient une erreur de taille : lis le message et corrige la ligne indiquée.
model = nn.Sequential(
    nn.Conv2d(1, 8, kernel_size=3),
    nn.ReLU(),
    nn.MaxPool2d(2),
    nn.Flatten(),
    nn.Linear(8 * 14 * 14, 10),
)
`,

  "Mini-Transformer (texte)": `import torch
import torch.nn as nn

# Le modèle lit une phrase : écris-la dans la carte « Entrée », à droite.
example_text = "le chat dort sur le canapé du salon"


class MiniTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 32)       # un vecteur de 32 nombres par mot
        self.position = nn.Embedding(64, 32)      # un vecteur par position dans la phrase
        self.block = nn.TransformerEncoderLayer(  # attention, puis petit réseau dense
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.head = nn.Linear(32, 3)              # 3 classes en sortie

    def forward(self, ids):
        where = torch.arange(ids.shape[1]).unsqueeze(0)
        x = self.embed(ids) + self.position(where)
        x = self.block(x)
        return self.head(x.mean(dim=1))           # moyenne des tokens -> une réponse


model = MiniTransformer()
`,

  "Mini-ViT (image en patchs)": `import torch
import torch.nn as nn

# Vision Transformer : l'image est découpée en patchs, traités comme les mots d'une phrase.


class MiniViT(nn.Module):
    def __init__(self):
        super().__init__()
        self.patches = nn.Conv2d(1, 32, kernel_size=7, stride=7)   # 28x28 -> 4x4 = 16 patchs
        self.cls = nn.Parameter(torch.zeros(1, 1, 32))             # token « classe » ajouté devant
        self.position = nn.Parameter(torch.randn(1, 17, 32) * 0.1) # où se trouve chaque patch
        self.block = nn.TransformerEncoderLayer(
            d_model=32, nhead=4, dim_feedforward=64, batch_first=True)
        self.head = nn.Linear(32, 10)

    def forward(self, image):
        x = self.patches(image).flatten(2).transpose(1, 2)         # 16 tokens de 32 nombres
        x = torch.cat([self.cls, x], dim=1) + self.position        # 17 tokens : le n° 0 est « classe »
        x = self.block(x)
        return self.head(x[:, 0])                                  # on lit la réponse sur le token 0


model = MiniViT()
`,

  "Mélange d'experts (MoE)": `import torch
import torch.nn as nn

example_text = "plus de paramètres sans plus de calcul"


class Experts(nn.Module):
    """Un routeur choisit, pour chaque token, à quel expert (petit réseau) faire confiance."""

    def __init__(self, size=32, n=4):
        super().__init__()
        self.router = nn.Linear(size, n)          # un score par expert
        self.choice = nn.Softmax(dim=-1)          # scores -> répartition qui somme à 100 %
        self.experts = nn.ModuleList(
            nn.Sequential(nn.Linear(size, 64), nn.ReLU(), nn.Linear(64, size)) for _ in range(n))

    def forward(self, x):
        share = self.choice(self.router(x))
        answers = torch.stack([expert(x) for expert in self.experts], dim=-1)
        return (answers * share.unsqueeze(-2)).sum(dim=-1)   # réponses pondérées par le routeur


class MoE(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(1000, 32)
        self.attention = nn.MultiheadAttention(32, num_heads=4, batch_first=True)
        self.norm1 = nn.LayerNorm(32)
        self.moe = Experts()
        self.norm2 = nn.LayerNorm(32)
        self.head = nn.Linear(32, 3)

    def forward(self, ids):
        x = self.embed(ids)
        seen, _ = self.attention(x, x, x)
        x = self.norm1(x + seen)                  # Add + Normalize
        x = self.norm2(x + self.moe(x))           # Add + Normalize
        return self.head(x.mean(dim=1))


model = MoE()
`,

  "ML classique : arbre de décision": `from sklearn.datasets import make_moons
from sklearn.tree import DecisionTreeClassifier

# Deux croissants de points mélangés : X = leur position, y = leur couleur (0 ou 1).
X, y = make_moons(n_samples=200, noise=0.25, random_state=0)

# Change max_depth : plus l'arbre est profond, plus sa frontière est découpée.
model = DecisionTreeClassifier(max_depth=3, random_state=0)
`,

  "ML classique : comparer les modèles": `from sklearn.datasets import make_moons
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC

X, y = make_moons(n_samples=200, noise=0.25, random_state=0)

# Garde une seule ligne « model = ... » active et regarde la frontière changer.
model = LogisticRegression()
# model = KNeighborsClassifier(n_neighbors=5)
# model = RandomForestClassifier(n_estimators=50, max_depth=4, random_state=0)
# model = SVC(kernel="rbf", C=1.0)
`,

  "ML classique : régression": `import numpy as np
from sklearn.linear_model import LinearRegression

# Surface d'un logement (m²) -> prix (k€), avec un peu de hasard.
rng = np.random.default_rng(0)
X = rng.uniform(20, 120, size=(80, 1))
y = 3.2 * X[:, 0] + 40 + rng.normal(0, 25, size=80)

model = LinearRegression()
`,

  "ML classique : regroupement (K-means)": `from sklearn.datasets import load_iris
from sklearn.cluster import KMeans

# 150 fleurs décrites par 4 mesures. Pas de y : le modèle cherche les groupes tout seul.
X = load_iris().data

model = KMeans(n_clusters=3, n_init=10, random_state=0)
`,

  "Agent IA (LangGraph)": `from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Un agent = une boucle « réfléchir -> utiliser un outil -> réfléchir -> ... -> répondre ».
# Ici le « cerveau » est une fonction simple, pour tourner sans clé d'API : remplace
# reflechir() par un appel à un vrai modèle de langage quand tu veux.


class Etat(TypedDict):
    question: str
    notes: list      # ce que l'agent a appris en route
    action: str      # ce qu'il a décidé de faire ensuite
    reponse: str


def reflechir(etat):
    notes = etat["notes"]
    if not any(n.startswith("recherche") for n in notes):
        return {"action": "chercher"}
    if not any(n.startswith("calcul") for n in notes):
        return {"action": "calculer"}
    return {"action": "repondre"}


def chercher(etat):   # outil 1 : une fausse recherche
    return {"notes": etat["notes"] + ["recherche : la tour Eiffel mesure 330 m"]}


def calculer(etat):   # outil 2 : une calculatrice
    return {"notes": etat["notes"] + [f"calcul : 330 m = {330 * 100} cm"]}


def repondre(etat):
    return {"reponse": "La tour Eiffel mesure 33 000 cm."}


graph = StateGraph(Etat)
graph.add_node("reflechir", reflechir)
graph.add_node("chercher", chercher)
graph.add_node("calculer", calculer)
graph.add_node("repondre", repondre)
graph.add_edge(START, "reflechir")
graph.add_conditional_edges("reflechir", lambda etat: etat["action"],
                            {"chercher": "chercher", "calculer": "calculer", "repondre": "repondre"})
graph.add_edge("chercher", "reflechir")   # après un outil, on revient réfléchir
graph.add_edge("calculer", "reflechir")
graph.add_edge("repondre", END)

state = {"question": "Combien de centimètres mesure la tour Eiffel ?",
         "notes": [], "action": "", "reponse": ""}
`,

  "Agent IA avec Claude (LangGraph)": `import ast
import operator
import os
from typing import TypedDict
from langgraph.graph import StateGraph, START, END

# Même boucle qu'avant, mais le « cerveau » est un vrai modèle de langage (Claude) qui choisit
# lui-même ses outils. neuro-lens met NEURO_LENS_LIVE=1 seulement quand tu cliques sur
# « Lancer avec Claude » : on ne paie pas un appel à chaque touche tapée.
VRAI_MODELE = os.environ.get("NEURO_LENS_LIVE") == "1" and bool(os.environ.get("ANTHROPIC_API_KEY"))

OUTILS = [
    {"name": "chercher", "description": "Cherche un fait dans une petite encyclopédie.",
     "input_schema": {"type": "object", "properties": {"sujet": {"type": "string"}}, "required": ["sujet"]}},
    {"name": "calculer", "description": "Calcule une opération simple, par exemple 330 * 100.",
     "input_schema": {"type": "object", "properties": {"operation": {"type": "string"}}, "required": ["operation"]}},
]
ENCYCLOPEDIE = {"tour eiffel": "La tour Eiffel mesure 330 m.", "everest": "L'Everest culmine à 8 849 m."}


class Etat(TypedDict):
    question: str
    messages: list   # la conversation avec le modèle : questions, appels d'outils, résultats
    outil: dict      # l'outil que le modèle veut utiliser maintenant
    action: str
    reponse: str


def reflechir(etat):
    if not VRAI_MODELE:   # sans clé : une règle fixe joue le rôle du modèle
        vus = len(etat["messages"]) // 2
        plan = [("chercher", {"sujet": "tour eiffel"}), ("calculer", {"operation": "330 * 100"})]
        if vus < len(plan):
            nom, entree = plan[vus]
            return {"action": nom, "outil": {"id": str(vus), "input": entree},
                    "messages": etat["messages"] + [{"role": "assistant", "content": f"j'utilise {nom}"}]}
        return {"action": "repondre", "reponse": "La tour Eiffel mesure 33 000 cm."}

    import anthropic
    reponse = anthropic.Anthropic().beta.messages.create(
        model="claude-opus-5-5", max_tokens=4000, output_config={"effort": "low"},
        betas=["server-side-fallback-2026-07-01"], fallbacks="default",   # un autre modèle prend le relais en cas de refus
        system="Réponds en français, en une phrase. Utilise les outils plutôt que ta mémoire.",
        tools=OUTILS, messages=etat["messages"] or [{"role": "user", "content": etat["question"]}],
    )
    messages = (etat["messages"] or [{"role": "user", "content": etat["question"]}]) + [
        {"role": "assistant", "content": reponse.content}]
    appel = next((b for b in reponse.content if b.type == "tool_use"), None)
    if appel:
        return {"action": appel.name, "outil": {"id": appel.id, "input": appel.input}, "messages": messages}
    texte = "".join(b.text for b in reponse.content if b.type == "text")
    return {"action": "repondre", "reponse": texte or f"(arrêt : {reponse.stop_reason})", "messages": messages}


def resultat(etat, texte):   # on rend le résultat de l'outil au modèle
    bloc = {"type": "tool_result", "tool_use_id": etat["outil"]["id"], "content": texte}
    return {"messages": etat["messages"] + [{"role": "user", "content": [bloc] if VRAI_MODELE else texte}]}


def chercher(etat):   # outil 1
    sujet = etat["outil"]["input"]["sujet"].lower()
    return resultat(etat, next((v for k, v in ENCYCLOPEDIE.items() if k in sujet), "Rien trouvé."))


def calculer(etat):   # outil 2 : une calculatrice qui n'accepte que + - * /
    ops = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}

    def valeur(n):
        if isinstance(n, ast.Constant) and isinstance(n.value, (int, float)):
            return n.value
        if isinstance(n, ast.BinOp) and type(n.op) in ops:
            return ops[type(n.op)](valeur(n.left), valeur(n.right))
        raise ValueError("opération non autorisée")

    try:
        texte = str(valeur(ast.parse(etat["outil"]["input"]["operation"], mode="eval").body))
    except (SyntaxError, ValueError) as err:
        texte = f"Erreur : {err}"
    return resultat(etat, texte)


def repondre(etat):
    return {}


graph = StateGraph(Etat)
graph.add_node("reflechir", reflechir)
graph.add_node("chercher", chercher)
graph.add_node("calculer", calculer)
graph.add_node("repondre", repondre)
graph.add_edge(START, "reflechir")
graph.add_conditional_edges("reflechir", lambda etat: etat["action"],
                            {"chercher": "chercher", "calculer": "calculer", "repondre": "repondre"})
graph.add_edge("chercher", "reflechir")
graph.add_edge("calculer", "reflechir")
graph.add_edge("repondre", END)

state = {"question": "Combien de centimètres mesure la tour Eiffel ?",
         "messages": [], "outil": {}, "action": "", "reponse": ""}
`,
};
