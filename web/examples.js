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
};
