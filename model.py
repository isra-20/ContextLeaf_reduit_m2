"""
model.py
--------
Modèle multimodal avec Gated Fusion (fusion par porte).

Ce script implémente le modèle `GatedFusionModel` qui fusionne :
  - des caractéristiques visuelles extraites par un ResNet-18 pré-entraîné ;
  - des caractéristiques contextuelles (5 variables agro-écologiques) encodées par un MLP.

Utilisation :
    from prepare_data2 import get_dataloaders
    from model import GatedFusionModel, build_model

    data = get_dataloaders()
    model = build_model(num_classes=data["num_classes"], ctx_dim=len(data["train_df"].filter(like="temp").columns))  # ou ctx_dim=5
    # ... boucle d'entraînement ...
"""

import torch
import torch.nn as nn
from torchvision import models


class GatedFusionModel(nn.Module):
    """
    Modèle multimodal avec fusion par porte (gated fusion).

    La porte `g` est un petit réseau qui "regarde" simultanément les deux embeddings
    et décide quelle modalité est la plus fiable pour cet échantillon. Si le contexte
    est bruité ou manquant, la porte peut apprendre à se rabattre sur l'image, et
    inversement. Cela rend le modèle plus robuste que la simple concaténation.
    """

    def __init__(self, num_classes: int, ctx_dim: int = 5):
        """
        Args:
            num_classes: Nombre de classes de maladie .
            ctx_dim: Dimension du vecteur contextuel .
        """
        super().__init__()

        # ------------------------------------------------------------------
        # 1) Encodeur visuel : ResNet-18 pré-entraîné sur ImageNet.
        #    On retire la couche de classification finale (`fc`) en la remplaçant
        #    par `Identity`, de sorte que la sortie soit un vecteur de 512 dimensions.
        # ------------------------------------------------------------------
        self.visual = models.resnet18(pretrained=True)
        visual_dim = self.visual.fc.in_features  # 512 pour ResNet-18
        self.visual.fc = nn.Identity()           # → sortie [B, 512]

        # ------------------------------------------------------------------
        # 2) Encodeur contextuel : MLP à deux couches cachées.
        #    5 features → 64 → 32. Un Dropout(0.2) est ajouté pour régulariser.
        #    La sortie est un embedding contextuel de dimension 32.
        # ------------------------------------------------------------------
        self.context = nn.Sequential(
            nn.Linear(ctx_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU()
        )

        # ------------------------------------------------------------------
        # 3) Mécanisme de porte (gate).
        #    Prend en entrée la concaténation des embeddings visuel et contextuel
        #    (dim = 512 + 32 = 544) et produit un scalaire ∈ [0, 1] via une sigmoïde.
        #    Ce scalaire sera utilisé pour pondérer l'embedding visuel ; son
        #    complément (1 - g) pondérera l'embedding contextuel.
        # ------------------------------------------------------------------
        self.gate = nn.Sequential(
            nn.Linear(visual_dim + 32, 1),
            nn.Sigmoid()
        )

        # ------------------------------------------------------------------
        # 4) Classifieur final.
        #    Prend en entrée le vecteur fusionné [g·v , (1-g)·c] de dimension
        #    (512 + 32) = 544, et produit les logits pour `num_classes`.
        #    On utilise une seule couche linéaire, mais on pourrait ajouter
        #    des couches cachées (ex. Linear(544, 128) + ReLU + Dropout + Linear).
        # ------------------------------------------------------------------
        self.classifier = nn.Linear(visual_dim + 32, num_classes)

    def forward(self, img: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
      
        # --- Extraction des caractéristiques ---
        v = self.visual(img)          # [B, 512]
        c = self.context(ctx)         # [B, 32]

        # --- Calcul du poids de la porte ---
        # On concatène les deux embeddings pour que la porte "voie" les deux
        # modalités simultanément avant de décider du poids.
        vc = torch.cat([v, c], dim=1)  # [B, 544]
        g = self.gate(vc)             # [B, 1]  ∈ [0, 1]

        # --- Fusion pondérée ---
        # g pondère l'image, (1 - g) pondère le contexte.
        fused = torch.cat([g * v, (1 - g) * c], dim=1)  # [B, 544]

        # --- Classification ---
        logits = self.classifier(fused)  # [B, num_classes]
        return logits


def build_model(num_classes: int, ctx_dim: int = 5, device: str = None) -> GatedFusionModel:
    """
    Instancie le modèle et le place sur le device spécifié .

   """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = GatedFusionModel(num_classes=num_classes, ctx_dim=ctx_dim).to(device)
    return model


# ======================================================================
# Exemple d'utilisation avec les DataLoaders de prepare_data2.py
# ======================================================================
if __name__ == "__main__":

    from prepare_data2 import get_dataloaders

    # 1) Chargement des données (le CSV est déjà généré par prepare_data2.py)
    data = get_dataloaders()
    train_loader = data["train_loader"]
    test_loader = data["test_loader"]
    num_classes = data["num_classes"]
    ctx_dim = 5  # les 5 features contextuelles 

    print(f"Nombre de classes : {num_classes}")
    print(f"Dimension du contexte : {ctx_dim}")

    # 2) Instanciation du modèle
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(num_classes=num_classes, ctx_dim=ctx_dim, device=device)
    print(f"Modèle créé sur {device}")

    # 3) Vérification rapide des dimensions sur un batch réel
    imgs, ctxs, labels = next(iter(train_loader))
    imgs, ctxs = imgs.to(device), ctxs.to(device)
    with torch.no_grad():
        logits = model(imgs, ctxs)
    print(f"Forme des logits : {logits.shape}  (attendu : [{imgs.size(0)}, {num_classes}])")
    print("✓ Le modèle est prêt pour l'entraînement")