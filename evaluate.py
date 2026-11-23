"""
evaluate.py
-----------
Métriques d'évaluation pour GatedFusionModel.

Spécificités :
  - Support d'un sous-ensemble de classes pour l'évaluation
    (`restrict_labels`) : utile quand dataset/test ne contient qu'un
    sous-ensemble des classes vues à l'entraînement (ex : AD vs PlantDoc).
  - Les prédictions sont masquées (logits = -inf) sur les classes
    non autorisées avant l'argmax : le modèle ne peut pas "sortir" du
    sous-ensemble évalué.
  - Met en avant MACRO-F1 et BALANCED ACCURACY (dataset déséquilibré).
"""

from typing import Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)


# =====================================================================
# 1) Poids de classes
# =====================================================================
def compute_class_weights(train_df, label2idx, device="cpu"):
    """Calcule les poids 'balanced' pour CrossEntropyLoss."""
    counts = train_df["label"].map(label2idx).value_counts().sort_index()
    n_classes = len(counts)
    n_total = counts.sum()
    weights = n_total / (n_classes * counts.values.astype(np.float64))
    return torch.tensor(weights, dtype=torch.float32, device=device)


# =====================================================================
# 2) Évaluation
# =====================================================================
@torch.no_grad()
def evaluate(
    model,
    loader,
    device,
    criterion: Optional[nn.Module] = None,
    idx2label: Optional[dict] = None,
    restrict_labels: Optional[Sequence[int]] = None,
):
    """Évalue le modèle et renvoie un dict de métriques.

    Args:
        restrict_labels : liste d'indices de classes autorisées pour la
            prédiction. Si fournie, les logits des autres classes sont
            mis à -inf avant l'argmax. L'évaluation (rapport, matrice
            de confusion, moyennes) est faite uniquement sur ces classes.
            `None` = comportement d'origine (toutes les classes).

    Returns: dict avec, en plus des clés originales :
        - eval_labels : liste des indices effectivement évalués.
    """
    model.eval()
    all_logits, all_labels = [], []
    total_loss, n_seen = 0.0, 0

    for img, ctx, y in loader:
        img, ctx, y = img.to(device), ctx.to(device), y.to(device)
        logits = model(img, ctx)
        if criterion is not None:
            total_loss += criterion(logits, y).item() * img.size(0)
        n_seen += img.size(0)
        all_logits.append(logits.detach().cpu())
        all_labels.append(y.detach().cpu())

    logits_full = torch.cat(all_logits)              # [N, C_total]
    y_true      = torch.cat(all_labels).numpy()      # labels complets

    # Masque : les classes non autorisées sont neutralisées
    if restrict_labels is not None:
        eval_labels = list(restrict_labels)
        mask = torch.full((logits_full.size(1),), float("-inf"))
        mask[eval_labels] = 0.0
        logits_used = logits_full + mask
    else:
        # Toutes les classes présentes dans y_true + y_pred
        logits_used = logits_full
        eval_labels = sorted(set(np.unique(y_true).tolist()) |
                             set(logits_used.argmax(dim=1).numpy().tolist()))

    y_pred = logits_used.argmax(dim=1).numpy()

    # Rapport lisible : noms alignés sur les indices évalués
    if idx2label is not None:
        target_names = [idx2label.get(i, f"class_{i}") for i in eval_labels]
    else:
        target_names = None

    cls_report_str = classification_report(
        y_true, y_pred,
        labels=eval_labels,
        target_names=target_names,
        digits=4, zero_division=0,
    )

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=eval_labels, average=None, zero_division=0,
    )

    return {
        "loss": (total_loss / n_seen) if criterion is not None else None,
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "balanced_acc": float(balanced_accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "classification_report": cls_report_str,
        "precision_per_class": precision.tolist(),
        "recall_per_class": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support_per_class": support.tolist(),
        "confusion": confusion_matrix(y_true, y_pred, labels=eval_labels),
        "eval_labels": eval_labels,          # ← nouveau
        "y_true": y_true,
        "y_pred": y_pred,
    }


# =====================================================================
# 3) Affichages
# =====================================================================
def format_metrics(metrics, prefix=""):
    loss_str = f"{metrics['loss']:.4f}" if metrics["loss"] is not None else "n/a"
    return (
        f"{prefix}Loss {loss_str} | "
        f"Acc {metrics['accuracy']:.4f} | "
        f"BalAcc {metrics['balanced_acc']:.4f} | "
        f"Macro-F1 {metrics['macro_f1']:.4f} | "
        f"Weighted-F1 {metrics['weighted_f1']:.4f}"
    )


def print_classification_report(metrics):
    print("\n=== Classification report ===")
    print(metrics["classification_report"])


def print_per_class_report(metrics, idx2label):
    """Affiche P/R/F1/support pour les classes effectivement évaluées."""
    eval_labels = metrics.get("eval_labels") or sorted(idx2label.keys())

    print("\n=== Rapport par classe (compact) ===")
    print(f"{'Classe':<30} {'Prec':>6} {'Recall':>6} {'F1':>6} {'Support':>8}")
    print("-" * 62)
    for i, label_idx in enumerate(eval_labels):
        label = idx2label.get(label_idx, f"class_{label_idx}")
        print(
            f"{label[:29]:<30} "
            f"{metrics['precision_per_class'][i]:>6.3f} "
            f"{metrics['recall_per_class'][i]:>6.3f} "
            f"{metrics['f1_per_class'][i]:>6.3f} "
            f"{metrics['support_per_class'][i]:>8d}"
        )
    print("-" * 62)


def plot_confusion_matrix(metrics, idx2label, save_path=None, normalize=True):
    """Matrice de confusion restreinte aux classes évaluées."""
    import matplotlib.pyplot as plt

    cm = metrics["confusion"].astype(float)
    if normalize:
        row_sums = cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        cm = cm / row_sums

    eval_labels = metrics.get("eval_labels") or sorted(idx2label.keys())
    labels = [idx2label.get(i, f"class_{i}") for i in eval_labels]

    fig, ax = plt.subplots(figsize=(max(6, len(labels)), max(5, len(labels) - 1)))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1 if normalize else None)
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yticklabels(labels)
    ax.set_xlabel("Prédit")
    ax.set_ylabel("Vrai")
    ax.set_title("Matrice de confusion" + (" (normalisée)" if normalize else ""))

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm[i, j] > 0.5 else "black",
                    fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"✓ Matrice de confusion sauvegardée : {save_path}")
    else:
        plt.show()
    plt.close(fig)