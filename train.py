"""
train.py
--------
Boucle d'entraînement pour le modèle GatedFusionModel.

GESTION DU DÉSÉQUILIBRE (PlantDoc+ AD) :
  1) Class weights dans CrossEntropyLoss (activés par défaut).
  2) Split stratifié train → train/val, pour que la validation contienne
     toutes les classes, même les plus rares.
  3) Early stopping ET sélection du meilleur modèle sur le MACRO-F1 de
     validation, jamais sur l'accuracy.
  4) Le test final n'est évalué qu' avec le meilleur checkpoint.

SORTIES (dossier `checkpoints/`) :
  - best_model.pt          : meilleur modèle (sur macro-F1 val)
  - history.json           : historique des époques
  - best_val_metrics.json  : métriques de validation du meilleur modèle
  - test_metrics.json      : métriques finales sur le test
  - classification_report.txt : rapport sklearn (test)
  - confusion_matrix.png   : matrice de confusion normalisée (test)
  - args.json              : hyperparamètres utilisés

Usage :
    python train.py
    python train.py --epochs 20 --lr 5e-5
  
    python train.py --no-class-weights
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader 

from prepare_data2 import (
    PlantDocMultimodal,
    get_transforms,
    get_dataloaders,
    CONTEXT_COLS,
    BATCH_SIZE,
    NUM_WORKERS,
)
from model import build_model
from evaluate import (
    compute_class_weights,
    evaluate,
    format_metrics,
    print_classification_report,
    print_per_class_report,
    plot_confusion_matrix,
)


# =====================================================================
# 1) Split stratifié train → train / validation
# =====================================================================
def split_train_val(train_df, label2idx, val_ratio=0.15, seed=42,
                    batch_size=BATCH_SIZE, num_workers=NUM_WORKERS):
    """Découpe le DataFrame d'entraînement en train/validation stratifiés."""
    tr_df, val_df = train_test_split(
        train_df,
        test_size=val_ratio,
        stratify=train_df["label"],
        random_state=seed,
    )
    tr_df = tr_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)

    train_tf, test_tf = get_transforms()
    tr_ds = PlantDocMultimodal(tr_df, label2idx, transform=train_tf)
    val_ds = PlantDocMultimodal(val_df, label2idx, transform=test_tf)

    tr_loader = DataLoader(
        tr_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )
    return tr_loader, val_loader, tr_df, val_df



# =====================================================================
# 2) Early stopping (sur macro-F1)
# =====================================================================
class EarlyStopping:
    def __init__(self, patience=5, min_delta=1e-4, mode="max"):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.best = -np.inf if mode == "max" else np.inf
        self.num_bad = 0
        self.should_stop = False

    def step(self, value):
        if self.mode == "max":
            improved = value > self.best + self.min_delta
        else:
            improved = value < self.best - self.min_delta

        if improved:
            self.best = value
            self.num_bad = 0
        else:
            self.num_bad += 1
            if self.num_bad >= self.patience:
                self.should_stop = True
        return improved


# =====================================================================
# 3) Une époque d'entraînement
# =====================================================================
def train_one_epoch(model, loader, optimizer, criterion, device, grad_clip=5.0):
    model.train()
    total_loss, n_seen = 0.0, 0
    for img, ctx, y in loader:
        img, ctx, y = img.to(device), ctx.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(img, ctx)
        loss = criterion(logits, y)
        loss.backward()
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=grad_clip)
        optimizer.step()
        total_loss += loss.item() * img.size(0)
        n_seen += img.size(0)
    return total_loss / n_seen


# =====================================================================
# 4) Helpers de sérialisation
# =====================================================================
def metrics_to_json(metrics):
    """Convertit un dict de métriques evaluate() en JSON-sérialisable."""
    return {
        "loss": metrics["loss"],
        "accuracy": metrics["accuracy"],
        "balanced_acc": metrics["balanced_acc"],
        "macro_f1": metrics["macro_f1"],
        "weighted_f1": metrics["weighted_f1"],
        "precision_per_class": metrics["precision_per_class"],
        "recall_per_class": metrics["recall_per_class"],
        "f1_per_class": metrics["f1_per_class"],
        "support_per_class": metrics["support_per_class"],
        "confusion": metrics["confusion"].tolist(),
        "classification_report": metrics["classification_report"],
    }


# =====================================================================
# 6) Point d'entrée
# =====================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--noise", type=float, default=2.0,
                        help="noise_level du contexte (cohérent avec prepare_data2)")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS)
    parser.add_argument("--no-class-weights", action="store_true")
    
    parser.add_argument("--out-dir", default="checkpoints",
                        help="dossier racine des sorties")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # ---- Reproductibilité ----
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device : {device}")

    # ---- Dossier de sortie : checkpoints/<timestamp>/ ----
    run_dir = Path(args.out_dir) / time.strftime("%Y%m%d_%H%M%S")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"Sortie : {run_dir}")

    # ---- Données ----
    data = get_dataloaders(noise_level=args.noise, batch_size=args.batch_size)
    train_loader, val_loader, tr_df, val_df = split_train_val(
        data["train_df"], data["label2idx"],
        val_ratio=args.val_ratio, seed=args.seed,
        batch_size=args.batch_size, num_workers=args.num_workers,
    )
    test_loader = data["test_loader"]
    idx2label = data["idx2label"]
    num_classes = data["num_classes"]
    ctx_dim = len(CONTEXT_COLS)

      # ---- Sous-ensemble de classes présent dans dataset/test ----
    test_df = data["test_df"]
    test_labels_present = sorted(test_df["label"].unique().tolist())
    idx2label_eval = {data["label2idx"][l]: l for l in test_labels_present}
    restrict_labels = sorted(idx2label_eval.keys())

    print(f"Train : {len(tr_df)} | Val : {len(val_df)} "
          f"| Test : {len(test_loader.dataset)}")
    print(f"Classes (train)   : {num_classes}")
    print(f"Classes (test)    : {len(idx2label_eval)} → {list(idx2label_eval.values())}")
    print(f"ctx_dim : {ctx_dim}")


    dist = tr_df["label"].value_counts()
    print("Distribution train :")
    for lbl, n in dist.items():
        print(f"  {lbl:<30} {n:>5}  ({n/len(tr_df)*100:5.1f}%)")

   

    # ---- Modèle ----
    model = build_model(num_classes=num_classes, ctx_dim=ctx_dim, device=device)

    # ---- Perte pondérée ----
    class_weights = None
    if not args.no_class_weights:
        class_weights = compute_class_weights(tr_df, data["label2idx"], device=device)
        print("Class weights :", np.round(class_weights.cpu().numpy(), 3))
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # ---- Optimiseur + scheduler ----
    optimizer = optim.AdamW(model.parameters(), lr=args.lr,
                            weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=2,
    )

    # ---- Boucle d'entraînement ----
    early = EarlyStopping(patience=args.patience, mode="max")
    history = []
    best_macro_f1 = -1.0
    best_epoch = -1
    best_val_metrics = None
    best_ckpt = run_dir / "best_model.pt"

    for epoch in range(1, args.epochs + 1):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device,
                               criterion=criterion, idx2label=idx2label)
        scheduler.step(val_metrics["macro_f1"])

        print(f"Epoch {epoch:02d} | train_loss {tr_loss:.4f} | "
              + format_metrics(val_metrics, prefix="[val] "))
        history.append({
            "epoch": epoch,
            "train_loss": tr_loss,
            "val_loss": val_metrics["loss"],
            "val_accuracy": val_metrics["accuracy"],
            "val_balanced_acc": val_metrics["balanced_acc"],
            "val_macro_f1": val_metrics["macro_f1"],
            "val_weighted_f1": val_metrics["weighted_f1"],
            "lr": optimizer.param_groups[0]["lr"],
        })

        improved = early.step(val_metrics["macro_f1"])
        if improved:
            best_macro_f1 = val_metrics["macro_f1"]
            best_epoch = epoch
            best_val_metrics = val_metrics
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch,
                "val_macro_f1": best_macro_f1,
                "val_accuracy": val_metrics["accuracy"],
                "val_balanced_acc": val_metrics["balanced_acc"],
                "val_weighted_f1": val_metrics["weighted_f1"],
                "label2idx": data["label2idx"],
                "idx2label": {int(k): v for k, v in idx2label.items()},
                "ctx_mean": data["ctx_mean"].to_dict(),
                "ctx_std": data["ctx_std"].to_dict(),
                "noise_level": args.noise,
            }, best_ckpt)
            print(f"  ↳ nouveau meilleur (macro-F1 {best_macro_f1:.4f}) → {best_ckpt}")

        if early.should_stop:
            print(f"Early stopping (patience={args.patience})")
            break

    # ---- Résumé du meilleur modèle (validation) ----
    print("\n=== Meilleur modèle (validation) ===")
    print(f"Epoch          : {best_epoch}")
    print(f"Macro-F1 (val) : {best_macro_f1:.4f}")
    if best_val_metrics is not None:
        print(format_metrics(best_val_metrics, prefix="[best val] "))

    # ---- Évaluation finale sur le TEST ----
    print("\n=== Évaluation finale sur le TEST (meilleur modèle) ===")
    print(f"Classes évaluées : {list(idx2label_eval.values())}\n")

    ckpt = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(ckpt["model_state_dict"])
    test_metrics = evaluate(
        model, test_loader, device,
        criterion=criterion,
        idx2label=idx2label_eval,       # noms des 4 classes AD
        restrict_labels=restrict_labels # masque les 4 autres classes
    )

    print(format_metrics(test_metrics, prefix="[test] "))
    print_classification_report(test_metrics)
    #print_per_class_report(test_metrics, idx2label_eval)

    try:
        plot_confusion_matrix(test_metrics, idx2label_eval,
                              save_path=str(run_dir / "confusion_matrix.png"))
    except Exception as e:
        print(f"⚠️  Matrice de confusion non générée : {e}")

    # ---- Sauvegardes dans checkpoints/<timestamp>/ ----
    with open(run_dir / "history.json", "w") as f:
        json.dump(history, f, indent=2)

    with open(run_dir / "best_val_metrics.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "best_val_macro_f1": best_macro_f1,
            "metrics": metrics_to_json(best_val_metrics) if best_val_metrics else None,
            "idx2label": {str(k): v for k, v in idx2label.items()},
        }, f, indent=2)

    with open(run_dir / "test_metrics.json", "w") as f:
        json.dump({
            "best_epoch": best_epoch,
            "eval_labels": test_metrics["eval_labels"],
            "metrics": metrics_to_json(test_metrics),
            "idx2label_eval": {str(k): v for k, v in idx2label_eval.items()},
            "idx2label_train": {str(k): v for k, v in idx2label.items()},
        }, f, indent=2)

    with open(run_dir / "classification_report.txt", "w") as f:
        f.write(test_metrics["classification_report"])

    with open(run_dir / "args.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    print(f"\n✓ Résultats sauvegardés dans {run_dir}")
    print(f"  - {best_ckpt.name} (meilleur modèle)")
    print("  - best_val_metrics.json (métriques de validation du meilleur modèle)")
    print("  - test_metrics.json, classification_report.txt, confusion_matrix.png")
    print("  - history.json, args.json")


if __name__ == "__main__":
    main()