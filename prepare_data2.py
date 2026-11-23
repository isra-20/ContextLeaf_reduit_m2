"""
prepare_data2.py
Prépare les données PlantDoc (tomate, 8 classes) pour un modèle multimodal :
  image + 5 features contextuelles synthétiques corrélées au label.

Lit directement depuis ContextLeaf_reduit/dataset 
Les métadonnées synthétiques sont sauvegardées dans un CSV 

Utilisation :
    from prepare_data2 import get_dataloaders
    data = get_dataloaders(noise_level=2.0)
    train_loader, test_loader = data["train_loader"], data["test_loader"]

Diagnostic du niveau de bruit (contexte seul) :
    python prepare_data2.py --diagnose

Régénérer le CSV  :
    python prepare_data2.py --rebuild
"""

import os
import json
import argparse
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# =============================================================================
# 1) CONFIGURATION
# =============================================================================
DATASET_ROOT = '/content/drive/MyDrive/ContextLeaf_reduit/dataset'
DEFAULT_CSV  = '/content/drive/MyDrive/ContextLeaf_reduit/dataset_metadata.csv'

# paramètres pour les métadonnées synthétiques
CONTEXT_COLS = ["temperature", "humidity", "rainfall", "soil_pH", "soil_organic_carbon"]
NOISE_LEVEL  = 2.0    # multiplicateur du bruit : plus grand = contexte moins informatif
SEED         = 42     # reproductibilité des métadonnées synthétiques

IMG_SIZE    = 224
BATCH_SIZE  = 32
NUM_WORKERS = 2

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

# =============================================================================
# 2) CONSTRUCTION DU DATAFRAME : une ligne par image (chemin, label, split)
# =============================================================================
def build_file_list(root_dir):
    """Parcourt root_dir/{train,test}/<classe>/ et liste toutes les images.
    `sorted` garantit un ordre stable des classes entre deux exécutions."""
    data = []
    for split in ["train", "test"]:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            continue
        for cls in sorted(os.listdir(split_dir)):
            cls_dir = os.path.join(split_dir, cls)
            if not os.path.isdir(cls_dir):
                continue
            for img_name in os.listdir(cls_dir):
                if img_name.lower().endswith((".jpg", ".jpeg", ".png")):
                    data.append({
                        "image_path": os.path.join(cls_dir, img_name),
                        "label": cls,
                        "split": split,
                    })
    return pd.DataFrame(data)


# =============================================================================
# 3) MÉTADONNÉES SYNTHÉTIQUES : feature = base + signal(label) + bruit
# =============================================================================
def add_synthetic_context(df, noise_level=NOISE_LEVEL, seed=SEED):
    """Ajoute 5 features corrélées au label.
    Le signal dépend du label ; le bruit est multiplié par `noise_level`.
    Les codes de classes sont calculés sur tout le df -> cohérents train/test."""
    df = df.copy()
    codes = df["label"].astype("category").cat.codes.values
    rng = np.random.default_rng(seed)
    n = len(df)

    # écart-types de base du bruit pour chaque feature
    base_std = {"temperature": 2, "humidity": 5, "rainfall": 10,
                "soil_pH": 0.3, "soil_organic_carbon": 0.2}
    noise = {k: rng.normal(size=n) * s * noise_level for k, s in base_std.items()}

    df["temperature"]          = 20 + 5 * np.sin(codes) + noise["temperature"]
    df["humidity"]             = 60 + 15 * np.cos(codes) + noise["humidity"]
    df["rainfall"]             = 50 + 40 * (codes % 3) + noise["rainfall"]
    df["soil_pH"]              = 6.5 + 0.5 * np.sin(codes) + noise["soil_pH"]
    df["soil_organic_carbon"]  = 1.5 + 0.8 * np.cos(codes) + noise["soil_organic_carbon"]
    return df


# =============================================================================
# 4) CACHE CSV : sauvegarde / chargement du DataFrame
# =============================================================================
def save_metadata_csv(df, csv_path, meta_path, root_dir, noise_level, seed):
    """Sauvegarde le DataFrame (avec métadonnées synthétiques) dans un CSV.
    Les chemins sont stockés en RELATIF à `root_dir` pour rester portables.
    Un JSON sidecar contient les paramètres de génération."""
    df_save = df.copy()
    df_save["image_path"] = df_save["image_path"].apply(
        lambda p: os.path.relpath(p, root_dir)
    )
    # ordre des colonnes : path, label, split, puis les 5 features
    cols = ["image_path", "label", "split"] + CONTEXT_COLS
    df_save = df_save[cols]

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)
    df_save.to_csv(csv_path, index=False)
    with open(meta_path, "w") as f:
        json.dump({
            "noise_level": noise_level,
            "seed": seed,
            "root_dir": os.path.abspath(root_dir),
        }, f, indent=2)


def load_metadata_csv(csv_path, meta_path, root_dir):
    """Charge le DataFrame depuis le CSV et reconstruit les chemins absolus."""
    df = pd.read_csv(csv_path)
    df["image_path"] = df["image_path"].apply(lambda p: os.path.join(root_dir, p))
    with open(meta_path, "r") as f:
        meta = json.load(f)
    return df, meta


def build_or_load_dataframe(root_dir=DATASET_ROOT,
                            csv_path=DEFAULT_CSV,
                            noise_level=NOISE_LEVEL,
                            seed=SEED,
                            force_rebuild=False):
    """Charge depuis le CSV si présent (source de vérité), sinon génère + sauvegarde.

    Le CSV est prioritaire : si vous l'avez édité à la main, il sera respecté.
    `force_rebuild=True` ignore le cache et régénère depuis les images.
    """
    meta_path = os.path.splitext(csv_path)[0] + ".json"

    if not force_rebuild and os.path.exists(csv_path):
        df, meta = load_metadata_csv(csv_path, meta_path, root_dir)
        if meta.get("noise_level") != noise_level or meta.get("seed") != seed:
            print(f"⚠️  CSV existant généré avec noise_level={meta.get('noise_level')}, "
                  f"seed={meta.get('seed')} — chargement du CSV tel quel. "
                  f"Utilisez force_rebuild=True (ou --rebuild) pour régénérer.")
        print(f"✓ CSV chargé : {csv_path} ({len(df)} lignes)")
        return df

    # Génération à partir des images
    raw_df = build_file_list(root_dir)
    df = add_synthetic_context(raw_df, noise_level=noise_level, seed=seed)
    save_metadata_csv(df, csv_path, meta_path, root_dir, noise_level, seed)
    print(f"✓ CSV sauvegardé : {csv_path} ({len(df)} lignes)")
    return df


# =============================================================================
# 5) DIAGNOSTIC : quelle part du label est « fuitée » dans le contexte ?
# =============================================================================
def context_only_accuracy(df):
    """Entraîne un Random Forest sur le contexte seul (sans image).
    ~100 % = fuite trop forte ; ~hasard (12,5 %) = contexte inutile."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score

    tr, te = df[df.split == "train"], df[df.split == "test"]
    clf = RandomForestClassifier(n_estimators=200, random_state=0, n_jobs=-1)
    clf.fit(tr[CONTEXT_COLS], tr["label"])
    return accuracy_score(te["label"], clf.predict(te[CONTEXT_COLS]))


def diagnose_noise_levels(raw_df, levels=(0.5, 1, 2, 4, 8)):
    """Affiche l'accuracy du contexte seul pour plusieurs niveaux de bruit."""
    majority = raw_df[raw_df.split == "test"]["label"].value_counts(normalize=True).max()
    print(f"Référence hasard (classe majoritaire) : {majority:.1%}")
    for nl in levels:
        acc = context_only_accuracy(add_synthetic_context(raw_df, noise_level=nl))
        print(f"noise_level={nl:<4} -> accuracy contexte seul : {acc:.1%}")


# =============================================================================
# 6) NORMALISATION : statistiques calculées sur le TRAIN uniquement
# =============================================================================
def normalize_context(train_df, test_df, cols=CONTEXT_COLS):
    """Standardise les features contextuelles (moyenne 0, écart-type 1).
    Les stats du train sont appliquées au test pour éviter toute fuite."""
    train_df, test_df = train_df.copy(), test_df.copy()
    mean, std = train_df[cols].mean(), train_df[cols].std()

    train_df[cols] = (train_df[cols] - mean) / std
    test_df[cols]  = (test_df[cols]  - mean) / std
    return train_df, test_df, mean, std


# =============================================================================
# 7) DATASET PYTORCH : renvoie (image, contexte, label)
# =============================================================================
class PlantDocMultimodal(Dataset):
    """Dataset multimodal. Les données sont converties en listes/tableaux NumPy
    dans __init__ pour éviter `df.iloc` (lent) à chaque accès."""

    def __init__(self, df, label2idx, context_cols=CONTEXT_COLS, transform=None):
        self.paths  = df["image_path"].tolist()
        self.labels = df["label"].map(label2idx).values
        self.ctx    = df[context_cols].values.astype(np.float32)
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        img = Image.open(self.paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        ctx   = torch.from_numpy(self.ctx[idx])
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        return img, ctx, label


# =============================================================================
# 8) TRANSFORMATIONS D'IMAGES : augmentation en train, simple resize en test
# =============================================================================
def get_transforms(img_size=IMG_SIZE):
    norm = transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)
    train_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        norm,
    ])
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        norm,
    ])
    return train_tf, test_tf


# =============================================================================
# 9) FONCTION PRINCIPALE : enchaîne toutes les étapes
# =============================================================================
def get_dataloaders(root_dir=DATASET_ROOT, csv_path=DEFAULT_CSV,
                    noise_level=NOISE_LEVEL, batch_size=BATCH_SIZE,
                    num_workers=NUM_WORKERS, save_stats_to=None,
                    force_rebuild=False):
    """Retourne un dict : loaders, datasets, mapping des labels, stats du contexte.
    `save_stats_to` (chemin .json) sauvegarde mean/std pour l'inférence future.
    `force_rebuild` ignore le CSV existant et régénère les métadonnées."""
    df = build_or_load_dataframe(root_dir, csv_path, noise_level,
                                 force_rebuild=force_rebuild)

    train_df = df[df.split == "train"].reset_index(drop=True)
    test_df  = df[df.split == "test"].reset_index(drop=True)
    assert set(test_df.label).issubset(set(train_df.label)), "Classes différentes train/test"

    train_df, test_df, ctx_mean, ctx_std = normalize_context(train_df, test_df)

    # mapping label <-> indice, défini une seule fois et partagé
    label2idx = {l: i for i, l in enumerate(sorted(train_df["label"].unique()))}
    idx2label = {i: l for l, i in label2idx.items()}

    train_tf, test_tf = get_transforms()
    train_ds = PlantDocMultimodal(train_df, label2idx, transform=train_tf)
    test_ds  = PlantDocMultimodal(test_df,  label2idx, transform=test_tf)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=batch_size, shuffle=False,
                              num_workers=num_workers, pin_memory=True)

    if save_stats_to:
        with open(save_stats_to, "w") as f:
            json.dump({"mean": ctx_mean.to_dict(), "std": ctx_std.to_dict(),
                       "label2idx": label2idx, "noise_level": noise_level}, f, indent=2)

    return {
        "train_loader": train_loader, "test_loader": test_loader,
        "train_ds": train_ds, "test_ds": test_ds,
        "train_df": train_df, "test_df": test_df,
        "label2idx": label2idx, "idx2label": idx2label,
        "num_classes": len(label2idx),
        "ctx_mean": ctx_mean, "ctx_std": ctx_std,
        "metadata_csv": csv_path,
    }


# =============================================================================
# 10) POINT D'ENTRÉE : diagnostic de bruit OU vérification rapide des shapes
# =============================================================================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnose", action="store_true",
                        help="teste plusieurs niveaux de bruit (contexte seul)")
    parser.add_argument("--noise", type=float, default=NOISE_LEVEL)
    parser.add_argument("--root",  default=DATASET_ROOT)
    parser.add_argument("--csv",   default=DEFAULT_CSV)
    parser.add_argument("--rebuild", action="store_true",
                        help="ignore le CSV existant et régénère les métadonnées")
    args = parser.parse_args()

    # vérification rapide de l'arborescence
    print(f"Contenu de {args.root} : {os.listdir(args.root)}")
    print(f"Classes dans train     : {sorted(os.listdir(os.path.join(args.root, 'train')))}")

    if args.diagnose:
        raw_df = build_file_list(args.root)
        diagnose_noise_levels(raw_df)
    else:
        data = get_dataloaders(root_dir=args.root, csv_path=args.csv,
                               noise_level=args.noise,
                               force_rebuild=args.rebuild,
                               save_stats_to="ctx_stats.json")
        imgs, ctx, labels = next(iter(data["train_loader"]))
        print("Train / Test :", len(data["train_ds"]), "/", len(data["test_ds"]))
        print("Shapes       :", tuple(imgs.shape), tuple(ctx.shape), tuple(labels.shape))
        print("Classes      :", data["idx2label"])
        print("CSV utilisé  :", data["metadata_csv"])