
"""
prepare_data1.py

Organisation finale :
  • dataset/train/  ← PlantDoc-OD (TRAIN + TEST) + PlantDoc-Dataset (train + test)
                       → 8 classes cibles via CLASS_MAP
  • dataset/test/   ← AD uniquement
                       → 4 classes : Healthy, Bacterial Spot, Early Blight, Mosaic Virus
"""

import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

# ----------------------------------------------------------------------
# 1. Configuration des chemins
# ----------------------------------------------------------------------
RAW_DIR = Path("/content/drive/MyDrive/ContextLeaf_reduit/raw_data")

PLANTDOC_OD_DIR = RAW_DIR / "PlantDoc-Object-Detection-Dataset"
CROPPEDPD_DIR   = RAW_DIR / "PlantDoc-Dataset"
AD_DIR          = RAW_DIR / "AD"

OUTPUT_DIR       = Path("/content/drive/MyDrive/ContextLeaf_reduit/dataset")
OUTPUT_TRAIN_DIR = OUTPUT_DIR / "train"
OUTPUT_TEST_DIR  = OUTPUT_DIR / "test"

# ----------------------------------------------------------------------
# 2. Mapping vers les 8 classes cibles (TRAIN)
# ----------------------------------------------------------------------
CLASS_MAP = {
    "Tomato leaf":                  "Healthy",
    "Tomato leaf bacterial spot":   "Bacterial Spot",
    "Tomato Early blight leaf":     "Early Blight",
    "Tomato leaf late blight":      "Late Blight",
    "Tomato mold leaf":             "Leaf Mold",
    "Tomato leaf mosaic virus":     "Mosaic Virus",
    "Tomato Septoria leaf spot":    "Septoria Leaf Spot",
    "Tomato leaf yellow virus":     "Yellow Leaf Curl",
}

TARGET_CLASSES = sorted(set(CLASS_MAP.values()))
print("Classes cibles (entraînement) :", TARGET_CLASSES)

# Classes AD attendues (4 classes pour le test)
AD_CLASSES = ["Healthy", "Bacterial Spot", "Early Blight", "Mosaic Virus"]
# lookup insensible à la casse pour être robuste aux variations de nommage
AD_LOOKUP = {c.lower(): c for c in AD_CLASSES}

# ----------------------------------------------------------------------
# 3. Fonctions utilitaires
# ----------------------------------------------------------------------
def normaliser_nom_classe(nom: str) -> str:
    """Remplace '_' par espace, trim, et supprime les espaces multiples."""
    return " ".join(nom.replace("_", " ").split()).strip()


def extraire_classes_xml(xml_path: Path) -> set:
    """Lit un XML et retourne l'ensemble des <object><name>.</name></object>."""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        return {
            normaliser_nom_classe(obj.find("name").text)
            for obj in root.findall("object")
            if obj.find("name") is not None and obj.find("name").text
        }
    except Exception as e:
        print(f"⚠️  Erreur lecture {xml_path.name} : {e}")
        return set()


def copier_image(src: Path, dst_dir: Path, verbose=False):
    """Copie src dans dst_dir si elle n'existe pas déjà."""
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    if not dst.exists():
        shutil.copy2(src, dst)
        if verbose:
            print(f"  → {src.name}  →  {dst_dir.name}/")
        return True
    return False


def trouver_image_pour_xml(xml_file: Path) -> Path | None:
    """Cherche l'image correspondant à un XML (essais d'extensions)."""
    for ext in (".jpg", ".jpeg", ".png", ".JPG", ".JPEG", ".PNG"):
        candidate = xml_file.with_suffix(ext)
        if candidate.exists():
            return candidate
    return None


# ----------------------------------------------------------------------
# 4. TRAIN – PlantDoc-OD (parcours XML → mapping → copie)
# ----------------------------------------------------------------------
def traiter_plantdoc_od(src_dir: Path, dst_root: Path, label: str):
    print(f"\n=== {label} ===")
    print(f"  Source : {src_dir}")
    if not src_dir.is_dir():
        print(f"  ❌ Dossier introuvable")
        return

    n_copies, n_skip_xml, n_skip_img = 0, 0, 0
    for xml_file in src_dir.glob("*.xml"):
        classes_presentes = extraire_classes_xml(xml_file)
        classes_cibles = {CLASS_MAP[c] for c in classes_presentes if c in CLASS_MAP}
        if not classes_cibles:
            n_skip_xml += 1
            continue

        img_file = trouver_image_pour_xml(xml_file)
        if img_file is None:
            n_skip_img += 1
            continue

        for classe in classes_cibles:
            if copier_image(img_file, dst_root / classe):
                n_copies += 1

    print(f"  → {n_copies} copies | {n_skip_xml} XML hors classes cibles | "
          f"{n_skip_img} images manquantes")


# ----------------------------------------------------------------------
# 5. TRAIN – PlantDoc-Dataset (classification, structure par dossiers)
# ----------------------------------------------------------------------
def traiter_croppedpd(src_dir: Path, dst_root: Path, label: str):
    print(f"\n=== {label} ===")
    print(f"  Source : {src_dir}")
    if not src_dir.is_dir():
        print(f"  ❌ Dossier introuvable")
        return

    n_copies = 0
    for classe_dir in sorted(src_dir.iterdir()):
        if not classe_dir.is_dir():
            continue
        nom_brut = normaliser_nom_classe(classe_dir.name)
        if nom_brut not in CLASS_MAP:
            continue
        cible = CLASS_MAP[nom_brut]
        dst_dir = dst_root / cible

        for img_file in classe_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if copier_image(img_file, dst_dir):
                    n_copies += 1

    print(f"  → {n_copies} copies")


# ----------------------------------------------------------------------
# 6. TEST – AD (4 classes)
# ----------------------------------------------------------------------
def traiter_ad(src_dir: Path, dst_root: Path, label: str):
    print(f"\n=== {label} ===")
    print(f"  Source : {src_dir}")
    if not src_dir.is_dir():
        print(f"  ❌ Dossier introuvable")
        return

    # Si AD contient un sous-dossier "test", on l'utilise ; sinon on scanne AD directement
    if (src_dir / "test").is_dir():
        print(f"  (sous-dossier 'test' détecté)")
        src_dir = src_dir / "test"

    n_copies = 0
    for classe_dir in sorted(src_dir.iterdir()):
        if not classe_dir.is_dir():
            continue
        nom_brut = normaliser_nom_classe(classe_dir.name)
        cible = AD_LOOKUP.get(nom_brut.lower())
        if cible is None:
            print(f"  ⏭️  Classe AD non ciblée : {classe_dir.name}")
            continue
        dst_dir = dst_root / cible

        for img_file in classe_dir.iterdir():
            if img_file.is_file() and img_file.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                if copier_image(img_file, dst_dir):
                    n_copies += 1

    print(f"  → {n_copies} copies")


# ----------------------------------------------------------------------
# 7. Résumé final
# ----------------------------------------------------------------------
def afficher_resume(root: Path):
    for split in ("train", "test"):
        split_dir = root / split
        if not split_dir.is_dir():
            continue
        print(f"\n📦 {split}/")
        total = 0
        for d in sorted(split_dir.iterdir()):
            if not d.is_dir():
                continue
            n = sum(1 for f in d.iterdir()
                    if f.suffix.lower() in {".jpg", ".jpeg", ".png"})
            total += n
            print(f"   {d.name:25s} : {n:5d} images")
        print(f"   {'TOTAL':25s} : {total:5d} images")


# ----------------------------------------------------------------------
# 8. Point d'entrée
# ----------------------------------------------------------------------
def main():
    OUTPUT_TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_TEST_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- TRAIN : tout PlantDoc (OD TRAIN+TEST, classification train+test) ----------
    traiter_plantdoc_od(PLANTDOC_OD_DIR / "TRAIN", OUTPUT_TRAIN_DIR, "PlantDoc-OD  TRAIN → train/")
    traiter_plantdoc_od(PLANTDOC_OD_DIR / "TEST",  OUTPUT_TRAIN_DIR, "PlantDoc-OD  TEST  → train/")
    traiter_croppedpd  (CROPPEDPD_DIR   / "train", OUTPUT_TRAIN_DIR, "PlantDoc-Cls train → train/")
    traiter_croppedpd  (CROPPEDPD_DIR   / "test",  OUTPUT_TRAIN_DIR, "PlantDoc-Cls test  → train/")

    # ---------- TEST : AD uniquement ----------
    traiter_ad(AD_DIR, OUTPUT_TEST_DIR, "AD → test/")

    print("\n" + "=" * 60)
    print("✅ Terminé")
    print("=" * 60)
    afficher_resume(OUTPUT_DIR)


if __name__ == "__main__":
    main()