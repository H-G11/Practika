from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

def ensure_dir(p: Path):
    p.mkdir(parents=True, exist_ok=True)

def copy_images_for_split(split: str):
    labels_dir = DATA / split / "labels"
    images_dir = DATA / split / "images"
    ensure_dir(images_dir)

    if not labels_dir.exists():
        raise FileNotFoundError(labels_dir)

    label_files = sorted(labels_dir.glob("*.txt"))
    copied = 0
    missed = 0

    for lf in label_files:
        # ищем картинку с таким же stem в корне data/
        stem = lf.stem
        candidates = [
            DATA / f"{stem}.jpg",
            DATA / f"{stem}.JPG",
            DATA / f"{stem}.png",
            DATA / f"{stem}.jpeg",
            DATA / f"{stem}.JPEG",
        ]
        img = next((c for c in candidates if c.exists()), None)
        if img is None:
            missed += 1
            continue

        shutil.copy2(img, images_dir / img.name)
        copied += 1

    print(f"{split}: labels={len(label_files)} copied_images={copied} missed={missed}")
    if missed:
        raise FileNotFoundError(f"{split}: {missed} labels have no matching source image")
    return copied, missed

def main():
    """Legacy label-only layout helper; use coco_to_yolo.py for COCO datasets."""
    # Existing labels are required; this utility never creates annotations.
    copy_images_for_split("train")
    copy_images_for_split("test")

    # An absent validation set must be prepared explicitly; do not imply it is ready.
    if not (DATA / "valid" / "labels").exists():
        raise FileNotFoundError("Missing validation labels; use the documented COCO converter")

if __name__ == "__main__":
    main()
