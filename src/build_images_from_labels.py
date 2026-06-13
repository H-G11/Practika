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
        print(f"[WARN] Нет папки: {labels_dir}")
        return 0, 0

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
    return copied, missed

def main():
    # train/test точно есть
    copy_images_for_split("train")
    copy_images_for_split("test")

    # valid может быть пустой — создадим из части train, если надо (следующий шаг)
    if not (DATA / "valid" / "labels").exists():
        ensure_dir(DATA / "valid" / "labels")
        ensure_dir(DATA / "valid" / "images")
        print("valid: папка создана (пока пустая)")

if __name__ == "__main__":
    main()