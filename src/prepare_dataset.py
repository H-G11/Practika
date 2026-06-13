from pathlib import Path
import random
import shutil

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}

def is_img(p):
    return p.suffix in IMG_EXT

def main():
    # Собираем все картинки из data/ (не из подпапок)
    images = [p for p in DATA.iterdir() if p.is_file() and is_img(p)]
    print(f"Найдено картинок: {len(images)}")

    if not images:
        raise RuntimeError("Картинки не найдены в папке data/")

    random.shuffle(images)
    n = len(images)
    n_train = int(0.8 * n)
    n_valid = int(0.1 * n)

    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_valid],
        "test":  images[n_train + n_valid:]
    }

    for split, imgs in splits.items():
        img_dir = DATA / split / "images"
        lbl_dir = DATA / split / "labels"
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in imgs:
            # Копируем картинку
            shutil.copy2(img_path, img_dir / img_path.name)

            # Создаём label (бокс на всё изображение, класс 0 = trash)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            lbl_path.write_text("0 0.5 0.5 1.0 1.0\n", encoding="utf-8")

    print(f"train: {len(splits['train'])} картинок")
    print(f"valid: {len(splits['valid'])} картинок")
    print(f"test:  {len(splits['test'])} картинок")
    print("Готово! ✅")

if __name__ == "__main__":
    main()