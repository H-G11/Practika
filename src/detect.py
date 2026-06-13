from pathlib import Path
import sys
from ultralytics import YOLO

def main():
    root = Path(__file__).resolve().parents[1]
    weights = root / "results" / "trash_detector-4" / "weights" / "best.pt"

    if not weights.exists():
        raise FileNotFoundError(f"Не найдены веса: {weights}")

    # если путь не передали — возьмём первое изображение из data/test/images
    if len(sys.argv) >= 2:
        source = sys.argv[1]
    else:
        test_dir = root / "data" / "test" / "images"
        imgs = sorted(test_dir.glob("*.*"))
        if not imgs:
            raise FileNotFoundError(f"Нет изображений в {test_dir}")
        source = str(imgs[0])
        print("No image path provided. Using:", source)

    model = YOLO(str(weights))
    model.predict(source=source, conf=0.25, save=True, project=str(root / "results"), name="predict")

if __name__ == "__main__":
    main()