from pathlib import Path
import sys
from ultralytics import YOLO

def main():
    root = Path(__file__).resolve().parents[1]

    weights = root / "results" / "trash_coco" / "weights" / "best.pt"
    if len(sys.argv) >= 2 and sys.argv[1].endswith(".pt"):
        weights = Path(sys.argv[1])

    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    # image path
    if len(sys.argv) >= 3:
        source = sys.argv[2]
    else:
        test_dir = root / "data_coco_yolo" / "test" / "images"
        imgs = sorted(test_dir.glob("*.*"))
        if not imgs:
            raise FileNotFoundError(f"No images in {test_dir}")
        source = str(imgs[0])
        print("Using:", source)

    model = YOLO(str(weights))
    model.predict(source=source, conf=0.25, save=True, project=str(root / "results"), name="predict_coco")

if __name__ == "__main__":
    main()