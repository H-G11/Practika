from pathlib import Path
from ultralytics import YOLO

def main():
    root = Path(__file__).resolve().parents[1]

    model = YOLO("yolov8s.pt")
    model.train(
        data=str(root / "data.yaml"),
        epochs=20,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,           # важно для Windows, чтобы не падало
        project=str(root / "results"),
        name="trash_detector",
    )

if __name__ == "__main__":
    main()