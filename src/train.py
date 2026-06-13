from pathlib import Path
from ultralytics import YOLO

def main():
    root = Path(__file__).resolve().parents[1]

    model = YOLO("yolov8n.pt")  # быстро и подходит для realtime
    model.train(
        data=str(root / "data_coco.yaml"),
        epochs=30,
        imgsz=640,
        batch=16,
        device=0,
        workers=0,
        project=str(root / "results"),
        name="trash_coco",
    )

if __name__ == "__main__":
    main()