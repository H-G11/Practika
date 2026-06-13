from pathlib import Path
import cv2
from ultralytics import YOLO

def main():
    print("Realtime started. Click the video window and press Q or ESC to quit.")
    root = Path(__file__).resolve().parents[1]
    weights = root / "results" / "trash_coco" / "weights" / "best.pt"
    if not weights.exists():
        raise FileNotFoundError(f"Weights not found: {weights}")

    model = YOLO(str(weights))

    cap = cv2.VideoCapture(0)  # 0 = встроенная/первая камера
    if not cap.isOpened():
        raise RuntimeError("Could not open camera. Try changing VideoCapture(0) to (1)")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = model.predict(frame, imgsz=640, conf=0.4, verbose=False)
        annotated = results[0].plot()

        cv2.imshow("Trash realtime (q/ESC to quit)", annotated)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):  # 27 = ESC
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()