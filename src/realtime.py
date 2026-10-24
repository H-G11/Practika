"""Local webcam inference. Camera and GUI resources are released on errors."""

import argparse
from pathlib import Path


def run_camera(model, cv2, *, camera: int, conf: float, imgsz: int, device: str, iou: float = .7) -> None:
    cap = cv2.VideoCapture(camera)
    try:
        if not cap.isOpened():
            raise RuntimeError(f"Could not open camera {camera}; select another --camera index")
        while True:
            ok, frame = cap.read()
            if not ok:
                raise RuntimeError("Camera stopped delivering frames")
            results = model.predict(frame, imgsz=imgsz, conf=conf, iou=iou, device=device, verbose=False)
            cv2.imshow("Waste detection (Q / ESC to exit)", results[0].plot())
            if cv2.waitKey(1) & 0xFF in (ord("q"), ord("Q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.4)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args(argv)
    if not args.weights.is_file() or args.camera < 0 or not 0 < args.conf <= 1 or not 0 < args.iou <= 1 or args.imgsz <= 0:
        parser.error("Require local weights, camera >= 0, conf/iou in (0, 1], imgsz > 0")
    try:
        import cv2
        from ultralytics import YOLO

        run_camera(YOLO(str(args.weights.resolve())), cv2, camera=args.camera,
                   conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device)
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Webcam inference failed: {exc}\n")


if __name__ == "__main__":
    main()
