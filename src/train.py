"""Train YOLOv8 against an explicitly selected dataset and local starting weights."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--weights", type=Path, required=True, help="Trusted local pretrained .pt file")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu", help="cpu, or a CUDA index such as 0")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if not args.data.is_file() or not args.weights.is_file():
        parser.error("Both data and initial weights must be existing local files")
    if min(args.epochs, args.imgsz, args.batch) <= 0 or args.workers < 0:
        parser.error("epochs, imgsz and batch must be positive; workers must be nonnegative")
    try:
        from ultralytics import YOLO

        model = YOLO(str(args.weights.resolve()))
        model.train(data=str(args.data.resolve()), epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=args.device, workers=args.workers,
                    seed=args.seed, deterministic=True, project=str(args.output.resolve()), name="waste_yolov8")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Training failed: {exc}\n")


if __name__ == "__main__":
    main()
