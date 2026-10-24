"""Run inference on a local image using explicitly supplied, trusted weights."""

import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Local image; relative paths use the current directory")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "results")
    args = parser.parse_args(argv)
    if not args.weights.is_file() or not args.source.is_file():
        parser.error("Both weights and source must be existing local files")
    if not 0 < args.conf <= 1 or args.imgsz <= 0:
        parser.error("Require 0 < conf <= 1 and imgsz > 0")
    try:
        from ultralytics import YOLO

        model = YOLO(str(args.weights.resolve()))
        model.predict(source=str(args.source.resolve()), conf=args.conf, imgsz=args.imgsz,
                      device=args.device, save=True, save_txt=True, save_conf=True,
                      project=str(args.output.resolve()), name="predict")
    except (ImportError, OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"Inference failed: {exc}\n")


if __name__ == "__main__":
    main()
