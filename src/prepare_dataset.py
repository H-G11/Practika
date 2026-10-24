"""Compatibility entry point: prepare real COCO annotations, never invent boxes."""

try:
    from .coco_to_yolo import main
except ImportError:  # python src/prepare_dataset.py
    from coco_to_yolo import main


if __name__ == "__main__":
    main()
