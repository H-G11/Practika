"""Shared local-file prediction boundary for CLI and interactive viewer."""

from __future__ import annotations

import math
from pathlib import Path


def predict_image(model, source: str | Path, *, confidence: float = .25, iou: float = .7,
                  imgsz: int = 640, device: str = 'cpu'):
    if not isinstance(source, (str, Path)):
        raise ValueError('Select an existing local image file')
    path = Path(source)
    if not path.is_file() or path.suffix.lower() not in {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}:
        raise ValueError('Select an existing local image file')
    if not all(math.isfinite(v) and 0 < v <= 1 for v in (confidence, iou)) or imgsz < 1:
        raise ValueError('confidence and iou must be finite in (0, 1]; imgsz must be positive')
    results = model.predict(source=str(path.resolve()), conf=confidence, iou=iou,
                            imgsz=imgsz, device=device, save=False, verbose=False)
    if not results:
        raise RuntimeError('Model returned no result object')
    return results[0]


def detection_report(result) -> dict:
    """Export actual model boxes; an empty detection result stays empty."""
    rows = []
    boxes = result.boxes
    if boxes is not None:
        for coordinates, class_id, confidence in zip(boxes.xyxy.tolist(), boxes.cls.tolist(), boxes.conf.tolist(), strict=True):
            index = int(class_id)
            rows.append({'class_id': index, 'label': result.names[index],
                         'confidence': float(confidence), 'xyxy': [float(v) for v in coordinates]})
    return {'detections': rows, 'count': len(rows), 'image_shape': list(result.orig_shape),
            'runtime_ms': {k: float(v) for k, v in result.speed.items() if isinstance(v, (int, float))}}


def export_comparison(result, destination: Path) -> dict:
    """Save actual raw/prediction pair, without replacing input pixels or boxes."""
    import cv2

    destination = Path(destination)
    if destination.exists():
        raise FileExistsError('Comparison output exists; choose a new path')
    destination.parent.mkdir(parents=True, exist_ok=True)
    pair = cv2.hconcat([result.orig_img, result.plot()])
    if not cv2.imwrite(str(destination), pair):
        raise OSError('OpenCV could not write comparison image')
    return detection_report(result)
