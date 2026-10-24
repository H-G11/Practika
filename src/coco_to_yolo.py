"""Convert real COCO boxes to YOLO labels with a reproducible image-level split."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import shutil
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def yolo_bbox(bbox: list[float], w: float, h: float) -> tuple[float, ...]:
    """Intersect a COCO box with the image, then normalize its corners."""
    if len(bbox) != 4 or not all(math.isfinite(float(v)) for v in [w, h, *bbox]):
        raise ValueError("Boxes and image dimensions must be finite numbers")
    x, y, bw, bh = map(float, bbox)
    if w <= 0 or h <= 0 or bw <= 0 or bh <= 0:
        raise ValueError("Image and box dimensions must be positive")
    left, top = max(0, x), max(0, y)
    right, bottom = min(w, x + bw), min(h, y + bh)
    if right <= left or bottom <= top:
        raise ValueError("Box does not intersect the image")
    return (left + right) / (2 * w), (top + bottom) / (2 * h), (right - left) / w, (bottom - top) / h


def image_path(directory: Path, name: str) -> Path:
    relative = Path(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("COCO image paths must stay inside images-dir")
    path = (directory / relative).resolve()
    if not path.is_relative_to(directory.resolve()) or not path.is_file():
        raise ValueError(f"Missing or unsafe image: {name}")
    if path.suffix.lower() not in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
        raise ValueError(f"Unsupported image extension: {name}")
    return path


def convert(annotations: Path, images_dir: Path, output: Path, seed: int = 42) -> dict:
    annotations, images_dir, output = map(Path, (annotations, images_dir, output))
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError("Output must be absent or empty; existing datasets are never overwritten")
    if output.resolve().is_relative_to(images_dir.resolve()) or images_dir.resolve().is_relative_to(output.resolve()):
        raise ValueError("Input images and output directories must not contain each other")
    raw = annotations.read_bytes()
    data = json.loads(raw)
    categories = sorted(data["categories"], key=lambda item: item["id"])
    images = sorted(data["images"], key=lambda item: item["id"])
    if not categories or len({c["id"] for c in categories}) != len(categories):
        raise ValueError("Categories must be nonempty with unique IDs")
    if any(not isinstance(c["name"], str) or not c["name"].strip() for c in categories):
        raise ValueError("Every category needs a nonempty name")
    category_map = {c["id"]: i for i, c in enumerate(categories)}
    if len(images) < 10 or len({im["id"] for im in images}) != len(images):
        raise ValueError("Need at least 10 images with unique IDs for an 80/10/10 split")
    if any(type(im["id"]) is not int or im["id"] < 0 for im in images):
        raise ValueError("Image IDs must be nonnegative integers")
    by_id = {im["id"]: im for im in images}
    labels = defaultdict(list)
    crowd_skipped = 0
    for annotation in data["annotations"]:
        if annotation["image_id"] not in by_id or annotation["category_id"] not in category_map:
            raise ValueError("Annotation references an unknown image or category")
        if annotation.get("iscrowd", 0):
            crowd_skipped += 1
            continue
        im = by_id[annotation["image_id"]]
        box = yolo_bbox(annotation["bbox"], im["width"], im["height"])
        labels[im["id"]].append(f"{category_map[annotation['category_id']]} " + " ".join(f"{v:.8f}" for v in box))
    sources = {}
    digests = set()
    # Validate all files first. Duplicate images can leak across train/val/test.
    for im in images:
        if not all(math.isfinite(float(im[k])) and im[k] > 0 for k in ("width", "height")):
            raise ValueError("Image dimensions must be positive and finite")
        path = image_path(images_dir, im["file_name"])
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest in digests:
            raise ValueError("Duplicate image content; deduplicate the source before splitting")
        digests.add(digest)
        sources[im["id"]] = (path, digest)
    random.Random(seed).shuffle(images)
    n_train, n_val = int(len(images) * 0.8), int(len(images) * 0.1)
    splits = {"train": images[:n_train], "valid": images[n_train:n_train + n_val], "test": images[n_train + n_val:]}
    manifest = {"seed": seed, "annotations_sha256": hashlib.sha256(raw).hexdigest(),
                "categories": categories, "crowd_annotations_skipped": crowd_skipped, "splits": {}}
    output.mkdir(parents=True, exist_ok=True)
    for split, items in splits.items():
        for kind in ("images", "labels"):
            (output / split / kind).mkdir(parents=True)
        manifest["splits"][split] = []
        for im in items:
            source, digest = sources[im["id"]]
            name = f"{im['id']}{source.suffix.lower()}"
            shutil.copy2(source, output / split / "images" / name)
            rows = sorted(labels[im["id"]])
            (output / split / "labels" / f"{im['id']}.txt").write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
            manifest["splits"][split].append({"id": im["id"], "source": im["file_name"], "sha256": digest, "boxes": len(rows)})
    # JSON scalars/arrays are valid YAML and safely quote names and Windows paths.
    config = f"path: {json.dumps(str(output.resolve()))}\ntrain: train/images\nval: valid/images\ntest: test/images\n"
    config += f"nc: {len(categories)}\nnames: {json.dumps([c['name'] for c in categories], ensure_ascii=False)}\n"
    (output / "data.yaml").write_text(config, encoding="utf-8")
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return manifest


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", type=Path, required=True)
    parser.add_argument("--images-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=ROOT / "data_coco_yolo")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)
    try:
        manifest = convert(args.annotations, args.images_dir, args.output, args.seed)
    except (OSError, ValueError, KeyError, TypeError) as exc:
        parser.exit(2, f"Dataset preparation failed: {exc}\n")
    print(json.dumps({k: len(v) for k, v in manifest["splits"].items()}))
    print(f"Dataset config: {args.output / 'data.yaml'}")


if __name__ == "__main__":
    main()
