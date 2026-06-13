from pathlib import Path
import json
import random
import shutil
from collections import defaultdict

random.seed(42)

ROOT = Path(__file__).resolve().parents[1]
ANN = ROOT / "data" / "data" / "annotations.json"

# ВАЖНО: где лежат изображения из COCO?
# В твоём датасете часто они лежат в data/data/ или data/ (надо проверить).
# Сначала попробуем найти автоматически:
CANDIDATE_IMG_DIRS = [
    ROOT / "data" / "data",
    ROOT / "data",
]

OUT = ROOT / "data_coco_yolo"

def find_image_dir(file_name: str) -> Path:
    for d in CANDIDATE_IMG_DIRS:
        if (d / file_name).exists():
            return d
    return None

def yolo_bbox(bbox, w, h):
    # COCO bbox: [x_min, y_min, width, height] in pixels
    x, y, bw, bh = bbox
    xc = (x + bw / 2) / w
    yc = (y + bh / 2) / h
    return xc, yc, bw / w, bh / h

def main():
    if not ANN.exists():
        raise FileNotFoundError(ANN)

    data = json.loads(ANN.read_text(encoding="utf-8"))

    images = data["images"]
    annotations = data["annotations"]
    categories = sorted(data["categories"], key=lambda c: c["id"])

    # category_id may be non-contiguous; map to 0..nc-1
    cat_id_to_idx = {c["id"]: i for i, c in enumerate(categories)}
    names = [c["name"] for c in categories]

    # group annotations by image_id
    ann_by_image = defaultdict(list)
    for a in annotations:
        ann_by_image[a["image_id"]].append(a)

    # filter images that exist on disk and have annotations
    usable = []
    for im in images:
        file_name = im["file_name"]
        img_dir = find_image_dir(file_name)
        if img_dir is None:
            continue
        if im["id"] not in ann_by_image:
            continue
        usable.append(im)

    print(f"COCO images in json: {len(images)}")
    print(f"Usable images found on disk with annotations: {len(usable)}")
    print(f"Classes (nc): {len(names)}")

    # split 80/10/10
    random.shuffle(usable)
    n = len(usable)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    splits = {
        "train": usable[:n_train],
        "valid": usable[n_train:n_train + n_val],
        "test": usable[n_train + n_val:],
    }

    # create folders
    for s in splits:
        (OUT / s / "images").mkdir(parents=True, exist_ok=True)
        (OUT / s / "labels").mkdir(parents=True, exist_ok=True)

    # write files
    for split, items in splits.items():
        for im in items:
            file_name = im["file_name"]
            img_dir = find_image_dir(file_name)
            src_img = img_dir / file_name

            dst_img = OUT / split / "images" / Path(file_name).name
            shutil.copy2(src_img, dst_img)

            w, h = im["width"], im["height"]
            lines = []
            for a in ann_by_image[im["id"]]:
                if a.get("iscrowd", 0) == 1:
                    continue
                bbox = a["bbox"]
                xc, yc, bw, bh = yolo_bbox(bbox, w, h)
                cls = cat_id_to_idx[a["category_id"]]
                # clip just in case
                xc = min(max(xc, 0.0), 1.0)
                yc = min(max(yc, 0.0), 1.0)
                bw = min(max(bw, 0.0), 1.0)
                bh = min(max(bh, 0.0), 1.0)
                if bw <= 0 or bh <= 0:
                    continue
                lines.append(f"{cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

            (OUT / split / "labels" / f"{dst_img.stem}.txt").write_text(
                "\n".join(lines) + ("\n" if lines else ""),
                encoding="utf-8",
            )

    # write data.yaml for this converted dataset
    yaml_path = ROOT / "data_coco.yaml"
    yaml_path.write_text(
        "path: ./data_coco_yolo\n"
        "train: train/images\n"
        "val: valid/images\n"
        "test: test/images\n\n"
        f"nc: {len(names)}\n"
        "names:\n" + "\n".join([f"  - {n}" for n in names]) + "\n",
        encoding="utf-8",
    )

    print("Done.")
    print("Created:", OUT)
    print("Wrote:", yaml_path)

if __name__ == "__main__":
    main()