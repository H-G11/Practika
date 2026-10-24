"""Validate prepared YOLO image/label pairs before training."""

import argparse
import math
from pathlib import Path

EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def check_dataset(data_path: str | Path) -> dict:
    root = Path(data_path)
    counts = {}
    seen = set()
    for split in ('train', 'valid', 'test'):
        images_dir, labels_dir = root / split / 'images', root / split / 'labels'
        if not images_dir.is_dir() or not labels_dir.is_dir():
            raise ValueError(f'Missing images or labels directory for {split}')
        images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in EXTENSIONS]
        stems = [p.stem for p in images]
        labels = {p.stem for p in labels_dir.glob('*.txt')}
        if not images or len(set(stems)) != len(stems) or set(stems) != labels:
            raise ValueError(f'{split}: require nonempty images and exactly one matching label per image')
        if seen & set(stems):
            raise ValueError('Image IDs overlap between splits')
        seen.update(stems)
        boxes = 0
        for stem in stems:
            for number, line in enumerate((labels_dir / f'{stem}.txt').read_text().splitlines(), 1):
                if not line.strip():
                    continue
                fields = line.split()
                if len(fields) != 5 or not fields[0].isdigit():
                    raise ValueError(f'{split}/{stem}:{number}: expected class xc yc w h')
                xc, yc, width, height = map(float, fields[1:])
                if not all(math.isfinite(v) for v in (xc, yc, width, height)) or not 0 < width <= 1 or not 0 < height <= 1:
                    raise ValueError(f'{split}/{stem}:{number}: invalid box dimensions')
                if not (width / 2 - 1e-7 <= xc <= 1 - width / 2 + 1e-7 and height / 2 - 1e-7 <= yc <= 1 - height / 2 + 1e-7):
                    raise ValueError(f'{split}/{stem}:{number}: box outside image')
                boxes += 1
        counts[split] = {'images': len(images), 'boxes': boxes}
    return counts


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path)
    args = parser.parse_args(argv)
    try:
        print(check_dataset(args.dataset))
    except (ValueError, OSError) as exc:
        parser.exit(2, f'Dataset validation failed: {exc}\n')


if __name__ == '__main__':
    main()
