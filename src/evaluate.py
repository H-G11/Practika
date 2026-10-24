"""Evaluate an actual held-out dataset and time repeated local image inference."""

import argparse
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def per_class_metrics(box, names) -> list[dict]:
    """Only report classes evaluated by the validator; never fill absent classes with mean AP."""
    return [{'class_id': int(class_id), 'label': names[int(class_id)],
             'precision': float(box.p[i]), 'recall': float(box.r[i]),
             'mAP50': float(box.ap50[i]), 'mAP50_95': float(box.ap[i])}
            for i, class_id in enumerate(box.ap_class_index)]


def confusion_counts(matrix: list[list[float]]) -> dict:
    """Ultralytics detection convention: rows=prediction, columns=target; background last.

    Include off-diagonal class confusions in both FP and FN. These counts use the
    validator's confusion threshold/matching policy, not the PR curve's best F1 point.
    """
    size = len(matrix)
    if size < 2 or any(len(row) != size for row in matrix):
        raise ValueError('Expected a square detection confusion matrix with background')
    classes = size - 1
    tp = sum(matrix[i][i] for i in range(classes))
    fp = sum(sum(matrix[i]) - matrix[i][i] for i in range(classes))
    fn = sum(sum(row[i] for row in matrix) - matrix[i][i] for i in range(classes))
    return {'true_positives': float(tp), 'false_positives': float(fp), 'false_negatives': float(fn)}


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--latency-source', type=Path, required=True, help='Single local image for repeated end-to-end timing')
    parser.add_argument('--split', choices=['val', 'test'], default='test')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--repeats', type=int, default=20)
    parser.add_argument('--warmup', type=int, default=3)
    parser.add_argument('--conf', type=float, default=0.25, help='Threshold for timed prediction only; validation uses 0.001')
    parser.add_argument('--output', type=Path, default=ROOT / 'results')
    args = parser.parse_args(argv)
    if not all(p.is_file() for p in (args.weights, args.data, args.latency_source)):
        parser.error('Weights, dataset YAML and latency source must be existing local files')
    if args.repeats < 2 or args.warmup < 1 or args.imgsz < 1 or not 0 < args.conf <= 1:
        parser.error('Require repeats >= 2, warmup >= 1, imgsz > 0, 0 < conf <= 1')
    try:
        import torch
        import ultralytics
        from ultralytics import YOLO

        model = YOLO(str(args.weights.resolve()))
        metrics = model.val(data=str(args.data.resolve()), split=args.split, device=args.device,
                            imgsz=args.imgsz, batch=1, workers=0, plots=True, conf=.001,
                            project=str(args.output.resolve()), name='evaluate')
        confusion = getattr(metrics, 'confusion_matrix', None)
        if confusion is None:
            raise RuntimeError('Installed Ultralytics does not expose confusion_matrix; inspect saved validation plots')
        matrix = confusion.matrix.tolist()
        durations = []
        for number in range(args.warmup + args.repeats):
            if torch.cuda.is_available() and args.device != 'cpu':
                torch.cuda.synchronize()
            started = time.perf_counter()
            model.predict(source=str(args.latency_source.resolve()), device=args.device, imgsz=args.imgsz,
                          conf=args.conf, save=False, verbose=False)
            if torch.cuda.is_available() and args.device != 'cpu':
                torch.cuda.synchronize()
            if number >= args.warmup:
                durations.append((time.perf_counter() - started) * 1000)
        with args.weights.open('rb') as stream:
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        report = {'weights_sha256': digest, 'split': args.split, 'imgsz': args.imgsz,
                  'mAP50': float(metrics.box.map50), 'mAP50_95': float(metrics.box.map),
                  'precision': float(metrics.box.mp), 'recall': float(metrics.box.mr),
                  'per_class': per_class_metrics(metrics.box, getattr(metrics, 'names', model.names)),
                  'confusion_matrix': matrix, 'confusion_counts': confusion_counts(matrix),
                  'confusion_policy': 'Ultralytics validator defaults; matrix rows=prediction, columns=target, background last',
                  'latency': {'scope': 'single-image model.predict including file IO, preprocess and postprocess; no plotting/saving',
                              'conf': args.conf, 'warmup': args.warmup, 'repeats': args.repeats,
                              'median_ms': statistics.median(durations),
                              'sequential_images_per_second': 1000 / statistics.mean(durations),
                              'p95_ms': sorted(durations)[max(0, int(.95 * len(durations) + .999999) - 1)],
                              'samples_ms': durations},
                  'hardware': {'platform': platform.platform(), 'processor': platform.processor(), 'logical_cpus': os.cpu_count(),
                               'device': args.device, 'cuda_available': torch.cuda.is_available(),
                               'gpus': [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]},
                  'versions': {'python': platform.python_version(), 'torch': torch.__version__, 'ultralytics': ultralytics.__version__}}
        destination = Path(metrics.save_dir) / 'evaluation.json'
        destination.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'Evaluation report: {destination}')
    except (ImportError, OSError, RuntimeError, ValueError, AttributeError) as exc:
        parser.exit(1, f'Evaluation failed: {exc}\n')


if __name__ == '__main__':
    main()
