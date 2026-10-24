"""Generate an authentic image comparison or annotated video with provenance."""

import argparse
import hashlib
import json
import math
import time
from pathlib import Path

try:
    from .prediction import detection_report, export_comparison, predict_image
except ImportError:
    from prediction import detection_report, export_comparison, predict_image


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='Fresh .jpg comparison or .mp4 video')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--conf', type=float, default=.25)
    parser.add_argument('--iou', type=float, default=.7)
    parser.add_argument('--imgsz', type=int, default=640)
    parser.add_argument('--max-frames', type=int, default=120)
    args = parser.parse_args(argv)
    sidecar = args.output.with_suffix('.json')
    if not args.source.is_file() or not args.weights.is_file() or args.output.exists() or sidecar.exists():
        parser.error('Require local source/weights and fresh output + sidecar paths')
    if args.output.suffix.lower() not in {'.jpg', '.mp4'} or args.max_frames < 1:
        parser.error('Output must be .jpg or .mp4; max-frames must be positive')
    if args.imgsz < 1 or not all(math.isfinite(v) and 0 < v <= 1 for v in (args.conf, args.iou)):
        parser.error('Require imgsz > 0 and finite confidence/IoU in (0, 1]')
    try:
        import cv2
        from ultralytics import YOLO

        model = YOLO(str(args.weights.resolve()))
        started = time.perf_counter()
        if args.output.suffix.lower() == '.jpg':
            result = predict_image(model, args.source, confidence=args.conf, iou=args.iou,
                                   imgsz=args.imgsz, device=args.device)
            reports = [export_comparison(result, args.output)]
        else:
            reports = []
            cap, writer = cv2.VideoCapture(str(args.source)), None
            try:
                if not cap.isOpened():
                    raise RuntimeError('Could not open source video')
                args.output.parent.mkdir(parents=True, exist_ok=True)
                fps = cap.get(cv2.CAP_PROP_FPS)
                if not 0 < fps <= 240:
                    raise ValueError('Source FPS is missing/invalid; transcode explicitly first')
                while len(reports) < args.max_frames:
                    ok, frame = cap.read()
                    if not ok:
                        break
                    result = model.predict(frame, conf=args.conf, iou=args.iou, imgsz=args.imgsz,
                                           device=args.device, verbose=False)[0]
                    annotated = result.plot()
                    if writer is None:
                        h, w = annotated.shape[:2]
                        writer = cv2.VideoWriter(str(args.output), cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                        if not writer.isOpened():
                            raise RuntimeError('Could not create MP4 writer')
                    writer.write(annotated)
                    reports.append(detection_report(result))
                if not reports:
                    raise RuntimeError('Video contains no readable frames')
            finally:
                cap.release()
                if writer is not None:
                    writer.release()
        hashes = {}
        for key, path in (('source', args.source), ('weights', args.weights)):
            with path.open('rb') as stream:
                hashes[key] = hashlib.file_digest(stream, 'sha256').hexdigest()
        report = {'kind': 'actual_yolo_inference', 'hashes': hashes, 'configuration':
                  {'device': args.device, 'confidence': args.conf, 'iou': args.iou, 'imgsz': args.imgsz},
                  'elapsed_seconds_including_export': time.perf_counter() - started, 'frames': reports}
        sidecar.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
        print(f'Actual output: {args.output}; provenance: {sidecar}')
    except (ImportError, RuntimeError, ValueError, OSError) as exc:
        parser.exit(1, f'Demo generation failed: {exc}\n')


if __name__ == '__main__':
    main()
