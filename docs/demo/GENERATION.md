# Produce genuine new demo artifacts

Current primary visual is an unchanged archived prediction output with a documented full-frame-label defect. **New localized image/video, raw-vs-prediction and success/failure examples are BLOCKED:** source images/COCO annotations and a checkpoint trained on corrected boxes are missing; ML packages cannot be downloaded inside the socket-restricted execution environment. Recovery of the referenced GitHub v0.2 weights was attempted with curl and failed with DNS exit 6. It was not silently replaced with a different model.

On a machine with authorized data and a verified environment, from the repository root:

```bash
python src/prepare_dataset.py --annotations data/data/annotations.json --images-dir data/data --output data_coco_yolo --seed 42
python src/train.py --data data_coco_yolo/data.yaml --weights yolov8n.pt --device cpu
python src/evaluate.py --data data_coco_yolo/data.yaml --weights results/waste_yolov8/weights/best.pt --latency-source path/to/held-out.jpg --split test --device cpu
python src/generate_demo.py --source path/to/held-out.jpg --weights results/waste_yolov8/weights/best.pt --output docs/demo/comparison.jpg --device cpu
python src/generate_demo.py --source path/to/authorized-video.mp4 --weights results/waste_yolov8/weights/best.pt --output docs/demo/inference.mp4 --max-frames 120 --device cpu
```

Substitute real local paths and the actual run directory. `generate_demo.py` writes a fresh raw-left/prediction-right image or a genuine per-frame annotated MP4, plus a JSON sidecar with source/checkpoint SHA256, thresholds, image size, device and actual detection records. Output and sidecar must not already exist. On export failure a partial output may remain; inspect it before retrying. MP4 preserves the source playback FPS, which must not be reported as inference throughput; generation time is recorded separately. Audio is not copied.

Optional GIF conversion, using ffmpeg on the actual generated MP4:

```bash
ffmpeg -i docs/demo/inference.mp4 -filter_complex "fps=8,scale=640:-1:flags=lanczos,split[a][b];[a]palettegen[p];[b][p]paletteuse" -loop 0 docs/demo/inference.gif
```

Do not commit generated patient/private images, secrets or checkpoints. Review image/data/model rights. Only then replace the archival README asset with real new outputs and link their sidecars. No files named `comparison.jpg`, `inference.mp4` or `inference.gif` are claimed to exist yet.
