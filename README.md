# Waste detection with YOLOv8

Python tools for preparing annotated waste images, training YOLOv8, running image or webcam inference, and evaluating a held-out split. This is an experimental project; a validated waste detector is not supplied.

The original `Practika` remote is preserved. Proposed repository name: `waste-detection-yolov8`.

## Results and demonstration status

The repository contains a historical [prediction montage](results/trash_detector-4/val_batch0_pred.jpg), [confusion matrix](results/trash_detector-4/confusion_matrix.png) and training log. They are actual retained training artifacts, **not evidence of correct object localization**: the old preparation script assigned every image a single full-frame box. The saved predictions show that failure mode. Do not use the historical high mAP as a portfolio benchmark.

The preparation code now requires real COCO annotations. Retraining, independent evaluation and a new authentic inference demo are **blocked** here: no source dataset is present, ML packages are absent, and this execution environment denies network sockets. Existing weights have not been executed during this audit. See [verification](docs/verification.md) and [historical results](docs/historical-results.md).

## Requirements and installation

Use Python 3.11 or 3.12. Webcam mode also needs a camera, desktop display and OpenCV GUI support. CPU is the default; CUDA is optional and requires a compatible PyTorch installation. Python 3.14 ML installation has not been validated.

Linux/macOS, from the repository root:

```bash
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m pip check
```

Windows PowerShell, from the repository root:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m pip check
```

No environment activation is required. Commands below use `python`; substitute the appropriate virtual-environment executable above. On CPU-only Linux you can first install `torch torchvision` from the [official CPU wheel index](https://download.pytorch.org/whl/cpu), then install requirements. The direct Ultralytics version is retained for checkpoint compatibility; the complete dependency lock and clean ML installation still require validation on a network-enabled machine.

## Dataset preparation

The original README referenced the [Kaggle TACO mirror](https://www.kaggle.com/datasets/kneroma/tacotrashdataset). The upstream project is [TACO](https://github.com/pedropro/TACO). The exact archive/version used for the historical run is unknown. Obtain an authorized copy and inspect its layout; no download command or archive contents have been verified here.

Use the directory against which COCO `file_name` values are relative. For an archive containing `data/data/annotations.json` and images under `data/data/`:

```bash
python src/prepare_dataset.py --annotations data/data/annotations.json --images-dir data/data --output data_coco_yolo --seed 42
python src/utils.py data_coco_yolo
```

`src/coco_to_yolo.py` is the same converter entry point. It validates references and files, clips boxes by intersecting their corners with the image, maps category IDs to contiguous classes and splits images 80/10/10 using a local RNG. Input order does not change membership. All images are checked for duplicate content before splitting; output names use image IDs to avoid basename collisions. This is an image-level split, not a scene-grouped or stratified benchmark. Review correlated scenes and class coverage before reporting results.

Empty labels are retained for images with no non-crowd annotations; crowd annotations are counted and skipped because this YOLO format has no crowd flag. Review crowd-only images before training. At least ten images are required. The converter refuses a nonempty output directory and missing images instead of silently reducing the dataset. A failed write can leave a partial output; inspect it and use a fresh directory for the next attempt.

Generated `data_coco_yolo/data.yaml` contains the actual class list and an absolute dataset path. `manifest.json` records source annotation SHA256, image hashes, split membership and box counts. Regenerate the config after relocating the dataset. Root `data.yaml` and `data_coco.yaml` are historical examples; training uses the explicit `--data` argument.

## Training

Obtain trusted local YOLOv8 starting weights, such as `yolov8n.pt`, through the official Ultralytics distribution. No script automatically downloads or substitutes weights.

```bash
python src/train.py --data data_coco_yolo/data.yaml --weights yolov8n.pt --epochs 30 --imgsz 640 --batch 16 --device cpu --seed 42
```

Use `--device 0` only with working CUDA. Reduce `--batch` for limited memory. Workers default to zero for Windows compatibility. Runs go to `results/waste_yolov8*`; use the actual run directory printed by Ultralytics. A deterministic seed helps repeatability but does not guarantee identical results across hardware/software versions.

## Image and webcam inference

After training, these commands assume the first run produced the named checkpoint. Relative input paths resolve from the current directory; run the examples from the repository root.

```bash
python src/detect.py path/to/image.jpg --weights results/waste_yolov8/weights/best.pt --device cpu --conf 0.25
python src/realtime.py --weights results/waste_yolov8/weights/best.pt --camera 0 --device cpu
```

Image predictions and confidence-bearing YOLO text outputs go to a new `results/predict*` directory. Webcam mode exits with Q or ESC and releases the camera on failures. On Windows, `run_realtime.bat results\waste_yolov8\weights\best.pt` uses the already-installed `.venv`; it performs no installation or model download.

Only load checkpoints from a trusted source. `.pt` files can involve Python deserialization. Historical `trash_detector-4` weights are not interchangeable with a newly trained multi-class TACO model. The previously referenced release `v0.2/best.pt` was not available for inspection in this environment and is no longer downloaded implicitly.

## Evaluation and a reproducible demo

Evaluate the untouched test split after selecting training settings using validation data:

```bash
python src/evaluate.py --data data_coco_yolo/data.yaml --weights results/waste_yolov8/weights/best.pt --latency-source path/to/test-image.jpg --split test --device cpu --warmup 3 --repeats 20
```

The script requests mAP50, mAP50-95, precision, recall and confusion plots from Ultralytics. It writes `evaluation.json` beside validation outputs with confusion-derived TP/FP/FN, weight hash, software/hardware details and repeated single-image end-to-end latency. FP/FN include class confusions; their threshold/matching policy differs from the PR curve's best-F1 operating point. Timing includes file loading, preprocessing and postprocessing, excludes plotting/saving, and synchronizes CUDA. The pinned runtime API still needs an actual dependency-enabled run; missing confusion-matrix support is an explicit error, never invented counts.

For a reviewable demo, run `detect.py` on an authorized held-out image, retain the real image/prediction, exact command, checkpoint hash and evaluation report, and describe actual errors. No new model-output asset has been generated in this audit.

## Verification

```bash
python -m unittest discover -s tests -v
python -m pip install -r requirements-dev.txt
python -m ruff check .
python scripts/check_format.py
python -m bandit -r src -lll
python -m pip_audit --progress-spinner off
```

Unit tests run without ML dependencies. Model calls are mocked in CLI tests; those tests do not establish model quality. CI includes Windows/Linux tests and a separate CPU dependency/import/security job. Local results and blocked checks are in [verification](docs/verification.md).

## Licensing and artifacts

Neither repository code nor historical weights have an explicit project license in the source tree. Owner selection/confirmation is required; this audit does not assign one. Ultralytics declares [AGPL-3.0 licensing](https://pypi.org/project/ultralytics/8.4.66/). Dataset, image, annotation and model redistribution permissions must be reviewed separately; the upstream toolkit license is not blanket permission for every image in a mirror.

Datasets and newly generated runs/weights are ignored. Existing historical artifacts remain to preserve provenance; no history was rewritten. See [artifact notes](docs/historical-results.md) before publishing or moving them to a release.
