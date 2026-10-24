# Waste detection with YOLOv8

Tools for COCO dataset preparation, YOLOv8 training, image/webcam inference and evaluation.

## Demo

<a href="docs/demo/validation-predictions.jpg"><img src="docs/demo/validation-predictions.jpg" width="420" alt="Archived YOLOv8 predictions with trash labels and nearly full-frame boxes"></a>

*Archived model output. The full-frame boxes reflect an error in the original labels; reliable localization requires retraining on corrected annotations. [Provenance and limitations](docs/demo/README.md).*

## Installation

Python 3.11/3.12. Run from the repository root.

```bash
# Linux/macOS
python3.11 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
```

```powershell
# Windows PowerShell
py -3.11 -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Use the virtual-environment Python for the commands below. [Platform and dependency details](docs/usage.md).

## Inference

```bash
python src/detect.py path/to/image.jpg --weights path/to/best.pt --device cpu
python src/realtime.py --weights path/to/best.pt --camera 0 --device cpu
```

Provide a local image and trusted checkpoint. Predictions are saved to `results/predict*`; Q/ESC exits webcam mode. Weights are not downloaded automatically.

For an image-upload viewer with confidence and NMS controls:

```bash
python -m pip install -r requirements-demo.txt
python src/viewer.py --weights path/to/best.pt --device cpu
```

Open `http://127.0.0.1:7861`. [Comparison image and video generation](docs/demo/GENERATION.md).

## Training and evaluation

Obtain authorized COCO images and annotations; the original project references [TACO on Kaggle](https://www.kaggle.com/datasets/kneroma/tacotrashdataset). Adjust paths to your dataset:

```bash
python src/prepare_dataset.py --annotations data/data/annotations.json --images-dir data/data --output data_coco_yolo --seed 42
python src/train.py --data data_coco_yolo/data.yaml --weights yolov8n.pt --device cpu
python src/evaluate.py --data data_coco_yolo/data.yaml --weights results/waste_yolov8/weights/best.pt --latency-source path/to/test-image.jpg --device cpu
```

Preparation records source hashes and image-level 80/10/10 splits. New model-quality and latency results have not been measured. [Evaluation methodology](docs/engineering.md).

## Architecture

![COCO preparation, training, evaluation and inference pipeline](docs/assets/pipeline.svg)

## Tests

```bash
python -m unittest discover -s tests -v
```

[Verified results and remaining checks](docs/verification.md) · [Usage](docs/usage.md) · [Historical results and artifact notes](docs/historical-results.md).
