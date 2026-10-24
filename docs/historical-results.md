# Historical run: provenance and invalid benchmark

`results/trash_detector-4/args.yaml` records YOLOv8s, 20 epochs, image size 640, batch 16, seed 0, device 0, workers 0, and the single-class root `data.yaml`. It contains an old local Windows path; that path is historical metadata, not a setup requirement. Exact GPU model, dataset snapshot, split membership, installed dependency versions and number of independent source scenes were not recorded in a reproducible manifest.

`results.csv`, epoch 20, records precision 0.99522, recall 1.0, mAP50 0.995 and mAP50-95 0.995. **These are historical log entries, not accepted waste-localization measurements.** The original `prepare_dataset.py` generated `0 0.5 0.5 1.0 1.0` for every image; the retained prediction montage shows full-frame detections. Its scores therefore cannot validate object boxes or transfer to TACO's 60-class labeling. The original source dataset is absent, so its exact composition cannot be reconstructed here.

The historical confusion-matrix PNG is preserved, but raw counts, FP/FN and inference-latency measurements cannot be reconstructed reliably from the CSV or assigned to a new model. New metrics require actual COCO annotations, retraining and a held-out evaluation. Training wall-clock time is not inference latency.

## Checkpoints

Both files were already tracked before this audit; no new model binaries were added, executed or deserialized. SHA256 values:

| File | Bytes | SHA256 |
| --- | ---: | --- |
| `weights/best.pt` | 22500963 | `871c14964a654fc42ee785fb0524a74200d684442d7f317317bd5bfed02a5ac9` |
| `weights/last.pt` | 22500963 | `a521c57aa9a110d6669a196c4041f9fd7f715c7a68c49b46a70c3dd94ef30dcf` |

Paths above are relative to `results/trash_detector-4/`. These ZIP-format checkpoints occupy about 43 MiB combined and remain historical evidence, not recommended default weights. Moving them to an appropriately licensed release and removing redundant future training artifacts should be reviewed by the owner; existing history and remote settings have not been changed. The release v0.2 checkpoint referenced by the old runner was not fetched and cannot be claimed equivalent.
