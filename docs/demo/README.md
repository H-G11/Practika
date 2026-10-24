# Demo asset provenance

`validation-predictions.jpg` is a byte-for-byte copy of the repository's retained Ultralytics validation output:

- Source: [`results/trash_detector-4/val_batch0_pred.jpg`](../../results/trash_detector-4/val_batch0_pred.jpg).
- Source run: `trash_detector-4`; [`args.yaml`](../../results/trash_detector-4/args.yaml) records YOLOv8s, 20 epochs, image size 640 and single-class labels.
- SHA256 of both original and copy: `1399809854bdfc97d3c8c60eb3ed96e5fc352231497cad2bb86aafa7d25745a5`.
- Bytes: 489478.

No cropping, retouching, relabeling, image generation or new inference was performed. The visible boxes/labels are the original model output, not ground-truth overlays. The image contains a batch montage rather than independently selected success examples.

**Known limitation:** the old preparation script generated full-frame target boxes. These predictions reflect that defect and do not demonstrate reliable object localization. Historical scores are not accepted benchmarks. See [historical results](../historical-results.md).

**BLOCKED — replacement localization demo:** the source dataset and corrected trained checkpoint are absent; Ultralytics, PyTorch and OpenCV are unavailable in the current execution environment. Historical single-class weights exist, but are not evidence that retraining on real COCO boxes has occurred. See [usage](../usage.md) for the actual preparation, training, inference and held-out evaluation procedure. Replace this archival demo only with genuine new prediction output and retain its provenance.

Original image/data redistribution terms still require owner review. The asset was already present in the public repository; copying it does not establish additional rights.
