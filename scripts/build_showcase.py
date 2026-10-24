"""Generate an implementation map; preserve existing model output as archival evidence."""

from pathlib import Path

from showcase_svg import MUTED, TEAL, arrow, panel, save, text

ROOT=Path(__file__).resolve().parents[1]


def main():
    body=text(32,72,'Real annotations in / explicit checkpoints out / no implicit download or fabricated boxes',15,MUTED)
    nodes=[(32,116,'COCO + images','source hashes + file validation'),(360,116,'YOLO dataset','80 / 10 / 10 image-level split'),
           (688,116,'YOLOv8 training','explicit weights / CPU or CUDA'),(32,300,'Outputs + provenance','actual boxes / raw-prediction pair'),
           (360,300,'Prediction boundary','confidence + NMS IoU / local files'),(688,300,'Held-out evaluation','mAP + per-class + timed prediction')]
    for x,y,title,subtitle in nodes:
        body+=panel(x,y,300,96)+text(x+16,y+36,title,20,weight=700)+text(x+16,y+66,subtitle,14,MUTED)
    for edge in ((332,164,358,164),(660,164,686,164),(838,212,838,298),(688,348,662,348),(360,348,334,348)):
        body+=arrow(*edge)
    body+=text(32,448,'Entrypoints: prepare_dataset.py → train.py → evaluate.py / viewer.py / generate_demo.py',15,TEAL)
    save(ROOT/'docs/assets/pipeline.svg','Waste detection / reproducible execution path',
         'Components implemented in src/coco_to_yolo.py, train.py, prediction.py, viewer.py, generate_demo.py and evaluate.py. Runtime inference is blocked in the current environment.',body)
    body=text(32,75,'Engineering decisions embodied in the current data and inference code',15,MUTED)
    rows=[('Real COCO boxes','Correct corner clipping; reject invalid source records','Avoid full-frame pseudo-labels'),
          ('Seeded image splits','Stable IDs + hash-based duplicate rejection','Scene grouping still needs dataset knowledge'),
          ('Explicit local weights','No hidden download; checkpoint hashes in demo exports','Model quality still requires held-out evaluation'),
          ('Confidence + NMS IoU','Adjustable in the local viewer','Precision/recall changes need actual measurements')]
    for i,(name,mechanism,tradeoff) in enumerate(rows):
        y=102+i*82
        body+=panel(24,y,992,72)+text(42,y+28,name,17,TEAL,700)+text(300,y+28,mechanism,16)+text(300,y+53,tradeoff,14,MUTED)
    save(ROOT/'docs/assets/decisions.svg','Design choices / explicit tradeoffs','No measured result is implied by the architecture or tradeoff diagram.',body)
    print('Generated pipeline.svg and decisions.svg from documented implementation boundaries')


if __name__=='__main__':
    main()
