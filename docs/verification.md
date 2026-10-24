# Verification — 2026-10-24

Environment: Linux x86_64, CPython 3.14.7. The available Python is newer than the documented ML target (3.11/3.12). No CUDA or ML inference was executed.

| Check | Actual result |
| --- | --- |
| `python -m unittest discover -s tests -v` | 21 passed, 0 failures, 0 errors, 0 skips |
| Python compilation, CLI `--help` | Passed |
| Whitespace formatting and `git diff --check` | Passed; no broad code formatter run |
| Clean ML install, actual training/inference/evaluation/webcam | BLOCKED: no dataset or ML dependencies; shell network sockets denied |
| Ruff / Bandit / pip-audit | BLOCKED: tools absent; cannot download packages/advisories |
| GitHub Actions | YAML parsed locally; jobs not executed remotely |
| Full dependency lock | BLOCKED until resolver, imports and security checks can run on target Python |

Machine-readable unit results: [verification-results.json](verification-results.json). Tests use tiny synthetic byte fixtures to verify file handling and mathematical box conversion, not actual image decoding or detection. CLI model calls are mocked. Camera cleanup is tested on simulated failures and normal exit.

The dependency list removes the stale fixed Torch/Torchvision/NumPy combination and unused direct TensorBoard/Pillow/Matplotlib pins. Ultralytics is kept at 8.4.66; remaining direct constraints require resolution and `pip check` before calling the ML setup verified. No clean-install claim is made.

Offline text-secret scanning covered all 9 original reachable commits (51 distinct blobs) and the working tree. No credential pattern matches were found. Twenty binary objects, including model files and images, require separate content/licensing review; binary contents were not secret-scanned. This limited scan does not prove absence of secrets, PII or vulnerable dependencies. Dataset licensing and an explicit project/checkpoint license remain owner-review blockers. No credentials were confirmed exposed and no rotation was performed.

## Showcase implementation follow-up

The local upload viewer (`src/viewer.py`) uses the shared `prediction.py` boundary, actual YOLO results, adjustable confidence/NMS and loopback-only Gradio. CLI launch with the existing checkpoint was attempted and failed explicitly because Gradio is absent; no model was loaded and no UI screenshot was fabricated.

`generate_demo.py` implements real raw/prediction image pairs and annotated MP4 with hash/configuration/detection sidecars; data and corrected weights are absent, so no new detection video/image is claimed. Recovery of v0.2 weights failed with curl DNS exit 6. `evaluate.py` adds evaluated per-class metrics (no filled-in absent classes) and sequential images/second. Actual dependency-enabled execution is still blocked.

Four added tests cover local-input/finite-threshold enforcement, truthful empty detections, coordinate/class/confidence export and evaluated-class reporting. All 21 tests pass. No installed model inference is substituted by mocked tests. Two accessible safe SVG diagrams and the unchanged archival JPEG passed `scripts/check_assets.py`; local rendering was inspected using rsvg-convert. Existing training/PR/confusion plots remain explicitly historical and invalid for localization claims.
