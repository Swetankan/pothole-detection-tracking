# Pothole Lab

Software-only pothole video processing: pretrained detection, optional promptable masks, camera-motion-aware tracking, review and manual depth labels.

## Quick start

Python 3.12 was used for this run. Install FFmpeg and Python dependencies in a virtual environment:

```bash
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows instead: .venv\Scripts\activate
pip install -r requirements.txt
python download_models.py
python pipeline.py analyze data/sample.mp4 --output outputs/new_run --sam mobile_sam.pt --imgsz 960 --conf 0.30 --mask-every 5
python reconcile.py outputs/new_run
python -m streamlit run review_app.py
```

The default detector downloads from Hugging Face when no local weights are present. MobileSAM downloads through Ultralytics when requested. The input video stays local; model downloads require internet. GPU is optional: add `--device 0` for CUDA, or `--device mps` on a compatible Mac. CPU is supported but slow.

Open the review app and enter `outputs/new_run`. Confirm/reject tracks, merge duplicates, enter measured depth in centimetres, and rebuild exports. No depth is inferred from the detector confidence score.

## Delivered sample

The delivered export uses a ten-observation minimum to suppress brief fragments. This is a development heuristic, not a validated optimum.

`outputs/sample/annotated.mp4` is the actual processed clean sample, not the user's pre-annotated video. `analysis.json` stores every sampled frame, model scores, polygons, track IDs, blur and motion diagnostics. The sample has no independent reference labels; counts remain provisional. Do not interpret the video overlay as validated ground truth.

The clean sample and pre-annotated sample show the same scene; they are not independent evaluation data. Model choices and thresholds were visually explored on this scene, so this is development data.

## Implemented architecture

1. Decode frames at native resolution with OpenCV; record timestamps as frame/fps for these constant-frame-rate clips.
2. YOLO26 pothole detector from DanielsStulpe/pothole-detection; optionally accept other Ultralytics detection/segmentation checkpoints via `--weights`.
3. Optional full-frame plus overlapping tiled detection with box NMS (`--tiles`). Not enabled for the delivered run; its benefit needs evaluation.
4. Optional MobileSAM box-prompted segmentation. Every new track receives a mask. It is refreshed every five frames in the sample; intermediate polygons are approximately transformed to the current detector box. Each detection records `mask_source`. Use `--mask-every 1` for independently inferred masks on every frame at higher cost.
5. Lucas–Kanade forward/backward feature checks and RANSAC homography for camera-motion compensation; Hungarian box association; limited gap tolerance.
6. Configurable observation filter for provisional tracks (default three; delivered export uses ten observations, 0.4 seconds at 25 fps). Detections are not displayed when absent; there are no invented detections in missed frames.
7. Reverse association generates suggestions for non-overlapping fragmented tracks. It does not automatically merge uncertain tracks.
8. Review statuses, merge validation, per-frame JSON mask corrections and nonnegative finite manual-depth validation. Raw predictions remain intact; frame_overrides.json records changes.
9. H.264 MP4, CSV/JSON register, evidence frames and provenance hashes.

## What is NOT implemented or claimed

- No automatic depth in centimetres; no physical scale or visible bottom reference exists for many water-filled holes.
- No SAM 3, 3D reconstruction or metric-depth foundation model is used in this delivered baseline.
- Reverse tracking suggests identity merges; it does not recover missed detections or propagate masks backward.
- The UI edits status, identity and depth and supports per-frame JSON polygon correction or instance insertion. It does not provide point-and-click polygon drawing or automatic correction propagation.
- No fine-tuning has been performed. The supplied training command is ready for correctly labelled independent road data.
- No claim of state-of-the-art, 95% accuracy, certified counts or calibrated uncertainty. Scores are model outputs, not measured accuracy probabilities.
- No audio is copied to the rendered output (the supplied sample has no audio).
- Variable-frame-rate footage should be converted to constant frame rate first. This implementation uses frame/fps timestamps.
- Review exports filter/merge existing predictions. For frame-level evaluation the evaluation script uses raw analysis predictions, not reviewed exports.

## CLI examples

```bash
# Full masks on every frame, optional CUDA:
python pipeline.py analyze data/sample.mp4 --output outputs/full_masks --sam mobile_sam.pt --mask-every 1 --device 0
# Explicit local checkpoint; boxes are supported even without SAM:
python pipeline.py analyze data/sample.mp4 --weights /path/to/best.pt --output outputs/custom
# Regenerate exports after review without model inference:
python pipeline.py export outputs/sample
# Unit tests:
python -m unittest discover -s tests
# Frame-level mask evaluation against fully annotated frames:
python evaluate.py outputs/sample/analysis.json labels/ground_truth.json
```

Ground-truth format:

```json
{"frames": [{"frame": 0, "polygons": [[[100,200],[180,200],[180,240],[100,240]]]}]}
```

This example is schema documentation only, not a label for the real clip. Label every visible pothole in each evaluated frame, including ambiguous water cases under a documented annotation policy. The evaluator reports mask precision/recall at IoU 0.5 and mean IoU of matched masks. It does not compute mAP, temporal ID metrics or depth error.

## Fine-tuning

Use YOLO segmentation-format labels and a dataset YAML with `train`, `val`, `test`, and `names`. Create a separate JSON manifest of physical road IDs:

```json
{"train":["road_A"],"val":["road_B"],"test":["road_C"]}
```

```bash
python train.py --data dataset.yaml --split-manifest splits.json --weights models/pothole-seg.pt --device 0
```

Supply an actual segmentation checkpoint for this command (the default video detector is a detection model). For example download `best.pt` from `keremberke/yolov8m-pothole-segmentation`. The manifest check prevents declared road-ID overlap; it cannot independently verify your image-to-road mapping. Check that mapping manually. Never put frames or duplicate versions of one physical pothole into different splits. Keep an untouched test road and evaluate the final selected weights separately.

Prioritise corrected masks, wet-road hard negatives, small potholes, complete instance labels and independent roads. More training epochs on the same 28-second scene will not demonstrate generalisation.

## Model provenance

- Selected detector: https://huggingface.co/DanielsStulpe/pothole-detection (`yolo26_best.pt`). Its model card declares MIT; review upstream model/framework and training-data terms before redistribution or commercial use.
- Compared detector: https://huggingface.co/Samdutse/pothole-yolov8 (`best.pt`).
- Compared segmentation model: https://huggingface.co/keremberke/yolov8m-pothole-segmentation (`best.pt`). Low response on the sample at initial settings; not selected.
- MobileSAM: downloaded by Ultralytics from its official assets repository. It segments prompted regions; it does not independently establish that a region is a pothole.
- Framework: Ultralytics. Review its AGPL/enterprise terms for your use. Third-party model licenses are separate from this project code.

Weights are not included in the source archive. `analysis.json` records the detector hash used. Dependencies and platform versions from the execution environment are in `environment-tested.txt`; `requirements.txt` contains the functional dependencies. This is a research baseline that needs held-out evaluation before deployment.
