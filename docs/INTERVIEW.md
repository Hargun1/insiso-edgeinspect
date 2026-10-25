# Walkthrough and interview preparation
Read the actual implementation and execute the pipeline before presenting it.

## Two-minute walkthrough
- 0:00–0:20: problem, four defect classes, local edge constraint.
- 0:20–0:45: show masks-to-box policy and custom supervised split; explain why official training has no defects.
- 0:45–1:10: real baseline vs augmentation results; what helped or hurt and how selection avoided test leakage.
- 1:10–1:35: show one actual FP and one miss, with evidence and qualified explanation.
- 1:35–2:00: upload an image, run real ONNX CPU inference, and show measured model/total p50/p95.

## Explain these without memorizing
- **Why YOLO11n?** Small pretrained detector as a practical baseline. Actual suitability depends on local CPU timing and small-defect recall.
- **Why Python and React?** Python supports training/export/runtime; React makes results inspectable. Database and Express add little to the required pipeline.
- **Mask to YOLO label:** component rectangle (x1,y1,x2,y2) -> center x/y and width/height divided by original image dimensions.
- **Why negatives?** A model must learn good metal surfaces and avoid false rejects. Good images have empty label files.
- **Why a custom split?** MVTec was built for anomaly detection. Its train set has no defective objects for supervised box training.
- **What does IoU mean?** Intersection area / union area. Correct class and IoU >=0.5 are required for a match.
- **Why can precision change without AP changing?** Precision is at one score threshold; AP integrates ranked predictions across thresholds.
- **Why not rotate arbitrarily?** Orientation can affect the meaning of a flip class, lighting and pose; review masks and validate augmentations.
- **How do duplicate predictions count?** The highest-confidence matching detection can be TP; another detection of the same truth is FP.
- **How do boxes reach original coordinates?** Invert letterbox padding and scale, then clip to image boundaries.
- **Why ONNX accuracy evaluation?** Export/decoder changes may affect outputs; successful serialization alone is insufficient.
- **Why not display PASS?** No predicted defect is not proof of an acceptable part. Real deployments need validated acceptance/reject rules.
- **10 ms?** First distinguish throughput and deadline; profile stages and tail latency, then measure hardware/precision/resolution tradeoffs.
- **What did AI do?** Explain the disclosure precisely and distinguish written code from experiments actually run.

## Files to read in order
src/prepare.py -> src/train.py -> src/metrics.py -> src/evaluate.py -> src/select.py -> src/runtime.py -> src/export_benchmark.py -> src/finalize.py -> src/api.py -> web/src/main.jsx.

