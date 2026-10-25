# Verification status
Updated 25 October 2026.

## Completed
- Downloaded the metal_nut category and official pretrained YOLO11n weights; retained source and checksums.
- Prepared and visually audited a fixed 199/67/69 supervised train/validation/test split.
- Trained both GPU experiments, selected by validation mAP50, and evaluated the selected checkpoint on the held-out test.
- Exported and checked ONNX; evaluated the exact custom deployment runtime on the same held-out test.
- Corrected boundary clipping/NMS order and retained original decoder outputs for traceability. Final PyTorch and ONNX detection metrics match.
- Measured CPU timing over 20 warmups and 100 iterations, with CPU identity, image size and runtime settings recorded.
- Visually reviewed five genuine held-out failure cases; report includes each image, observation and qualified explanation.
- Completed written scenario answers A/B/C, model card, attribution, source and model checksums.
- Built the React production frontend. 19 tests passed, including NMS boundary-order regression, label conversion, metric matching and API input handling.
- Verified real ONNX inference and image decoding through API route functions; integration_checks.json states the exact scope.
- Final verification rebuilt all 335 labels/split assignments from the 430 checksum-verified data files. Both validation runs and the selected PyTorch/ONNX test evaluations reproduced the recorded metrics.
- Re-exported the selected checkpoint to ONNX in a temporary directory, checked the export, and reproduced its held-out metrics. Repeated 20 warmups and 100 CPU timing iterations as a separate verification measurement; the original benchmark remains preserved.
- Started the actual app with start.ps1. Verified HTTP health/results routes, built frontend assets, good/flipped sample inference, original-coordinate boxes, and 400/413/415/422 input-error responses. See artifacts/final_verification.json.
- Re-ran all 19 tests, Python compilation, dependency consistency checks and the React production build successfully.

## Practical limits
- HTTP transport, real inference and frontend asset serving passed the final verification. Interactive browser control was denied by the app approval system ("Computer Use was not approved to use Microsoft Edge"), so interactive browser actions remain unverified. Use start.ps1 for the local demo.
- The detector has modest localization performance: precision 0.3438, recall 0.4074, mAP50 0.5182. It is not ready to serve as an unattended production reject system.
- One split/seed and very few defect images per class limit generalization claims. No quantization, edge-controller measurement or production pilot was performed.

## External submission
Repository: https://github.com/Hargun1/insiso-edgeinspect. Source, documentation, built frontend, selected ONNX/PyTorch models, both training runs' best checkpoints, and measured results are included together. artifacts/model_checksums.json identifies the selected models; artifacts/repository_manifest.json records SHA-256 hashes and sizes for the published files (excluding the manifest itself). No submission email has been sent. Reply in the original thread with studying status and confirmed earliest availability. Review docs/REPORT.md and docs/INTERVIEW.md before presenting it.

