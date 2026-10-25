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

## Practical limits
- Unit/route-function checks and the production build passed. Full HTTP transport and interactive browser validation could not be completed because this execution session restricts sockets. Use start.ps1 from a normal terminal for the local demo.
- The detector has modest localization performance: precision 0.3438, recall 0.4074, mAP50 0.5182. It is not ready to serve as an unattended production reject system.
- One split/seed and very few defect images per class limit generalization claims. No quantization, edge-controller measurement or production pilot was performed.

## External submission
Repository: https://github.com/Hargun1/insiso-edgeinspect. Source, documentation, built frontend, selected ONNX/PyTorch models, both training runs' best checkpoints, and measured results are included together. artifacts/model_checksums.json identifies the selected models; artifacts/repository_manifest.json records SHA-256 hashes and sizes for the published files (excluding the manifest itself). No submission email has been sent. Reply in the original thread with studying status and confirmed earliest availability. Review docs/REPORT.md and docs/INTERVIEW.md before presenting it.

