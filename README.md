# EdgeInspect — Metal Nut Defect Detection
Independent InSiSo computer-vision internship take-home by Hargun Madan.

**Status: experiments, ONNX, CPU benchmark and technical write-up completed.**

Selected test results: precision **0.3438**, recall **0.4074**, mAP@0.5 **0.5182**. These are modest localization results, not production-readiness claims. See [the report](docs/REPORT.md) for five real failures and limitations.

**Quick demo:** run start.ps1 from a normal PowerShell terminal, then open http://127.0.0.1:8000. The packaged ONNX model runs locally. Real good/flipped sample images are included.

## Submission files
- [Technical report and five real failure cases](docs/REPORT.md)
- [ONNX model](artifacts/metal-nut.onnx) and [selected PyTorch weights](artifacts/best-model.pt)
- [Model card](docs/MODEL_CARD.md), [measured CPU benchmark](artifacts/benchmark.json), and [test results](artifacts/onnx_test.json)

## What this project does
Converts MVTec AD metal_nut masks to boxes, trains a small four-class YOLO11n detector, evaluates localization, exports ONNX, benchmarks a CPU-only runtime, and serves a local React inspection workspace. The demo supports uploaded images, real detection overlays, a threshold control, measured runtime breakdown, experiment results, and in-memory history.

Python handles vision and FastAPI; React handles the UI. A database and an Express proxy would not improve this assignment, so they are intentionally absent. Once dependencies and weights are installed, inference stays local.

## Setup (Windows PowerShell)
Use Python 3.11 or 3.12 if dependency wheels are unavailable for your Python version. Commands below run from the project root.

~~~powershell
python -m venv .venv
.\.venv\Scripts\python -m pip install --upgrade pip
.\.venv\Scripts\python -m pip install -r requirements.txt
~~~

Optional NVIDIA training: install an appropriate CUDA-enabled PyTorch wheel from https://pytorch.org/get-started/locally/ before the remaining dependencies. CPU training is supported. The discovered development GPU was a GTX 1650 (4 GB); both completed training runs used this GPU.

If environment creation fails at ensurepip, try Python 3.12 from an ordinary terminal. Do not assume the partially created environment works.

## 1. Get the data and create labels
Download the **metal_nut** category from the official MVTec AD page:
https://www.mvtec.com/research-teaching/datasets/mvtec-ad

Extract it to data/raw/metal_nut. Expected structure:

~~~text
data/raw/metal_nut/
  train/good/*.png
  test/{good,bent,color,flip,scratch}/*.png
  ground_truth/{bent,color,flip,scratch}/*_mask.png
~~~

~~~powershell
.\.venv\Scripts\python -m src.prepare --raw data/raw/metal_nut
.\.venv\Scripts\python -m src.audit_labels
~~~

**Protocol:** MVTec's official training data has no defects. For this supervised assignment, pool images by category and make a custom stratified 60/20/20 train/validation/test split, seed 42. This repurposes some original test images for supervised training; it is not comparable to the official anomaly-detection benchmark. Remove exact byte duplicates before splitting. Keep the generated manifest fixed across experiments. Near-duplicate scenes may remain: exact hashes do not establish production independence. If capture IDs become available, split by physical part/capture session instead.

Every nonzero connected mask component becomes one tight, axis-aligned box. Preserve tiny components rather than silently filtering hard labels. Pixel coordinates use exclusive maximum edges. Good parts have empty label files. Defect class order is bent, color, flip, scratch. The source folder supplies class identity. Disconnected components may represent one physical defect, and flipped parts may have large masks: review the label contact sheet before training. Boxes discard mask shape and include background; discuss this limitation.

The audit writes label_preview.jpg and split_audit.json. Verify every class in every split, tiny boxes, flipped-part masks, and data quality. Changing label policy requires a new frozen dataset version.

## 2. Train a baseline and one improvement
For the GPU setup used here, run train-gpu.ps1. It verifies CUDA and runs both experiments, selection, ONNX export and CPU benchmarking. The checkpoints in this package are already trained.

~~~powershell
.\.venv\Scripts\python -m src.train --experiment baseline --device cpu
.\.venv\Scripts\python -m src.train --experiment augmented --device cpu
~~~

For a configured CUDA environment replace --device cpu with --device 0. Reduce --batch to 2 if GPU memory is insufficient; keep both arms identical. Default: pretrained yolo11n.pt, 320 input, 30 epoch maximum, seed 42, AdamW, early stopping patience 15. The best checkpoint is selected using validation performance by the training framework. Actual epochs can differ because of early stopping; disclose them and elapsed time.

**Improvement hypothesis:** mild geometric and lighting augmentation may reduce overfitting on limited defect images. Both arms use the same initialization, split and training budget; augmentation is the changed factor. Avoid aggressive hue shifts because color is itself a defect class. Neither arm uses mosaic. A single seed is an exploratory comparison, not statistically reliable proof. If augmentation hurts validation results, report that honestly and select the baseline.

## 3. Validate, freeze selection, then evaluate
~~~powershell
.\.venv\Scripts\python -m src.evaluate --model runs/baseline/weights/best.pt --name baseline_val
.\.venv\Scripts\python -m src.evaluate --model runs/augmented/weights/best.pt --name augmented_val
.\.venv\Scripts\python -m src.select
~~~

Selection uses higher validation mAP@0.5; a tie favors the simpler baseline. The operating threshold remains predeclared at 0.25. The validation threshold sweep is diagnostic only. To change the threshold, document a validation-only decision before touching test. Never select it from test results.

~~~powershell
.\.venv\Scripts\python -m src.finalize
~~~

This runs selected-model test evaluation, ONNX export, benchmark and ONNX test evaluation. It refuses to overwrite an existing selection or final result, so the final test cannot silently become a tuning loop. If a run is interrupted, complete its remaining commands manually after inspecting which outputs already exist.

Metrics are class-aware with one-to-one greedy matching at IoU >=0.5. Precision/recall are micro-aggregated counts at the fixed confidence threshold. AP uses all-points interpolated area under the confidence-ranked precision–recall curve, then macro-averages classes with ground truth. Predictions are collected down to confidence 0.001; AP below that floor is unmeasured. This AP implementation is not COCO's 101-point convention. Report the convention with the numbers. Good images contribute false positives. Report per-class AP and counts, not just an aggregate.

False positives/misses generate overlays in artifacts/failures. Green = ground truth, red = false positive, blue = matched prediction. Visually inspect at least five **distinct held-out images**, then replace each selected case's pending analysis in docs/REPORT.md with an observation and a clearly qualified causal hypothesis. If there are fewer than five errors at the frozen operating point, report all of them and transparently inspect an additional prespecified stress/shift set; do not invent misses or tune on test to generate them.

## 4. ONNX export and CPU benchmarking
The finalizer calls src.export_benchmark. For a manual run:

~~~powershell
.\.venv\Scripts\python -m src.export_benchmark --model runs/baseline/weights/best.pt --threads 1 --iterations 100
~~~

Use the selected checkpoint, not necessarily the baseline shown in this example. The exported artifact is artifacts/metal-nut.onnx: FP32, fixed batch 1, 320 x 320, opset 17. The runtime uses ONNX Runtime CPUExecutionProvider, RGB conversion, letterboxing, /255 normalization, and class-aware NMS.

The benchmark records CPU name, OS, runtime version, thread count, model size, 20 warmups, 100 iterations, and mean/p50/p95 for preprocessing, model execution, postprocessing and total. Images are preloaded. File loading/decoding, HTTP and display are excluded. GPU inference is never substituted for the required CPU timing. Repeated runs on different CPUs are not directly comparable.

The exporter records a decoded-detection count smoke check against PyTorch, not a proof of numerical parity. Finalization also evaluates ONNX through the **same custom runtime used by the demo**, for a real deployment accuracy check. Inspect any accuracy change before submitting.

## 5. Run the inspection UI
~~~powershell
.\.venv\Scripts\python -m uvicorn src.api:app --host 127.0.0.1 --port 8000
~~~

In another terminal:
~~~powershell
cd web
npm install
npm run dev
~~~

Open http://127.0.0.1:5173. If no weights exist, the UI shows the missing-model state; inference returns HTTP 503. Upload PNG/JPEG up to 10 MiB and 16 megapixels. Detection boxes are in original-image coordinates. The API corrects EXIF orientation. The UI uses the same correction in modern browsers; use upright, orientation-free dataset images for the demo.

Production preview: run npm run build inside web, then restart FastAPI and open http://127.0.0.1:8000. The frontend uses relative API URLs, so this is a single local origin. Google Fonts is optional; system fonts keep the interface usable offline. No inference or image upload uses an external service. Session history is in memory; reloading clears it.

“No defect detected” is a model observation, not a guarantee of a good part. This demo is not a certified production reject controller. Threshold changes clear the displayed prediction so an old result cannot appear to use a new threshold.

## Validation
~~~powershell
python -m unittest discover -s tests -v
python -m compileall -q src
.\.venv\Scripts\python -m src.check_submission
~~~

Metric and selection tests are dependency-free; the complete test suite requires the installed vision/API dependencies. Training, mask conversion, export and inference were executed; see the verification status for the exact API/browser test scope. See docs/STATUS.md for what was actually executed.

## Deliverables and submission
- Code and reproduction instructions in this repository.
- docs/REPORT.md: approach, results, five reviewed failures, experiment conclusion and written answers A/B/C.
- artifacts/metal-nut.onnx (or a release/download link). Large model artifacts should go in a GitHub release or Git LFS.
- artifacts/selection.json, benchmark.json, *_val.json and *_test.json, plus frozen manifest/split audit and environment versions.
- Optional 2-minute recording: explain split, compare baseline/improvement, inspect one failure, run local inference, show CPU timings.
- Reply in the original email thread with the GitHub link, studying status/year, and your confirmed earliest start date.

**AI disclosure:** Codex assisted with implementation, evaluation design, documentation and the UI. Training results must come from actual runs. The applicant should review and explain every submitted component. See docs/INTERVIEW.md.

## Attribution
Dataset: Paul Bergmann et al., MVTec AD. CC BY-NC-SA 4.0, non-commercial use. Do not redistribute the dataset casually; prefer download instructions and image IDs. Review image licensing before posting failure overlays publicly.
Model/framework: Ultralytics YOLO11, pretrained on COCO. Review Ultralytics' AGPL-3.0 / enterprise terms for distribution and any future commercial use.
Design direction: industrial edge-vision workflows and InSiSo's Develop / Evaluate / ONNX / local deployment emphasis. EdgeInspect is independent, uses its own name, and does not claim to be InSiSoNet or an official company product.


## Included measured submission
See docs/REPORT.md, docs/MODEL_CARD.md and docs/SUBMISSION.md. Models are in artifacts/; selected weights and ONNX checksums are in artifacts/model_checksums.json. Five reviewed failure overlays are in docs/images/. The original decoder results are retained as *_decoder_v1.json; the current decoder corrects NMS/clipping order. No test-driven model/threshold change was made.

## Inference-only installation
For a smaller installation that does not need PyTorch, create the environment and install requirements-runtime.txt, then run start.ps1. The ONNX model and sample images are included; training is optional for using the demo.
