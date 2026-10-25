# Metal Nut Defect Detection
Hargun Madan | InSiSo computer-vision internship take-home | 25 October 2026

## Result and practical limits
A four-class YOLO11n detector was trained on a custom supervised MVTec AD metal_nut split and exported to a local CPU ONNX runtime. Augmentation raised validation mAP@0.5 from 0.3893 to 0.4628. The selected ONNX model achieved **mAP@0.5 0.5182, precision 0.3438, recall 0.4074** at confidence 0.25 and IoU 0.5 on 69 held-out images.

These are modest results, not production-readiness claims. The model misses 16 of 27 annotated components and produces 21 extra boxes. At the simpler image-alarm level it flags 16/20 defective images and falsely flags 0/49 good images, but this does not establish a reliable industrial false-alarm rate: the test is small and four defective parts still escape. Part-level alarm performance must not be substituted for localization recall.

## Labels and split
Each connected, nonzero pixel-mask component becomes one tight axis-aligned box with exclusive maximum pixel coordinates. The folder supplies bent/color/flip/scratch class identity; good images get empty label files. All components are kept. I visually reviewed the training label contact sheet before training.

This policy is deterministic, but disconnected fragments are counted as separate detection instances even when they belong to one physical defect. It can create several nearby or overlapping boxes, while flip masks usually cover the whole part. Thus the metric reflects the declared component-label policy, not a guaranteed count of physical defect events. A future labeling study should agree on physical defect instances with an inspector before evaluating a revised policy.

MVTec's original training split has only good parts. For supervised detection, I repurposed some original test images into training and validation, after deduplicating exact image bytes and making a seed-42, stratified 60/20/20 split. **This is not the official unsupervised MVTec protocol and is not comparable to its anomaly benchmark scores.** Exact hashes do not rule out near duplicates; capture-session/physical-part grouped evaluation is the next safeguard if those IDs become available. Training augmentation happens after the split. Selection uses validation only.

| Split | Good | Bent | Color | Flip | Scratch | Total |
|---|---:|---:|---:|---:|---:|---:|
| Train | 145 | 15 | 13 | 13 | 13 | 199 |
| Validation | 48 | 5 | 4 | 5 | 5 | 67 |
| Test | 49 | 5 | 5 | 5 | 5 | 69 |

## Controlled experiment and results
Both arms start from the same official COCO-pretrained yolo11n.pt, seed 42, input 320x320, batch 4, FP32, AdamW at initial learning rate 0.001, at most 30 epochs and patience 15. Training ran on an NVIDIA GeForce GTX 1650 (4 GB). The first CPU attempt was interrupted during epoch one without a usable checkpoint; it is not an experiment result. Before comparing validation results, the completed runs were configured with the same GPU budget. The baseline stopped after 28 epochs and augmentation after 29.

| Run | Split | Precision | Recall | mAP@0.5 | TP / FP / FN |
|---|---|---:|---:|---:|---|
| Baseline | val | 0.6250 | 0.3125 | 0.3893 | 10 / 6 / 22 |
| Mild augmentation | val | 0.4783 | 0.3438 | 0.4628 | 11 / 12 / 21 |
| Selected PyTorch | test | 0.3438 | 0.4074 | 0.5182 | 11 / 21 / 16 |
| Selected ONNX | test | 0.3438 | 0.4074 | 0.5182 | 11 / 21 / 16 |

Precision and recall are micro-aggregated detection counts at confidence 0.25. Matching is one-to-one, class-aware and uses IoU >=0.5. AP uses all-points interpolated area under the ranked precision-recall curve, with predictions collected down to confidence 0.001; this is not COCO's 101-point convention. mAP is the macro average over classes with ground truth. Good images count toward false positives. The framework selects the checkpoint within each run using its validation fitness; the independent evaluator selects between the two best checkpoints by validation mAP@0.5, with ties favoring baseline. The diagnostic validation threshold sweep was not used to change the predeclared operating threshold.

### Did the improvement work?
The changed factor is modest augmentation: +/-15-degree rotation, translation 0.05, scale 0.15, horizontal/vertical flips at probability 0.5, and small saturation/value changes. Mosaic and mixup stay off. Strong hue shifts are avoided because color is a defect class. Flip here means the nut is physically turned over; reflecting an image is not the same physical operation, but its realism still deserves validation.

Aggregate validation AP improved by **0.0735 absolute (7.35 percentage points)**, so the augmented checkpoint was selected before final-test evaluation. This is not a uniform win: validation precision fell from 0.6250 to 0.4783 while recall rose from 0.3125 to 0.3438. Bent and color AP increased, scratch AP fell from 0.1715 to 0.0400, and flip AP remained 1.0. The small validation sample and a single seed limit confidence in any causal claim. The augmentation may broaden pose/color coverage while harming subtle scratch cues; the class-specific results support investigating that tradeoff, not claiming a universal improvement.

### Selected ONNX per-class test AP
| Class | Annotated components | AP@0.5 |
|---|---:|---:|
| bent | 9 | 0.2074 |
| color | 5 | 0.3233 |
| flip | 5 | 1.0000 |
| scratch | 8 | 0.5421 |

Flip masks are large and visually distinct; five positive examples are insufficient to infer production reliability. The aggregate score is partly driven by this easier class. Bent/color localization remain weak, and no test-driven retraining or threshold selection was performed.

## Export, runtime parity and CPU speed
The selected weights export to **FP32 ONNX, opset 17, fixed batch 1, 320x320**, 10.44 MB. ONNX checker passes. Deployment uses RGB conversion, letterboxing, division by 255 and class-aware NMS on CPUExecutionProvider.

Reference-versus-runtime checking found a decoder error: clipping boxes before NMS changed IoU and suppressed extra boundary boxes. I moved clipping after NMS, added a boundary regression test and rechecked deployment output. Training, checkpoint selection, labels and confidence were unchanged. Original decoder results are retained as *_decoder_v1.json. The final ONNX precision, recall and mAP now match the PyTorch test metrics. This is a deployment correction, not a model-improvement experiment or test-set threshold tuning.

CPU: **Intel(R) Core(TM) i5-10300H CPU @ 2.50GHz**, Windows 11, ONNX Runtime 1.23.1, one intra-op thread and one inter-op thread. Twenty warmup iterations precede 100 measurements over preloaded validation images. The following is the current-runtime benchmark, not a GPU timing.

| Stage | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
| preprocess | 1.99 | 1.99 | 2.62 |
| inference | 40.45 | 39.11 | 50.74 |
| postprocess | 1.18 | 1.00 | 1.80 |
| total | 43.62 | 41.81 | 55.19 |

Disk reading, decoding, HTTP, display and physical actuator timing are excluded. Laptop load and thermal state affect timings; the earlier benchmark is retained for traceability and is not presented as a controlled speedup. The model does not meet a 10 ms CPU budget on this machine.

## Five visually reviewed held-out failures
Green boxes are ground truth, red boxes are false positives, and blue/cyan boxes are matched predictions. The causes below are reasoned hypotheses based on visible evidence; the measured error type is separate from the hypothesis.

### 1. test_bent_015 - Miss: 4 components
No detection above 0.25, despite visible damage on the upper lug and left edge.

Likely explanation: The damage spans several disconnected mask components and scales. Limited pose/defect examples may leave this appearance uncovered; this is a hypothesis, not an observed internal cause.

![test_bent_015 ground truth and predictions](images/test_bent_015.jpg)

### 2. test_bent_023 - Localization FP + miss
A bent prediction at confidence 0.420 covers only part of the upper-edge defect; IoU is 0.486, below the fixed 0.5 matching cutoff.

Likely explanation: The box follows the visually strongest notch but underestimates the annotated extent. This demonstrates why finding a defect-looking region is not sufficient for correct localization.

![test_bent_023 ground truth and predictions](images/test_bent_023.jpg)

### 3. test_color_004 - Miss: 1 component
The dark blue patch beside the inner ring has no prediction above the operating threshold.

Likely explanation: The relatively small, dark patch blends into the inner-rim shadow and metallic texture. More representative dark-color flaws and higher-resolution validation are plausible next experiments.

![test_color_004 ground truth and predictions](images/test_color_004.jpg)

### 4. test_color_013 - 3 extra detections
The red patch is correctly detected as color at confidence 0.878, but two bent boxes and a second color box remain as false positives.

Likely explanation: The same high-contrast region triggers multiple class heads. Class-aware NMS intentionally keeps different classes; partial boxes also escape same-class suppression. This is classification and duplicate-counting failure, not a missed red patch.

![test_color_013 ground truth and predictions](images/test_color_013.jpg)

### 5. test_scratch_018 - 2 FPs + 1 miss
The upper scratch is matched at confidence 0.307, but higher-scoring color (0.753) and bent (0.614) boxes overlap it. The separate lower-left scratch is missed.

Likely explanation: A dark scar can resemble staining or deformation, while the second scar has different orientation and local texture. The pattern suggests class ambiguity and incomplete coverage across multiple defects.

![test_scratch_018 ground truth and predictions](images/test_scratch_018.jpg)

## Priorities before a factory pilot
Agree on physical-defect instance labeling, gather more independent bent/color/scratch examples, and validate by later production lot/shift. Compare higher resolution or a controlled part crop using a new validation-only experiment; do not choose changes from the final test. Evaluate false rejects at realistic prevalence, confidence calibration, tail latency and reject-actuator timing. The app reports REVIEW or NO_DEFECT_DETECTED rather than certifying PASS. For the limited-data and latency scenarios, the following answers describe broader options rather than unrun experiments.

## A. Only 30 defective images and 3,000 good images
I would first clarify the cost of a missed defect versus a false reject, the expected defect sizes, and whether those 30 images cover independent physical parts, lighting conditions and defect mechanisms. I would audit labels and remove duplicates, then reserve evaluation images by capture session or production lot. A random image split could leak almost identical views across training and evaluation. With very few positives I would report counts and uncertainty, including per-defect recall, rather than treating a high aggregate score as reliable.

The good images are useful data. I would establish a normal-only anomaly-detection baseline, for example a pretrained feature extractor with a compact normal-reference bank or a teacher/student normal model. The 30 defect images help assess localization, understand uncovered modes and tune an alarm threshold on validation data. A compact detector fine-tuned with conservative augmentation is a second baseline if localized defect examples are sufficient. Neither method can be assumed to cover unseen defect types. If an ensemble is considered, it must fit the measured runtime and reduce costly errors in independent evaluation.

I would augment only training images using physically plausible changes. Strong color shifts could hide color flaws; copied synthetic defects may introduce artifacts, so they should supplement genuine data and be validated separately. Extra views of one part remain grouped. I would collect additional real defects deliberately through controlled production sampling, ask an inspector to review uncertain and novel examples, and log the reason for each rejection. The operating threshold should reflect the customer’s miss/false-reject costs and review capacity. Evaluation must include a realistic stream of good parts: even a low false-positive rate can overwhelm the line when good parts dominate. Before automatic rejection, I would use a monitored pilot with human review, documented capture controls and a rollback plan.

## B. Good test scores, then missed defects after two weeks
1. **Contain and verify.** Preserve missed-part images, model outputs, timestamps and line settings; route uncertain production to manual inspection. Confirm that the failures are real defects, not changed acceptance criteria or label mistakes. The immediate aim is to prevent further escapes while preserving evidence.
2. **Inspect acquisition first.** Check focus, motion blur, exposure, gain, illumination, reflections, lens contamination, camera position and trigger timing. Compare current raw frames to training and commissioning frames. A mechanical or lighting change can produce a large shift without a software change.
3. **Check the complete deployed path.** Verify model checksum/version, class mapping, threshold, ROI, resize/letterbox, color order, normalization and quantization. Re-run saved missed images through both the deployed runtime and reference implementation. Inspect frame drops, stale frames and whether detections reached the PLC/reject actuator in time. Good offline detection cannot compensate for a late physical reject.
4. **Check production changes and drift.** Ask about material suppliers, finish, part geometry, defect size/type, speed, temperature and camera replacement. Compare image distributions and results by lot, shift and defect type. Random commissioning tests may have underrepresented the new conditions.
5. **Relabel and quantify.** Build a reviewed recent-data set containing misses, random good samples and difficult non-defects. Estimate per-class recall and false rejects under the actual operating point. Keep an independent holdout from later batches; avoid training and testing on the same rescue examples.
6. **Fix the demonstrated cause, then validate.** Restore capture settings for acquisition drift, repair integration bugs, or retrain with representative new data. Revisit the threshold using validation costs rather than blindly lowering it. Validate accuracy, latency and reject timing together, then shadow/canary the change, version it and monitor ongoing escapes with a rollback path.

## C. 10 ms budget, 40 ms model
First define whether 40 ms is model execution or the complete image-to-decision path, and whether the 10 ms limit is latency per part or steady-state throughput. Measure batch-1 p50/p95 on the actual controller after warmup, separating acquisition, preprocessing, model, postprocessing and actuator communication. CPU model and thread contention matter. Batching may improve throughput but increases individual latency and is usually inappropriate for a strict per-part deadline.

I would remove avoidable overhead first: reuse the loaded session and buffers, avoid repeated image conversions, use optimized runtime graph execution and tune CPU threads. Then compare a smaller detector or backbone, a smaller input or a tightly controlled part ROI. These reduce compute, but smaller inputs can erase scratches and small bent regions, so measure per-class recall and false rejects again. Reducing postprocessing candidates may save time if NMS is dominant; it cannot fix a 40 ms convolution bottleneck.

Next consider supported acceleration: a controller NPU, GPU, or vendor runtime such as OpenVINO where the hardware supports it. INT8 post-training quantization needs representative calibration data drawn from training, including real capture variation, and independent accuracy/latency evaluation; FP16 helps only on hardware with efficient support. If post-training quantization damages sensitive defects, quantization-aware training or distillation into a smaller model may help. Pruning requires a runtime that realizes the reduced computation; sparse weights alone do not guarantee lower latency.

A staged system could use a cheap part/quality gate and run a detector only when needed, but the worst-case defective image must still meet the deadline, and the gate's false negatives count as system misses. Frame skipping likewise risks missing short-lived defects. If an optimized, validated pipeline still cannot meet the hard 10 ms bound, I would propose better hardware or a redesigned station with adequate buffering and an agreed timing budget, rather than claiming that average FPS satisfies a hard deadline. Every speed change needs a measured accuracy tradeoff and a documented fallback.


## Reproduction and AI assistance
Follow README.md. Code, frozen split hashes, per-run configuration, raw predictions, metrics, failure overlays, ONNX, environment versions and a local React inspection UI are included in the submission package. Real model/API function checks and test results are listed in docs/STATUS.md.

Codex assisted with implementation, protocol design, documentation, debugging, visual failure review and the UI. The initial online setup and GPU pipeline were run in a normal terminal because the agent session restricts networking. All numerical results come from saved experiment outputs, not simulated values. The applicant must review and understand the submitted code and explain its decisions in the follow-up.

Applicant: Hargun Madan, currently studying B.Tech AI & ML at MSRIT, expected graduation May 2028. Availability belongs in the email reply, not the model evaluation.

## Sources and licensing
- MVTec AD: https://www.mvtec.com/research-teaching/datasets/mvtec-ad
- Ultralytics YOLO11: https://docs.ultralytics.com/models/yolo11/
- ONNX export: https://docs.ultralytics.com/integrations/onnx/
- ONNX Runtime: https://onnxruntime.ai/docs/performance/

MVTec AD images and derivatives are CC BY-NC-SA 4.0, for non-commercial use. The five annotated images above derive from MVTec AD, by Paul Bergmann et al.; annotations/overlays were added for this evaluation and retain the dataset's attribution and license. Full dataset bytes are not bundled. Project code is distributed under AGPL-3.0; see LICENSE and THIRD_PARTY_NOTICES.md. This is an independent take-home project, not an official InSiSo product.
