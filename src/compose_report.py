"""Build the report from measured artifacts and reviewed, real failures."""
import csv
import hashlib
import json
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
def load(name):
    return json.loads((ROOT/'artifacts'/(name+'.json')).read_text(encoding='utf-8-sig'))
def fmt(x): return f'{x:.4f}'

def main():
    docs = ROOT/'docs'
    images = docs/'images'
    images.mkdir(exist_ok=True)
    baseline, augmented = load('baseline_val'), load('augmented_val')
    selected, onnx, benchmark = load('selected_test'), load('onnx_test'), load('benchmark')
    names = ['test_bent_015','test_bent_023','test_color_004','test_color_013','test_scratch_018']
    reviewed = [
      ('test_bent_015','Miss: 4 components','No detection above 0.25, despite visible damage on the upper lug and left edge.',
       'The damage spans several disconnected mask components and scales. Limited pose/defect examples may leave this appearance uncovered; this is a hypothesis, not an observed internal cause.'),
      ('test_bent_023','Localization FP + miss','A bent prediction at confidence 0.420 covers only part of the upper-edge defect; IoU is 0.486, below the fixed 0.5 matching cutoff.',
       'The box follows the visually strongest notch but underestimates the annotated extent. This demonstrates why finding a defect-looking region is not sufficient for correct localization.'),
      ('test_color_004','Miss: 1 component','The dark blue patch beside the inner ring has no prediction above the operating threshold.',
       'The relatively small, dark patch blends into the inner-rim shadow and metallic texture. More representative dark-color flaws and higher-resolution validation are plausible next experiments.'),
      ('test_color_013','3 extra detections','The red patch is correctly detected as color at confidence 0.878, but two bent boxes and a second color box remain as false positives.',
       'The same high-contrast region triggers multiple class heads. Class-aware NMS intentionally keeps different classes; partial boxes also escape same-class suppression. This is classification and duplicate-counting failure, not a missed red patch.'),
      ('test_scratch_018','2 FPs + 1 miss','The upper scratch is matched at confidence 0.307, but higher-scoring color (0.753) and bent (0.614) boxes overlap it. The separate lower-left scratch is missed.',
       'A dark scar can resemble staining or deformation, while the second scar has different orientation and local texture. The pattern suggests class ambiguity and incomplete coverage across multiple defects.')
    ]
    rows=[]
    raw={x['id']:x for x in onnx['failures']}
    for name,kind,observation,hypothesis in reviewed:
        source=ROOT/'artifacts/failures/onnx_test'/(name+'.jpg')
        shutil.copy2(source,images/source.name)
        item=dict(id=name,type=kind,observation=observation,hypothesis=hypothesis,
                  image='docs/images/'+source.name,false_positives=raw[name]['false_positives'],
                  misses=raw[name]['misses'])
        rows.append(item)
    (ROOT/'artifacts/reviewed_failures.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    epochs={}
    for name in ['baseline','augmented']:
        records=list(csv.DictReader((ROOT/'runs'/name/'results.csv').open()))
        epochs[name]=len(records)
        shutil.copy2(ROOT/'runs'/name/'provenance.json',ROOT/'artifacts'/(name+'_provenance.json'))
    timings=benchmark['timing_ms']
    size=benchmark['model_bytes']/1_000_000
    old=(docs/'REPORT.md').read_text(encoding='utf-8-sig')
    answer_file=docs/'WRITTEN_ANSWERS.md'
    if answer_file.is_file():
        answers=answer_file.read_text(encoding='utf-8')[len('# Written scenario answers\n\n'):]
    else:
        answers=old[old.index('## A. Only 30'):old.index('## AI assistance')]
    text=f"""# Metal Nut Defect Detection
Hargun Madan | InSiSo computer-vision internship take-home | 25 October 2026

## Result and practical limits
A four-class YOLO11n detector was trained on a custom supervised MVTec AD metal_nut split and exported to a local CPU ONNX runtime. Augmentation raised validation mAP@0.5 from {fmt(baseline['metrics']['map50'])} to {fmt(augmented['metrics']['map50'])}. The selected ONNX model achieved **mAP@0.5 {fmt(onnx['metrics']['map50'])}, precision {fmt(onnx['metrics']['precision'])}, recall {fmt(onnx['metrics']['recall'])}** at confidence 0.25 and IoU 0.5 on 69 held-out images.

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
Both arms start from the same official COCO-pretrained yolo11n.pt, seed 42, input 320x320, batch 4, FP32, AdamW at initial learning rate 0.001, at most 30 epochs and patience 15. Training ran on an NVIDIA GeForce GTX 1650 (4 GB). The first CPU attempt was interrupted during epoch one without a usable checkpoint; it is not an experiment result. Before comparing validation results, the completed runs were configured with the same GPU budget. The baseline stopped after {epochs['baseline']} epochs and augmentation after {epochs['augmented']}.

| Run | Split | Precision | Recall | mAP@0.5 | TP / FP / FN |
|---|---|---:|---:|---:|---|
"""
    for label,obj in [('Baseline',baseline),('Mild augmentation',augmented),('Selected PyTorch',selected),('Selected ONNX',onnx)]:
        m=obj['metrics']
        text+=f"| {label} | {obj['split']} | {fmt(m['precision'])} | {fmt(m['recall'])} | {fmt(m['map50'])} | {m['tp']} / {m['fp']} / {m['fn']} |\n"
    text+="""
Precision and recall are micro-aggregated detection counts at confidence 0.25. Matching is one-to-one, class-aware and uses IoU >=0.5. AP uses all-points interpolated area under the ranked precision-recall curve, with predictions collected down to confidence 0.001; this is not COCO's 101-point convention. mAP is the macro average over classes with ground truth. Good images count toward false positives. The framework selects the checkpoint within each run using its validation fitness; the independent evaluator selects between the two best checkpoints by validation mAP@0.5, with ties favoring baseline. The diagnostic validation threshold sweep was not used to change the predeclared operating threshold.

### Did the improvement work?
The changed factor is modest augmentation: +/-15-degree rotation, translation 0.05, scale 0.15, horizontal/vertical flips at probability 0.5, and small saturation/value changes. Mosaic and mixup stay off. Strong hue shifts are avoided because color is a defect class. Flip here means the nut is physically turned over; reflecting an image is not the same physical operation, but its realism still deserves validation.

Aggregate validation AP improved by **0.0735 absolute (7.35 percentage points)**, so the augmented checkpoint was selected before final-test evaluation. This is not a uniform win: validation precision fell from 0.6250 to 0.4783 while recall rose from 0.3125 to 0.3438. Bent and color AP increased, scratch AP fell from 0.1715 to 0.0400, and flip AP remained 1.0. The small validation sample and a single seed limit confidence in any causal claim. The augmentation may broaden pose/color coverage while harming subtle scratch cues; the class-specific results support investigating that tradeoff, not claiming a universal improvement.

### Selected ONNX per-class test AP
| Class | Annotated components | AP@0.5 |
|---|---:|---:|
"""
    for name,m in onnx['metrics']['per_class'].items():
        text+=f"| {name} | {m['instances']} | {fmt(m['ap50'])} |\n"
    text+=f"""
Flip masks are large and visually distinct; five positive examples are insufficient to infer production reliability. The aggregate score is partly driven by this easier class. Bent/color localization remain weak, and no test-driven retraining or threshold selection was performed.

## Export, runtime parity and CPU speed
The selected weights export to **FP32 ONNX, opset 17, fixed batch 1, 320x320**, {size:.2f} MB. ONNX checker passes. Deployment uses RGB conversion, letterboxing, division by 255 and class-aware NMS on CPUExecutionProvider.

Reference-versus-runtime checking found a decoder error: clipping boxes before NMS changed IoU and suppressed extra boundary boxes. I moved clipping after NMS, added a boundary regression test and rechecked deployment output. Training, checkpoint selection, labels and confidence were unchanged. Original decoder results are retained as *_decoder_v1.json. The final ONNX precision, recall and mAP now match the PyTorch test metrics. This is a deployment correction, not a model-improvement experiment or test-set threshold tuning.

CPU: **{benchmark['cpu']}**, Windows 11, ONNX Runtime {benchmark['onnxruntime']}, one intra-op thread and one inter-op thread. Twenty warmup iterations precede 100 measurements over preloaded validation images. The following is the current-runtime benchmark, not a GPU timing.

| Stage | Mean ms | p50 ms | p95 ms |
|---|---:|---:|---:|
"""
    for name,m in timings.items():
        text+=f"| {name} | {m['mean']:.2f} | {m['p50']:.2f} | {m['p95']:.2f} |\n"
    text+="""
Disk reading, decoding, HTTP, display and physical actuator timing are excluded. Laptop load and thermal state affect timings; the earlier benchmark is retained for traceability and is not presented as a controlled speedup. The model does not meet a 10 ms CPU budget on this machine.

## Five visually reviewed held-out failures
Green boxes are ground truth, red boxes are false positives, and blue/cyan boxes are matched predictions. The causes below are reasoned hypotheses based on visible evidence; the measured error type is separate from the hypothesis.
"""
    for i,(name,kind,observation,hypothesis) in enumerate(reviewed,1):
        text+=f"\n### {i}. {name} - {kind}\n{observation}\n\nLikely explanation: {hypothesis}\n\n![{name} ground truth and predictions](images/{name}.jpg)\n"
    text+="""
## Priorities before a factory pilot
Agree on physical-defect instance labeling, gather more independent bent/color/scratch examples, and validate by later production lot/shift. Compare higher resolution or a controlled part crop using a new validation-only experiment; do not choose changes from the final test. Evaluate false rejects at realistic prevalence, confidence calibration, tail latency and reject-actuator timing. The app reports REVIEW or NO_DEFECT_DETECTED rather than certifying PASS. For the limited-data and latency scenarios, the following answers describe broader options rather than unrun experiments.

"""
    text+=answers
    text+="""
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
"""
    (docs/'REPORT.md').write_text(text,encoding='utf-8')
    shutil.copy2(ROOT/'runs/augmented/weights/best.pt',ROOT/'artifacts/best-model.pt')
    manifest={}
    for name in ['metal-nut.onnx','best-model.pt']:
        path=ROOT/'artifacts'/name
        manifest[name]=dict(bytes=path.stat().st_size,sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (ROOT/'artifacts/model_checksums.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (docs/'MODEL_CARD.md').write_text(f"""# Model card
- Task: four-class metal_nut defect localization; box output from mask components.
- Architecture: Ultralytics YOLO11n, COCO-pretrained then fine-tuned.
- Selected run: augmented, selected using validation mAP50 only.
- Dataset: MVTec AD metal_nut; custom supervised split, seed 42; not official anomaly protocol.
- Input: RGB, letterbox 320x320, float32 /255, fixed batch 1.
- Output: raw [1, 8, 2100] detector tensor; four box coordinates and four class scores. Runtime applies class-aware NMS 0.5 and confidence 0.25.
- Classes: 0 bent; 1 color; 2 flip; 3 scratch.
- Test: precision {fmt(onnx['metrics']['precision'])}, recall {fmt(onnx['metrics']['recall'])}, mAP50 {fmt(onnx['metrics']['map50'])}; 69 images.
- CPU model execution: {timings['inference']['mean']:.2f} ms mean, {timings['inference']['p95']:.2f} ms p95 on {benchmark['cpu']}.
- Intended use: non-commercial take-home demonstration and research.
- Limitations: small positive sample, fragmented mask labels, limited appearance coverage, missed defects, extra and wrong-class boxes, no unknown-defect guarantee, no production calibration/actuator validation.
- Use REVIEW / NO_DEFECT_DETECTED as advisory outputs. A negative result does not certify a safe or acceptable part.
- Training source, versions, checksums and measured outputs: artifacts/.
- Dataset/derived-image license: CC BY-NC-SA 4.0. Framework/project code: AGPL-3.0; see notices.
""",encoding='utf-8')
    print('Report, five reviewed images, model card and model checksums written.')


if __name__=='__main__':
    main()

