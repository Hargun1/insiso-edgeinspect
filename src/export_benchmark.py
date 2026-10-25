import argparse,json,platform,shutil,subprocess
from pathlib import Path

def main():
    import cv2,numpy as np,onnx,onnxruntime as ort
    from ultralytics import YOLO
    from src.runtime import Detector
    p=argparse.ArgumentParser();p.add_argument('--model',required=True)
    p.add_argument('--images',default='data/prepared/images/val');p.add_argument('--threads',type=int,default=1)
    p.add_argument('--iterations',type=int,default=100);a=p.parse_args()
    if a.iterations<10:raise ValueError('Use >=10 benchmark iterations')
    out=Path('artifacts');out.mkdir(exist_ok=True)
    model=YOLO(a.model)
    exported=model.export(format='onnx',imgsz=320,batch=1,dynamic=False,simplify=False,opset=17,half=False,device='cpu')
    dest=out/'metal-nut.onnx';shutil.copy2(exported,dest)
    onnx.checker.check_model(onnx.load(str(dest)))
    detector=Detector(dest,a.threads)
    paths=sorted(Path(a.images).glob('*.png'))
    if not paths:raise ValueError('No benchmark images')
    images=[cv2.imread(str(p)) for p in paths[:20]]
    for i in range(20):detector.predict(images[i%len(images)])
    times=[detector.predict(images[i%len(images)])['timing_ms'] for i in range(a.iterations)]
    summary={k:dict(mean=float(np.mean([t[k] for t in times])),p50=float(np.percentile([t[k] for t in times],50)),
                    p95=float(np.percentile([t[k] for t in times],95))) for k in times[0]}
    # Compare decoded output of the selected weights and ONNX on representative validation images.
    comparisons=[]
    for path,image in zip(paths[:5],images[:5]):
        pt=model.predict(str(path),imgsz=320,conf=.25,iou=.5,device='cpu',verbose=False)[0]
        op=detector.predict(image)['detections']
        comparisons.append(dict(image=path.name,pytorch_detections=len(pt.boxes),onnx_detections=len(op)))
    cpu=platform.processor()
    if platform.system()=='Windows':
        try:cpu=subprocess.check_output(['powershell','-NoProfile','-Command','(Get-CimInstance Win32_Processor).Name'],text=True).strip()
        except (OSError,subprocess.SubprocessError):pass
    report=dict(status='measured',cpu=cpu,platform=platform.platform(),onnxruntime=ort.__version__,
                provider='CPUExecutionProvider',threads=a.threads,image_size=[320,320],batch=1,
                warmup=20,iterations=a.iterations,model_bytes=dest.stat().st_size,timing_ms=summary,
                excluded='Disk reading, image decoding, HTTP and UI rendering',
                parity_counts=comparisons,parity_note='Counts are a smoke check; evaluate ONNX on the frozen test set too.')
    (out/'benchmark.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

