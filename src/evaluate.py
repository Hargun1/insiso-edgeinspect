"""Save raw predictions, independent AP, operational errors, and real error overlays."""
import argparse,json
from pathlib import Path
from src.metrics import evaluate,match
from src.prepare import NAMES

def run():
    import cv2
    from ultralytics import YOLO
    p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--data',default='data/prepared')
    p.add_argument('--split',choices=['val','test'],default='val');p.add_argument('--name',required=True)
    p.add_argument('--confidence',type=float,default=.25);a=p.parse_args()
    if not 0<a.confidence<1:raise ValueError('Confidence must be in (0,1)')
    root=Path(a.data);manifest=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    is_onnx=Path(a.model).suffix.lower()=='.onnx'
    if is_onnx:
        from src.runtime import Detector
        model=Detector(a.model)
    else:
        model=YOLO(a.model)
    records=[]
    for item in manifest['items']:
        if item['split']!=a.split:continue
        path=root/'images'/a.split/(item['id']+'.png')
        truth=[dict(class_id=NAMES.index(item['category']),box=b) for b in item['boxes']]
        if is_onnx:
            pred=model.predict(cv2.imread(str(path)),confidence=.001)['detections']
        else:
            r=model.predict(str(path),imgsz=320,conf=.001,iou=.5,max_det=300,device='cpu',verbose=False)[0]
            pred=[dict(class_id=int(c),score=float(s),box=b.tolist()) for b,c,s in
                  zip(r.boxes.xyxy.cpu().numpy(),r.boxes.cls.cpu().numpy(),r.boxes.conf.cpu().numpy())]
        records.append(dict(id=item['id'],category=item['category'],image=str(path),truth=truth,predictions=pred))
    out=Path('artifacts');out.mkdir(exist_ok=True)
    metrics=evaluate(records,NAMES,a.confidence)
    # Validation-only threshold sweep. Never choose a threshold from test scores.
    sweep=[]
    if a.split=='val':
        for t in [.1,.15,.2,.25,.3,.4,.5,.6,.7]:
            m=evaluate(records,NAMES,t);sweep.append({k:m[k] for k in ['confidence','precision','recall','fp','fn']})
    errors=[]
    dest=out/'failures'/a.name;dest.mkdir(parents=True,exist_ok=True)
    for r in records:
        pairs,misses=match([v for v in r['predictions'] if v['score']>=a.confidence],r['truth'])
        fps=[v for v,ok in pairs if not ok]
        if not fps and not misses:continue
        im=cv2.imread(r['image'])
        for g in r['truth']:
            x1,y1,x2,y2=map(int,g['box']);cv2.rectangle(im,(x1,y1),(x2,y2),(70,210,70),2)
        for pr,ok in pairs:
            x1,y1,x2,y2=map(int,pr['box']);color=(230,160,0) if ok else (40,40,240)
            cv2.rectangle(im,(x1,y1),(x2,y2),color,2)
            cv2.putText(im,f"{NAMES[pr['class_id']]} {pr['score']:.2f}",(max(0,x1),max(15,y1-5)),
                        cv2.FONT_HERSHEY_SIMPLEX,.5,color,1)
        cv2.imwrite(str(dest/(r['id']+'.jpg')),im)
        errors.append(dict(id=r['id'],category=r['category'],false_positives=len(fps),misses=len(misses),
                           image=f'failures/{a.name}/{r["id"]}.jpg',
                           analysis='PENDING visual review: describe observed evidence; state causal explanations as hypotheses.'))
    result=dict(status='measured',name=a.name,split=a.split,model=a.model,metrics=metrics,
                threshold_sweep=sweep,failures=errors)
    (out/(a.name+'.json')).write_text(json.dumps(result,indent=2),encoding='utf-8')
    (out/('predictions_'+a.name+'.json')).write_text(json.dumps(records,indent=2),encoding='utf-8')
    print(json.dumps(metrics,indent=2));print(f'{len(errors)} genuine error images saved; visually review at least five.')

if __name__=='__main__':run()

