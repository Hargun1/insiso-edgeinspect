"""CPU-only ONNX inference, shared by the benchmark and API. No torch dependency."""
import time
import cv2
import numpy as np
import onnxruntime as ort
from src.metrics import iou
from src.prepare import NAMES

class Detector:
    def __init__(self,path,threads=1):
        opts=ort.SessionOptions();opts.intra_op_num_threads=threads;opts.inter_op_num_threads=1
        self.session=ort.InferenceSession(str(path),sess_options=opts,providers=['CPUExecutionProvider'])
        self.input=self.session.get_inputs()[0]
        self.size=int(self.input.shape[2])
        if self.input.shape!=[1,3,self.size,self.size]:raise ValueError('Expected fixed batch-1 square NCHW model')

    def preprocess(self,image):
        h,w=image.shape[:2];r=min(self.size/w,self.size/h)
        nw,nh=round(w*r),round(h*r)
        left=(self.size-nw)//2;top=(self.size-nh)//2
        canvas=np.full((self.size,self.size,3),114,np.uint8)
        canvas[top:top+nh,left:left+nw]=cv2.resize(image,(nw,nh))
        tensor=np.ascontiguousarray(canvas[:,:,::-1].transpose(2,0,1)[None],dtype=np.float32)/255.
        return tensor,(r,left,top,w,h)

    def postprocess(self,raw,info,confidence=.25,nms=.5):
        r,left,top,w,h=info
        rows=raw[0][0].T # YOLO11 detect: [1, 4+classes, candidates], no objectness channel.
        if rows.shape[1]!=4+len(NAMES):raise ValueError(f'Unexpected ONNX output {rows.shape}')
        classes=rows[:,4:].argmax(1);scores=rows[:,4:].max(1)
        candidates=[]
        for row,c,s in zip(rows,classes,scores):
            if s<confidence:continue
            x,y,bw,bh=row[:4]
            # NMS must operate before boundary clipping, matching the reference decoder.
            box=[float((x-bw/2-left)/r),float((y-bh/2-top)/r),
                 float((x+bw/2-left)/r),float((y+bh/2-top)/r)]
            if box[2]>box[0] and box[3]>box[1]:
                candidates.append(dict(class_id=int(c),label=NAMES[int(c)],score=float(s),box=box))
        kept=[]
        for p in sorted(candidates,key=lambda p:-p['score'])[:1000]:
            if not any(q['class_id']==p['class_id'] and iou(p['box'],q['box'])>nms for q in kept):
                kept.append(p)
            if len(kept)>=300:break
        # Affine inversion preserves IoU; clipping does not, so clip after suppression.
        for detection in kept:
            box=detection['box']
            detection['box']=[min(float(w),max(0.,box[0])),min(float(h),max(0.,box[1])),
                              min(float(w),max(0.,box[2])),min(float(h),max(0.,box[3]))]
        return [d for d in kept if d['box'][2]>d['box'][0] and d['box'][3]>d['box'][1]]

    def predict(self,image,confidence=.25):
        t=time.perf_counter();tensor,info=self.preprocess(image);p=time.perf_counter()
        raw=self.session.run(None,{self.input.name:tensor});q=time.perf_counter()
        detections=self.postprocess(raw,info,confidence);end=time.perf_counter()
        return dict(detections=detections,width=info[3],height=info[4],
                    decision='REVIEW' if detections else 'NO_DEFECT_DETECTED',
                    timing_ms=dict(preprocess=(p-t)*1000,inference=(q-p)*1000,
                                   postprocess=(end-q)*1000,total=(end-t)*1000))

