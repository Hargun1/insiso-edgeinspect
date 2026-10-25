"""Convert MVTec metal_nut masks into a reproducible supervised YOLO split."""
import argparse,hashlib,json,random,shutil
from pathlib import Path
NAMES=['bent','color','flip','scratch']

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def mask_boxes(mask):
    import cv2
    n,_,stats,_=cv2.connectedComponentsWithStats((mask>0).astype('uint8'),8)
    # Every nonempty component is kept: do not silently delete tiny defects.
    return [[int(x),int(y),int(x+w),int(y+h)] for x,y,w,h,area in stats[1:] if area>0]

def prepare(root,out,seed=42):
    import cv2
    root,out=Path(root),Path(out)
    if (root/'metal_nut').is_dir():root=root/'metal_nut'
    if not (root/'ground_truth').is_dir():raise ValueError('Expected metal_nut/ground_truth directory')
    if out.exists():raise FileExistsError(f'{out} exists. Choose a new output to protect the frozen split.')
    groups={'good':sorted((root/'train/good').glob('*.png'))+sorted((root/'test/good').glob('*.png'))}
    groups.update({name:sorted((root/'test'/name).glob('*.png')) for name in NAMES})
    if any(len(v)<5 for v in groups.values()):raise ValueError('Incomplete dataset; each defect type needs >=5 images')
    rng=random.Random(seed); seen={}; items=[]
    for name,paths in groups.items():
        # Deduplicate BEFORE splitting. Also reject conflicting labels for identical bytes.
        unique=[]
        for p in paths:
            sha=digest(p)
            if sha in seen:
                if seen[sha]!=name:raise ValueError('Conflicting labels for duplicate image')
                continue
            seen[sha]=name;unique.append((p,sha))
        if len(unique)<5:raise ValueError(f'{name} needs >=5 unique images after deduplication')
        rng.shuffle(unique)
        a=max(1,int(.6*len(unique)));b=max(a+1,int(.8*len(unique)))
        for i,(p,sha) in enumerate(unique):
            split='train' if i<a else 'val' if i<b else 'test'
            img=cv2.imread(str(p))
            if img is None:raise ValueError(f'Unreadable image: {p}')
            h,w=img.shape[:2];boxes=[];mask_sha=None
            if name!='good':
                mp=root/'ground_truth'/name/(p.stem+'_mask.png')
                mask=cv2.imread(str(mp),0)
                if mask is None or mask.shape!=(h,w):raise ValueError(f'Invalid mask: {mp}')
                mask_sha=digest(mp);boxes=mask_boxes(mask)
                if not boxes:raise ValueError(f'Empty defective mask: {mp}')
            stem=p.relative_to(root).as_posix().replace('/','_').removesuffix('.png')
            ip=out/'images'/split/(stem+'.png');lp=out/'labels'/split/(stem+'.txt')
            ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(p,ip)
            labels=[]
            for x1,y1,x2,y2 in boxes:
                labels.append(f'{NAMES.index(name)} {(x1+x2)/2/w:.8f} {(y1+y2)/2/h:.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
            lp.write_text('\n'.join(labels),encoding='utf-8')
            items.append(dict(id=stem,source=p.relative_to(root).as_posix(),sha256=sha,mask_sha256=mask_sha,
                              split=split,category=name,width=w,height=h,boxes=boxes))
    out.mkdir(exist_ok=True)
    manifest=dict(seed=seed,protocol='Custom stratified supervised 60/20/20; NOT official MVTec anomaly protocol',
                  classes=NAMES,items=items)
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    (out/'dataset.yaml').write_text(f'path: {out.resolve().as_posix()}\ntrain: images/train\nval: images/val\ntest: images/test\nnames: '+json.dumps(NAMES)+'\n',encoding='utf-8')
    print(json.dumps({s:{n:sum(i['split']==s and i['category']==n for i in items) for n in groups}
                      for s in ['train','val','test']},indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--raw',required=True);p.add_argument('--out',default='data/prepared');p.add_argument('--seed',type=int,default=42)
    a=p.parse_args();prepare(a.raw,a.out,a.seed)

