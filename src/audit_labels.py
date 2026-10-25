"""Visual label audit and leakage/count summary; run before training."""
import argparse,json
from pathlib import Path

def main():
    import cv2,numpy as np
    p=argparse.ArgumentParser();p.add_argument('--data',default='data/prepared');a=p.parse_args()
    root=Path(a.data);m=json.loads((root/'manifest.json').read_text(encoding='utf-8-sig'))
    items=m['items'];seen={};tiny=[];counts={}
    for it in items:
        if it['sha256'] in seen:raise ValueError('Duplicate bytes across prepared items')
        seen[it['sha256']]=it['split']
        key=it['split']+'/'+it['category'];counts[key]=counts.get(key,0)+1
        for b in it['boxes']:
            if min((b[2]-b[0])*320/it['width'],(b[3]-b[1])*320/it['height'])<3:tiny.append(it['id'])
    cards=[]
    for category in ['good']+m['classes']:
        group=[i for i in items if i['category']==category and i['split']=='train'][:4]
        for it in group:
            im=cv2.imread(str(root/'images/train'/(it['id']+'.png')))
            for b in it['boxes']:
                x,y,x2,y2=b;cv2.rectangle(im,(x,y),(x2,y2),(40,230,100),2)
            card=cv2.resize(im,(220,220));card=cv2.copyMakeBorder(card,30,0,0,0,cv2.BORDER_CONSTANT,value=(25,30,35))
            cv2.putText(card,it['id'][:30],(6,20),cv2.FONT_HERSHEY_SIMPLEX,.35,(230,230,230),1)
            cards.append(card)
    while len(cards)%4:cards.append(np.zeros_like(cards[0]))
    grid=np.vstack([np.hstack(cards[i:i+4]) for i in range(0,len(cards),4)])
    out=Path('artifacts');out.mkdir(exist_ok=True);cv2.imwrite(str(out/'label_preview.jpg'),grid)
    report=dict(counts=counts,unique_images=len(seen),tiny_box_image_ids=sorted(set(tiny)),
                note='Exact-byte duplicate check only. Review contact sheet and tiny labels manually.')
    (out/'split_audit.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__':main()

