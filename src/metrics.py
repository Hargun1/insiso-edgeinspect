"""Dependency-free class-aware detection metrics at IoU 0.5."""
def iou(a, b):
    area = lambda x: max(0, x[2]-x[0])*max(0, x[3]-x[1])
    inter = max(0, min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
    union = area(a)+area(b)-inter
    return inter/union if union else 0.

def match(pred, truth, threshold=.5):
    used, pairs = set(), []
    for p in sorted(pred, key=lambda x: -x['score']):
        best = max(((iou(p['box'],g['box']),j) for j,g in enumerate(truth)
                    if j not in used and p['class_id']==g['class_id']), default=(0,-1))
        ok = best[1]>=0 and best[0]>=threshold
        if ok: used.add(best[1])
        pairs.append((p,ok))
    return pairs,[g for j,g in enumerate(truth) if j not in used]

def evaluate(records, names, confidence=.25):
    totals, ranked = [0]*len(names), [[] for _ in names]
    tp=fp=fn=0
    for r in records:
        for g in r['truth']: totals[g['class_id']]+=1
        pairs,_=match(r['predictions'],r['truth'])
        for p,ok in pairs: ranked[p['class_id']].append((p['score'],ok))
        pairs,misses=match([p for p in r['predictions'] if p['score']>=confidence],r['truth'])
        tp+=sum(ok for _,ok in pairs); fp+=sum(not ok for _,ok in pairs); fn+=len(misses)
    classes={}
    for c,name in enumerate(names):
        if not totals[c]:
            classes[name]={'instances':0,'ap50':None}; continue
        count=0; rec=[0.]; pre=[1.]
        for k,(_,ok) in enumerate(sorted(ranked[c],key=lambda x:-x[0]),1):
            count+=ok; rec.append(count/totals[c]); pre.append(count/k)
        rec.append(1.); pre.append(0.)
        for k in range(len(pre)-2,-1,-1): pre[k]=max(pre[k],pre[k+1])
        ap=sum((rec[k]-rec[k-1])*pre[k] for k in range(1,len(rec)))
        classes[name]={'instances':totals[c],'ap50':ap}
    aps=[v['ap50'] for v in classes.values() if v['ap50'] is not None]
    return dict(precision=tp/(tp+fp) if tp+fp else 0.,recall=tp/(tp+fn) if tp+fn else 0.,
                map50=sum(aps)/len(aps) if aps else None,tp=tp,fp=fp,fn=fn,
                confidence=confidence,iou=.5,images=len(records),per_class=classes,
                ap_method='all-points interpolated AP, class macro average')

