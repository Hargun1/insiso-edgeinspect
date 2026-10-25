"""Freeze experiment selection before final test access."""
import json
from pathlib import Path

def choose(baseline,augmented):
    for x in [baseline,augmented]:
        if x.get('split')!='val' or x.get('status')!='measured':raise ValueError('Measured validation results required')
        if x['metrics']['map50'] is None:raise ValueError('Missing validation AP')
    return augmented if augmented['metrics']['map50']>baseline['metrics']['map50'] else baseline

def main():
    root=Path('artifacts');target=root/'selection.json'
    if target.exists():raise FileExistsError('Selection already frozen. Do not tune on the final test.')
    a=json.loads((root/'baseline_val.json').read_text(encoding='utf-8-sig'))
    b=json.loads((root/'augmented_val.json').read_text(encoding='utf-8-sig'))
    best=choose(a,b)
    selection=dict(model=best['model'],experiment=best['name'],criterion='Highest validation mAP50; tie -> baseline',
                   confidence=.25,validation_map50=best['metrics']['map50'])
    with target.open('x',encoding='utf-8') as f:json.dump(selection,f,indent=2)
    print(json.dumps(selection,indent=2))

if __name__=='__main__':main()

