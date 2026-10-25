"""Download data mirror and official pretrained weights. Run on a network-enabled terminal."""
import argparse,concurrent.futures,hashlib,json,time,urllib.request
from pathlib import Path,PurePosixPath
REPO='MSherbinii/mvtec-ad-metal-nut'
BASE='https://huggingface.co'
AGENT={'User-Agent':'EdgeInspect-take-home/0.1'}

def get_json(url):
    with urllib.request.urlopen(urllib.request.Request(url,headers=AGENT),timeout=60) as r:
        return json.load(r)

def download(url,dest,size=None):
    dest=Path(dest);dest.parent.mkdir(parents=True,exist_ok=True)
    if dest.is_file() and (size is not None and dest.stat().st_size==size):return
    tmp=dest.with_suffix(dest.suffix+'.partial')
    for attempt in range(3):
        try:
            with urllib.request.urlopen(urllib.request.Request(url,headers=AGENT),timeout=90) as r,tmp.open('wb') as f:
                while chunk:=r.read(1024*1024):f.write(chunk)
            if size is not None and tmp.stat().st_size!=size:raise ValueError('Download size mismatch')
            if not tmp.stat().st_size:raise ValueError('Empty download')
            tmp.replace(dest);return
        except Exception:
            if attempt==2:raise
            time.sleep(attempt+1)

def main():
    p=argparse.ArgumentParser();p.add_argument('--raw',default='data/raw');p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();raw=Path(a.raw);raw.mkdir(parents=True,exist_ok=True)
    print('MVTec AD data is CC BY-NC-SA 4.0. Official source: https://www.mvtec.com/research-teaching/datasets/mvtec-ad',flush=True)
    print('Downloading the metal_nut-only public mirror; origin and immutable revision are recorded.',flush=True)
    meta=get_json(f'{BASE}/api/datasets/{REPO}')
    revision=meta['sha']
    if not revision.isalnum():raise ValueError('Invalid revision')
    # This category has <1000 entries. Abort rather than silently accepting pagination.
    tree=get_json(f'{BASE}/api/datasets/{REPO}/tree/{revision}?recursive=true&limit=1000')
    if len(tree)>=1000:raise ValueError('Unexpected large listing. Use the official archive instead.')
    files=[]
    for obj in tree:
        path=PurePosixPath(obj['path'])
        if obj['type']!='file' or path.parts[0]!='metal_nut' or '..' in path.parts:continue
        if path.suffix not in ['.png','.txt']:continue
        files.append(obj)
    images=[x for x in files if x['path'].endswith('.png')]
    if len(images)<300:raise ValueError('Unexpected/incomplete category listing')
    print(f'{len(files)} files; revision {revision}.',flush=True)
    def work(obj):
        download(f'{BASE}/datasets/{REPO}/resolve/{revision}/{obj["path"]}',raw/obj['path'],obj['size'])
        sha=hashlib.sha256((raw/obj['path']).read_bytes()).hexdigest()
        lfs=obj.get('lfs') or {}
        if lfs.get('oid') and sha!=lfs['oid']:raise ValueError('LFS checksum mismatch: '+obj['path'])
        return dict(path=obj['path'],sha256=sha,size_bytes=obj['size'])
    done=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1,min(a.workers,8))) as pool:
        for i,record in enumerate(pool.map(work,files),1):
            done.append(record)
            if i%25==0:print(f'{i}/{len(files)} complete',flush=True)
    (raw/'download_manifest.json').write_text(json.dumps(dict(repository=REPO,revision=revision,
        official_source='https://www.mvtec.com/research-teaching/datasets/mvtec-ad',
        license='CC-BY-NC-SA-4.0',files=done),indent=2),encoding='utf-8')
    weight=Path('yolo11n.pt')
    url='https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n.pt'
    if not weight.is_file():download(url,weight)
    manifest=dict(url=url,sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),size_bytes=weight.stat().st_size)
    Path('artifacts').mkdir(exist_ok=True)
    Path('artifacts/pretrained_source.json').write_text(json.dumps(manifest,indent=2),encoding='utf-8')
    print('Assets ready. Return to Codex to audit labels and begin training.',flush=True)

if __name__=='__main__':main()

