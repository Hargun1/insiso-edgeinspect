"""Create a portable, allowlisted source/model submission archive."""
import hashlib
import json
import zipfile
from pathlib import Path
from src.check_submission import check
ROOT = Path(__file__).resolve().parents[1]


def main():
    issues = check(ROOT)
    if issues:
        raise RuntimeError('\n'.join(issues))
    files = []
    for name in ['README.md','LICENSE','THIRD_PARTY_NOTICES.md','requirements.txt',
                 'requirements-runtime.txt','setup.ps1','train-gpu.ps1','start.ps1','.gitignore']:
        path = ROOT/name
        if path.is_file():
            files.append(path)
    for folder in ['src','tests','docs','web/src','web/public','web/dist']:
        for path in (ROOT/folder).rglob('*'):
            if path.is_file() and '__pycache__' not in path.parts and path.suffix != '.pyc':
                files.append(path)
    for name in ['index.html','package.json','package-lock.json','vite.config.js']:
        files.append(ROOT/'web'/name)
    for path in (ROOT/'artifacts').iterdir():
        if path.is_file() and path.suffix in ['.json','.onnx','.pt'] and path.name != 'package_verification.json':
            files.append(path)
    files += [ROOT/'artifacts/environment-lock.txt',ROOT/'artifacts/tests-final.log']
    output = ROOT/'InSiSo-EdgeInspect-Submission.zip'
    entries = sorted({path.relative_to(ROOT).as_posix(): path for path in files}.items())
    if any('node_modules/' in name or '.venv/' in name or '.cache/' in name for name,_ in entries):
        raise ValueError('Unexpected dependency/private directory in submission')
    with zipfile.ZipFile(output,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=6) as archive:
        for name,path in entries:
            archive.write(path,name)
    with zipfile.ZipFile(output) as archive:
        corrupt = archive.testzip()
        if corrupt:
            raise ValueError('Corrupt entry: '+corrupt)
        for name,expected in json.loads((ROOT/'artifacts/model_checksums.json').read_text()).items():
            if hashlib.sha256(archive.read('artifacts/'+name)).hexdigest() != expected['sha256']:
                raise ValueError('Packaged model checksum mismatch')
        for name in ['README.md','docs/REPORT.md','artifacts/metal-nut.onnx','start.ps1','web/dist/index.html']:
            if name not in archive.namelist():
                raise ValueError('Missing packaged entry '+name)
    summary = dict(archive=output.name,files=len(entries),bytes=output.stat().st_size,
                   sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                   integrity_check='passed; packaged model hashes match')
    (ROOT/'artifacts/package_verification.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()

