"""Immediate sequential experiment job with recoverable checkpoints and progress file."""
import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', default='auto')
    args = parser.parse_args()
    import torch
    device = ('0' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    batch = '4' if device != 'cpu' else '8'
    print(f'Training device: {device}; batch: {batch}', flush=True)
    root = Path(__file__).resolve().parents[1]
    artifacts = root/'artifacts'
    artifacts.mkdir(exist_ok=True)
    status = artifacts/'pipeline_status.json'

    def stage(name, state='running', **extra):
        status.write_text(json.dumps(dict(stage=name, state=state,
            updated_at=datetime.now(timezone.utc).isoformat(), **extra), indent=2), encoding='utf-8')
        print(name, state, flush=True)

    def run(module, *args):
        stage(module + ' ' + ' '.join(args))
        subprocess.run([sys.executable, '-u', '-m', module, *args], cwd=root, check=True)

    try:
        if not (root/'data/prepared/manifest.json').is_file():
            run('src.prepare', '--raw', 'data/raw/metal_nut')
        if not (artifacts/'split_audit.json').is_file():
            run('src.audit_labels')
        for name in ['baseline', 'augmented']:
            folder = root/'runs'/name
            checkpoint = folder/'weights/last.pt'
            if not (folder/'provenance.json').is_file():
                args = ['--experiment', name, '--device', device, '--batch', batch]
                if checkpoint.is_file():
                    args += ['--resume-from', str(checkpoint)]
                run('src.train', *args)
            if not (artifacts/(name+'_val.json')).is_file():
                run('src.evaluate', '--model', f'runs/{name}/weights/best.pt', '--name', name+'_val')
        if not (artifacts/'selection.json').is_file():
            run('src.select')
        selection = json.loads((artifacts/'selection.json').read_text(encoding='utf-8-sig'))
        if not (artifacts/'selected_test.json').is_file():
            run('src.evaluate', '--model', selection['model'], '--split', 'test',
                '--name', 'selected_test', '--confidence', str(selection['confidence']))
        if not (artifacts/'benchmark.json').is_file():
            run('src.export_benchmark', '--model', selection['model'])
        if not (artifacts/'onnx_test.json').is_file():
            run('src.evaluate', '--model', 'artifacts/metal-nut.onnx', '--split', 'test',
                '--name', 'onnx_test', '--confidence', str(selection['confidence']))
        lock = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
        (artifacts/'environment-lock.txt').write_text(lock, encoding='utf-8')
        import shutil
        shutil.copy2(root/'data/prepared/manifest.json', artifacts/'split_manifest.json')
        stage('experiments_complete', state='complete')
    except Exception as error:
        stage('interrupted_or_failed', state='failed', error=str(error))
        raise


if __name__ == '__main__':
    main()

