"""Validate that a portable submission contains real metrics, models and five reviewed failures."""
import hashlib
import json
from pathlib import Path


def check(root):
    issues = []
    required = ['artifacts/metal-nut.onnx', 'artifacts/best-model.pt',
                'artifacts/selection.json', 'artifacts/benchmark.json',
                'artifacts/baseline_val.json', 'artifacts/augmented_val.json',
                'artifacts/selected_test.json', 'artifacts/onnx_test.json',
                'artifacts/split_audit.json', 'artifacts/split_manifest.json',
                'artifacts/environment-lock.txt', 'artifacts/reviewed_failures.json',
                'artifacts/model_checksums.json', 'docs/REPORT.md', 'LICENSE']
    for name in required:
        path = root/name
        if not path.is_file() or path.stat().st_size == 0:
            issues.append('Missing: '+name)
    report = root/'docs/REPORT.md'
    if report.is_file() and 'pending' in report.read_text(encoding='utf-8-sig').lower():
        issues.append('Report contains unfinished work')
    for name, split in [('baseline_val','val'), ('augmented_val','val'),
                        ('selected_test','test'), ('onnx_test','test')]:
        path = root/'artifacts'/(name+'.json')
        if not path.is_file():
            continue
        try:
            result = json.loads(path.read_text(encoding='utf-8-sig'))
            metrics = result['metrics']
            if result['status'] != 'measured' or result['split'] != split:
                raise ValueError('incorrect status/split')
            if metrics['images'] <= 0 or metrics['map50'] is None:
                raise ValueError('missing measurements')
            for key in ['precision', 'recall', 'map50']:
                if not 0 <= metrics[key] <= 1:
                    raise ValueError('metric outside [0,1]')
        except (ValueError, KeyError, TypeError) as error:
            issues.append(f'Invalid {name}: {error}')
    reviewed = root/'artifacts/reviewed_failures.json'
    if reviewed.is_file():
        try:
            cases = json.loads(reviewed.read_text(encoding='utf-8-sig'))
            if len({case['id'] for case in cases}) < 5:
                issues.append('Fewer than five distinct reviewed cases')
            for case in cases:
                if not case['observation'] or not case['hypothesis'] or not (root/case['image']).is_file():
                    issues.append('Incomplete failure review: '+case['id'])
        except (ValueError, KeyError, TypeError) as error:
            issues.append('Invalid failure review: '+str(error))
    checksums = root/'artifacts/model_checksums.json'
    if checksums.is_file():
        try:
            for name, expected in json.loads(checksums.read_text(encoding='utf-8-sig')).items():
                path = root/'artifacts'/name
                if path.is_file() and hashlib.sha256(path.read_bytes()).hexdigest() != expected['sha256']:
                    issues.append('Model checksum mismatch: '+name)
        except (ValueError, KeyError, TypeError) as error:
            issues.append('Invalid checksums: '+str(error))
    return issues


if __name__ == '__main__':
    problems = check(Path('.'))
    for problem in problems:
        print('[INCOMPLETE] '+problem)
    if problems:
        raise SystemExit(1)
    print('PASS: portable artifacts, measured metrics, model checksums and five reviewed failures.')
    print('GitHub publication and the email reply are separate external submission steps.')

