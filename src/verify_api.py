"""Verify the real ONNX-backed route functions without network/event-loop access."""
import io
import json
from pathlib import Path
from fastapi import UploadFile
from starlette.datastructures import Headers
from src import api
from src.runtime import Detector


def main():
    api.detector = Detector('artifacts/metal-nut.onnx')
    checks = []
    assert api.health()['model_loaded']
    checks.append('health function: real ONNX model loaded')
    sample = Path('data/prepared/images/test/test_flip_021.png')
    if not sample.is_file():
        sample = Path('web/public/samples/flip.png')
    image = sample.read_bytes()
    upload = UploadFile(filename='part.png', file=io.BytesIO(image),
                        headers=Headers({'content-type': 'image/png'}))
    result = api.inspect(upload, confidence=.25)
    assert any(d['label'] == 'flip' for d in result['detections'])
    assert (result['width'], result['height']) == (700, 700)
    checks.append('inspection function: real upload decode, ONNX inference, coordinates and flip detection')
    assert 'onnx_test' in api.results()['experiments']
    checks.append('results function: measured experiment files served')
    assert Path('web/dist/index.html').is_file()
    checks.append('React production build exists')
    report = dict(status='passed', transport='Direct route functions with real ONNX; no HTTP transport',
                  checks=checks,
                  limitations='Socket and browser integration not verified in the restricted execution session.')
    Path('artifacts/integration_checks.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()

