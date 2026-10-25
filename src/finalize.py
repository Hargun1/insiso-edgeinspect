"""Sequential final evaluation; never selects from final-test scores."""
import json,subprocess,sys
from pathlib import Path

def main():
    root=Path('artifacts');selection=json.loads((root/'selection.json').read_text(encoding='utf-8-sig'))
    if (root/'selected_test.json').exists() or (root/'onnx_test.json').exists():
        raise FileExistsError('Final test already accessed. Inspect completed artifacts; do not silently repeat tuning.')
    def run(module,*args):subprocess.run([sys.executable,'-m',module,*args],check=True)
    run('src.evaluate','--model',selection['model'],'--split','test','--name','selected_test',
        '--confidence',str(selection['confidence']))
    run('src.export_benchmark','--model',selection['model'])
    run('src.evaluate','--model','artifacts/metal-nut.onnx','--split','test','--name','onnx_test',
        '--confidence',str(selection['confidence']))
    lock=subprocess.check_output([sys.executable,'-m','pip','freeze'],text=True)
    (root/'environment-lock.txt').write_text(lock,encoding='utf-8')
    print('Final artifacts produced. Review failures, ONNX accuracy, and report before submitting.')

if __name__=='__main__':main()

