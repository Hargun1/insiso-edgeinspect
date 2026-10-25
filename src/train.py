"""Controlled fine-tuning: same split, seed and budget; only augmentation changes."""
import argparse
import json
import platform
import time
from pathlib import Path


def main():
    from ultralytics import YOLO
    import torch
    import ultralytics
    import ultralytics.utils.torch_utils as torch_utils

    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='data/prepared/dataset.yaml')
    parser.add_argument('--experiment', choices=['baseline', 'augmented'], required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--resume-from')
    args = parser.parse_args()
    torch_utils.NUM_THREADS = args.threads
    torch.set_num_threads(args.threads)
    # Strong hue changes could erase the color-flaw signal.
    augmentation = dict(degrees=15., translate=.05, scale=.15, fliplr=.5, flipud=.5,
                        hsv_h=0., hsv_s=.1, hsv_v=.1, mosaic=0., mixup=0.)
    if args.experiment == 'baseline':
        augmentation = {key: 0. for key in augmentation}
    config = dict(data=args.data, epochs=args.epochs, imgsz=320, batch=args.batch,
                  device=args.device, seed=42, deterministic=True, workers=0,
                  amp=False, patience=15, optimizer='AdamW', lr0=.001,
                  project='runs', name=args.experiment, exist_ok=True,
                  plots=True, **augmentation)
    from ultralytics import settings
    settings.update({'sync': False})
    if not args.resume_from and (Path('runs')/args.experiment/'provenance.json').is_file():
        raise FileExistsError('Completed run exists. Use a fresh project copy to reproduce without overwriting frozen results.')
    model = YOLO(args.resume_from or 'yolo11n.pt')
    started = time.time()
    if args.resume_from:
        model.train(resume=True, device=args.device, workers=0)
    else:
        model.train(**config)
    output = Path(model.trainer.save_dir)
    provenance = dict(config=config, elapsed_seconds_this_invocation=time.time()-started,
                      resume_from=args.resume_from, python=platform.python_version(),
                      torch=torch.__version__, ultralytics=ultralytics.__version__,
                      threads=args.threads, cuda=torch.cuda.is_available(),
                      device=torch.cuda.get_device_name(0) if torch.cuda.is_available()
                      else platform.processor())
    (output/'provenance.json').write_text(json.dumps(provenance, indent=2), encoding='utf-8')


if __name__ == '__main__':
    main()

