"""Keep third-party runtime settings within the project workspace."""
import os
from pathlib import Path
_CACHE=Path(__file__).resolve().parents[1]/'.cache'
for key,folder in [('YOLO_CONFIG_DIR','yolo'),('MPLCONFIGDIR','matplotlib'),('TORCH_HOME','torch')]:
    target=Path(os.environ.setdefault(key,str(_CACHE/folder)))
    target.mkdir(parents=True,exist_ok=True)
# Ultralytics appends this name and checks existence before checking write permission.
(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True)

