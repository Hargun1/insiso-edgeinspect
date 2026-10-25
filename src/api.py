"""Local inspection API. Bind to loopback; images stay on this machine."""
import io,json,os,threading,warnings
from contextlib import asynccontextmanager
from pathlib import Path
from fastapi import FastAPI,UploadFile,File,HTTPException,Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

ROOT=Path(__file__).resolve().parents[1]
MODEL=Path(os.environ.get('MODEL_PATH',str(ROOT/'artifacts/metal-nut.onnx')))
detector=None
lock=threading.Lock()

@asynccontextmanager
async def lifespan(app):
    global detector
    if MODEL.is_file():
        from src.runtime import Detector
        detector=Detector(MODEL)
    yield

app=FastAPI(title='EdgeInspect — Metal Nut Inspection',lifespan=lifespan)
app.add_middleware(CORSMiddleware,allow_origins=['http://localhost:5173','http://127.0.0.1:5173'],
                   allow_methods=['GET','POST'],allow_headers=['Content-Type'])

@app.get('/api/health')
def health():
    return dict(model_loaded=detector is not None,model='YOLO11n / ONNX',provider='CPUExecutionProvider',
                cloud_inference=False,status='ready' if detector else 'awaiting_trained_model')

@app.get('/api/results')
def results():
    data={}
    for name in ['baseline_val','augmented_val','selected_test','onnx_test','benchmark']:
        path=ROOT/'artifacts'/(name+'.json')
        if path.is_file():data[name]=json.loads(path.read_text(encoding='utf-8-sig'))
    return dict(status='measured' if data else 'pending_experiments',experiments=data)

@app.post('/api/inspect')
def inspect(file:UploadFile=File(...),confidence:float=Query(.25,ge=.01,le=.99)):
    if detector is None:raise HTTPException(503,'No trained ONNX model. Complete training and export first.')
    if file.content_type not in ['image/png','image/jpeg']:raise HTTPException(415,'Use a PNG or JPEG image.')
    raw=file.file.read(10*1024*1024+1)
    if len(raw)>10*1024*1024:raise HTTPException(413,'Image must be at most 10 MiB.')
    from PIL import Image,ImageOps,UnidentifiedImageError
    import numpy as np
    try:
        with warnings.catch_warnings():
            warnings.simplefilter('error',Image.DecompressionBombWarning)
            image=Image.open(io.BytesIO(raw))
            if image.width*image.height>16_000_000:raise ValueError('Maximum decoded image size is 16 megapixels')
            image=ImageOps.exif_transpose(image).convert('RGB')
            bgr=np.asarray(image)[:,:,::-1].copy()
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombWarning,Image.DecompressionBombError) as e:
        raise HTTPException(400,'Invalid image or image exceeds decode limit.') from e
    with lock:return detector.predict(bgr,confidence)

# One-origin production preview after npm run build.
if (ROOT/'web/dist').is_dir():
    app.mount('/',StaticFiles(directory=ROOT/'web/dist',html=True),name='web')

