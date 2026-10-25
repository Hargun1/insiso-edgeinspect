import React,{useEffect,useRef,useState} from 'react';
import{createRoot}from'react-dom/client';
import './style.css';

function App(){
 const[tab,setTab]=useState('Inspect'),[health,setHealth]=useState(null),[results,setResults]=useState(null),
 [file,setFile]=useState(null),[url,setUrl]=useState(''),[prediction,setPrediction]=useState(null),
 [confidence,setConfidence]=useState(.25),[busy,setBusy]=useState(false),[error,setError]=useState(''),
 [history,setHistory]=useState([]);
 const input=useRef(null),abort=useRef(null),requestId=useRef(0);
 useEffect(()=>{let active=true;const load=()=>Promise.all([
 fetch('/api/health').then(r=>{if(!r.ok)throw Error();return r.json()}),
 fetch('/api/results').then(r=>{if(!r.ok)throw Error();return r.json()})
 ]).then(([h,r])=>{if(active){setHealth(h);setResults(r)}}).catch(()=>{if(active)setHealth({model_loaded:false,status:'API offline'})});
 load();const timer=setInterval(load,15000);return()=>{active=false;clearInterval(timer);abort.current?.abort()}},[]);
 useEffect(()=>()=>{if(url)URL.revokeObjectURL(url)},[url]);
 function choose(f){
  if(!f)return;
  if(!['image/png','image/jpeg'].includes(f.type)){setError('Choose a PNG or JPEG.');return}
  if(f.size>10*1024*1024){setError('Maximum file size is 10 MiB.');return}
  requestId.current++;abort.current?.abort();setBusy(false);setFile(f);setUrl(URL.createObjectURL(f));setPrediction(null);setError('');
 }
 async function loadSample(name){
  const id=++requestId.current;abort.current?.abort();setBusy(false);setPrediction(null);setError('');
  try{const response=await fetch('/samples/'+name+'.png');if(!response.ok)throw Error('Sample image unavailable');
   const blob=await response.blob();if(id===requestId.current)choose(new File([blob],'sample-'+name+'.png',{type:'image/png'}));
  }catch(e){if(id===requestId.current)setError(e.message)}
 }
 async function run(){
  if(!file||busy)return;
  const id=++requestId.current;const controller=new AbortController();abort.current=controller;
  setBusy(true);setError('');const body=new FormData();body.append('file',file);
  const timeout=setTimeout(()=>controller.abort(),30000);
  try{
   const r=await fetch('/api/inspect?confidence='+confidence,{method:'POST',body,signal:controller.signal});
   const data=await r.json();if(!r.ok)throw Error(data.detail||'Inspection failed');
   if(id!==requestId.current)return;
   setPrediction(data);setHistory(h=>[{name:file.name,...data},...h].slice(0,8));
  }catch(e){if(id===requestId.current)setError(e.name==='AbortError'?'Inspection cancelled or timed out.':e.message)}
  finally{clearTimeout(timeout);if(id===requestId.current)setBusy(false)}
 }
 const measured=results?.experiments||{};
 return <div className="shell">
  <aside><a className="brand" href="./"><span className="mark">◈</span> edgeinspect<span className="dot">.</span></a>
   <div className="workspace">ENGINEERING WORKSPACE<span>Metal nut · MVTec AD</span></div>
   <nav aria-label="Workspace">{['Inspect','Experiments','Deployment','Method'].map((x,i)=><button key={x} className={tab===x?'active':''} onClick={()=>setTab(x)}><span>0{i+1}</span>{x}<b>↗</b></button>)}</nav>
   <div className="aside-footer"><span className="mini-label">TAKE-HOME / 2026</span><p>Built for the edge.<br/>Measured with care.</p><small>Hargun Madan<br/>Independent internship submission</small></div>
  </aside>
  <main>
   <header><div><span className="crumb">WORKSPACE</span><span className="slash">/</span>{tab}</div><span className={'status '+(health?.model_loaded?'ready':'')}><i/>{health?.model_loaded?'CPU runtime ready':health?.status==='API offline'?'API offline':'Awaiting trained model'}</span></header>
   <div className="content">
   <div className="eyebrow"><span/> VISION QUALITY CONTROL · LOCAL RUNTIME</div>
   <div className="title-row"><div><h1>{tab==='Inspect'?'Small parts. Clear decisions.':tab==='Experiments'?'Evidence before confidence.':tab==='Deployment'?'From model to machine.':'Every decision, explained.'}</h1><p className="subtitle">A metal nut defect inspection station. Four defect classes. One local pipeline.</p></div><span className="station">STATION_01<br/><b>METAL NUT</b></span></div>
   {tab==='Inspect'&&<>
    <section className="metrics" aria-label="Runtime configuration">{[['MODEL','YOLO11n','Nano object detector'],['INPUT','320 × 320','RGB · letterboxed'],['EXECUTION','ONNX / CPU','No cloud inference'],['DEFECT TYPES','04','Bent · Color · Flip · Scratch']].map(([a,b,c])=><article key={a}><small>{a}</small><strong>{b}</strong><span>{c}</span></article>)}</section>
    <div className="inspection-grid">
     <section className="panel viewer"><div className="panel-head"><h2>Inspection viewport</h2><span>{file?'IMAGE LOADED':'AWAITING INPUT'}</span></div>
      <div className={'canvas '+(url?'loaded':'')} onDragOver={e=>e.preventDefault()} onDrop={e=>{e.preventDefault();choose(e.dataTransfer.files[0])}}>
       {url?<div className="image-wrap"><img src={url} alt="Selected part for inspection"/>{prediction&&<svg viewBox={'0 0 '+prediction.width+' '+prediction.height} aria-label="Detected defect bounding boxes">{prediction.detections.map((d,i)=><g key={i}><rect x={d.box[0]} y={d.box[1]} width={d.box[2]-d.box[0]} height={d.box[3]-d.box[1]}/><text x={d.box[0]} y={Math.max(18,d.box[1]-7)}>{d.label+' '+Math.round(d.score*100)+'%'}</text></g>)}</svg>}</div>
        :<div className="empty"><div className="scan-icon">⌗</div><h3>Bring a part into focus.</h3><p>Drop a metal nut image here<br/>or select a PNG or JPEG.</p><button className="secondary" onClick={()=>input.current.click()}>＋ Select image</button><small>Images remain on your local machine.</small></div>}
       <span className="corner tl"/><span className="corner tr"/><span className="corner bl"/><span className="corner br"/>
      </div>
      <div className="viewer-footer"><span>{file?file.name:'No image selected'}</span><button className="text-button" onClick={()=>input.current.click()}>Change image ↗</button></div>
      <div className="sample-row"><span>TRY A REAL SAMPLE</span><button className="text-button" onClick={()=>loadSample('good')}>Good part</button><button className="text-button" onClick={()=>loadSample('flip')}>Flipped part</button></div>
      <input ref={input} type="file" accept="image/png,image/jpeg" hidden onChange={e=>choose(e.target.files[0])}/>
     </section>
     <section className="panel control"><div className="panel-head"><h2>Inspection controls</h2><span>01</span></div>
      <div className="control-body"><label htmlFor="confidence">Confidence threshold <b>{confidence.toFixed(2)}</b></label><input id="confidence" type="range" min=".05" max=".95" step=".05" value={confidence} onChange={e=>{requestId.current++;abort.current?.abort();setBusy(false);setConfidence(Number(e.target.value));setPrediction(null)}}/>
      <p className="hint">Lower thresholds may catch more defects and increase false alarms. Validate the operating point before deployment.</p>
      <button className="primary" disabled={!file||!health?.model_loaded||busy} onClick={run}>{busy?'Inspecting…':'Run inspection'}<span>↗</span></button>
      {!health?.model_loaded&&<p className="notice">Model not loaded. Train and export the detector to enable real inference.</p>}
      {error&&<p role="alert" className="error">{error}</p>}
      <div className="decision"><small>INSPECTION RESULT</small><h3>{prediction?(prediction.detections.length?'Review required':'No defect detected'):'Ready when you are.'}</h3><p>{prediction?'Model output is advisory. A negative prediction does not certify a good part.':'Results and localized defects appear after inference.'}</p></div>
      {prediction&&<><div className="latency"><span>CPU inference</span><b>{prediction.timing_ms.inference.toFixed(1)} ms</b></div><div className="latency"><span>Processing total</span><b>{prediction.timing_ms.total.toFixed(1)} ms</b></div>{prediction.detections.map((d,i)=><div className="detection" key={i}><span>{d.label}</span><b>{(d.score*100).toFixed(1)}%</b></div>)}</>}
      </div>
     </section>
    </div>
    <div className="lower"><span>LOCAL BY DESIGN</span><p>Image → letterbox → ONNX Runtime → class-aware NMS → inspection decision</p><span>NO CLOUD ROUND TRIP ↗</span></div>
    {history.length>0&&<section className="panel history"><div className="panel-head"><h2>Session history</h2><span>IN MEMORY ONLY</span></div>{history.map((h,i)=><div className="history-row" key={i}><span>{h.name}</span><b>{h.detections.length} detections</b><span>{h.timing_ms.total.toFixed(1)} ms</span></div>)}</section>}
   </>}
   {tab==='Experiments'&&<section className="panel prose"><div className="panel-head"><h2>Baseline → augmentation study</h2><span>FROZEN SPLIT</span></div><div className="prose-body"><p>Same pretrained model, seed, input size and training budget. The improvement arm adds modest rotation, translation, scale and brightness variation. Selection uses validation data only.</p><table><thead><tr><th>Experiment</th><th>Precision</th><th>Recall</th><th>mAP@0.5</th></tr></thead><tbody>{['baseline_val','augmented_val','selected_test','onnx_test'].map(n=><tr key={n}><td>{n.replaceAll('_',' ')}</td>{['precision','recall','map50'].map(k=><td key={k}>{typeof measured[n]?.metrics?.[k]==='number'?measured[n].metrics[k].toFixed(3):'Pending'}</td>)}</tr>)}</tbody></table><p className="notice">Pending values mean experiments have not been completed. There are no simulated scores.</p><h3>Failure analysis</h3><p>Real false positives and misses are generated by the evaluator. Review overlays and write evidence-based explanations in the report. Green = ground truth; red = false positive; blue = matched prediction.</p></div></section>}
   {tab==='Deployment'&&<section className="panel prose"><div className="panel-head"><h2>CPU deployment profile</h2><span>FP32 / BATCH 1</span></div><div className="prose-body"><h3>Export once. Inspect locally.</h3><p>A fixed-shape ONNX model runs through CPUExecutionProvider. Preprocessing, inference and postprocessing are timed separately with 20 warmups and 100 measured iterations.</p><table><tbody>{[['CPU',measured.benchmark?.cpu],['Runtime',measured.benchmark?.onnxruntime],['Threads',measured.benchmark?.threads],['Inference p50',measured.benchmark?.timing_ms?.inference?.p50?.toFixed(2)+' ms'],['Inference p95',measured.benchmark?.timing_ms?.inference?.p95?.toFixed(2)+' ms']].map(([a,b])=><tr key={a}><th>{a}</th><td>{!b||String(b).includes('undefined')?'Pending benchmark':b}</td></tr>)}</tbody></table><p>Timing excludes file loading, image decoding, networking and display. This station is a demonstration, not a validated production reject mechanism.</p></div></section>}
   {tab==='Method'&&<section className="panel prose"><div className="panel-head"><h2>Reproducible by design</h2><span>SEED 42</span></div><div className="prose-body"><h3>01 / Masks become boxes</h3><p>Each connected mask component becomes a tight bounding box. Good images have empty label files. Tiny components are retained. Labels inherit the defect class from the source directory.</p><h3>02 / Split before augmentation</h3><p>Exact duplicate images are removed before a stratified 60/20/20 split. The manifest records image and mask hashes. This is a custom supervised split, not the official unsupervised MVTec benchmark.</p><h3>03 / Measure honestly</h3><p>Precision and recall use a fixed confidence threshold and IoU 0.5. AP integrates the confidence-ranked precision–recall curve, then averages over classes with ground truth. Good images count toward false positives.</p><h3>04 / Understand the limits</h3><p>Small data, reflective metal and mask-derived boxes limit conclusions. Review tiny labels, class confusion and unseen lighting. Hold out future production batches before making reliability claims.</p><h3>Independent take-home project</h3><p>Built by Hargun Madan with disclosed AI assistance. Inspired by industrial edge-vision workflows; not an official InSiSo product.</p></div></section>}
   <footer><span>EDGEINSPECT / ENGINEERING DEMO</span><span>Develop · Evaluate · Export · Inspect</span></footer>
   </div>
  </main>
 </div>
}
createRoot(document.getElementById('root')).render(<App/>);

