$ErrorActionPreference = 'Stop'
Set-Location -LiteralPath $PSScriptRoot
$trainingPython = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
if (-not (Test-Path -LiteralPath $trainingPython)) { throw 'Run setup.ps1 first.' }
Write-Host 'Installing CUDA-enabled PyTorch from the official PyTorch wheel index...'
& $trainingPython -m pip install 'torch==2.9.1+cu128' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'PyTorch CUDA installation failed.' }
& $trainingPython -m pip install --force-reinstall --no-deps 'torchvision==0.24.1+cu128' --index-url https://download.pytorch.org/whl/cu128
if ($LASTEXITCODE -ne 0) { throw 'CUDA installation failed. Copy the error above.' }
Write-Host 'Checking a real GPU computation and torchvision NMS...'
& $trainingPython -c "import torch,torchvision; assert torch.cuda.is_available(), 'CUDA is still unavailable'; x=torch.ones((64,64),device='cuda'); y=x@x; torch.cuda.synchronize(); assert float(y[0,0])==64.; boxes=torch.tensor([[0.,0.,10.,10.]],device='cuda'); scores=torch.tensor([.9],device='cuda'); torchvision.ops.nms(boxes,scores,.5); print('Verified:',torch.cuda.get_device_name(0),torch.__version__)"
if ($LASTEXITCODE -ne 0) { throw 'The GPU check failed; training has not started.' }
Write-Host 'Running baseline, augmentation, held-out evaluation, ONNX export and CPU benchmark...'
& $trainingPython -u -m src.run_experiments --device 0
if ($LASTEXITCODE -ne 0) { throw 'Pipeline stopped. Preserve all files; copy the last error for checkpoint recovery.' }
Write-Host 'Experiments complete. Return to Codex for failure-case analysis and the final report.'
