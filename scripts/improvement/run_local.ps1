param(
    [Parameter(Mandatory=$true)][string]$InputPack,
    [Parameter(Mandatory=$true)][string]$Receipt,
    [ValidateSet('multimodal','price')][string]$Mode = 'multimodal',
    [ValidateRange(1,100)][int]$Epochs = 30,
    [ValidateRange(0,2)][int]$Seed = 0,
    [string]$PythonPath = ''
)
$ErrorActionPreference = 'Stop'
$repoRoot = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
if (-not $PythonPath) { $PythonPath = Join-Path $repoRoot '.venv-local-gpu\Scripts\python.exe' }
if (-not (Test-Path -LiteralPath $PythonPath)) { throw 'Working GPU Python missing; follow LOCAL_GPU_RUNBOOK.md.' }
$packPath = (Resolve-Path -LiteralPath $InputPack).Path
$receiptPath = (Resolve-Path -LiteralPath $Receipt).Path
$record = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
if ($record.includes_prices -ne $true) { throw 'Nicole must export with --include-prices.' }
if ((Get-FileHash -LiteralPath $packPath -Algorithm SHA256).Hash -ne $record.sha256) {
    throw 'Input pack checksum differs from the supplied export receipt.'
}
& $PythonPath -c "import torch; assert torch.cuda.is_available(), 'CUDA is unavailable'; print(torch.__version__); print(torch.cuda.get_device_name(0)); x=torch.ones(1,device='cuda'); print('CUDA check:',x.item())"
if ($LASTEXITCODE -ne 0) { throw 'GPU environment check failed; no CPU fallback or training was started.' }
$outputRoot = Join-Path $repoRoot 'campaign-local\runs'
New-Item -ItemType Directory -Force -Path $outputRoot | Out-Null
$outputPath = Join-Path $outputRoot ('compact-' + $Mode + '-s' + $Seed + '-' + (Get-Date -Format 'yyyyMMdd-HHmmss-fff'))
$runArgs = @('scripts/improvement/train_compact.py', '--input-pack', $packPath,
    '--input-pack-sha256', $record.sha256, '--out', $outputPath, '--device', 'cuda',
    '--epochs', "$Epochs", '--seed', "$Seed", '--seq-len', '35')
if ($Mode -eq 'multimodal') { $runArgs += '--use-text' }
Push-Location -LiteralPath $repoRoot
try {
    & $PythonPath @runArgs
    if ($LASTEXITCODE -ne 0) { throw "Training failed with exit code $LASTEXITCODE. Preserve $outputPath for diagnosis." }
    Write-Output "Saved validation results and model to: $outputPath"
} finally {
    Pop-Location
}
