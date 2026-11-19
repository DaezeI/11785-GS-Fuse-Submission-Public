# Inference only; keeps the training directories and frozen snapshot unchanged.
[CmdletBinding()]
param(
    [string] $SnapshotRoot = (Join-Path $HOME 'gs-fuse-runs/raw-snapshot-v1'),
    [string] $ValidationRoot = (Join-Path $HOME 'gs-fuse-runs/raw-path-validation-v1'),
    [string] $OutRoot = (Join-Path $HOME 'gs-fuse-runs/raw-path-test-v1'),
    [string] $Python
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-LocalPath([string] $Path) {
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

$repo = Resolve-LocalPath (Join-Path $PSScriptRoot '../..')
$SnapshotRoot = Resolve-LocalPath $SnapshotRoot
$ValidationRoot = Resolve-LocalPath $ValidationRoot
$OutRoot = Resolve-LocalPath $OutRoot
if ($Python) {
    $Python = Resolve-LocalPath $Python
} else {
    $Python = Join-Path $repo '.venv-local-gpu/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Existing Python executable not found: $Python. Pass -Python with its path."
}
if (Test-Path -LiteralPath $OutRoot) {
    throw "Output already exists and was not changed: $OutRoot"
}

$evaluator = Join-Path $repo 'scripts/improvement/eval_compact_path_test.py'
$originalHubOffline = $env:HF_HUB_OFFLINE
$originalTransformersOffline = $env:TRANSFORMERS_OFFLINE
Push-Location -LiteralPath $repo
try {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    & $Python $evaluator --snapshot-root $SnapshotRoot --validation-root $ValidationRoot --out $OutRoot --device cuda
    if ($LASTEXITCODE -ne 0) {
        throw "Inference-only evaluator failed (exit $LASTEXITCODE)."
    }
} finally {
    Pop-Location
    $env:HF_HUB_OFFLINE = $originalHubOffline
    $env:TRANSFORMERS_OFFLINE = $originalTransformersOffline
}
Write-Host "Done. Results: $(Join-Path $OutRoot 'test-summary.json')"
