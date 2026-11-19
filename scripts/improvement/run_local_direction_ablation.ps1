# Run the compact direction-loss comparison sequentially; no test split scoring.
[CmdletBinding()]
param(
    [string] $SnapshotRoot = (Join-Path $HOME 'gs-fuse-runs/raw-snapshot-v1'),
    [string] $OutRoot = (Join-Path $HOME 'gs-fuse-runs/compact-direction-v1'),
    [string] $Python
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$repo = [IO.Path]::GetFullPath((Join-Path $PSScriptRoot '../..'))
$SnapshotRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($SnapshotRoot)
$OutRoot = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($OutRoot)
if ($Python) {
    $Python = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Python)
} else {
    $Python = Join-Path $repo '.venv-local-gpu/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing Python: $Python" }
if (Test-Path -LiteralPath $OutRoot) { throw "Choose a fresh -OutRoot. Existing files unchanged: $OutRoot" }
$rootBoundary = $SnapshotRoot.TrimEnd([char[]] @('\', '/')) + [IO.Path]::DirectorySeparatorChar
if ($OutRoot.StartsWith($rootBoundary, [StringComparison]::OrdinalIgnoreCase)) { throw 'Output must be outside the frozen snapshot.' }
$scriptFile = Join-Path $repo 'scripts/improvement/train_compact_direction_ablation.py'
if (-not (Test-Path -LiteralPath $scriptFile -PathType Leaf)) { throw "Missing trainer: $scriptFile" }

function Invoke-Direction([string[]] $JobArgs, [string] $LogPath) {
    & $Python -u $scriptFile @JobArgs | Tee-Object -FilePath $LogPath
    if ($LASTEXITCODE -ne 0) { throw "Direction ablation stopped (exit $LASTEXITCODE). See $LogPath and terminal output." }
}

$common = @('--snapshot-root', $SnapshotRoot, '--device', 'cuda')
Push-Location -LiteralPath $repo
try {
    [void][IO.Directory]::CreateDirectory($OutRoot)
    Write-Host 'Smoke: seed 0, one epoch per loss variant'
    Invoke-Direction ($common + @('--out', (Join-Path $OutRoot 'smoke'), '--seeds', '0', '--epochs', '1')) (Join-Path $OutRoot 'smoke.log')
    Write-Host 'Full queue: seeds 0, 1, 2; two loss variants per seed; one run at a time'
    Invoke-Direction ($common + @('--out', (Join-Path $OutRoot 'validation'), '--seeds', '0', '1', '2', '--epochs', '30')) (Join-Path $OutRoot 'validation.log')
} finally {
    Pop-Location
}
Write-Host "Done: $(Join-Path $OutRoot 'validation/direction-ablation-summary.json')"
