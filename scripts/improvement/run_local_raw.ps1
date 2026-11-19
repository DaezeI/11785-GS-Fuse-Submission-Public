# Windows PowerShell 5.1+; run from the GS-Fuse repository.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Prepare', 'Smoke', 'Full')]
    [string] $Stage,
    [Parameter(Mandatory = $true)]
    [string] $SnapshotRoot,
    [string] $SourceRoot,
    [string] $OutRoot,
    [string] $SmokeOut,
    [string] $Python,
    [switch] $Exploratory
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Get-LocalPath([string] $Path) {
    return $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
}

function Require-Directory([string] $Path, [string] $Label) {
    if (-not (Test-Path -LiteralPath $Path -PathType Container)) {
        throw "$Label is not a directory: $Path"
    }
}

function Require-FreshPath([string] $Path, [string] $Label) {
    if (Test-Path -LiteralPath $Path) {
        throw "$Label already exists. Pick a fresh path; no files were removed: $Path"
    }
}

function Invoke-CheckedPython([string[]] $Arguments) {
    Write-Host ('Running: ' + $Arguments[0])
    & $script:Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python failed (exit $LASTEXITCODE): $($Arguments[0]). No later stage ran."
    }
}

function Get-Checkpoint([string] $Root, [string] $Mode, [int] $Seed) {
    return Join-Path (Join-Path (Join-Path $Root $Mode) "seed-$Seed") 'best.pt'
}

function Require-Checkpoint([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Expected checkpoint is missing: $Path"
    }
}

$repo = Get-LocalPath (Join-Path $PSScriptRoot '../..')
$prepare = Join-Path $repo 'scripts/improvement/prepare_raw_inputs.py'
$train = Join-Path $repo 'scripts/improvement/train_compact_raw.py'
$SnapshotRoot = Get-LocalPath $SnapshotRoot
if ($SourceRoot) { $SourceRoot = Get-LocalPath $SourceRoot }
if ($OutRoot) { $OutRoot = Get-LocalPath $OutRoot }
if ($SmokeOut) { $SmokeOut = Get-LocalPath $SmokeOut }
if (-not $Python) {
    $Python = Join-Path $repo '.venv-local-gpu/Scripts/python.exe'
} else {
    $Python = Get-LocalPath $Python
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Existing Python executable not found: $Python. Pass -Python with its path; this runner does not install anything."
}

$extra = @()
if ($Exploratory) {
    $extra += '--exploratory'
    Write-Warning 'EXPLORATORY: do not label these runs as the canonical-hash reproduction.'
}

# Protect against implicit Hugging Face fetches during this process only.
$originalOffline = $env:HF_HUB_OFFLINE
$originalTransformersOffline = $env:TRANSFORMERS_OFFLINE
$env:HF_HUB_OFFLINE = '1'
$env:TRANSFORMERS_OFFLINE = '1'
Push-Location -LiteralPath $repo
try {
    if ($Stage -eq 'Prepare') {
        if (-not $SourceRoot) { throw 'Prepare requires -SourceRoot.' }
        if ($OutRoot -or $SmokeOut) { throw 'Do not pass -OutRoot or -SmokeOut to Prepare.' }
        Require-Directory $SourceRoot 'Extracted source root'
        Require-FreshPath $SnapshotRoot 'Snapshot root'
        Require-FreshPath "${SnapshotRoot}.receipt.json" 'Snapshot receipt'
        Invoke-CheckedPython (@($prepare, '--source-root', $SourceRoot, '--snapshot-root', $SnapshotRoot) + $extra)
        return
    }

    if ($SourceRoot) { throw 'Supply -SourceRoot only to Prepare.' }
    if (-not $OutRoot) { throw "$Stage requires -OutRoot." }
    if ($Stage -eq 'Smoke' -and $SmokeOut) { throw 'Do not pass -SmokeOut to Smoke.' }
    Require-Directory $SnapshotRoot 'Snapshot root'
    Require-FreshPath $OutRoot 'Output root'
    $snapshotManifest = Join-Path $SnapshotRoot 'manifest.json'
    if (-not (Test-Path -LiteralPath $snapshotManifest -PathType Leaf)) {
        throw "Prepared manifest is missing: $snapshotManifest"
    }
    $snapshotHash = (Get-FileHash -LiteralPath $snapshotManifest -Algorithm SHA256).Hash
    $snapshotRecord = Join-Path $SnapshotRoot 'snapshot.json'
    if (-not (Test-Path -LiteralPath $snapshotRecord -PathType Leaf)) {
        throw "Frozen input record is missing: $snapshotRecord"
    }
    $snapshotRecordHash = (Get-FileHash -LiteralPath $snapshotRecord -Algorithm SHA256).Hash

    if ($Stage -eq 'Full') {
        if (-not $SmokeOut) { throw 'Full requires -SmokeOut from a successful Smoke run.' }
        $receiptPath = Join-Path $SmokeOut 'RAW_SMOKE_COMPLETE.json'
        if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
            throw "Successful smoke receipt is missing: $receiptPath"
        }
        $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
        if ($receipt.snapshot_root -ne $SnapshotRoot -or
            $receipt.snapshot_manifest_sha256 -ne $snapshotHash -or
            $receipt.snapshot_record_sha256 -ne $snapshotRecordHash -or
            $receipt.exploratory -ne [bool] $Exploratory) {
            throw 'The smoke receipt does not match this snapshot and run mode. Re-run Smoke with the same inputs first.'
        }
        foreach ($mode in @('price', 'residual', 'joint')) {
            Require-Checkpoint (Get-Checkpoint $SmokeOut $mode 0)
        }
    }

    $epochs = if ($Stage -eq 'Smoke') { '1' } else { '30' }
    $seeds = if ($Stage -eq 'Smoke') { @(0) } else { @(0, 1, 2) }
    [void] (New-Item -ItemType Directory -Path $OutRoot)
    foreach ($seed in $seeds) {
        foreach ($mode in @('price', 'residual', 'joint')) {
            $out = Join-Path (Join-Path $OutRoot $mode) "seed-$seed"
            [void] (New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out))
            $arguments = @(
                $train, '--snapshot-root', $SnapshotRoot, '--out', $out,
                '--device', 'cuda', '--mode', $mode, '--seed', [string] $seed,
                '--epochs', $epochs, '--batch-size', '32'
            ) + $extra
            if ($mode -ne 'price') {
                $priceCheckpoint = Get-Checkpoint $OutRoot 'price' $seed
                Require-Checkpoint $priceCheckpoint
                $arguments += @('--price-checkpoint', $priceCheckpoint)
            }
            Invoke-CheckedPython $arguments
            Require-Checkpoint (Get-Checkpoint $OutRoot $mode $seed)
        }
    }

    if ($Stage -eq 'Smoke') {
        [ordered]@{
            snapshot_root = $SnapshotRoot
            snapshot_manifest_sha256 = $snapshotHash
            snapshot_record_sha256 = $snapshotRecordHash
            exploratory = [bool] $Exploratory
        } | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $OutRoot 'RAW_SMOKE_COMPLETE.json') -Encoding UTF8
    }
} finally {
    Pop-Location
    $env:HF_HUB_OFFLINE = $originalOffline
    $env:TRANSFORMERS_OFFLINE = $originalTransformersOffline
}
