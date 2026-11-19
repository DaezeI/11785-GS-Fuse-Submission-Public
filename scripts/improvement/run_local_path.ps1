# Windows PowerShell 5.1+. Run from the GS-Fuse campaign checkout.
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Smoke', 'Full')]
    [string] $Stage,
    [Parameter(Mandatory = $true)]
    [string] $SnapshotRoot,
    [Parameter(Mandatory = $true)]
    [string] $OutRoot,
    [string] $SmokeOut,
    [string] $Python
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

function Resolve-LocalPath([string] $Path) {
    $absolute = $ExecutionContext.SessionState.Path.GetUnresolvedProviderPathFromPSPath($Path)
    return [IO.Path]::GetFullPath($absolute)
}

function Get-RequiredHash([string] $Path) {
    if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) {
        throw "Required file is missing: $Path"
    }
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Get-RunPath([string] $Root, [string] $Mode, [int] $Seed) {
    return Join-Path (Join-Path $Root $Mode) "seed-$Seed"
}

function Invoke-Train([string[]] $Arguments) {
    & $script:Python @Arguments
    if ($LASTEXITCODE -ne 0) {
        throw "Compact path trainer failed (exit $LASTEXITCODE). No later run was started."
    }
}

function Get-RunProvenance([string] $Root, [string] $Mode, [int] $Seed) {
    $run = Get-RunPath $Root $Mode $Seed
    $configPath = Join-Path $run 'resolved_config.json'
    $configHash = Get-RequiredHash $configPath
    $config = Get-Content -LiteralPath $configPath -Raw | ConvertFrom-Json
    if ($config.task -ne 'c1-ohlc-return-path-v1' -or $config.mode -ne $Mode -or
        $config.seed -ne $Seed -or $config.device -ne 'cuda' -or
        $config.test_scored -ne $false -or $config.vali.events -ne 2302) {
        throw "Unexpected whole-path validation configuration: $configPath"
    }
    $result = [ordered]@{
        checkpoint_sha256 = Get-RequiredHash (Join-Path $run 'best.pt')
        config_sha256 = $configHash
        validation_sha256 = Get-RequiredHash (Join-Path $run 'vali.npz')
    }
    if ($Mode -eq 'residual') {
        $result['controls_sha256'] = Get-RequiredHash (Join-Path $run 'vali-text-controls.npz')
    }
    return $result
}

function Get-SourceProvenance([string] $Root) {
    $result = [ordered]@{}
    foreach ($relative in @(
        'model/compact_return.py', 'model/compact_raw_text.py', 'model/compact_news_vocab.py',
        'scripts/improvement/train_compact_path.py', 'scripts/improvement/compact_path_metrics.py',
        'scripts/improvement/train_compact_raw.py', 'scripts/improvement/prepare_raw_inputs.py',
        'scripts/freeze_c1_inputs.py', 'data/c1_protocol.py', 'data/forecast_task.py'
    )) {
        $result[$relative] = Get-RequiredHash (Join-Path $Root $relative)
    }
    return $result
}

$repo = Resolve-LocalPath (Join-Path $PSScriptRoot '../..')
$train = Join-Path $repo 'scripts/improvement/train_compact_path.py'
$SnapshotRoot = Resolve-LocalPath $SnapshotRoot
$OutRoot = Resolve-LocalPath $OutRoot
if ($SmokeOut) { $SmokeOut = Resolve-LocalPath $SmokeOut }
if ($Python) {
    $Python = Resolve-LocalPath $Python
} else {
    $Python = Join-Path $repo '.venv-local-gpu/Scripts/python.exe'
}
if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) {
    throw "Existing Python executable not found: $Python. Pass -Python with its path; no packages are installed."
}
if (Test-Path -LiteralPath $OutRoot) {
    throw "Use a fresh -OutRoot; no prior output was changed: $OutRoot"
}
$snapshotBoundary = $SnapshotRoot.TrimEnd([char[]] @('\', '/')) + [IO.Path]::DirectorySeparatorChar
if ($OutRoot.StartsWith($snapshotBoundary, [StringComparison]::OrdinalIgnoreCase)) {
    throw '-OutRoot must be outside the frozen snapshot.'
}

$manifestHash = Get-RequiredHash (Join-Path $SnapshotRoot 'manifest.json')
$recordPath = Join-Path $SnapshotRoot 'snapshot.json'
$recordHash = Get-RequiredHash $recordPath
$sources = Get-SourceProvenance $repo
$pythonHash = Get-RequiredHash $Python
$record = Get-Content -LiteralPath $recordPath -Raw | ConvertFrom-Json
$canonicalManifest = '1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1'
$canonicalContent = '049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf'
if ($manifestHash -ne $canonicalManifest -or $record.content_sha256 -ne $canonicalContent) {
    throw 'This study requires the existing canonical screened C1 snapshot. There is no exploratory override.'
}

$modes = @('price', 'residual')
$receiptName = 'RAW_PATH_SMOKE_COMPLETE.json'
if ($Stage -eq 'Smoke' -and $SmokeOut) {
    throw 'Do not pass -SmokeOut to Smoke.'
}
if ($Stage -eq 'Full') {
    if (-not $SmokeOut) { throw 'Full requires -SmokeOut from a successful path Smoke run.' }
    $receiptPath = Join-Path $SmokeOut $receiptName
    if (-not (Test-Path -LiteralPath $receiptPath -PathType Leaf)) {
        throw "Smoke receipt is missing: $receiptPath"
    }
    $receipt = Get-Content -LiteralPath $receiptPath -Raw | ConvertFrom-Json
    if ($receipt.schema -ne 'gs-fuse-raw-path-smoke-v1' -or
        $receipt.snapshot_root -ne $SnapshotRoot -or
        $receipt.manifest_sha256 -ne $manifestHash -or
        $receipt.snapshot_record_sha256 -ne $recordHash -or
        $receipt.python_path -ne $Python -or $receipt.python_sha256 -ne $pythonHash) {
        throw 'Smoke provenance differs. Run Smoke again with these inputs and this trainer.'
    }
    foreach ($relative in $sources.Keys) {
        if ($receipt.sources.PSObject.Properties[$relative].Value -ne $sources[$relative]) {
            throw "Source changed after Smoke ($relative); run Smoke again."
        }
    }
    foreach ($mode in $modes) {
        $actual = Get-RunProvenance $SmokeOut $mode 0
        $expected = $receipt.runs.PSObject.Properties[$mode].Value
        foreach ($artifact in $actual.Keys) {
            if ($actual[$artifact] -ne $expected.PSObject.Properties[$artifact].Value) {
                throw "Smoke artifacts changed for $mode; run Smoke again."
            }
        }
    }
}

$epochs = if ($Stage -eq 'Smoke') { '1' } else { '30' }
$seeds = if ($Stage -eq 'Smoke') { @(0) } else { @(0, 1, 2) }
$runProvenance = [ordered]@{}

# Process-local offline flags. No data or pretrained models are fetched.
$originalOffline = $env:HF_HUB_OFFLINE
$originalTransformersOffline = $env:TRANSFORMERS_OFFLINE
Push-Location -LiteralPath $repo
try {
    $env:HF_HUB_OFFLINE = '1'
    $env:TRANSFORMERS_OFFLINE = '1'
    [void] [IO.Directory]::CreateDirectory($OutRoot)
    foreach ($seed in $seeds) {
        foreach ($mode in $modes) {
            $out = Get-RunPath $OutRoot $mode $seed
            $arguments = @(
                $train, '--snapshot-root', $SnapshotRoot, '--out', $out,
                '--device', 'cuda', '--mode', $mode, '--seed', [string] $seed,
                '--epochs', $epochs, '--batch-size', '32'
            )
            if ($mode -eq 'residual') {
                $priceCheckpoint = Join-Path (Get-RunPath $OutRoot 'price' $seed) 'best.pt'
                $arguments += @('--price-checkpoint', $priceCheckpoint)
            }
            Write-Host "Running $Stage / $mode / seed $seed"
            Invoke-Train $arguments
            $provenance = Get-RunProvenance $OutRoot $mode $seed
            if ($Stage -eq 'Smoke') { $runProvenance[$mode] = $provenance }
        }
    }

    if ($Stage -eq 'Smoke') {
        # Refuse to certify a smoke run if inputs or code changed while it ran.
        if ((Get-RequiredHash (Join-Path $SnapshotRoot 'manifest.json')) -ne $manifestHash -or
            (Get-RequiredHash $recordPath) -ne $recordHash -or
            (Get-RequiredHash $Python) -ne $pythonHash) {
            throw 'Provenance changed during Smoke. No smoke receipt was written.'
        }
        $sourcesNow = Get-SourceProvenance $repo
        foreach ($relative in $sources.Keys) {
            if ($sources[$relative] -ne $sourcesNow[$relative]) {
                throw "Source changed during Smoke ($relative). No smoke receipt was written."
            }
        }
        [ordered]@{
            schema = 'gs-fuse-raw-path-smoke-v1'
            snapshot_root = $SnapshotRoot
            manifest_sha256 = $manifestHash
            snapshot_record_sha256 = $recordHash
            python_path = $Python
            python_sha256 = $pythonHash
            sources = $sources
            runs = $runProvenance
        } | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath (Join-Path $OutRoot $receiptName) -Encoding UTF8
    }
} finally {
    Pop-Location
    $env:HF_HUB_OFFLINE = $originalOffline
    $env:TRANSFORMERS_OFFLINE = $originalTransformersOffline
}
