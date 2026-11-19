# GS-Fuse compact event-summary experiment: local Windows run

Use the existing `.venv-local-gpu` environment from the GS-Fuse checkout. Nothing in the runner installs packages, downloads the dataset, queues a job, or runs test-set evaluation. These commands target the train/validation experiment only. Keep the archive, extracted source, prepared snapshot, and outputs in separate local directories. Do not commit any of them.

## 1. Locate or extract the original dataset

Use the original [GS-FUSE_Dataset.zip](https://drive.google.com/file/d/1fH436rkOHVYIG2JPROrFTXXemnY7JPdB/view?usp=sharing). Drive metadata identified it as a ZIP of **260,350,898 bytes**; that does not verify its content or hash. If the correct archive is already local, use it. If not, use the Drive link to download the ZIP manually, then check its size and record its SHA-256. Do not substitute a processed-only copy, a price-only Drive folder, or cached embeddings for the archived summary text.

From Windows PowerShell, starting in the repository root:

```powershell
$archive = Join-Path $HOME 'Downloads\GS-FUSE_Dataset.zip'
$extract = Join-Path $HOME 'gs-fuse-inputs\source-extracted-v1' # must not already exist
$archiveInfo = Get-Item -LiteralPath $archive
if ($archiveInfo.Length -ne 260350898) { throw "Unexpected archive size: $($archiveInfo.Length) bytes" }
Get-FileHash -LiteralPath $archive -Algorithm SHA256
if (Test-Path -LiteralPath $extract) { throw "Use a fresh extraction directory: $extract" }

# Check that ZIP entries cannot resolve outside the new directory before extracting.
Add-Type -AssemblyName System.IO.Compression.FileSystem
$extract = [IO.Path]::GetFullPath($extract)
$boundary = $extract.TrimEnd([char[]] @('\', '/')) + [IO.Path]::DirectorySeparatorChar
$zip = [IO.Compression.ZipFile]::OpenRead($archive)
try {
    foreach ($entry in $zip.Entries) {
        $candidate = [IO.Path]::GetFullPath([IO.Path]::Combine($extract, $entry.FullName))
        if (-not $candidate.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Unsafe archive entry: $($entry.FullName)"
        }
    }
} finally {
    $zip.Dispose()
}
Expand-Archive -LiteralPath $archive -DestinationPath $extract
```

Point `$source` to the **actual data directory** inside the extraction. It must directly contain `event\` and `series\SP500.csv`. Use the printed candidates to choose it; an extra wrapper directory in the archive is not the source root.

```powershell
Get-ChildItem -LiteralPath $extract -Filter SP500.csv -File -Recurse |
    ForEach-Object { $_.Directory.Parent.FullName }

$source = Join-Path $extract 'data' # adjust to observed candidate
Get-Item -LiteralPath (Join-Path $source 'series\SP500.csv')
Get-ChildItem -LiteralPath (Join-Path $source 'event') -Directory | Select-Object -First 3
```

The staging code expects data in the canonical repository layout, including paths such as `event/<category>/<YYYYMMDD>.txt/` with archived event summaries such as `*_full_summary.txt`. These are not original articles; [CAMEF documents its files with this suffix as Llama-generated](https://github.com/lakebodhi/CAMEF#-file-naming-convention). The local encoder does not run Llama or read cached embeddings. A folder date is not proof that a news item was published before the price forecast cutoff. Do not use the original paper's availability assumptions to claim verified publication timing.

## 2. Prepare a fresh frozen snapshot

```powershell
$runner = '.\scripts\improvement\run_local_raw.ps1'
$snapshot = Join-Path $HOME 'gs-fuse-runs\raw-snapshot-v1' # does not exist yet
& $runner -Stage Prepare -SourceRoot $source -SnapshotRoot $snapshot
```

The default path is strict. Read the receipt written beside the snapshot (`<snapshot-root>.receipt.json`) and confirm the committed C1 split membership (6,167 train / 2,302 validation / 2,236 test events). Those counts describe admission checks, **not** an instruction to train or evaluate on test. Do not create replacements, repair event dates, or proceed on a population mismatch. Keep the snapshot unchanged across conditions and seeds.

If staging reports **only** a canonical hash mismatch while the C1 population and all other required checks match, an explicit `-Exploratory` run is available. Use fresh separately named snapshot and output paths, pass `-Exploratory` to all three stages, and label every result **exploratory, not canonical reproduction**. Never use it to excuse missing data, altered split membership, guessed provenance, or an unrelated error.

## 3. Smoke all three modes before the full run

```powershell
$smoke = Join-Path $HOME 'gs-fuse-runs\raw-smoke-v1' # must be fresh
& $runner -Stage Smoke -SnapshotRoot $snapshot -OutRoot $smoke
```

This runs on CUDA with batch size 32, seed 0, and one epoch per mode: price-only, then residual, then joint. For residual and joint, the runner supplies the matching seed's `price/seed-0/best.pt`. Residual uses the frozen price checkpoint; joint uses it only as a comparably seeded price baseline, not for initialization. Results and checkpoints stay under `<out>/<mode>/seed-<seed>/`. A successful smoke writes `RAW_SMOKE_COMPLETE.json` only after all three subprocesses exit successfully and their checkpoints exist.

Inspect each mode's validation outputs, required control coverage, and warnings before proceeding. A CUDA/device, input, non-finite loss, checkpoint, or incomplete control error is a stop, not a reason to launch a full run.

## 4. Run the validation-only matrix

```powershell
$full = Join-Path $HOME 'gs-fuse-runs\raw-validation-v1' # must be fresh
& $runner -Stage Full -SnapshotRoot $snapshot -OutRoot $full -SmokeOut $smoke
```

Full uses seeds 0, 1, and 2, 30 epochs, batch size 32, and CUDA, in the same per-seed price → residual → joint order. It requires a matching successful smoke receipt; it intentionally does not contain a test-set switch. Review results using paired validation controls and the matched price baseline. A smoke pass is not evidence of performance, and no validation outcome should be reported as a test result.

If the checkout's existing virtual environment is elsewhere, pass `-Python 'C:\path with spaces\python.exe'`. On any partial failure retain the logs and use a fresh output path for a retry; the runner will not overwrite prior outputs.

### Python CLI equivalents

The PowerShell runner calls these Python entry points with explicit paths and writes a smoke receipt. These are one-seed examples; they do not issue a smoke receipt or enable any full run.

```powershell
$python = '.\.venv-local-gpu\Scripts\python.exe'
& $python .\scripts\improvement\prepare_raw_inputs.py --source-root $source --snapshot-root $snapshot
& $python .\scripts\improvement\train_compact_raw.py --snapshot-root $snapshot --out '<fresh-price-output>' --device cuda --mode price --seed 0 --epochs 1 --batch-size 32
& $python .\scripts\improvement\train_compact_raw.py --snapshot-root $snapshot --out '<fresh-residual-output>' --device cuda --mode residual --seed 0 --epochs 1 --batch-size 32 --price-checkpoint '<price-output>\best.pt'
& $python .\scripts\improvement\train_compact_raw.py --snapshot-root $snapshot --out '<fresh-joint-output>' --device cuda --mode joint --seed 0 --epochs 1 --batch-size 32 --price-checkpoint '<price-output>\best.pt'
```

## Results follow-up

This endpoint experiment predicts only the close at bars 35, 70 and 140. The follow-on [whole-path study](RAW_PATH_VALIDATION.md) gives separate validation scores and explains why neither endpoint MAE is comparable to R3's full-path score. Keep local datasets, checkpoints, and prediction archives out of Git.
