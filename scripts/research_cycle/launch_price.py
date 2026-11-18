#!/usr/bin/env python3
"""Render a versioned screened-price launch package for Nicole's personal GPU resource.

Render locally; transfer the directory to the resource, inspect it, then run
submit.sh there only after checking existing scheduler records and capacity.
All paths supplied here are paths on that resource. No jobs are submitted here.
"""
import argparse
import hashlib
import os
from pathlib import Path
import re
import shlex

COMMIT = 'bb09396881cc92c5e654c4210944ce5c6da17ced'
MANIFEST = '1b04373a4361238b09c0b1d952163973403d2d7839dbb9e7215c429af14b23d1'
SERIES = '6a873b1d72d27f9188c6bf12acb756775916be8ed9ff42d9e9dd98028f68de2a'


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--root', required=True, help='Absolute remote project root')
    p.add_argument('--project', required=True, help='Local scheduler project setting')
    p.add_argument('--queue', required=True, help='Local scheduler queue setting')
    p.add_argument('--resource-type', required=True, help='Scheduler one-H200 resource type')
    p.add_argument('--python-module', required=True, help='Environment module name')
    p.add_argument('--output', type=Path, required=True, help='New local package directory')
    a = p.parse_args()
    if a.seed <= 0 or not a.root.startswith('/') or any(c in a.root for c in '\n\r'):
        p.error('seed must be positive and root must be an absolute single-line path')
    for name in ('project', 'queue', 'resource_type', 'python_module'):
        if not re.fullmatch(r'[A-Za-z0-9_.\-/]+', getattr(a, name)):
            p.error(f'invalid {name}')
    if a.output.exists():
        p.error('output already exists; choose a new versioned package directory')
    run = f'screened-price-seed{a.seed}-v1'
    package = f'price-seed{a.seed}-v1'
    root = shlex.quote(a.root.rstrip('/'))
    common = f'''#!/bin/bash
set -euo pipefail
umask 007
ROOT={root}
ASSETS="$ROOT/gsfuse-assets-v1"
SNAPSHOT="$ASSETS/data/snapshots/c1-screened-return-v1"
REPO="$ROOT/code/main-v1"
PY="$ROOT/envs/gsfuse-v1/bin/python"
RUN="$ROOT/runs/{run}"
export TMPDIR="$ROOT/tmp" HF_HOME="$ROOT/hf-cache" PIP_CACHE_DIR="$ROOT/pip-cache"
export TRANSFORMERS_OFFLINE=1 HF_HUB_OFFLINE=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=8 OPENBLAS_NUM_THREADS=8 MKL_NUM_THREADS=8
module load {shlex.quote(a.python_module)}
cd "$REPO"
test "$(git rev-parse HEAD)" = {COMMIT}
test -z "$(git status --porcelain)"
test -f "$ROOT/logs/environment-ready-v1"
test -f "$ROOT/logs/assets-verified-v1.json"
test -f "$ROOT/logs/inputs-ready-v1"
"$PY" - "$ROOT/logs/assets-verified-v1.json" <<'PYCODE'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as fingerprint_file:
    fingerprints = json.load(fingerprint_file)
expected = {{
    "code_commit": "{COMMIT}",
    "snapshot_content_sha256": "049aa089d2eaa77644be5deba3abad3fa33c0a18b4375553664cafc71370c2bf",
    "manifest_sha256": "{MANIFEST}",
    "price_sha256": "{SERIES}",
}}
if any(fingerprints.get(key) != value for key, value in expected.items()):
    raise SystemExit("Frozen input provenance mismatch")
PYCODE
test "$(sha256sum "$SNAPSHOT/manifest.json" | cut -d' ' -f1)" = {MANIFEST}
test "$(sha256sum "$SNAPSHOT/series/SP500.csv" | cut -d' ' -f1)" = {SERIES}
COMMON=(--forecast-task price --c1-manifest "$SNAPSHOT/manifest.json" --c1-timing-policy category4_1400_strict_before --event-dir "$SNAPSHOT/event" --series-dir "$SNAPSHOT/series" --moment-path "$ASSETS/pretrained/MOMENT-1-large" --text-model-path "$ASSETS/pretrained/Llama-3.2-1B" --text-cache "$ASSETS/cache/text_inst_llama1b_tf32.pt" --no-4bit --tf32 --fusion-type crossattn --gate-type softmax)
printf 'PBS_JOBID=%s\\n' "${{PBS_JOBID:-manual}}"
git rev-parse HEAD
nvidia-smi
'''
    controls = '''--c1-manifest "$SNAPSHOT/manifest.json" --c1-timing-policy category4_1400_strict_before --event-dir "$SNAPSHOT/event" --series-dir "$SNAPSHOT/series" --moment-path "$ASSETS/pretrained/MOMENT-1-large" --text-model-path "$ASSETS/pretrained/Llama-3.2-1B" --text-cache "$ASSETS/cache/text_inst_llama1b_tf32.pt" --no-4bit --tf32'''
    def header(stage, wall):
        # PBS directives are parsed before shell expansion; paths must be literal.
        log = a.root.rstrip('/') + f'/logs/price-seed{a.seed}-{stage}-v1.log'
        if not re.fullmatch(r'/[A-Za-z0-9_./-]+', log):
            p.error('root must contain only letters, digits, _, -, /, and . for PBS logs')
        return f'''#!/bin/bash
#PBS -N ly_gsf_price_s{a.seed}_{stage}
#PBS -P {a.project}
#PBS -q {a.queue}
#PBS -v RTYPE={a.resource_type}
#PBS -l select=1
#PBS -l walltime={wall}
#PBS -j oe
#PBS -o {log}
source {shlex.quote(a.root.rstrip('/') + '/scripts/' + package + '/common.sh')}
'''
    evidence = '''test ! -e "$OUTPUT"
mkdir -p "$OUTPUT"
cp "$ROOT/logs/assets-verified-v1.json" "$OUTPUT/input-fingerprints.json"
cp "$ROOT/logs/python-environment-v1.txt" "$OUTPUT/python-environment.txt"
cp "$ROOT/scripts/''' + package + '''/sha256sums" "$OUTPUT/launcher-sha256.txt"
printf '%s\\n' "$PBS_JOBID" > "$OUTPUT/scheduler-job.txt"
'''
    smoke = header('smoke', '01:00:00') + 'OUTPUT="$RUN/smoke"\n' + evidence + f'''"$PY" -u smoke_train.py "${{COMMON[@]}}" --raw-ts-residual --no-learnable-last-value --seed {a.seed} --batch-size 2 --n-event-train 16 --n-event-eval 8 --n-sw-train 32 --n-sw-eval 16 --n-interpret 0 --stage1-epochs 1 --stage2-epochs 1 --num-epochs 1 --output-path "$OUTPUT" --wandb --wandb-mode offline --wandb-project 11785-GS-Fuse --wandb-name {run}-smoke
"$PY" -u scripts/run_text_controls.py --model-path "$OUTPUT/best_model.pth" {controls} --max-events 8 --shuffles 3 --seed 23 --batch-size 2 --checkpoint-training-split screened_c1_price_smoke_v1 --out "$OUTPUT/paired.npz"
"$PY" scripts/analyze_text_predictions.py --predictions "$OUTPUT/paired.npz" --out "$OUTPUT/paired.analysis.json"
sha256sum "$OUTPUT/best_model.pth" "$OUTPUT/paired.npz" > "$OUTPUT/artifacts.sha256"
'''
    full = header('full', '02:00:00') + 'OUTPUT="$RUN/full"\n' + evidence + f'''"$PY" -u train.py "${{COMMON[@]}}" --raw-ts-residual --no-learnable-last-value --seed {a.seed} --batch-size 8 --sw-stride 10 --stage1-epochs 2 --stage2-epochs 2 --num-epochs 2 --output-path "$OUTPUT" --wandb --wandb-mode offline --wandb-project 11785-GS-Fuse --wandb-name {run}-full
test -s "$OUTPUT/best_model.pth"
sha256sum "$OUTPUT/best_model.pth" > "$OUTPUT/checkpoint.sha256"
'''
    def evaluation(stage, mode):
        return header(stage, '01:00:00') + f'''OUTPUT="$RUN/paired"
mkdir -p "$OUTPUT"
test ! -e "$OUTPUT/{mode}.npz"
"$PY" -u scripts/run_text_controls.py --model-path "$RUN/full/best_model.pth" {controls} --max-events 0 --shuffles 3 --seed 23 --batch-size 8 --no-text-mode {mode} --checkpoint-training-split screened_c1_price_v1 --out "$OUTPUT/{mode}.npz"
"$PY" scripts/analyze_text_predictions.py --predictions "$OUTPUT/{mode}.npz" --out "$OUTPUT/{mode}.analysis.json"
sha256sum "$OUTPUT/{mode}.npz" "$OUTPUT/{mode}.analysis.json" > "$OUTPUT/{mode}.sha256"
'''
    submit = f'''#!/bin/bash
set -euo pipefail
umask 007
ROOT={root}
DIR="$ROOT/scripts/{package}"
RECORD="$ROOT/logs/price-seed{a.seed}-chain-v1.env"
test ! -e "$ROOT/runs/{run}"
(cd "$DIR" && sha256sum -c sha256sums)
# A partial launch also reserves this record; inspect it before any retry.
(set -o noclobber; : > "$RECORD")
submit() {{
 local label="$1" job
 shift
 job=$(qsub "$@")
 # Record every successful scheduler response before validating its format.
 printf '%s=%s\\n' "$label" "$job" >> "$RECORD"
 [[ "$job" =~ ^[0-9]+\\.[A-Za-z0-9._-]+$ ]] || {{ printf 'Unexpected scheduler response: %s\\n' "$job" >&2; exit 1; }}
 printf '%s' "$job"
}}
SMOKE=$(submit SMOKE_JOB "$DIR/smoke.pbs")
FULL=$(submit FULL_JOB -W "depend=afterok:$SMOKE" "$DIR/full.pbs")
ZERO=$(submit ZERO_JOB -W "depend=afterok:$FULL" "$DIR/zero.pbs")
submit BYPASS_JOB -W "depend=afterok:$ZERO" "$DIR/bypass.pbs" >/dev/null
cat "$RECORD"
'''
    a.output.mkdir(parents=True)
    files = {'common.sh': common, 'smoke.pbs': smoke, 'full.pbs': full, 'zero.pbs': evaluation('zero', 'zeroed_article_tokens'), 'bypass.pbs': evaluation('bypass', 'fusion_bypass_ts_inst'), 'submit.sh': submit}
    for name, content in files.items():
        path = a.output / name
        path.write_text(content)
        os.chmod(path, 0o700)
    (a.output / 'sha256sums').write_text(''.join(f'{hashlib.sha256(content.encode()).hexdigest()}  {name}\n' for name, content in files.items()))
    print(f'Rendered {a.output}; install as $ROOT/scripts/{package}. No jobs submitted.')


if __name__ == '__main__':
    main()
