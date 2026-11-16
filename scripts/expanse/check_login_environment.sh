#!/bin/bash
set -euo pipefail

echo "This is a login-node metadata check; it must not run model computation."
hostname
date -Is
id

command -v expanse-client >/dev/null
expanse-client user -r expanse_gpu
expanse-client resource

if [[ -n "${EXPANSE_PROJECT:-}" ]]; then
  expanse-client project "$EXPANSE_PROJECT" -p
else
  echo "EXPANSE_PROJECT is unset; project-specific lookup skipped."
fi

type module >/dev/null 2>&1 || source /etc/profile.d/modules.sh
module spider python 2>&1 | head -n 80 || true
module spider cuda 2>&1 | head -n 80 || true

echo "No GPU job was requested by this script."
