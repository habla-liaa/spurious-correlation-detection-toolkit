#!/usr/bin/env bash
set -euo pipefail

mapfile -t configs < <(
    find configs -type f -name '*.yaml' \
        ! -path 'configs/*/lists/*' \
        ! -name 'base.yaml' \
        ! -name 'base-*.yaml' \
        | sort
)

if [ "${#configs[@]}" -eq 0 ]; then
    echo "No YAML configs found under configs/"
    exit 1
fi

for i in "${!configs[@]}"; do
    config="${configs[$i]}"
    echo "[line $LINENO] Running ($((i + 1))/${#configs[@]}): python src/pipeline.py -c -cf ${config}"
    python src/pipeline.py -c -cf "${config}"
done
