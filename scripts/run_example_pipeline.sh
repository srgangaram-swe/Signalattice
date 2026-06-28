#!/usr/bin/env bash
set -euo pipefail

CONFIG="${1:-configs/example.yaml}"

python -m quant_platform.cli run-full-pipeline --config "${CONFIG}" --force
