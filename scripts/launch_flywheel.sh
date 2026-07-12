#!/usr/bin/env bash
# Launch one turn of the active-learning flywheel in a detached container.
#
# Bakes in --env-file (written by deploy_gbt.sh) so the Tier-2 verify step can
# reach Materials Project for convex-hull placement, and pins to a healthy GPU
# (some devices on the box have flaky HBM/ECC -- override with GPU=<n>).
#
# Usage:  bash scripts/launch_flywheel.sh          # detached, GPU 4
#         GPU=1 NAME=phlog_fly2 bash scripts/launch_flywheel.sh
set -euo pipefail

NAME="${NAME:-phlog_fly}"
GPU="${GPU:-4}"
IMAGE="${PHLOGISTON_IMAGE:-phlogiston:rocm}"
ENV_FILE="${ENV_FILE:-$HOME/.phlogiston.env}"
REPO="${REPO:-$HOME/Phlogiston}"
DATA="${DATA:-$HOME/phlogiston-data}"

[ -f "$ENV_FILE" ] || { echo "missing $ENV_FILE (run deploy_gbt.sh sync first)" >&2; exit 1; }

docker rm -f "$NAME" >/dev/null 2>&1 || true
docker run -d --name "$NAME" \
    --ipc=host --network=host \
    --device=/dev/kfd --device=/dev/dri --group-add video \
    --security-opt label=disable \
    --env-file "$ENV_FILE" \
    -e ROCR_VISIBLE_DEVICES="$GPU" -e HIP_VISIBLE_DEVICES=0 \
    -w /workspace/Phlogiston \
    -v "$REPO":/workspace/Phlogiston \
    -v "$DATA":/workspace/Phlogiston/data \
    "$IMAGE" bash /workspace/Phlogiston/scripts/run_flywheel.sh
echo "launched $NAME on GPU $GPU"
