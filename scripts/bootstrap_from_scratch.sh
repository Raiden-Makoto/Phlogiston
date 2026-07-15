#!/usr/bin/env bash
# Phlogiston bootstrap on a fresh GPU box: fetch -> featurize. Stops there;
# CDVAE training / consistency workflow are launched separately once this
# completes and the data is inspected.
set -euo pipefail

IMAGE="${PHLOGISTON_IMAGE:-phlogiston:rocm}"
REPO="${REPO:-$HOME/Phlogiston}"
DATA="${DATA:-$HOME/phlogiston-data}"
ENV_FILE="${ENV_FILE:-$HOME/.phlogiston.env}"
GPU_BASE="${GPU_BASE:-4}"
N_GPUS="${N_GPUS:-4}"
LOG="${LOG:-$HOME/phlogiston-bootstrap.log}"

exec > >(tee -a "$LOG") 2>&1
echo "[bootstrap] $(date -Is) starting on $(hostname)"

wait_image() {
    while ! docker image inspect "$IMAGE" >/dev/null 2>&1; do
        echo "[bootstrap] waiting for docker image $IMAGE ..."
        sleep 60
    done
    echo "[bootstrap] image $IMAGE ready"
}

run_cpu() {
    local name=$1; shift
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run --rm --name "$name" \
        --network=host \
        --env-file "$ENV_FILE" \
        -w /workspace/Phlogiston \
        -v "$REPO":/workspace/Phlogiston \
        -v "$DATA":/workspace/Phlogiston/data \
        "$IMAGE" "$@"
}

wait_image
mkdir -p "$DATA"
[ -f "$ENV_FILE" ] || { echo "[bootstrap] missing $ENV_FILE (need MP API key)" >&2; exit 1; }

echo "[bootstrap] === fetch GNoME ==="
run_cpu phlog_fetch_gnome python -m phlogiston.cli fetch-gnome \
    --keys summary_pbe mp_snapshot structures_by_id

echo "[bootstrap] === fetch Materials Project (stable + near-stable) ==="
run_cpu phlog_fetch_mp python -m phlogiston.cli fetch-mp \
    --max-energy-above-hull 0.1 --exclude-radioactive

echo "[bootstrap] === featurize ==="
run_cpu phlog_featurize python -m phlogiston.cli featurize --workers 16

echo "[bootstrap] $(date -Is) featurization complete -- stopping here as requested."
echo "[bootstrap] next: train-cdvae on GPUs ${GPU_BASE}-$((GPU_BASE + N_GPUS - 1))"
