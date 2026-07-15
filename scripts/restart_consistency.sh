#!/usr/bin/env bash
# Consistency workflow: pairs corpus + fine-tune (requires trained CDVAE + featurized shards).
#
# Usage:
#   bash scripts/restart_consistency.sh pairs    # generate consistency corpus only
#   bash scripts/restart_consistency.sh train    # fine-tune only (corpus must exist)
#   bash scripts/restart_consistency.sh all      # pairs then train
set -euo pipefail

IMAGE="${PHLOGISTON_IMAGE:-phlogiston:rocm}"
REPO="${REPO:-$HOME/Phlogiston}"
DATA="${DATA:-$HOME/phlogiston-data}"
GEN="${GEN:-data/runs/cdvae_long/cdvae_best.pt}"
PAIRS_ROOT="${PAIRS_ROOT:-data/runs/consistency_corpus}"
OUT="${OUT:-data/runs/cdvae_consistency}"
# First GPU index for 4-way DDP / worker pinning (stick to GPUs 4-7)
GPU_BASE="${GPU_BASE:-4}"
N_GPUS="${N_GPUS:-4}"

die() { echo "ERROR: $*" >&2; exit 1; }

[ -f "$DATA/runs/cdvae_long/cdvae_best.pt" ] || die "missing $DATA/runs/cdvae_long/cdvae_best.pt"
[ -d "$DATA/processed/shards" ] || die "missing $DATA/processed/shards"

run_docker() {
    local name=$1 gpu=$2
    shift 2
    docker rm -f "$name" >/dev/null 2>&1 || true
    docker run -d --name "$name" \
        --ipc=host --network=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --security-opt label=disable \
        --shm-size=64g \
        -e ROCR_VISIBLE_DEVICES="$gpu" \
        -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
        -w /workspace/Phlogiston \
        -v "$REPO":/workspace/Phlogiston \
        -v "$DATA":/workspace/Phlogiston/data \
        "$IMAGE" "$@"
}

pairs() {
    echo "=== consistency pairs corpus (${N_GPUS} workers, GPUs ${GPU_BASE}-$((GPU_BASE + N_GPUS - 1))) ==="
    for k in $(seq 0 $((N_GPUS - 1))); do
        gpu=$((GPU_BASE + k))
        run_docker "phlog_pairs_$k" "$gpu" \
            python -m phlogiston.cli distill-corpus \
            --generator "$GEN" \
            --out-root "$PAIRS_ROOT" \
            --n-samples 700 \
            --gen-batch-size 350 \
            --relax-steps 300 \
            --keep-fmax 0.2 \
            --store-disp \
            --shard-size 2000 \
            --shard-start "$k" \
            --tag "w$k" \
            --seed "$k"
        echo "  launched phlog_pairs_$k on GPU $gpu"
    done
}

train() {
    echo "=== consistency fine-tune (DDP ${N_GPUS} GPUs) ==="
    local gpus
    gpus=$(seq -s, "$GPU_BASE" $((GPU_BASE + N_GPUS - 1)))
    docker rm -f phlog_ft >/dev/null 2>&1 || true
    docker run -d --name phlog_ft \
        --ipc=host --network=host \
        --device=/dev/kfd --device=/dev/dri \
        --group-add video --security-opt label=disable \
        --shm-size=64g \
        -e ROCR_VISIBLE_DEVICES="$gpus" \
        -e OMP_NUM_THREADS=8 -e MKL_NUM_THREADS=8 -e OPENBLAS_NUM_THREADS=8 \
        -w /workspace/Phlogiston \
        -v "$REPO":/workspace/Phlogiston \
        -v "$DATA":/workspace/Phlogiston/data \
        "$IMAGE" \
        torchrun --nproc_per_node="$N_GPUS" -m phlogiston.cli train-cdvae \
            --init-ckpt "$GEN" \
            --n-layers 2 \
            --max-shards 40 \
            --consistency-root "$PAIRS_ROOT" \
            --consistency-weight 2.0 \
            --consistency-batch-size 64 \
            --epochs 12 --warmup-epochs 1 --patience 3 \
            --lr 3e-4 --batch-size 256 --num-workers 6 \
            --out-dir "$OUT"
    echo "  launched phlog_ft on GPUs $gpus"
}

cmd="${1:-all}"
case "$cmd" in
    pairs) pairs ;;
    train) train ;;
    all)   pairs; echo "waiting for pairs workers — run: bash scripts/restart_consistency.sh train";;
    *) die "usage: $0 pairs|train|all" ;;
esac
