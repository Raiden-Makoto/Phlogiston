#!/usr/bin/env bash
# Deploy Phlogiston to the gbt350 box (8x AMD Instinct MI350X, gfx950).
#
# Local Docker is unavailable here (and shipping a 31GB ROCm image over SSH is
# impractical), so we sync the repo up and build on the box, where the ROCm
# PyTorch base image is already cached.
#
# Usage:
#   docker/deploy_gbt.sh sync     # rsync repo -> box:~/Phlogiston
#   docker/deploy_gbt.sh build    # build image on the box
#   docker/deploy_gbt.sh run      # start an interactive container on the box
#   docker/deploy_gbt.sh all      # sync + build (then use `run`)
set -euo pipefail

SSH_HOST="${GBT_HOST:-gbt}"                 # matches the `Host gbt` ssh config entry
REMOTE_DIR="${GBT_REMOTE_DIR:-/home/macui/Phlogiston}"
IMAGE="${PHLOGISTON_IMAGE:-phlogiston:rocm}"
CONTAINER="${PHLOGISTON_CONTAINER:-phlogiston}"
LOCAL_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

sync() {
    echo ">> Syncing ${LOCAL_DIR}/ -> ${SSH_HOST}:${REMOTE_DIR}"
    # Keep .git (needed for in-container `git pull`); drop venv/data/artifacts.
    rsync -az --delete \
        --exclude '.venv/' \
        --exclude 'data/raw/' --exclude 'data/cache/' --exclude 'data/processed/' \
        --exclude 'runs/' \
        --exclude '__pycache__/' --exclude '*.pyc' \
        -e "ssh" \
        "${LOCAL_DIR}/" "${SSH_HOST}:${REMOTE_DIR}/"
}

build() {
    echo ">> Building ${IMAGE} on ${SSH_HOST}"
    ssh "${SSH_HOST}" "cd ${REMOTE_DIR} && docker build -t ${IMAGE} ."
}

# Resolve the GitHub PAT for in-container git pull/push from the CROISSANT env
# var (this token has Contents access to the repo). Prefer an already exported
# value; otherwise read it out of the local ~/.bashrc (scripts run
# non-interactively, so we can't rely on it being sourced).
PAT_VAR="${PAT_VAR:-CROISSANT}"
resolve_pat() {
    GITHUB_PAT="${!PAT_VAR:-}"
    if [ -z "${GITHUB_PAT}" ]; then
        GITHUB_PAT="$(grep -E "^\s*export\s+${PAT_VAR}=" "${HOME}/.bashrc" 2>/dev/null \
            | tail -1 | sed -E "s/^\s*export\s+${PAT_VAR}=//" | tr -d '"'\''')"
    fi
    [ -n "${GITHUB_PAT}" ] || { echo "${PAT_VAR} not set and not found in ~/.bashrc" >&2; exit 1; }
}

run() {
    echo ">> Launching container ${CONTAINER} on ${SSH_HOST}"
    resolve_pat
    # Ship the PAT to the box in a 0600 env-file (removed after the session),
    # then pass it into the container so git can auth over HTTPS.
    ssh "${SSH_HOST}" "umask 077 && printf 'GITHUB_PAT=%s\n' '${GITHUB_PAT}' > ~/.phlogiston.env"
    # Pass the host's numeric render/video GIDs (names don't reliably map) so
    # the container can access /dev/kfd and /dev/dri.
    ssh -t "${SSH_HOST}" 'docker run -it --rm \
        --name '"${CONTAINER}"' \
        --env-file ~/.phlogiston.env \
        --device=/dev/kfd --device=/dev/dri \
        --group-add=$(getent group render | cut -d: -f3) \
        --group-add=$(getent group video | cut -d: -f3) \
        --ipc=host --network host \
        --cap-add=SYS_PTRACE --security-opt seccomp=unconfined \
        --shm-size 16G \
        '"${IMAGE}"' bash; rm -f ~/.phlogiston.env'
}

cmd="${1:-all}"
case "${cmd}" in
    sync)  sync ;;
    build) build ;;
    run)   run ;;
    all)   sync && build && echo ">> Done. Start a session with: docker/deploy_gbt.sh run" ;;
    *) echo "Unknown command: ${cmd}" >&2; echo "Use: sync | build | run | all" >&2; exit 1 ;;
esac
