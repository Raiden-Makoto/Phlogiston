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
# Host dir persisted across container runs and mounted at the repo's data/ dir
# (datasets are large; we don't want to re-download them every session).
DATA_DIR="${GBT_DATA_DIR:-/home/macui/phlogiston-data}"
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
# Read an env var, falling back to its `export VAR=...` line in ~/.bashrc
# (scripts run non-interactively, so ~/.bashrc isn't sourced for us).
resolve_var() {
    local name="$1" val="${!1:-}"
    if [ -z "${val}" ]; then
        val="$(grep -E "^\s*export\s+${name}=" "${HOME}/.bashrc" 2>/dev/null \
            | tail -1 | sed -E "s/^\s*export\s+${name}=//" | tr -d '"'\''')"
    fi
    printf '%s' "${val}"
}

# Materials Project API key env var (contents fetched into the container).
MP_KEY_VAR="${MP_KEY_VAR:-MP54AC}"

resolve_pat() {
    GITHUB_PAT="$(resolve_var "${PAT_VAR}")"
    [ -n "${GITHUB_PAT}" ] || { echo "${PAT_VAR} not set and not found in ~/.bashrc" >&2; exit 1; }
}

run() {
    echo ">> Launching container ${CONTAINER} on ${SSH_HOST}"
    resolve_pat
    MP_KEY="$(resolve_var "${MP_KEY_VAR}")"   # optional; only needed for MP fetches
    # Ship secrets to the box in a 0600 env-file (removed after the session),
    # then pass them into the container: GITHUB_PAT for git auth over HTTPS and
    # MP_API_KEY for Materials Project fetches.
    ssh "${SSH_HOST}" "umask 077 && { printf 'GITHUB_PAT=%s\n' '${GITHUB_PAT}'; printf 'MP_API_KEY=%s\n' '${MP_KEY}'; } > ~/.phlogiston.env && mkdir -p '${DATA_DIR}'"
    # Pass the host's numeric render/video GIDs (names don't reliably map) so
    # the container can access /dev/kfd and /dev/dri.
    ssh -t "${SSH_HOST}" 'docker run -it --rm \
        --name '"${CONTAINER}"' \
        --env-file ~/.phlogiston.env \
        -v '"${DATA_DIR}"':/workspace/Phlogiston/data \
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
