#!/usr/bin/env bash
# One-time rsync of phlogiston-data from the old gbt350 box to the current GPU box.
# Run when BOTH boxes have active Conductor reservations.
#
# Usage (from your laptop):
#   bash docker/sync_data_from_old.sh
#
# Transfers runs/ + processed/shards/ (~tens of GB). Skips raw GNoME downloads.
set -euo pipefail

OLD_HOST="${OLD_HOST:-gbt350-odcdh2-b13-1.png-odc.dcgpu}"
NEW_HOST="${NEW_HOST:-smci355-ccs-aus-m12-17.cs-aus.dcgpu}"
USER="${USER:-macui}"
SRC="${USER}@${OLD_HOST}:phlogiston-data/"
DST="${USER}@${NEW_HOST}:phlogiston-data/"

echo ">> rsync ${SRC} -> ${DST}"
ssh "${USER}@${NEW_HOST}" "mkdir -p ~/phlogiston-data/runs ~/phlogiston-data/processed"
rsync -az --info=progress2 \
    --include 'runs/***' \
    --include 'processed/shards/***' \
    --include 'processed/manifest.json' \
    --exclude '*' \
    -e ssh "${SRC}" "${DST}"
echo ">> done. Verify on new box:"
echo "   ssh ${USER}@${NEW_HOST} 'ls -la ~/phlogiston-data/runs/cdvae_long/cdvae_best.pt'"
