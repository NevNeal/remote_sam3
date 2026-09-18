#!/bin/bash
# push_docker_image_hpg.sh — build the cu128 HiPerGator image and push it.
# Run on your local workstation (Docker Desktop), same as the CHTC script.
#
# HiPerGator cannot run Docker (no root on compute nodes), so the image is
# pulled and converted to an Apptainer .sif on a HPG login node afterwards —
# see hpg/build_sif.sh.
#
# Usage:
#   DOCKERHUB_USER=nevneal ./hpg/push_docker_image_hpg.sh

set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:-nevneal}"
TAG="${TAG:-hpg}"
IMAGE="$DOCKERHUB_USER/remote_sam3:$TAG"

cd "$(dirname "$0")/.."     # build context = repo root

echo "[push] building $IMAGE from hpg/Dockerfile.hpg ..."
docker build -f hpg/Dockerfile.hpg -t "$IMAGE" .

echo "[push] pushing $IMAGE ..."
docker push "$IMAGE"

echo "[push] done."
echo "[push] Next, on a HiPerGator login node:"
echo "        DOCKER_IMAGE=$IMAGE ./hpg/build_sif.sh"
