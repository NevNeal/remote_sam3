#!/bin/bash
# build_apptainer.sh — build remote_sam3.sif and stage it for HTCondor.
#
# Two supported flows depending on where you start:
#
#   A. You already pushed the Docker image to a registry (Docker Hub or GHCR).
#      Build the SIF directly on a CHTC submit node from the registry URL.
#      Recommended — no local Apptainer install needed.
#
#   B. You only have the Dockerfile locally. Build a Docker image, save it as
#      a tar, scp it up, and convert on the submit node.
#
# This script implements (A). For (B) see instructions.md.

set -euo pipefail

DOCKER_IMAGE="${DOCKER_IMAGE:-ghcr.io/<GITHUB_USER>/remote_sam3:latest}"
NETID="${NETID:-<NETID>}"
STAGING="/staging/${NETID}"
SIF_NAME="remote_sam3.sif"

if [[ "$NETID" == "<NETID>" ]]; then
    echo "ERROR: set NETID env var, e.g. NETID=jdaniels ./chtc/build_apptainer.sh"
    exit 1
fi

if [[ "$DOCKER_IMAGE" == *"<GITHUB_USER>"* ]]; then
    echo "ERROR: set DOCKER_IMAGE env var, e.g."
    echo "  DOCKER_IMAGE=ghcr.io/nevneal/remote_sam3:latest ./chtc/build_apptainer.sh"
    exit 1
fi

mkdir -p "$STAGING"

echo "[build] pulling $DOCKER_IMAGE into Apptainer SIF..."
apptainer build "$STAGING/$SIF_NAME" "docker://$DOCKER_IMAGE"

echo "[build] done. SIF is at: $STAGING/$SIF_NAME"
echo "[build] In run_tests.sub, container_image should be:"
echo "        container_image = osdf:///chtc/staging/$NETID/$SIF_NAME"
