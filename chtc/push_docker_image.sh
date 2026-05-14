#!/bin/bash
# push_docker_image.sh — build the Docker image and push it to Docker Hub.
# Run this on your local workstation (where Docker Desktop is running).
# CHTC compute nodes will pull this image directly via `container_image = docker://...`.
#
# Usage:
#   DOCKERHUB_USER=yourname ./chtc/push_docker_image.sh

set -euo pipefail

DOCKERHUB_USER="${DOCKERHUB_USER:-nevneal}"
TAG="${TAG:-latest}"
IMAGE="$DOCKERHUB_USER/remote_sam3:$TAG"

echo "[push] building $IMAGE..."
docker build -t "$IMAGE" .

echo "[push] pushing $IMAGE to Docker Hub..."
docker push "$IMAGE"

echo "[push] done."
echo "[push] In chtc/run_tests.sub, container_image should be:"
echo "        container_image = docker://$IMAGE"
