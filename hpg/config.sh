#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Single source of truth for all HiPerGator paths and identifiers.
# Every hpg/*.sh script sources this file.  EDIT THE TOP BLOCK ONCE.
# ─────────────────────────────────────────────────────────────────────────────

# ── EDIT ME ──────────────────────────────────────────────────────────────────
# Your UF group (= SLURM account).  `id -gn` on a login node, or check the
# output of `slurmInfo`.  Investment QOS is <GROUP>, burst QOS is <GROUP>-b,
# but note: GPU partitions have NO burst QOS, so GPU jobs must use <GROUP>.
HPG_GROUP="${HPG_GROUP:-CHANGEME}"

# Your GatorLink username.
HPG_USER="${HPG_USER:-$USER}"

# Docker Hub tag holding the cu128 image (see Dockerfile.hpg / build_sif.sh).
DOCKER_IMAGE="${DOCKER_IMAGE:-nevneal/remote_sam3:hpg}"
# ─────────────────────────────────────────────────────────────────────────────

# ── Derived paths — all work lives on /blue, never in $HOME (40 GB quota) ────
BLUE="/blue/${HPG_GROUP}/${HPG_USER}"
PROJECT_DIR="${PROJECT_DIR:-${BLUE}/remote_sam3}"   # git checkout
SIF_DIR="${SIF_DIR:-${BLUE}/sif}"                   # container images
SIF="${SIF:-${SIF_DIR}/remote_sam3_hpg.sif}"
PARQUET="${PARQUET:-${BLUE}/data/inat_photos.parquet}"
RESULTS_ROOT="${RESULTS_ROOT:-${BLUE}/results}"      # per-run output trees
LOG_DIR="${LOG_DIR:-${BLUE}/logs}"

# Shared HuggingFace cache.  CRITICAL: without this every array task would
# re-download SAM3 (~3 GB) — 300 tasks would pull ~900 GB and rate-limit you.
# Warm it once with setup_blue.sh, then tasks read it offline.
export HF_HOME="${HF_HOME:-${BLUE}/hf_cache}"

# Apptainer caches are large; keep them off /home.
export APPTAINER_CACHEDIR="${APPTAINER_CACHEDIR:-${BLUE}/.apptainer/cache}"
export APPTAINER_TMPDIR="${APPTAINER_TMPDIR:-${BLUE}/.apptainer/tmp}"
