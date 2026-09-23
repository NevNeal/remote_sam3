#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# THE ONLY FILE YOU EDIT.
#
# Sourced by setup_env.sh and by every .sbatch script, always from the project
# root (the flower_sam3 checkout). Everything the project touches lives inside
# that directory, so nothing here needs your group name:
#
#     flower_sam3/
#     ├── data/inat_photos.parquet     the index (already there)
#     ├── .conda/sam3/                 the conda env      (setup_env.sh)
#     ├── hf_cache/                    SAM3 weights       (setup_env.sh)
#     ├── results/<taxon>_<prompt>/    run outputs        (the jobs)
#     └── logs/                        SLURM .out/.err    (the jobs)
#
# Anything here can be overridden per-run from the command line, because sbatch
# passes your environment through to the job:
#
#     TAXON_ID=62741 PROMPT=petal sbatch --array=0-39 slurm/array.sbatch
# ─────────────────────────────────────────────────────────────────────────────

# The project root. Every caller cd's there before sourcing this file.
PROJECT="${PROJECT:-$PWD}"

# A prefix env inside the project rather than a named one, so it lands on /blue
# no matter how conda's envs_dirs is configured, and never in the 40 GB $HOME.
CONDA_ENV="${CONDA_ENV:-${PROJECT}/.conda/sam3}"

PARQUET="${PARQUET:-${PROJECT}/data/inat_photos.parquet}"
RESULTS="${RESULTS:-${PROJECT}/results}"

# Shared, pre-warmed model cache — see setup_env.sh.
export HF_HOME="${HF_HOME:-${PROJECT}/hf_cache}"

# ── Images already on /blue, instead of a taxon from S3 ──────────────────────
# Set LOCAL_INDEX and every job segments the files listed in that parquet, in
# place, with no downloads — TAXON_ID and PARQUET are then unused. IMAGE_ROOT is
# the directory the index's leading 'data/' maps to:
#
#   index:  data/phenobase_inat_data/images/medium/batch_1/9208.webp
#   disk:   $IMAGE_ROOT/phenobase_inat_data/images/medium/batch_1/9208.webp
#
# slurm/local.sbatch sets LOCAL_INDEX for you; leave it empty for taxon runs.
LOCAL_INDEX="${LOCAL_INDEX:-}"
IMAGE_ROOT="${IMAGE_ROOT:-/home/neal.nevyn/blue_guralnick/share}"

# ── What to segment ──────────────────────────────────────────────────────────
TAXON_ID="${TAXON_ID:-160559}"
PROMPT="${PROMPT:-flower}"

# Only keep detections at or above this confidence.
MIN_SCORE="${MIN_SCORE:-0.9}"

# Threads reading images ahead of the GPU, per shard. An hpg-turin node has 96
# cores for 3 L4s, so this can go much higher than it could on CHTC — see README
# "Axis 2". On a /blue run these are local reads, not downloads, so the ceiling
# is the filesystem rather than iNat's patience.
WORKERS="${WORKERS:-16}"

# ── How hard one GPU is driven ───────────────────────────────────────────────
# Images per forward pass. 1 is one image at a time, which is all a 24 GB L4 has
# room for at these resolutions. A B200 has 180 GB and is wasted at 1 — see
# README "6d. One B200". SAVE_WORKERS must go up with it, or the GPU finishes a
# batch of 32 and then waits for one thread to write 32 overlays.
BATCH_SIZE="${BATCH_SIZE:-1}"
SAVE_WORKERS="${SAVE_WORKERS:-1}"

# BF16=1 runs the forward pass in bfloat16: roughly twice the throughput on
# Blackwell, and slightly different mask edges. Off by default because every run
# so far has been fp32 and the two are not bit-comparable.
BF16="${BF16:-}"

# MAX_SECONDS caps the segmentation loop instead of the photo list, for timed
# tests ("how many images in an hour?"). Everything finished is kept and the
# next run resumes from it. Empty means run until the shard's rows are done.
MAX_SECONDS="${MAX_SECONDS:-}"

# ── Derived ──────────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-${RESULTS}/${TAXON_ID}_${PROMPT// /_}}"
