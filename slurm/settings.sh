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

# ── What to segment ──────────────────────────────────────────────────────────
TAXON_ID="${TAXON_ID:-160559}"
PROMPT="${PROMPT:-flower}"

# Only keep detections at or above this confidence.
MIN_SCORE="${MIN_SCORE:-0.9}"

# Parallel image downloads per GPU. An hpg-turin node has 96 cores for 3 L4s,
# so this can go much higher than it could on CHTC — see README "Axis 2".
WORKERS="${WORKERS:-16}"

# ── Derived ──────────────────────────────────────────────────────────────────
OUT_DIR="${OUT_DIR:-${RESULTS}/${TAXON_ID}_${PROMPT// /_}}"
