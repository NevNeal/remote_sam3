#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# THE ONLY FILE YOU EDIT.
#
# Sourced by setup_env.sh and by every .sbatch script. Anything here can also be
# overridden per-run from the command line, because sbatch passes your
# environment through to the job:
#
#     TAXON_ID=62741 PROMPT=petal sbatch --array=0-39 slurm/array.sbatch
# ─────────────────────────────────────────────────────────────────────────────

# ── Your UF group. `id -gn` on a login node tells you. ───────────────────────
GROUP="${GROUP:-CHANGEME}"

# ── Where everything lives. /blue is the fast shared filesystem; $HOME has a ──
# ── 40 GB quota and is not for job I/O.                                     ──
BLUE="/blue/${GROUP}/${USER}"

CONDA_ENV="${CONDA_ENV:-sam3}"
PARQUET="${PARQUET:-${BLUE}/data/inat_photos.parquet}"
RESULTS="${RESULTS:-${BLUE}/results}"

# Shared, pre-warmed model cache — see setup_env.sh.
export HF_HOME="${HF_HOME:-${BLUE}/hf_cache}"

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
