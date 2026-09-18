#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One shard on one GPU. Not submitted directly — every .sbatch that segments
# images ends by running this, so all of them do exactly the same thing:
#
#     test.sbatch        1 shard,  100 images
#     test_2gpu.sbatch   2 shards, 200 images, both GPUs at once
#     array.sbatch       N shards, the whole taxon
#
# Inputs (environment): SHARD, NUM_SHARDS, and optionally LIMIT and OUT_DIR,
# plus everything in settings.sh.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"

if [[ ! -f segment.py || ! -f slurm/settings.sh ]]; then
    echo "ERROR: submit from the project root (the flower_sam3 checkout)," >&2
    echo "       not from $SLURM_SUBMIT_DIR" >&2
    exit 1
fi
source slurm/settings.sh

: "${SHARD:?SHARD not set}"
: "${NUM_SHARDS:?NUM_SHARDS not set}"

# ── Preflight: fail in seconds, not after a queue wait and a model load ─────
fail=0
if [[ ! -f "$PARQUET" ]]; then
    echo "ERROR: no parquet at $PARQUET" >&2
    ls -la "$(dirname "$PARQUET")"/*.parquet 2>/dev/null >&2 \
        && echo "       set PARQUET=<one of the above> in slurm/settings.sh" >&2
    fail=1
fi
if [[ ! -d "$CONDA_ENV/conda-meta" ]]; then
    echo "ERROR: no conda env at $CONDA_ENV — run ./setup_env.sh first" >&2
    fail=1
fi
if [[ ! -d "$HF_HOME/hub" ]]; then
    echo "ERROR: no SAM3 weights under $HF_HOME — run ./setup_env.sh first" >&2
    fail=1
fi
(( fail == 0 )) || exit 1

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u    # conda's activate scripts trip over set -u

export HF_HUB_OFFLINE=1          # weights are pre-cached by setup_env.sh
# Without this, every shard's BLAS spawns a thread per core on the node and
# three shards sharing an hpg-turin node thrash each other.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

# Who am I and where did I land. For a parallel run, compare these lines across
# the shards' logs: different GPU UUIDs and overlapping start times mean the
# shards genuinely ran side by side on separate GPUs.
echo "shard      : $SHARD of $NUM_SHARDS"
echo "job        : ${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:--}} task ${SLURM_ARRAY_TASK_ID:--}"
echo "node       : $(hostname)"
echo "gpu        : $(nvidia-smi --query-gpu=name,uuid,memory.total --format=csv,noheader)"
echo "started    : $(date -Iseconds)"

args=(
    --taxon-id   "$TAXON_ID"
    --parquet    "$PARQUET"
    --out        "$OUT_DIR"
    --prompt     "$PROMPT"
    --min-score  "$MIN_SCORE"
    --workers    "$WORKERS"
    --shard      "$SHARD"
    --num-shards "$NUM_SHARDS"
)
# LIMIT takes the first N photos of the taxon BEFORE sharding, so
# LIMIT=200 with 2 shards gives each shard 100.
if [[ -n "${LIMIT:-}" ]]; then
    args+=(--limit "$LIMIT")
fi

python segment.py "${args[@]}"

echo "finished   : $(date -Iseconds)"
