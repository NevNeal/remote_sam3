#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# run_shard.sh — launch one shard inside the Apptainer container.
#
# This is the HiPerGator equivalent of chtc/run_staging.sh.  It is called from
# the sbatch scripts, never directly.  Arguments:
#
#   run_shard.sh <taxon_id> <output_dir> <prompt> <shard> <num_shards> [gpu_index]
#
# gpu_index is optional and only used by sbatch_b200_node.sh, which packs
# several shards onto one 8-GPU node and pins each to its own device.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "$HERE/config.sh"

TAXON_ID="${1:?usage: run_shard.sh <taxon_id> <output_dir> <prompt> <shard> <num_shards> [gpu_index]}"
OUTPUT_DIR="${2:?missing output_dir}"
PROMPT="${3:-flower}"
SHARD="${4:-0}"
NUM_SHARDS="${5:-1}"
GPU_INDEX="${6:-}"

WORKERS="${WORKERS:-8}"
AMP_FLAG="${AMP_FLAG:---amp}"
LIMIT_FLAG="${LIMIT_FLAG:-}"

# Pin this process to one GPU when several shards share a node.
if [[ -n "$GPU_INDEX" ]]; then
    export CUDA_VISIBLE_DEVICES="$GPU_INDEX"
fi

# The HF cache on /blue is pre-warmed by setup_blue.sh.  Going offline keeps
# hundreds of concurrent shards from hammering (and being throttled by)
# huggingface.co, and makes the job independent of HF uptime.
export HF_HOME
export HF_HUB_OFFLINE="${HF_HUB_OFFLINE:-1}"
export TOKENIZERS_PARALLELISM=false

# Match the thread count to what SLURM actually gave us, or BLAS will spawn 96
# threads per shard and thrash a shared node.
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"

mkdir -p "$OUTPUT_DIR"

echo "[shard ${SHARD}/${NUM_SHARDS}] host=$(hostname)  job=${SLURM_JOB_ID:--}  task=${SLURM_ARRAY_TASK_ID:--}"
echo "[shard ${SHARD}/${NUM_SHARDS}] taxon=$TAXON_ID  prompt='$PROMPT'  out=$OUTPUT_DIR"
echo "[shard ${SHARD}/${NUM_SHARDS}] gpu=${CUDA_VISIBLE_DEVICES:-all}  workers=$WORKERS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv || echo "(no nvidia-smi)"

[[ -f "$PARQUET" ]] || { echo "ERROR: parquet not found at $PARQUET"; exit 1; }
[[ -f "$SIF"     ]] || { echo "ERROR: container not found at $SIF (run hpg/build_sif.sh)"; exit 1; }

module purge
module load apptainer

# --nv exposes the NVIDIA driver/devices.  /blue is auto-mounted on compute
# nodes but bind it explicitly so the path inside the container is identical.
exec apptainer exec --nv \
    --bind "/blue/${HPG_GROUP}:/blue/${HPG_GROUP}" \
    --env "HF_HOME=$HF_HOME" \
    --env "HF_HUB_OFFLINE=$HF_HUB_OFFLINE" \
    --env "OMP_NUM_THREADS=$OMP_NUM_THREADS" \
    --env "TOKENIZERS_PARALLELISM=false" \
    "$SIF" \
    python /app/hpg_pipeline.py \
        --taxon-id   "$TAXON_ID"   \
        --output-dir "$OUTPUT_DIR" \
        --parquet    "$PARQUET"    \
        --prompt     "$PROMPT"     \
        --shard      "$SHARD"      \
        --num-shards "$NUM_SHARDS" \
        --workers    "$WORKERS"    \
        $AMP_FLAG $LIMIT_FLAG
