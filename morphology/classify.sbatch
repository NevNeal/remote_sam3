#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Flower morphology per genus on one B200: select 10 photos per genus from run
# 2, ask Qwen3-VL-8B the questions in morphology/questions.py about each, and
# summarize per genus and family. ~90k photos x 6 questions.
#
# The job stops itself at MAX_SECONDS (7.5 h) and still writes the summaries,
# so it always finishes inside the 8 h limit; submitting it again picks up the
# photos that are left. The log prints photos/s and an ETA as it goes.
#
#   METADATA   run 2's all_genera_segment_metadata.csv (the report's export)
#   RUN_DIR    run 2's results dir; the metadata's mask paths are relative to it
#   MORPH_OUT  where photos.csv, genus_morphology.csv, ... go
#   LIMIT=2000 for a short test first
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=morphology
#SBATCH --partition=hpg-b200
#SBATCH --gpus=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128gb
#SBATCH --time=07:59:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
#SBATCH --error=logs/%x-%j.err

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source slurm/settings.sh
METADATA="${METADATA:-${PROJECT}/export/all_genera_run2/all_genera_segment_metadata.csv}"
RUN_DIR="${RUN_DIR:-${RESULTS}/all_genera_run2_${PROMPT// /_}}"
INDEX="${INDEX:-${PROJECT}/data/morphology_index.parquet}"
MORPH_OUT="${MORPH_OUT:-${RESULTS}/morphology}"
MORPH_MODEL="${MORPH_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
PER_GENUS="${PER_GENUS:-10}"
BATCH_SIZE="${BATCH_SIZE:-64}"
MAX_SECONDS="${MAX_SECONDS:-27000}"
LIMIT="${LIMIT:-0}"
export MORPH_MODEL

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
export OMP_NUM_THREADS=1
export TOKENIZERS_PARALLELISM=false
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader

# The weights (~17 GB) come from $HF_HOME if they are there, else are fetched
# once; `python morphology/classify.py download` on a login node does it ahead.
if ! HF_HUB_OFFLINE=1 python -c "from huggingface_hub import snapshot_download as s; s('$MORPH_MODEL', local_files_only=True)" 2>/dev/null; then
    echo "[morphology] $MORPH_MODEL not cached in $HF_HOME - downloading"
    python morphology/classify.py download --model "$MORPH_MODEL"
fi
export HF_HUB_OFFLINE=1

if [[ ! -f "$INDEX" ]]; then
    if [[ ! -f "$METADATA" ]]; then
        echo "ERROR: no run 2 metadata at $METADATA - set METADATA=..." >&2
        exit 1
    fi
    python morphology/classify.py select --metadata "$METADATA" --out "$INDEX" \
        --per-genus "$PER_GENUS"
fi

python morphology/classify.py classify --index "$INDEX" --run-dir "$RUN_DIR" \
    --out-dir "$MORPH_OUT" --model "$MORPH_MODEL" --batch-size "$BATCH_SIZE" \
    --workers $(( ${SLURM_CPUS_PER_TASK:-16} - 4 )) --max-seconds "$MAX_SECONDS" \
    --limit "$LIMIT"
python morphology/classify.py summarize --out-dir "$MORPH_OUT"
