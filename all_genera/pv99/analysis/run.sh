#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# The pv99 analysis on HPG, written to export/pv99/analysis/. From the project
# root (the flower_sam3 checkout), after the pv99 export:
#
#     bash all_genera/pv99/analysis/run.sh
#
# On the login node: installs reportlab into the env if missing and caches
# Qwen3-VL-8B-Instruct (~17 GB, once) under $HF_HOME. Then two chained jobs:
#
#   1. prep.sbatch    CPU    histogram, top 100 segments of the 30 biggest
#                            families (cut-outs, context crops, contact sheets),
#                            review figures for 30 photos per family
#   2. morph.sbatch   B200   flower-type calls for the 3,000 segments, the
#                            family comparison figure, the methodology PDF
#
# SKIP_PREP=1 submits only step 2 (prep outputs already there).
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
if [[ ! -f segment.py || ! -f all_genera/pv99/analysis/run.sh ]]; then
    echo "ERROR: run from the project root (the flower_sam3 checkout)" >&2
    exit 1
fi
if [[ ! -f export/pv99/pv99_segment_metadata.parquet ]]; then
    echo "ERROR: no export/pv99/pv99_segment_metadata.parquet - run the pv99 export first" >&2
    exit 1
fi
source slurm/settings.sh
mkdir -p logs export/pv99/analysis

module purge
module load conda
set +u; conda activate "$CONDA_ENV"; set -u
python -c "import reportlab" 2>/dev/null || pip install --quiet reportlab
python - <<'EOF'
from huggingface_hub import snapshot_download
path = snapshot_download("Qwen/Qwen3-VL-8B-Instruct")
print(f"model      : {path}")
EOF

dep=""
if [[ -z "${SKIP_PREP:-}" ]]; then
    prep=$(sbatch --parsable all_genera/pv99/analysis/prep.sbatch)
    echo "1. prep    : job $prep   (CPU)"
    dep="--dependency=afterok:$prep"
fi
morph=$(sbatch --parsable $dep all_genera/pv99/analysis/morph.sbatch)
echo "2. morph   : job $morph   (hpg-b200)"
echo
echo "watch      : squeue -u \$USER"
echo "logs       : logs/pv99_analysis_*-<job>.out"
echo "outputs    : export/pv99/analysis/   (PDF: flower_types_methodology.pdf)"
