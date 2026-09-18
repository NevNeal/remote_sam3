#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time environment setup. Run on a HiPerGator LOGIN node, from the project
# root (the flower_sam3 checkout).
#
#   export HF_TOKEN=hf_...        # facebook/sam3 is a gated repo
#   ./setup_env.sh
#
# 1. creates the conda env from environment.yml at ./.conda/sam3
#    (Python 3.12, torch 2.9.1+cu128, requirements.txt)
# 2. checks torch was built for the GPUs HiPerGator has
# 3. pre-downloads the SAM3 weights into ./hf_cache so jobs never touch the
#    network for the model
#
# Takes ~15 minutes, almost all of it downloading wheels. Safe to rerun: an
# existing env is kept, and a warm cache is a no-op.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"
source slurm/settings.sh

: "${HF_TOKEN:?export HF_TOKEN=hf_... first — facebook/sam3 is gated}"

module purge
module load conda

# $HOME is 40 GB. Keep conda's package cache inside the project and skip pip's
# cache entirely (torch + its CUDA libraries are ~4 GB of wheels).
export CONDA_PKGS_DIRS="${PROJECT}/.conda/pkgs"
export PIP_NO_CACHE_DIR=1

if [[ -d "${CONDA_ENV}/conda-meta" ]]; then
    echo "[env] ${CONDA_ENV} already exists — keeping it"
    echo "      (to rebuild: conda env remove -p ${CONDA_ENV} && ./setup_env.sh)"
else
    echo "[env] creating ${CONDA_ENV} from environment.yml"
    conda env create -f environment.yml -p "${CONDA_ENV}"
fi

# shellcheck disable=SC1091
set +u; conda activate "${CONDA_ENV}"; set -u    # conda's activate scripts trip over set -u

echo
echo "── verification ─────────────────────────────────────────────────────────"
python - <<'PY'
import re, torch, transformers
print("python       :", __import__("sys").version.split()[0])
print("torch        :", torch.__version__)
print("torch cuda   :", torch.version.cuda)
print("transformers :", transformers.__version__)
# Not torch.cuda.get_arch_list(): it returns [] whenever no GPU is visible,
# which is always the case on a login node. The flags it wraps are compiled in.
arches = (torch._C._cuda_getArchFlags() or "").split()
print("arch_list    :", arches)

# Code built for sm_XY runs on any GPU with the same major version X and minor
# >= Y, so PyTorch ships sm_86 and no separate sm_89: the L4 runs the sm_86 code.
def runs_on(major, minor):
    for a in arches:
        m = re.fullmatch(r"sm_(\d+?)(\d)a?", a)
        if m and int(m[1]) == major and int(m[2]) <= minor:
            return True
    return False

assert runs_on(8, 9), "no sm_8x <= sm_89 — this build will NOT run on the L4s"
assert runs_on(10, 0), "no sm_100 — this build will NOT run on hpg-b200"
print("L4 (sm_89) and B200 (sm_100) both supported")
# A login node has no GPU, so is_available() is expected to be False here.
# The jobs print the GPU they land on.
print("gpu here     :", torch.cuda.is_available(), "(False is normal on a login node)")
PY

# Pre-download SAM3 once into a cache that every job reads.
#
# Without this, an N-task array means N concurrent ~3 GB downloads from
# HuggingFace and near-certain throttling. With it, jobs run HF_HUB_OFFLINE=1
# and never touch the network for weights at all.
echo
echo "[env] warming the model cache at ${HF_HOME}"
mkdir -p "${HF_HOME}"
python - <<'PY'
from transformers import Sam3Model, Sam3Processor
Sam3Model.from_pretrained("facebook/sam3")
Sam3Processor.from_pretrained("facebook/sam3")
print("model cache warm")
PY
du -sh "${HF_HOME}"

mkdir -p logs
echo
echo "── next ─────────────────────────────────────────────────────────────────"
if [[ -f "${PARQUET}" ]]; then
    echo "  parquet found: ${PARQUET}"
else
    echo "  !! no parquet at ${PARQUET} — set PARQUET in slurm/settings.sh"
fi
echo "  1. Smoke test, 1 GPU:     sbatch slurm/test.sbatch"
echo "  2. Parallel test, 2 GPUs: sbatch slurm/test_2gpu.sbatch"
echo "  3. Full run:              sbatch --array=0-59 slurm/array.sbatch"
