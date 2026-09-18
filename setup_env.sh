#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# One-time environment setup. Run on a HiPerGator LOGIN node.
#
#   export HF_TOKEN=hf_...        # facebook/sam3 is a gated repo
#   ./setup_env.sh
#
# Creates the `sam3` conda env, installs PyTorch for CUDA 12.8, installs
# requirements.txt, and pre-downloads the SAM3 weights into a shared cache so
# that jobs never touch the network for the model.
#
# Takes ~15 minutes, almost all of it downloading wheels.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")"
source slurm/settings.sh

: "${HF_TOKEN:?export HF_TOKEN=hf_... first — facebook/sam3 is gated}"

module purge
module load conda

# HiPerGator points envs_dirs/pkgs_dirs at /blue/<group>/<user>/.conda on the
# first `module load conda`, so envs land on Lustre and not in your 40 GB $HOME.
echo "── conda storage locations ───────────────────────────────────────────────"
conda config --show envs_dirs pkgs_dirs

if conda env list | grep -qE "^${CONDA_ENV}\s"; then
    echo "[env] '${CONDA_ENV}' already exists — skipping create (delete it to rebuild:"
    echo "      conda env remove -n ${CONDA_ENV} )"
else
    echo "[env] creating '${CONDA_ENV}' from environment.yml"
    conda env create -f environment.yml -n "${CONDA_ENV}"
fi

# shellcheck disable=SC1091
conda activate "${CONDA_ENV}"

# PyTorch FIRST, from PyTorch's cu128 index.
#
# Why cu128 specifically: HiPerGator's large GPUs are B200 (Blackwell, compute
# capability sm_100). CUDA 12.8 is the first toolkit that can emit sm_100 code,
# and torch 2.7 was the first release with cu128 wheels. A cu124 build fails on
# a B200 with "no kernel image is available for execution on the device". The
# cu128 build still covers sm_75 and sm_89, so one install runs on the L4s too.
echo "[env] installing torch (cu128)"
pip install --no-cache-dir torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128

# Everything else from PyPI. Installed second so nothing drags in a cu126 torch.
echo "[env] installing requirements.txt"
pip install --no-cache-dir -r requirements.txt

echo
echo "── verification ─────────────────────────────────────────────────────────"
python - <<'PY'
import torch
print("torch        :", torch.__version__)
print("torch cuda   :", torch.version.cuda)
print("arch_list    :", torch.cuda.get_arch_list())
assert any(a.endswith("_100") for a in torch.cuda.get_arch_list()), \
    "sm_100 missing — this build will NOT run on hpg-b200"
print("sm_100 present — B200-capable")
PY

# Pre-download SAM3 once into a cache on /blue that every job reads.
#
# Without this, a 60-task array means 60 concurrent ~3 GB downloads from
# HuggingFace (~180 GB of traffic) and near-certain throttling. With it, jobs
# run HF_HUB_OFFLINE=1 and never touch the network for weights at all.
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

echo
echo "── next ─────────────────────────────────────────────────────────────────"
echo "  1. Get the parquet index to ${PARQUET}"
echo "       from your workstation:"
echo "         scp inat_photos.parquet ${USER}@hpg.rc.ufl.edu:${PARQUET}"
echo "       or build it here:  sbatch slurm/build_index.sbatch"
echo "  2. Smoke test:          sbatch slurm/test.sbatch"
echo "  3. Full run:            sbatch --array=0-59 slurm/array.sbatch"
