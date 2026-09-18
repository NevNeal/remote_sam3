# remote_sam3 on UF HiPerGator

Port of the CHTC/HTCondor pipeline to the University of Florida's HiPerGator
(SLURM). The segmentation logic is unchanged; the scheduling, container and
storage layers are all different.

---

## 1. The five things that actually had to change

| | CHTC | HiPerGator | Why it matters |
|---|---|---|---|
| **Scheduler** | HTCondor `.sub` files, `queue N` | SLURM `#SBATCH`, `--array=0-N` | Total rewrite of the submit layer. `hpg/sbatch_*.sh` replace `chtc/*.sub`. |
| **Container runtime** | `container_image = docker://…` — HTCondor pulls Docker directly | Apptainer only; no Docker, no root | Image must be converted to a `.sif` once (`hpg/build_sif.sh`) and invoked with `apptainer exec --nv`. |
| **CUDA / PyTorch** | PyTorch 2.6 + CUDA 12.4 | **must be ≥ PyTorch 2.7 / CUDA 12.8** | HPG's big GPUs are **B200 (Blackwell, sm_100)**. CUDA 12.4 predates sm_100 and cannot emit code for it — a B200 job dies with `no kernel image is available for execution on the device`. `hpg/Dockerfile.hpg` uses `pytorch/pytorch:2.9.1-cuda12.8-cudnn9-runtime`. **This is the one change that silently breaks everything if you skip it.** |
| **Data movement** | `transfer_input_files` / `output_destination = pelican://…`; job scratch is ephemeral | `/blue` Lustre is mounted on login *and* compute nodes | The whole staging/transfer dance disappears. The parquet sits on `/blue` and is read in place; shards write straight into one shared output tree, so results are merged on disk the moment the array ends. |
| **Archival target** | UW Research Drive over SMB, from `transfer.chtc.wisc.edu` only | `/orange` (investment tier), or Globus off-cluster | `hpg/archive_to_orange.sh` replaces `chtc/push_to_researchdrive.sh`. |

Two smaller but load-bearing changes:

- **Shared, pre-warmed HuggingFace cache.** On CHTC each job pulled `facebook/sam3`
  into its own scratch. With a 300-task array that would be ~900 GB of
  HuggingFace traffic and near-certain throttling. `hpg/setup_blue.sh` downloads
  the model once into `/blue/<group>/<user>/hf_cache`, and every shard runs with
  `HF_HUB_OFFLINE=1`.
- **Per-shard result CSVs.** `results_shard_NNN.csv` instead of one `results.csv`.
  Concurrent appends from 60 tasks to one file on Lustre will corrupt it;
  per-shard files make concurrency safe and keep resume working per shard.
  `hpg/merge_shards.py` concatenates them (and finally exists — the CHTC branch
  referenced a `merge_shards.py` that was never written).

---

## 2. Hardware on HiPerGator

HiPerGator 4 (production since ~Sept 2025). **The A100s are gone** — the 140 DGX
A100 nodes were retired and replaced with 63 DGX B200 nodes, so
`parallel_A100/`'s premise does not carry over.

### GPUs

| GPU | VRAM | Count | Per node | Partition | Arch (CC) |
|---|---|---|---|---|---|
| NVIDIA L4 | 24 GB | ~600 | 3 | `hpg-turin` | Ada, sm_89 |
| NVIDIA B200 | 180 GB | ~504 (63 × 8) | 8 | `hpg-b200` | Blackwell, sm_100 |
| NVIDIA L40 | 48 GB | 32 | — | (hybrid/HWGUI) | Ada, sm_89 |
| Quadro RTX 6000 | 24 GB | legacy | — | `hpg-rtx6000` | Turing, sm_75 |

The cu128 image covers sm_75 / sm_89 / sm_100, so one `.sif` runs on every one
of these partitions.

### Nodes

- **`hpg-turin` (L4 hosts):** AMD EPYC 9655P, 96 cores, ~768 GB RAM, 3 × L4.
  That's **32 CPU cores per GPU** — an enormous download budget compared to the
  4 cores/GPU this pipeline got on CHTC.
- **`hpg-b200`:** 2 × Intel Xeon Platinum 8570 (56c each = 112 cores), 2 TB RAM,
  8 × B200 with NVLink between them.
- **CPU-only:** ~60,000 cores total, 8 GB RAM/core, `hpg-default` / `hpg-milan`
  / `bigmem`.
- **Storage:** 11 PB all-flash parallel filesystem; `/home` 40 GB (do not work
  here), `/blue` = main high-performance group storage, `/orange` = bulk
  investment tier, `/red` = short-term highest-I/O by Director approval.

### Scheduler limits worth knowing

| | |
|---|---|
| Max walltime, `hpg-turin` / `hpg-b200` / `hpg-rtx6000` | **14 days** |
| Max walltime, `hpg-default` / `bigmem` | 31 days |
| Interactive GPU session | 12 hours |
| **Burst QOS on GPU partitions** | **none** — GPU jobs must use the investment QOS (`--qos=<group>`) |
| Jobs per user | 3,000 |
| Max array task ID | 3,000 |
| CPU per GPU | ≥ 1 core per GPU or the job is rejected |

Your real ceiling is your **group's GPU investment**, not the cluster totals.
Check it before sizing an array:

```bash
slurmInfo                       # your group's allocation and current usage
showQos <group>                 # QOS limits
sacctmgr show assoc where account=<group> format=account,qos,grptres
sinfo -p hpg-turin -o '%P %a %l %D %t %G'   # what's idle right now
```

---

## 3. How to take advantage of parallelization

This workload is about as parallel-friendly as it gets: every image is
independent, the model is small (~4 GB VRAM of a 24 GB L4), and each image costs
one HTTP download from S3 plus one forward pass. There are three axes to pull,
and they compose.

### Axis 1 — job array across many single-GPU shards (do this first)

`hpg/sbatch_l4_array.sh`. Shard *i* takes rows where `global_index % N == i`.
No coordination, no pre-partitioning, no merge step beyond concatenating CSVs.

```bash
TAXON_ID=160559 PROMPT=flower ./hpg/submit.sh array 60
```

Why interleaved rather than contiguous blocks: iNat photo counts per row are
uniform, but download latency is not, and contiguous blocks correlate (photos
uploaded around the same time live in the same S3 prefix and often fail
together). Interleaving spreads both the fast and the dead URLs evenly, so
shards finish within minutes of each other instead of the array's wall-clock
being set by one unlucky block.

**Sizing the array.** Two competing pressures: more shards = more parallelism but
more queue positions competing for your group's GPU allocation, and each shard
pays a fixed ~60–90 s startup (container + model load). Rule of thumb: size
shards so each runs 2–12 hours.

```
shards ≈ total_photos × sec_per_image / target_hours / 3600
```

At ~1 s/image on an L4: a 500k-photo taxon at 60 shards ≈ 2.3 hours/shard. Use
`%` to throttle so you stay a good neighbour and your array isn't rejected for
exceeding the group limit: `./hpg/submit.sh array 300 40` queues 300 shards but
runs at most 40 at once — same total work, gentler on the allocation, and the
scheduler drip-feeds them as GPUs free up. **This is usually the right answer**:
a wide array with a concurrency cap beats a narrow array, because a dead shard
costs 1/300 of the run instead of 1/20.

### Axis 2 — saturate the CPU side of each GPU

This is the axis most people leave on the table, and on `hpg-turin` it is nearly
free. Each L4 comes with ~32 CPU cores. The pipeline's GPU is idle whenever it's
waiting on an S3 download, and SAM3 on a single image is maybe 0.3–1 s of GPU
work against 0.2–2 s of network. `--workers` (default 8, `WORKERS=` to override)
controls the download prefetch pool.

```bash
WORKERS=16 ./hpg/submit.sh array 60      # with --cpus-per-task=8, 16 is about right
```

Raise `--cpus-per-task` to 12–16 and `WORKERS` to 16–24 and the GPU stops
waiting. Watch for it: if `seconds/image` in `results_shard_NNN.csv` is much
larger than what `nvidia-smi` utilization implies, you are network-bound and
should add workers, not GPUs. Practical cap is iNat's S3 tolerance, not the node
— if you start seeing 429s and `dl_failed` climb, back off.

### Axis 3 — pack multiple shards per node

`hpg/sbatch_b200_node.sh` takes one whole DGX B200 node and runs 8 shards on it,
each pinned to its own GPU via `CUDA_VISIBLE_DEVICES`. One allocation, one queue
position, 8× throughput. The same trick works on `hpg-turin` (3 L4s per node) if
you'd rather hold 20 nodes than 60 array slots.

**When to prefer this over the array:** you want a single allocation rather than
N queue positions; you're benchmarking per-image throughput across GPU types;
you eventually batch several images per forward pass and actually need the VRAM.

**When not to:** right now. `hpg-b200` is the contended partition, RC asks that
it be reserved for work that needs the VRAM or the FLOPs, and one-image-at-a-time
SAM3 uses 4 GB of 180. Sixty L4s you get immediately beat eight B200s you waited
a day for.

### What is *not* worth doing

- **Multi-GPU per process** (DDP, `device_map="auto"`). This is inference over
  independent images; there is no gradient to sync and no tensor too big for one
  GPU. One process per GPU is strictly simpler and strictly faster.
- **`--gpus=2` for one shard.** The pipeline uses one device. You'd idle the second.

### The next real speedup: batched inference

Currently `hpg_pipeline.py` does one image per forward pass — inherited from the
CHTC code, and the reason a 180 GB B200 looks pointless. Batching 8–32 images per
`model(**inputs)` call is where the remaining 3–10× lives, and it is what would
make the B200s worth their queue time. It needs same-size padding/collation in
the processor call and per-image un-padding in post-processing, so it is a real
change, not a flag. The `a100-336671-batched-rd` branch has prior work on this
worth rebasing.

### Rough budget

| Setup | Effective rate | 500k photos |
|---|---|---|
| 1 × L4, 8 workers | ~1 img/s | ~6 days |
| 60 × L4 array | ~60 img/s | ~2.3 hours |
| 300 × L4 array, 40 concurrent | ~40 img/s sustained | ~3.5 hours |
| 1 × B200 node, 8 shards | ~12–20 img/s | ~8 hours |

L4 and B200 per-image numbers are close *because the workload is currently
network-bound, not compute-bound* — which is exactly why axis 1 and axis 2 beat
axis 3 until batching lands.

---

## 4. Running it

### One-time setup

```bash
# 1. On your workstation: build and push the cu128 image
DOCKERHUB_USER=nevneal ./hpg/push_docker_image_hpg.sh

# 2. Edit hpg/config.sh — set HPG_GROUP (and HPG_USER if it isn't $USER)

# 3. On a HiPerGator login node
ssh <gatorlink>@hpg.rc.ufl.edu
cd /blue/<group>/<user> && git clone -b hipergator https://github.com/NevNeal/remote_sam3.git
cd remote_sam3
./hpg/build_sif.sh                      # docker:// -> .sif; prints arch_list
export HF_TOKEN=hf_...                  # facebook/sam3 is gated
./hpg/setup_blue.sh                     # dirs + warm the HF cache

# 4. Get the parquet index onto /blue (3.9 GB)
#    from your workstation:
scp inat_db/data/inat_photos.parquet \
    <gatorlink>@hpg.rc.ufl.edu:/blue/<group>/<user>/data/
```

### Every run

```bash
./hpg/submit.sh test                          # 1 L4, 100 images — always first
TAXON_ID=160559 PROMPT=flower ./hpg/submit.sh array 60
squeue -u $USER
./hpg/submit.sh merge --num-shards 60         # -> results.csv + completeness report
./hpg/archive_to_orange.sh /blue/<group>/<user>/results/160559_flower
```

`merge` names any shard that never reported and prints the `sbatch --array=`
line to rerun exactly those. Reruns are safe: each shard skips the `photo_id`s
already in its own CSV. **Keep `--num-shards` constant across reruns** — it
defines the row-to-shard mapping.

### Useful while it runs

```bash
squeue -u $USER -o '%.10i %.9P %.10j %.2t %.10M %R'
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS,ReqTRES%40
scancel <jobid>_<taskid>              # kill one shard
tail -f /blue/<group>/<user>/logs/sam3_l4-<jobid>_0.out
```

---

## 5. Open items

- `HPG_GROUP` in `hpg/config.sh` is `CHANGEME` — needs your UF group/account.
- Group GPU allocation unverified; run `slurmInfo` before sizing an array past
  ~20 concurrent shards.
- `--cpus-per-task=8` / `WORKERS=8` are conservative first guesses for
  `hpg-turin`'s 32 cores/GPU. Tune after the first array using the
  seconds-per-image column.
- Batched inference (§3) is the highest-value remaining change.
- `parallel_A100/` is left untouched on this branch but is dead on HiPerGator —
  no A100s exist there any more.

---

## Sources

- [GPU Access — UFIT-RC docs](https://docs.rc.ufl.edu/scheduler/gpu_access/)
- [SLURM Partition Limits](https://docs.rc.ufl.edu/scheduler/partition_limits/)
- [Job Arrays](https://docs.rc.ufl.edu/scheduler/job_arrays/)
- [Apptainer on HiPerGator](https://docs.rc.ufl.edu/software/apps/apptainer/)
- [Storage / practical storage](https://docs.rc.ufl.edu/quickstart/practical_storage/)
- [HiPerGator overview](https://www.rc.ufl.edu/hipergator)
- [NVIDIA Blackwell Compatibility Guide (sm_100 / CUDA 12.8)](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
