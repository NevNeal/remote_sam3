# remote_sam3 — SAM3 segmentation of iNaturalist photos on UF HiPerGator

Give it an iNaturalist taxon ID and a text prompt. It finds every research-grade
photo of that taxon, runs Meta's SAM3 on each one, and saves masks, overlays,
transparent cut-outs and the average colour inside each mask.

Built for HiPerGator: SLURM, a conda environment, and `/blue` for storage. No
Docker, no containers, no file staging.

```
build_index.py     make the metadata index (once, then monthly)
segment.py         the pipeline — one GPU, one shard
collect.py         merge the shard CSVs and report on a finished run
setup_env.sh       create the conda env from environment.yml, cache the model
environment.yml    the HiPerGator conda env: python 3.12 + torch cu128 + pins
requirements.txt   pip packages, pinned (pulled in by environment.yml)
slurm/settings.sh  ← THE ONLY FILE YOU EDIT
slurm/run_shard.sh one shard on one GPU — what every GPU job runs
slurm/bench.sh     benchmark a taxon: timed run + report, outputs deleted
bench_report.py    turns a benchmark's raw timing into report.md
slurm/*.sbatch     the jobs: test (1 GPU), test_2gpu (2 GPUs), array (N GPUs),
                   build_index (CPU)
```

On HiPerGator it lives in `flower_sam3/`, and everything it creates stays
inside that directory:

```
flower_sam3/
├── data/inat_photos.parquet     the index
├── .conda/sam3/                 the conda env      (setup_env.sh)
├── hf_cache/                    SAM3 weights       (setup_env.sh)
├── results/<taxon>_<prompt>/    run outputs        (the jobs)
└── logs/                        SLURM .out/.err    (the jobs)
```

The CHTC/HTCondor version of this project lives on the `main` branch. It is a
different cluster with a different scheduler, container runtime and storage
model, so there is nothing shared worth abstracting — this branch is a rebuild,
not a port.

---

## 1. The parquet index, and why it exists

`inat_photos.parquet` is **pure metadata — no images.** Seven columns:

| column | example |
|---|---|
| `taxon_id` | `62741` |
| `photo_id` | `50805` |
| `extension` | `jpeg` |
| `taxon_name` | `Rudbeckia hirta` |
| `quality_grade` | `research` |
| `latitude` / `longitude` | `42.15`, `-71.65` |

289 million rows, 4.6 GB (from the 2026-08-27 dumps). The photo URL isn't stored, because it doesn't need
to be — it's two columns glued together:

```
https://inaturalist-open-data.s3.amazonaws.com/photos/{photo_id}/original.{extension}
```

iNaturalist publishes its whole database as three tab-separated dumps on a public
S3 bucket (`photos`, `observations`, `taxa`). `build_index.py` downloads them,
joins them, keeps research-grade rows only, and writes the parquet.

**Why bother.** The alternative is the iNaturalist REST API, which is paginated
and rate-limited: ~20 minutes of careful paging to enumerate one large taxon, and
the whole thing falls over if you run 60 workers against it at once. A filtered
parquet read is **under a second**, needs no network, no token, and no politeness
budget. Sixty shards can each read it simultaneously because it's a read-only
file on Lustre.

That's it. The index is a phone book. All the actual data still comes from S3 at
run time.

---

## 2. What a shard is

A **shard** is one worker's slice of the photo list. Nothing is split or copied —
they're just dividing up the work.

Say taxon 160559 has 600 photos, and you run 4 shards. `segment.py` numbers the
photos 0…599, then each shard keeps the rows where `row_index % 4` equals its own
number:

```
photo row:   0   1   2   3   4   5   6   7   8   9  10  11  ...
shard 0:     ✓               ✓               ✓                 (0, 4, 8, …)
shard 1:         ✓               ✓               ✓             (1, 5, 9, …)
shard 2:             ✓               ✓               ✓         (2, 6, 10, …)
shard 3:                 ✓               ✓               ✓     (3, 7, 11, …)
```

Four GPUs, 150 photos each, done in a quarter of the time. No shard needs to know
what any other shard is doing — there is no coordination, no lock, no message
passing. They each read the same parquet, compute their own slice arithmetically,
and get on with it.

**Why interleaved (`0, 4, 8, …`) rather than blocks (`0-149`, `150-299`, …).**
Both divide the work evenly on paper. In practice, contiguous blocks finish at
wildly different times: photos uploaded around the same time sit next to each
other in the list, and they share properties — same era of camera, same S3
prefix, and crucially, deleted photos come in clumps. One block can be full of
dead URLs that return instantly while another is all 8-megapixel images. The run
is only finished when the *slowest* shard is, so you want every shard to get a
fair mix of the good and the bad. Interleaving guarantees that.

**Three things follow from this design:**

- **Each shard writes its own `results_shard_NNN.csv`.** Sixty processes
  appending to one file on a shared filesystem would interleave half-written
  lines and corrupt it. Separate files make concurrency a non-issue. `collect.py`
  concatenates them at the end.
- **Images, masks and overlays all go to the same folders.** That's safe because
  every filename contains a unique `photo_id`, so two shards can never write the
  same file. When the array finishes you have one complete output tree, not 60
  partial ones to stitch together.
- **Duplicate photos are dropped before sharding.** iNat's dumps contain ~0.1%
  byte-identical duplicate photo rows. Left alone, the two copies get different
  `row_index` values, land in *different* shards, and those shards segment the
  same photo and write the same filenames concurrently. `segment.py` de-duplicates
  on `photo_id` before `row_index` is assigned, and reports how many it dropped.
- **The shard count must stay the same across reruns**, because it's what defines
  who owns which row. Change it and the row→shard mapping changes underneath you.
  `slurm/array.sbatch` reads it from `SLURM_ARRAY_TASK_COUNT` so it can't drift;
  for a partial rerun you pass it explicitly, and `collect.py` prints the exact
  command.

Resume is free and per-shard: a shard reads its own CSV on startup and skips any
`photo_id` already recorded. Already-downloaded images are reused too. So a job
killed at its walltime loses only the image in flight — resubmit and it picks up.

---

## 3. Hardware on HiPerGator

**The A100s are gone.** UF retired the 140 DGX A100 nodes and replaced them with
63 DGX B200 nodes, so any A100 guidance you find (including on the `main` branch)
is out of date.

| GPU | VRAM | Count | Per node | Partition | Arch |
|---|---|---|---|---|---|
| **L4** | 24 GB | ~600 | 3 | `hpg-turin` | Ada, sm_89 |
| **B200** | 180 GB | ~504 | 8 | `hpg-b200` | Blackwell, sm_100 |
| RTX 6000 | 24 GB | legacy | — | `hpg-rtx6000` | Turing, sm_75 |

- **`hpg-turin`** (where these jobs run): AMD EPYC 9655P, 96 cores, ~768 GB RAM,
  3 × L4 per node. That's **32 CPU cores per GPU**, which matters a lot — see
  Axis 2 below.
- **`hpg-b200`:** 2 × Xeon Platinum 8570 (112 cores), 2 TB RAM, 8 × B200.
- **CPU-only:** ~60,000 cores, `hpg-default` / `hpg-milan` / `bigmem`.
- **Storage:** 11 PB all-flash parallel filesystem. `/home` is 40 GB and is not
  for job I/O; `/blue` is the working filesystem; `/orange` is the bulk
  investment tier for finished runs.

**One hard constraint on software.** B200 is Blackwell, compute capability
`sm_100`. CUDA 12.8 is the first toolkit that can emit sm_100 code, and PyTorch
2.7 the first release with cu128 wheels. A cu124 build fails on a B200 with
`no kernel image is available for execution on the device` — which is why
`setup_env.sh` installs torch from PyTorch's cu128 index and then asserts
`sm_100` is in `torch.cuda.get_arch_list()`. The cu128 build also covers sm_75
and sm_89, so one environment runs everywhere.

**Scheduler limits:**

| | |
|---|---|
| Max walltime, GPU partitions | 14 days |
| Max walltime, `hpg-default` | 31 days |
| Jobs per user / max array task ID | 3,000 |
| Burst QOS on GPU partitions | **none** — GPU jobs use the investment QOS |
| CPU per GPU | at least 1 core per GPU, or the job is rejected |

Your real ceiling is **your group's GPU investment**, not the cluster totals.
Check before sizing a wide array:

```bash
slurmInfo                                    # your group's allocation and usage
sinfo -p hpg-turin -o '%P %a %l %D %t %G'    # what is idle right now
```

---

## 4. Parallelization

Three axes. They compose, and the order below is the order of payoff — because
this workload is currently **network-bound, not compute-bound.** At ~1 s per
image, most of that second is waiting on S3, not running SAM3.

### Axis 1 — more shards (the job array)

```bash
sbatch --array=0-59 slurm/array.sbatch        # 60 GPUs
sbatch --array=0-299%40 slurm/array.sbatch    # 300 shards, 40 running at a time
```

Size shards so each runs **2–12 hours**:

```
shards ≈ total_photos × sec_per_image ÷ target_hours ÷ 3600
```

A 500k-photo taxon at ~1 s/image and 60 shards is ~2.3 hours per shard.

**Prefer wide-with-a-cap over narrow.** `--array=0-299%40` does the same total
work as `--array=0-39` but a dead shard costs you 1/300 of the run instead of
1/40, the scheduler drip-feeds tasks as GPUs free up instead of making you wait
for 40 at once, and you stay a good neighbour on your group's allocation.

The floor is startup cost: each task spends ~60–90 s loading conda and SAM3
weights. Shards shorter than ~30 minutes are mostly overhead.

### Axis 2 — feed each GPU harder (the one people skip)

On `hpg-turin` each L4 comes with ~32 CPU cores, and the GPU sits idle every time
it waits for a download. `WORKERS` controls how many images are fetched in
parallel:

```bash
WORKERS=24 sbatch --array=0-59 slurm/array.sbatch
```

`--cpus-per-task=12` and `WORKERS=16` are the defaults here, deliberately
conservative. **The diagnostic is in `collect.py`'s output:** it prints median
seconds per photo and flags the run as download-bound when that number is well
above what the GPU actually needs. If it is, raise `WORKERS` — adding GPUs won't
help a GPU that's already waiting.

The ceiling is iNat's S3 tolerance, not the node. If `download_failed` starts
climbing and the errors mention 429s, you've gone too far; back off.

### Axis 3 — bigger GPUs (not yet worth it)

`hpg-b200` gives you 180 GB of VRAM per GPU. SAM3 on a single image uses about
4 GB of it. So the B200s buy you very little today, while being the contended
partition that RC asks you to reserve for work that genuinely needs the VRAM.
Sixty L4s you get in minutes beat eight B200s you wait a day for.

**What would change that: batched inference.** `segment.py` runs one image per
forward pass, inherited from the original pipeline. Batching 8–32 images per
`model(**inputs)` call is where the remaining 3–10× lives, and it's the thing
that would make a 180 GB GPU make sense. It needs same-size padding and
collation in the processor call plus per-image un-padding in post-processing, so
it's a real change rather than a flag. Worth doing after the first full run tells
you where the time actually goes.

### Not worth doing

- **Multi-GPU per process** (DDP, `device_map="auto"`). This is inference over
  independent images — no gradients to synchronise, nothing too big for one GPU.
  One process per GPU is simpler and faster.
- **`--gpus=2` for one shard.** `segment.py` uses one device; the second idles.

### Rough budget

| setup | throughput | 500k photos |
|---|---|---|
| 1 L4 | ~1 img/s | ~6 days |
| 60 L4 array | ~60 img/s | ~2.3 hours |
| 300 L4 array, 40 concurrent | ~40 img/s sustained | ~3.5 hours |

---

## 5. Setup

Once, on a HiPerGator **login node**. About 15 minutes, almost all of it pip
downloading wheels.

### Step 1 — put the code in `flower_sam3`

`flower_sam3/` already exists and holds `data/`, so turn it into a checkout in
place rather than cloning into a new folder (git won't clone into a non-empty
directory). `data/`, `results/`, `hf_cache/`, `.conda/` and `logs/*` are all
gitignored, so git never touches them.

```bash
ssh neal.nevyn@hpg.rc.ufl.edu
cd ~/blue_guralnick/neal.nevyn/flower_sam3
git init
git remote add origin https://github.com/NevNeal/remote_sam3.git
git fetch origin
git checkout flower-sam3
ls data/          # the parquet should be data/inat_photos.parquet
```

If the parquet has a different name, set `PARQUET` in `slurm/settings.sh`. No
other edits are needed: every path is relative to `flower_sam3/`, so nothing
depends on your group name.

Later updates are just `git pull`.

### Step 2 — build the conda environment

```bash
export HF_TOKEN=hf_...        # huggingface.co/settings/tokens; facebook/sam3 is gated
./setup_env.sh
```

What `setup_env.sh` does, in order:

1. **`module load conda`.** HiPerGator's conda comes as a module; there is
   nothing to install yourself.
2. **`conda env create -f environment.yml -p .conda/sam3`.** That creates the
   environment as a folder inside the project rather than a named env in
   `$HOME`. `$HOME` is capped at 40 GB and this env is ~8 GB. Conda's package
   cache is also pointed inside the project, and pip's cache is switched off.
   `environment.yml` is the whole definition:
   - conda: Python 3.12 and pip, nothing else
   - pip: `torch==2.9.1+cu128` from PyTorch's CUDA 12.8 index, then
     `requirements.txt` (transformers, pandas, pyarrow, … at the versions
     proven against SAM3)

   The torch wheel carries its own CUDA runtime, so there is no
   `module load cuda`; all it needs is the NVIDIA driver on the GPU node.
3. **A build check.** It asserts torch was compiled for `sm_89` (the L4s) and
   `sm_100` (the B200s), so a bad build fails here and not in a queued job.
   `gpu here: False` is expected, because login nodes have no GPU.
4. **A model-cache warm-up.** It downloads the SAM3 weights (~3 GB) once into
   `hf_cache/`. Jobs run with `HF_HUB_OFFLINE=1` and read from there, so N
   parallel shards make zero HuggingFace requests. N simultaneous 3 GB pulls
   would get throttled.

It is safe to rerun: an existing env is updated to match `environment.yml`
(a no-op if nothing changed), and a warm cache is a no-op. To
rebuild from scratch: `conda env remove -p .conda/sam3 && ./setup_env.sh`.

To use the env interactively (e.g. on a login node or in an `srun` session):

```bash
module load conda
conda activate ./.conda/sam3          # from flower_sam3/
```

To try the GPU by hand before submitting anything:

```bash
srun -p hpg-turin --gpus=1 --cpus-per-task=4 --mem=16gb --time=00:20:00 --pty bash -i
module load conda && conda activate ./.conda/sam3
python -c "import torch; print(torch.cuda.get_device_name(0))"     # NVIDIA L4
```

### Rebuilding the index

The index is already in `data/`. For the monthly refresh, when iNat
republishes the dumps, build it on HiPerGator so the 28 GB of CSVs never touch
your laptop:

```bash
sbatch slurm/build_index.sbatch      # ~4-8 hours, no GPU
```

---

## 6. Running

Always from `flower_sam3/`. There are three steps, each a bigger version of the
one before:

```bash
sbatch slurm/test.sbatch              # 1 L4, 100 images: does anything work?
sbatch slurm/test_2gpu.sbatch         # 2 L4s at once, 200 images: does it parallelise?
sbatch --array=0-59 slurm/array.sbatch   # the real run, 60 L4s
squeue -u $USER
python collect.py results/160559_flower --num-shards 60
```

All three GPU jobs run the same `slurm/run_shard.sh`, so they differ only in
width and image count. `test_2gpu.sbatch` is `array.sbatch` at `--array=0-1`
with `LIMIT=200`.

### Checking the 2-GPU test

Each array task is its own 1-GPU job. SLURM may put both tasks on one node (an
`hpg-turin` node has three L4s) or on two nodes; the shards never talk to each
other, so either is fine. A pass looks like this:

```bash
sacct -j <jobid> --format=JobID,NodeList,Start,End,Elapsed,State
#   <jobid>_0 and <jobid>_1 both COMPLETED, Start/End windows overlapping

grep -h -E '^(shard|node|gpu|started|finished)' logs/sam3_2gpu-<jobid>_*.out
#   two different GPU UUIDs

python collect.py results/160559_flower_test2gpu --num-shards 2
#   200 photos, shards 0 and 1 both present, no duplicates
```

If one task sat `PENDING` while the other ran, SLURM still did its job, but you
haven't tested parallelism yet. That usually means your group's GPU allocation
is saturated (`slurmInfo` shows it).

Different taxon or prompt, no file edits needed:

```bash
TAXON_ID=62741 PROMPT=petal sbatch --array=0-39 slurm/array.sbatch
LIMIT=1000 sbatch slurm/test_2gpu.sbatch       # bigger parallel test
```

### Output

```
<results>/160559_flower/
├── images/batch_00001/     Symphyotrichum_novaeangliae_95238471.jpg
├── masks/batch_00001/      ..._instance_0.npy         uint8 0/1, most confident first
├── overlays/batch_00001/   ..._overlay.png            red masks, yellow boxes, scores
├── segments/batch_00001/   ..._segment_0.png          transparent cut-out, cropped
├── results_shard_000.csv   one row per photo, per shard
└── results.csv             written by collect.py
```

`results.csv` carries the parquet metadata plus `status`, `num_masks`, every
output path, the per-mask confidence list, `mask_avg_rgb` (mean colour inside
each mask, `R|G|B` per mask), and `seconds` for the forward pass. Files are
bucketed 1,000 to a `batch_NNNNN` folder so no directory holds a million entries.

### While it runs

```bash
squeue -u $USER -o '%.10i %.9P %.10j %.2t %.10M %R'
tail -f logs/sam3-<jobid>_0.out
sacct -j <jobid> --format=JobID,State,Elapsed,MaxRSS
scancel <jobid>_<taskid>              # kill one shard
```

### Archiving a finished run

`/blue` is the fast tier and quota-limited; a full taxon is tens to hundreds of
GB of PNGs. Move finished runs to `/orange` if your group has an investment
(`orange_quota` to check), and use Globus rather than scp for copies off the
cluster:

```bash
rsync -ah --info=progress2 results/160559_flower/ \
                           /orange/<group>/$USER/sam3/160559_flower/
```

---

## 6b. Benchmarking a taxon (timing only, nothing kept)

```bash
TAXON_ID=62741 bash slurm/bench.sh          # 10 GPUs; pass 0-19 for 20
```

This runs the whole taxon across the GPUs exactly like a real run, but with
`--discard-outputs`. Every image, mask, overlay and cut-out is written, so save
time and file sizes are real. Each is then measured and deleted immediately, so
the run needs almost no space on `/blue`. When every GPU task has ended,
a second job (`bench_report.sbatch`) saves the evidence to
`benchmarks/<taxon>_<prompt>_<timestamp>/`, writes `report.md`, and deletes the
run's `results/` directory.

What gets measured:

| | where |
|---|---|
| per photo: download seconds, bytes, MB/s | `download_s`, `download_bytes`, `download_mbps` |
| per photo: GPU idle waiting for that download | `wait_s` |
| per photo: CPU preprocessing, GPU forward pass (cuda-synced), post-processing, saving | `prep_s`, `infer_s`, `post_s`, `save_s` |
| per photo: bytes it would have used on disk | `output_bytes` |
| per shard: parquet load, model load, loop time, node, GPU UUID | `raw/timing_shard_NNN.json` |
| per GPU: utilisation, memory, power, clocks every 5 s | `raw/telemetry/gpu_shard_NNN.csv` |
| per task: node, elapsed, CPU time, peak RAM | `sacct.txt` |

`report.md` rolls these up into wall-clock time and throughput. Per node it
gives allocated vs busy GPU-hours and download-wait hours. Downloads get
median/p95 speeds and times plus the most common errors, and there is an
estimate of the storage a kept run would need. `per_shard.csv`,
`per_node.csv` and `summary.json` hold the same numbers for plotting.

## 7. Open items

- Run `slurm/test_2gpu.sbatch` to confirm two L4s run shards side by side,
  before widening the array.
- Group GPU allocation unverified. Run `slurmInfo` before going past ~20
  concurrent shards.
- `--cpus-per-task` / `WORKERS` are conservative first guesses against
  `hpg-turin`'s 32 cores per GPU. Tune with `collect.py`'s median-seconds output.
- Batched inference (§4, Axis 3) is the highest-value remaining change.
- `--bf16` exists on `segment.py` but is off by default, since it changes mask
  numerics slightly and runs so far have been fp32.

---

## Sources

- [GPU Access](https://docs.rc.ufl.edu/scheduler/gpu_access/) ·
  [Partition Limits](https://docs.rc.ufl.edu/scheduler/partition_limits/) ·
  [Job Arrays](https://docs.rc.ufl.edu/scheduler/job_arrays/)
- [Conda on HiPerGator](https://docs.rc.ufl.edu/software/conda_environments/) ·
  [Conda configuration](https://docs.rc.ufl.edu/software/conda_configuration/)
- [Storage](https://docs.rc.ufl.edu/quickstart/practical_storage/) ·
  [HiPerGator overview](https://www.rc.ufl.edu/hipergator)
- [NVIDIA Blackwell compatibility (sm_100 / CUDA 12.8)](https://docs.nvidia.com/cuda/blackwell-compatibility-guide/)
- [iNaturalist open data on AWS](https://github.com/inaturalist/inaturalist-open-data)
