# Running remote_sam3 on CHTC

This document walks through the workflow for running the SAM3 segmentation
pipeline on UW–Madison's CHTC (Center for High Throughput Computing) and
pushing results to Research Drive.

> **Scheduler note:** CHTC's GPUs live on the **HTCondor** ("HTC") system, not
> the Slurm HPC cluster. The HPC partition (`shared` / `int` / `pre`) is CPU-only.
> All submit files in this repo are HTCondor `.sub` files.

---

## Things I need you to fill in

The script files reference these placeholders. Search for `<NETID>`, etc., or
just set them as environment variables before running the build/push scripts.

| Placeholder        | What it is                                                  | Where used                            | Your value |
| ------------------ | ----------------------------------------------------------- | ------------------------------------- | ---------- |
| `<NETID>`          | Your CHTC username (typically your campus NetID)            | `chtc/run_tests.sub`, `build_apptainer.sh` | `___`      |
| `<PI_SHARE_NAME>`  | Your PI's Research Drive share (the path after `//research.drive.wisc.edu/`) | `push_to_researchdrive.sh` | `___`      |
| `<GITHUB_USER>`    | GitHub user/org if pulling the Docker image from GHCR       | `build_apptainer.sh`                  | `___`      |
| `HF_TOKEN`         | HuggingFace token (read scope) for the gated `facebook/sam3` repo | submit env                      | `hf_...`   |
| Research Drive auth | Are you using Kerberos (`-k`) or password auth?            | `push_to_researchdrive.sh`            | `___`      |
| GPU Lab access     | Has CHTC granted you GPU Lab access?                        | required for `+WantGPULab = true`     | yes / no   |

> Until "GPU Lab access" is granted, your jobs will sit in IDLE forever. Apply
> via the form linked from <https://chtc.cs.wisc.edu/uw-research-computing/gpu-lab>.

---

## One-time setup

### 1. Get the code onto your CHTC submit node

```bash
ssh <NETID>@ap2001.chtc.wisc.edu
git clone https://github.com/NevNeal/remote_sam3.git
cd remote_sam3
```

### 2. Build (or fetch) the Apptainer SIF

You have two options:

**A. Pull from a Docker registry (recommended).** Push your local Docker image
to GHCR or Docker Hub first, then on the submit node:

```bash
# On your workstation:
docker build -t ghcr.io/<GITHUB_USER>/remote_sam3:latest .
docker push  ghcr.io/<GITHUB_USER>/remote_sam3:latest

# On the CHTC submit node:
NETID=<NETID> DOCKER_IMAGE=ghcr.io/<GITHUB_USER>/remote_sam3:latest \
  ./chtc/build_apptainer.sh
```

**B. Build locally, scp the SIF up.** If you can't push to a registry:

```bash
# On your workstation (needs apptainer installed):
apptainer build remote_sam3.sif docker-daemon://remote-sam3:latest
scp remote_sam3.sif <NETID>@transfer.chtc.wisc.edu:/staging/<NETID>/
```

After either path, the SIF should be at `/staging/<NETID>/remote_sam3.sif` and
`chtc/run_tests.sub` should reference `osdf:///chtc/staging/<NETID>/remote_sam3.sif`.

### 3. Export your HuggingFace token in the submit shell

```bash
export HF_TOKEN=hf_yourtokenhere
```

The submit file has `getenv = HF_TOKEN`, so the running job inherits it.

---

## Running the test job

```bash
mkdir -p logs
condor_submit chtc/run_tests.sub
```

Track it:

```bash
condor_q                         # while running
condor_history -limit 1          # after it ends
```

The job will:

1. Pull the SIF from `/staging`.
2. Run `chtc/run_tests.sh` inside the container, which runs `test_on_chtc.py`.
3. `test_on_chtc.py` performs four tests (environment / network / HuggingFace
   / pipeline) and writes `test_results.json`.
4. The full `test_output/` directory plus `test_results.json` come back to
   your submit node via HTCondor's default output transfer.

---

## Analyzing results

```bash
# pick the latest log:
ls -t logs/test.*.log | head -1
# strip the .log to pass the prefix:
python chtc/analyze_results.py logs/test.<cluster>.<process>
```

The output covers:

- **Lifecycle**: submit time, start time, queue wait, wall clock, exit code
- **Test breakdown**: which of the 4 tests passed / failed / were skipped, and
  how long each took
- **Output inventory**: how many images / masks / overlays the pipeline produced
- **Stderr tail**: last 30 lines of `.err` for quick failure triage
- **Research Drive push** receipt (only after step below)

---

## Pushing results to Research Drive

Compute nodes **cannot** reach Research Drive. You run this from the transfer
node:

```bash
ssh <NETID>@transfer.chtc.wisc.edu
cd ~/remote_sam3
PI_SHARE=<PI_SHARE_NAME> ./chtc/push_to_researchdrive.sh test_output sam3/$(date +%F)/591507
```

This will:

1. Tar `test_output/` into `/tmp/<name>_<timestamp>.tar.gz`
2. `smbclient` into `//research.drive.wisc.edu/<PI_SHARE>` with Kerberos auth
3. Create the remote subpath and `put` the tar
4. Verify with a listing, write `push_receipt.json`

Then re-run `analyze_results.py` and it will include the push receipt.

---

## Common failure modes (and where they show up)

| Symptom                                            | Where it appears                                | Likely cause                              |
| -------------------------------------------------- | ----------------------------------------------- | ----------------------------------------- |
| Job sits IDLE forever                              | `condor_q` shows IDLE; no `001` event in log    | GPU Lab access not granted, or no GPUs free for your request constraints |
| `environment` FAIL: "CUDA not available"           | `test_results.json` -> environment.error        | Wrong base image, missing nvidia-docker integration |
| `network` FAIL                                     | `test_results.json` -> network.error            | Compute node firewall blocked iNat API    |
| `huggingface` FAIL: 401                            | `test_results.json` -> huggingface.error        | `HF_TOKEN` not passed through; check `getenv` in submit file |
| `pipeline` FAIL: HF download timeout               | `test_results.json` -> pipeline.stderr_tail     | Model wasn't pre-cached; consider mounting an HF cache from `/staging` |
| `smbclient: session setup failed`                  | `push_to_researchdrive.sh` stderr               | Kerberos ticket expired (`kinit`) or wrong PI share name |
| Held with "Disk usage exceeded request_disk"       | `analyze_results.py` -> "HELD REASON"           | Increase `request_disk` in `run_tests.sub` |

---

## File overview

```
remote_sam3/
├── segmentation_pipeline.py     # the main pipeline (CLI: taxon_id, output_folder, --prompt, --limit)
├── test_on_chtc.py              # 4-test harness; runs inside the container
├── Dockerfile                   # PyTorch CUDA 12.4 + deps
├── requirements.txt
├── chtc/
│   ├── run_tests.sub            # HTCondor submit file (GPU + Apptainer)
│   ├── run_tests.sh             # container entrypoint
│   ├── build_apptainer.sh       # build SIF on submit node from Docker URL
│   ├── push_to_researchdrive.sh # smbclient push from transfer node
│   └── analyze_results.py       # log + JSON summarizer
└── instructions.md              # this file
```

---

## Docs I used to write this

- CHTC GPU jobs (HTCondor): <https://chtc.cs.wisc.edu/uw-research-computing/gpu-jobs>
- CHTC HTCondor submit guide: <https://chtc.cs.wisc.edu/uw-research-computing/htcondor-job-submission>
- CHTC Apptainer on HTC: <https://chtc.cs.wisc.edu/uw-research-computing/apptainer-htc>
- CHTC ML on HTC: <https://chtc.cs.wisc.edu/uw-research-computing/machine-learning-htc>
- CHTC Research Drive transfer: <https://chtc.cs.wisc.edu/uw-research-computing/transfer-data-researchdrive>
- CHTC HPC (Slurm) submit guide (not used; CPU-only): <https://chtc.cs.wisc.edu/uw-research-computing/hpc-job-submission>
