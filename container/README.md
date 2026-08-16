# EasyBFE + AMBER 26 container (multi-GPU ABFE)

A single image that runs the whole `easybfe abfe pipeline` — ligand
parameterization, leg setup, GPU MD, MBAR analysis — on a multi-GPU node.

| Component | What is in the image |
|---|---|
| `pmemd.cuda`, `pmemd.cuda.MPI` | AMBER 26, compiled from `build/pmemd26.tar.bz2` into `/opt/amber/pmemd26`. CUDA + MPI + NCCL, native code for **sm_70 / sm_75 / sm_80 / sm_89** (V100 / T4 / A100 / L4) |
| CUDA | the full `nvidia/cuda:12.6.3-runtime-ubuntu22.04` library set, NCCL included |
| AmberTools (`tleap`, `antechamber`, `parmchk2`, `ambpdb`, …) | conda-forge, in `/opt/conda/envs/easybfe` |
| EasyBFE + python stack (OpenMM, RDKit, ParmEd, MDAnalysis, alchemlyb, openfe, vina, meeko, plip…) | same conda env, from `environment.yml` **unmodified** |
| MPI | system OpenMPI 4.1 (`/usr/bin/mpirun`), what `pmemd.cuda.MPI` is linked against |
| Test inputs | **none — deliberately.** See below |

**Software and test cases are separate.** The image contains no configs, no
structures and no reference outputs; those live in `e2e-abfe-test/inputs/` and are
mounted read-only at run time. Changing the test system, or fixing a typo in a
config, never rebuilds or re-pushes the image.

> **Licence.** AMBER 26 is proprietary and not redistributable.
> `build/pmemd26.tar.bz2` is not in git, and the built image must stay in a
> **private** registry.

## Layout

```
container/
├── build/          the image: Dockerfile, build.sh, build-time patches,
│                   and the three scripts that ship inside it
└── e2e-abfe-test/       the test case: inputs, the runner, GPU-node automation,
                    and archived results
```

## 1. Prerequisites

* `build/pmemd26.tar.bz2` — the licensed AMBER 26 pmemd source.
* Docker with the NVIDIA container runtime (`--gpus`) on the build and run
  hosts. A GPU is *not* needed to build.
* NVIDIA driver ≥ 525 on the run host (CUDA 12.x minor-version compatibility).

## 2. Build

```bash
./container/build/build.sh                  # -> easybfe-amber26:latest
```

| Flag | Default | Notes |
|---|---|---|
| `--sm-archs "A B"` | `70 75 80 89` | `70` = V100, `75` = T4, `80` = A100, `89` = L4. Each target is another full nvcc pass **and** more fatbin in the binaries, so trim the list for a faster build. A card whose SM is absent has no native code in the image — AMBER ships no PTX-only fallback. |
| `--no-nccl` | NCCL on | Build `pmemd.cuda.MPI` without NCCL. On by default so multi-GPU-per-simulation runs have it; note the ABFE pipeline itself never reaches AMBER's NCCL path (one rank per λ window ⇒ one GPU per simulation group). See `DEVELOP.md` §4. |
| `--tag TAG` | `latest` | |
| `--cuda 12.6.3` | `12.6.3` | AMBER 26 accepts CUDA ≥ 11.8 and < 12.9; the base image's gcc must match (`DEVELOP.md` §3). |
| `--push` | off | Also tags and pushes to Artifact Registry (`--project`, `--region`, `--ar-repo`). |

Three gates fail the build rather than shipping a bad image:

* `cuobjdump -lelf` must show **every** requested `sm_*` in both binaries.
* `pmemd.cuda.MPI` must actually link `libnccl` when NCCL was requested, and
  that library must resolve in the runtime stage.
* the conda environment must import
  `numpy, scipy.sparse, openmm, rdkit, parmed, MDAnalysis, alchemlyb, pymbar, openfe, easybfe`
  and `easybfe --version` must run.

The build compiles pmemd twice (serial CUDA, and CUDA+MPI+NCCL). On 32 vCPUs
that is ~50 min cold for four SM targets, plus ~15 min of conda solve.

```bash
./container/build/build.sh --push --project abfe-server-test --region us-central1
# -> us-central1-docker.pkg.dev/abfe-server-test/easybfe/easybfe-amber26:latest
```

## 3. Check an image

```bash
docker run --rm --gpus all easybfe-amber26:latest selfcheck.sh
```

Reports the visible GPUs and whether their SM is compiled in, whether CUDA MPS
is available, that `pmemd.cuda`, `pmemd.cuda.MPI`, `mpirun`, `tleap` and
`easybfe` resolve and link cleanly, and whether NCCL is linked.

## 4. Run the end-to-end test

```bash
./container/e2e-abfe-test/run-test.sh
./container/e2e-abfe-test/run-test.sh --image <registry>/easybfe-amber26:latest
./container/e2e-abfe-test/run-test.sh --inputs /path/to/other/system
```

Inputs default to `e2e-abfe-test/inputs/` and are mounted read-only at
`/work/inputs`; outputs land in `--outdir` (default `./abfe-test-<timestamp>`):

```
run/ligand/              parameterized ligand
run/abfe/solvent|complex|restraint/
run/abfe/result.json     decomposed and total ΔG
run/abfe.log             master log
test.log                 everything the container printed
gpu-usage.csv            per-GPU utilisation sampled every 60 s
```

### The shipped test protocol

`e2e-abfe-test/inputs/config.yaml`, on the 5USZ system (JH2 WT protein + ligand),
24 λ windows × 3 legs = 72 independent simulations:

| stage | type | steps | dt | length |
|---|---|---|---|---|
| `01.em` | minimisation | `maxcyc 2000` | — | — |
| `02.heat` | NVT, restrained | 2 500 | 2 fs | 5 ps |
| `03.pres` | NPT, restrained | 2 500 | 2 fs | 5 ps |
| `04.pre_prod` | NPT | 12 500 | 4 fs | 50 ps |
| `05.prod` | **H-REMD** | 125 × `numexchg 200` | 4 fs | 100 ps |

Common `cntrl`: Langevin `ntt=3` at `gamma_ln 2.0`, `temp0 298.15`, MC barostat,
`cut 10.0`, HMR at 3.024 amu, alchemical stages add `icfe=1 ifsc=1 ifmbar=1`
with `scalpha 0.5 / scbeta 1.0`. The config also drives an MD-based Boresch
restraint search (`boresch.algorithm: rxrx-md` plus a `boresch_md` block) and
sets `early_stop_threshold` high so the full path always runs.

`05.prod` is where throughput is read from: `use_remd: true` turns the launch
into `mpirun -np 24 pmemd.cuda.MPI -ng 24 -groupfile 05.prod.groupfile -rem 3`,
all 24 replicas resident at once, exchanging every 125 steps.

> **`01.em` must stay `use_mpi: false`.** AMBER refuses to minimise under
> `pmemd.cuda.MPI` even with one rank per group and aborts the whole launch.
> `DEVELOP.md` §5.

This protocol validates the image and measures throughput. It is **not** a
converged affinity — production is far shorter than a scientific config, and the
reported `total_std` is MBAR statistical error only.

### Reading ns/day

```bash
python3 container/e2e-abfe-test/scripts/prod-rate.py <outdir>/run --stage 05.prod
```

Reads each window's `05.prod.out` (mdout) and reports per-window mean/min/max
plus the node total. It uses **only** the `Average timings for all steps`
block — every AMBER mdout also carries a `ns/day` under `Average timings for
last N steps` where N is often 1, and mixing the two produces nonsense. It
used to read `.info` (mdinfo) instead; `.info`'s final snapshot before a
stage ends turned out to understate ns/day, by as much as ~15-20% for some
leg/GPU/stage combinations — see `DEVELOP.md` §6.1's correction note and
`e2e-abfe-test/results/a100x4-nccl-mpi5-localssd/README.md` §4 for the
evidence. `.out` does not show this effect.

### Measured

This image, this config, 24 λ windows packed 6 per GPU. Per-window ns/day from
`05.prod.out`, and the whole-pipeline wall clock:

| | 4×A100 | 4×L4 | 4×T4 |
|---|---:|---:|---:|
| solvent | **343.4** | 332.3 | 155.7 |
| complex | **126.1** | 56.2 | 23.6 |
| restraint | **125.2** | 56.5 | 23.7 |
| total wall clock | **19m 47s** | 24m 01s | 38m 34s |
| peak memory / GPU | 6.4 GB | 4.8 GB | 4.1 GB |

A100 leads or ties L4 on every leg/stage measured here (complex/restraint by
2.2-2.4×; solvent by a much smaller margin — six solvent windows don't
saturate an A100 the way complex/restraint do, so its edge is thinner there,
just not negative). An earlier revision of this table, read from `.info`,
had L4 *beating* A100 on the `04.pre_prod` solvent leg and concluded "buying
A100s for a solvent-dominated workload would be close to wasted money" —
that was the `.info` measurement bug above, not a real effect; see
`DEVELOP.md` §6.1.

Three runs of the same software on the same inputs gave ΔG of −16.4, −17.2 and
−19.2 kcal/mol against a reported ±0.6. At this sampling length the MBAR error
bar understates the real uncertainty by roughly 5×.

`DEVELOP.md` §6 has the full tables, the per-GPU utilisation evidence, and what
does and does not move throughput (MPS: 6.1×; windows-per-GPU: the dominant term;
NCCL, CMA, NUMA, CPU count: nothing).

## 5. Run your own job

```bash
docker run --rm --gpus all --ipc=host --shm-size=8g \
    --ulimit memlock=-1 --ulimit stack=67108864 \
    -v /path/to/work:/work \
    easybfe-amber26:latest \
    easybfe abfe pipeline /work/config.yaml \
        -p /work/protein_fixed.pdb -l /work/ligand.sdf -o /work/run
```

Anything on the image's `PATH` works as the command (`easybfe`, `tleap`,
`pmemd.cuda`, `python`, `bash`); the entrypoint activates the conda environment
first.

### How the GPUs are used

EasyBFE writes one `run.sh` per leg that launches **one MPI rank per λ window**,
and `pmemd.cuda.MPI` assigns rank *i* to GPU *i mod n_gpus* — so on a 4-GPU node
the 24 windows sit 6 per card. The generated script starts a **CUDA MPS** daemon
when `nvidia-cuda-mps-control` is present so those 6 run concurrently rather
than time-slicing; that is worth 6.1× per window, not a tuning detail
(`EASYBFE_DISABLE_MPS=1` to disable).

The image also installs `/opt/easybfe/bin/pmemd.cuda.MPI`, a shim ahead of the
real binary on `PATH` that pins each rank to one GPU via `CUDA_VISIBLE_DEVICES`.
It removes stragglers rather than raising the mean, worth 14–17% of stage wall
clock. `EASYBFE_DISABLE_GPU_PIN=1` bypasses it.

The `docker run` flags matter:

* `--gpus all` — all GPUs visible to the container.
* `--ipc=host --shm-size=8g` — OpenMPI shared-memory transport between ranks.
* `--ulimit memlock=-1 --ulimit stack=67108864` — usual MPI/CUDA limits.

The image sets `OMPI_ALLOW_RUN_AS_ROOT=1` (containers run as root) and
`OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1` (OpenMPI otherwise counts physical
cores as slots and refuses to launch 24 ranks on a 48-vCPU machine).

It does not hard-disable OpenMPI's CMA fast path: `entrypoint.sh` probes whether
`process_vm_readv` is permitted and only sets
`OMPI_MCA_btl_vader_single_copy_mechanism=none` when the seccomp profile blocks
it. Benchmarking found no measurable gain from relaxing seccomp for this
workload, so the default flags do not.

## 6. On Google Cloud

Build once on a cheap CPU VM, run on the GPU node — the build needs no GPU, and
A100 time is ~30× the price of the build machine.

**Build node.**

```bash
gcloud compute instances create easybfe-build --zone=us-central1-b \
    --machine-type=n2-standard-32 \
    --image-family=common-cu129-ubuntu-2204-nvidia-580 \
    --image-project=deeplearning-platform-release \
    --boot-disk-size=250GB --scopes=cloud-platform

gcloud compute ssh easybfe-build --zone=us-central1-b \
    --command "bash -s" < container/e2e-abfe-test/scripts/provision-gce-docker.sh
# copy the repo + build/pmemd26.tar.bz2 up, then on the node:
#   sudo ./container/build/build.sh --push
```

**GPU node.** One command creates the node, installs Docker and the NVIDIA
runtime, pulls the image, uploads and unpacks `e2e-abfe-test/` from your working
tree, runs the pipeline, extracts ns/day and uploads the artefacts:

```bash
S=./container/e2e-abfe-test/scripts/run-gpu-test.sh
$S --label a100x4 --machine a2-highgpu-4g
$S --label l4x4   --machine g2-standard-48
$S --label t4x4   --machine n1-standard-32 --accelerator type=nvidia-tesla-t4,count=4
```

4-GPU shapes are often unavailable, so the script tries each provisioning model
against each zone and takes the first that lands: `FLEX_START` (Dynamic Workload
Scheduler — below on-demand price, and the model meant for this) then
`STANDARD`. `--models "SPOT FLEX_START STANDARD"` puts the cheapest first;
`--zones "..."` widens the search. `--max-run` (default 4h) makes the node
delete itself, so a forgotten node cannot bill indefinitely.

The node is **not** deleted automatically (its disk is the only record of a
failure). Delete it yourself:

```bash
gcloud compute instances delete easybfe-a100x4 --zone=us-central1-b --quiet
```

Two things that bite on a fresh project — quota and capacity, which fail
differently:

* This project has `NVIDIA_A100_GPUS = 1` on demand but
  `PREEMPTIBLE_NVIDIA_A100_GPUS = 16`, so 4×A100 is reachable on **spot** only.
* Capacity is separate from quota: `g2-standard-48` + 4×L4 returned
  `ZONE_RESOURCE_POOL_EXHAUSTED` in every US zone tried, spot *and* on demand,
  with 8 L4 of quota free. The zone hints in those errors are stale — each zone
  suggested the other two while all three refused. `run-gpu-test.sh` walks
  models and zones for this reason.
* The default compute service account may have **no IAM roles**, so the node
  cannot pull from Artifact Registry:

  ```bash
  gcloud artifacts repositories add-iam-policy-binding easybfe --location=us-central1 \
      --member=serviceAccount:PROJECT_NUMBER-compute@developer.gserviceaccount.com \
      --role=roles/artifactregistry.reader
  ```

### Image naming

There is exactly one deliverable tag:

```
us-central1-docker.pkg.dev/abfe-server-test/easybfe/easybfe-amber26:latest
```

`build/scripts/registry-cleanup.sh` prunes stale manifests:

```bash
./container/build/scripts/registry-cleanup.sh            # dry run
./container/build/scripts/registry-cleanup.sh --delete
```

> It is scoped to **one image name** on purpose. The same repository also holds
> `easybfe-server`, whose commit-SHA tags look like disposable build output but
> are pinned by the live Cloud Run service and by Batch job definitions. A
> repo-wide sweep would take production down.

## 7. Troubleshooting

| Symptom | Cause / fix |
|---|---|
| `selfcheck.sh` reports no GPUs | missing `--gpus all`, or no NVIDIA container runtime on the host |
| `selfcheck.sh` warns the GPU's SM is not compiled in | rebuild with that architecture in `--sm-archs` |
| `nvidia-cuda-mps-control not found` | the driver's MPS binaries were not mounted in; the run continues without MPS, ~4× slower |
| a run is inexplicably ~4× slow | look for `CUDA MPS: unavailable` in the leg log first |
| `mpirun ... as root` error | `OMPI_ALLOW_RUN_AS_ROOT{,_CONFIRM}` were overridden |
| `There are not enough slots available` | `OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1` was overridden, or there are more λ windows than hardware threads |
| `Minimization is NOT supported in parallel on GPUs` | an `em` stage has `use_mpi: true`; it cannot |
| A leg exits immediately | a stale `done.tag`/`error.tag` in the leg directory — delete it and re-run; the scripts are idempotent |
| `cudaMemcpy ... out of memory` | too many λ windows per GPU for the box size; use fewer ranks or more GPUs |

`DEVELOP.md` records how the image was put together, what was measured, and what
is still unverified.
