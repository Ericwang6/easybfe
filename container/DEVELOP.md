# Development log — EasyBFE + AMBER 26 container

Why the image is put together the way it is, what was measured rather than
assumed, and what is still unverified.

`README.md` is the how-to. This file is the reasoning, including the parts that
turned out to be wrong.

---

## 1. Layout

```
container/
├── build/                    everything that goes into the image
│   ├── Dockerfile
│   ├── Dockerfile.dockerignore
│   ├── build.sh
│   ├── pmemd26.tar.bz2       licensed AMBER 26 source (git-ignored)
│   └── scripts/
│       ├── limit-cuda-arch.sh    build-time patch: restrict SM targets
│       ├── enable-nccl-tool.sh   build-time patch: register nccl (§4)
│       ├── entrypoint.sh         conda activation + MPI transport probe
│       ├── selfcheck.sh          runtime probe, shipped in the image
│       ├── pmemd-pin-gpu.sh      one-GPU-per-rank shim, shipped in the image
│       └── registry-cleanup.sh   prune old Artifact Registry manifests
└── e2e-abfe-test/                 everything that is NOT in the image
    ├── run-test.sh           drives one `docker run` of the whole pipeline
    ├── inputs/               config.yaml + 5USZ protein/ligand
    ├── scripts/
    │   ├── run-gpu-test.sh          create a GPU node and collect results
    │   ├── gpu-test-node-startup.sh unattended node setup + run + upload
    │   ├── provision-gce-docker.sh  manual Docker/NVIDIA-runtime install
    │   └── prod-rate.py             ns/day out of the per-window mdinfo
    └── results/              a100x4/ l4x4/ t4x4/ + archive-prev-image/
```

**The split is load-bearing.** Test data used to be `COPY`d into the image at
`/opt/easybfe/data`. That coupled the deliverable to its fixtures: fixing a typo
in a YAML meant rebuilding and re-pushing a 12 GB image, and "the image I tested"
stopped being obviously equal to "the image I shipped". Inputs are now mounted
read-only at `/work/inputs`, `run-gpu-test.sh` tars `e2e-abfe-test/` out of the
working tree and uploads it, and the image contains software only. Swapping the
test case costs an upload.

## 2. What the job actually needs

From `easybfe/abfe/piepline.py` and `easybfe/amber/workflow.py`:

* `easybfe abfe pipeline` = parameterize ligand → set up `solvent`/`complex`/
  `restraint` legs → run each leg's generated `run.sh` → MBAR analysis.
* Every stage with `use_mpi: true` launches

  ```
  mpirun -np <num_procs> pmemd.cuda.MPI -ng <n_lambda> -groupfile <stage>.groupfile [-rem 3 -remlog ...]
  ```

  `num_procs: -1` resolves to *one rank per λ window*
  (`AmberFepSimulationConfig.validate_nproc`), i.e. **24 ranks** for this job,
  regardless of how many GPUs exist. `pmemd.cuda.MPI` maps rank *i* to GPU
  *i mod n_gpus*, so 4 GPUs carry 6 ranks each.
* `workflow.py` emits a CUDA MPS block in every generated script so those 6
  ranks run concurrently. MPS binaries come from the *driver*, not the CUDA
  toolkit, so their presence inside a container depends on the NVIDIA container
  runtime; the script degrades gracefully without them (expensively — §6.4).
* Executables that must be on `PATH`: `pmemd.cuda`, `pmemd.cuda.MPI`, `mpirun`,
  `ambpdb`, plus `tleap`/`antechamber`/`parmchk2` via acpype.
* Python imports this code path reaches that `environment.yml` does **not**
  list: `MDAnalysis`, `joblib`, `tqdm`, `pyyaml`, `openmmforcefields`. The image
  appends them rather than forking the file.

## 3. Image design

**Two stages.** Stage 1 (`nvidia/cuda:12.6.3-devel-ubuntu22.04`) compiles
pmemd26 twice — serial CUDA, and CUDA+MPI+NCCL. Stage 2
(`nvidia/cuda:12.6.3-runtime-ubuntu22.04`) takes only the installed
`/opt/amber/pmemd26` tree. The ~4 GB devel toolchain never ships, and neither
does the AMBER source: it is deleted in stage 1, so no layer of the shipped
image contains it. That is the *only* thing deliberately stripped.

**Full `-runtime` base, not `-base` plus named libraries.** An earlier round
installed the seven CUDA libraries `pmemd.cuda` links (`cudart`, `cufft`,
`curand`, `cusolver`, `cusparse`, `cublas`, `nvJitLink`) on the `-base` image
and saved ~3.7 GB. That optimisation has been reverted on purpose: the full
runtime set means nothing can fail to load for want of a library nobody thought
to list, and `libnccl.so.2` — which the NCCL-linked `pmemd.cuda.MPI` now needs —
comes with it.

**Conda environment is `environment.yml` unmodified.** Nothing dropped, nothing
swapped for a `-base` variant, no post-install pruning of the environment (only
the micromamba *package cache* is removed). `openfe`, `vina`, `meeko`, `plip`,
`pytest` and full `matplotlib` are all present, as is the `pip:` docs block.
The only edit is additive — the five packages from §2. The result is 12.8 GB on
disk, and an environment identical to a developer's.

Two things learned the hard way that argue for leaving it alone:

* Deleting `tests`/`testing` directories from site-packages looks like ~150 MB
  of free space and is not: `import scipy.sparse` reaches `numpy.testing`
  through array_api_compat's `from numpy import *`, which trips numpy's lazy
  `__getattr__`. A trimmed image imported nothing and every `easybfe` entry
  point died with `ModuleNotFoundError: No module named 'numpy.testing'`. It
  reached the registry before anything noticed.
* So the Dockerfile runs an **import smoke test as a build step**. Whatever else
  changes, that gate stays.

**CUDA 12.6.3 / Ubuntu 22.04 (gcc 11.4).** `pmemd26_src/cmake/CudaConfig.cmake`
hard-fails outside CUDA ≥ 7.5, < 12.9, and separately checks gcc against CUDA:
gcc < 13.3 with CUDA 12.4–12.6 is accepted, which is exactly what this base
image provides. CUDA 12.x minor-version compatibility means the image runs on
any driver ≥ 525.

**GPU targets: `sm_70 sm_75 sm_80 sm_89`** = V100, T4, A100, L4, so one tag
covers the fleet. AMBER otherwise compiles SM 5.0 → 9.0, eleven nvcc passes,
twice. `limit-cuda-arch.sh` appends a filter *after* AMBER's own arch selection
(dropping the `-gencode` pairs and re-adding the requested ones) rather than
editing the vendor's version logic. A card whose SM is missing has no native
code in the image — AMBER ships no PTX-only fallback — so the build verifies
with `cuobjdump -lelf` that every requested `sm_*` is really in both binaries
and fails if not.

**AmberTools from conda, pmemd from source.** `AMBERHOME` points at the conda
env (that is where `tleap`/`antechamber` find their data) and pmemd is found
through `PATH` only. `/opt/amber/pmemd26/amber.sh` is deliberately *not* sourced
— it would repoint `AMBERHOME` at a tree with no leap data.

**MPI.** pmemd links the system OpenMPI 4.1 from apt. `/opt/easybfe/bin/mpirun`
is a symlink to `/usr/bin/mpirun` and `/opt/easybfe/bin` is first on `PATH`, so
the unqualified `mpirun` in EasyBFE's generated scripts can never resolve to a
conda MPI that pmemd was not linked against.

Two OpenMPI settings are baked in, both container-specific:

* `OMPI_ALLOW_RUN_AS_ROOT{,_CONFIRM}` — containers run as root and OpenMPI
  otherwise refuses to start.
* `OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1` — OpenMPI counts *physical
  cores* as slots, but a cloud vCPU is a hyperthread, so a 48-vCPU machine
  advertises 24 slots and `mpirun -np 24` aborts with "not enough slots" on
  anything smaller. `a2-highgpu-4g` has exactly 24 physical cores for exactly 24
  λ windows, so the original run worked by coincidence. Counting hardware
  threads reports the machine honestly; preferred over `--oversubscribe`, which
  tells OpenMPI to ignore its own accounting.

The CMA/shared-memory decision is **not** baked in. `entrypoint.sh` probes at
start-up whether `process_vm_readv` is permitted and only sets
`OMPI_MCA_btl_vader_single_copy_mechanism=none` when the seccomp profile
actually blocks it.

## 4. AMBER bug: `-DNCCL=TRUE` cannot succeed as shipped

Building with NCCL fails at configure time:

```
CMake Error at cmake/PMEMDCompilerFlags.cmake:285 (message):
  NCCL is selected for inter-GPU communications but was not found.
```

The message is misleading. Nothing was ever looked for. NCCL is present in the
base image (`/usr/include/nccl.h`, `libnccl.so → libnccl.so.2.23.4`,
`libnccl_static.a`, packages `libnccl2` / `libnccl-dev`, apt-marked *held* — so
`apt-get install libnccl-dev` is not a fix and fails with "held broken
packages"). The failure is in AMBER's build system:

| step | file | what happens |
|---|---|---|
| need declared | `CMakeLists.txt:178` | CUDA build appends `nccl` to `NEEDED_3RDPARTY_TOOLS` |
| flags computed | `3rdPartyTools.cmake:103` | `foreach(TOOL ${3RDPARTY_TOOLS})` — iterates the **master** list |
| master list | `3rdPartyTools.cmake:6` | `blas lapack netcdf netcdf-fortran zlib libbz2 kmmd libm mkl plumed` — **no nccl** |
| consumer | `3rdPartyTools.cmake:388` | `if(NEED_nccl) find_package(NCCL) ...` — dead code |
| gate | `PMEMDCompilerFlags.cmake:284` | `if(NCCL AND NOT nccl_ENABLED)` → FATAL_ERROR |

`NEED_nccl` is never defined, so `find_package(NCCL)` never runs. `FindNCCL.cmake`
is present and correct; it is simply never called.

Three cmake probes, ~30 s each, identical except for one variable:

| configuration | result |
|---|---|
| `NCCL_HOME` / `NCCL_INCLUDE_DIR` / `NCCL_LIBRARY` as environment variables | `FATAL_ERROR` |
| the same as cmake **cache** variables (`-DNCCL_INCLUDE_DIR=…`) | `FATAL_ERROR` — so it is not a find failure |
| `enable-nccl-tool.sh` applied, **no** path variables at all | `-- Found NCCL: /usr/include` · `-- NCCL: ON` · `Configuring done` |

The third row is the one that settles it: once the find actually runs, CMake's
default search path locates the library unaided.

`build/scripts/enable-nccl-tool.sh` adds `nccl` to the master list and a matching
description to the parallel `3RDPARTY_TOOL_USES` list (they are indexed against
each other; editing one alone misaligns the build report). Two lines, applied to
the extracted copy in stage 1 — no vendor source is forked, and `src/` is not
touched. It is applied *after* the serial build so editing it does not
invalidate that layer's cache.

The env vars are kept because `FindNCCL.cmake` does read `$ENV{NCCL_HOME}`; the
other two are inert (find_path/find_library consult cache variables, not the
environment) and cost nothing.

**Scope, stated plainly.** Linking NCCL is a capability, not a speed-up for this
pipeline. `src/pmemd/src/cuda/gpu.cpp`:

```c
#ifdef NCCL
  if (bSingleNode && gpu->nGpus > 2) { bNCCL = true; } else
#endif
```

`gpu->nGpus` is the number of ranks cooperating on **one** simulation. EasyBFE
runs `-ng 24` with 24 ranks, so every group has `nGpus == 1`: `bNCCL` is never
set and `gpu_allreduce` returns early. NCCL engages only when a single system is
decomposed across >2 GPUs. `build.sh --no-nccl` builds without it.

Because the feature is unreachable as shipped, it is plausibly untested upstream.
It is not untested here: §7 exercises it on four A100s and AMBER reports
`NCCL support: ENABLED`.

## 5. AMBER constraint: `01.em` cannot run under MPI

An earlier round recorded that setting `use_mpi: true` on `01.em` would fold the
24 sequential minimisations into the same 24-rank launch as every other stage,
and claimed ~37% of MD wall clock back. **That change was never run.** The first
attempt to execute it aborted the whole job within a minute:

```
Running multipmemd version of pmemd Amber24
   Total processors =    24
   Number of groups =    24
...
CUDA (GPU): Minimization is NOT supported in parallel on GPUs.
            Please use the single GPU code for minimizations.
MPI_ABORT was invoked on rank 0 in communicator MPI_COMM_WORLD
```

Note the shape of that launch: `-ng 24` with 24 ranks is 24 groups of **one**
rank — one GPU per minimisation, not one minimisation split across GPUs. AMBER
refuses regardless. `01.em` must stay `use_mpi: false`; EasyBFE runs the
minimisations one at a time and a serial `pmemd.cuda` always takes device 0.

Recovering that time needs EasyBFE to launch N concurrent single-GPU
`pmemd.cuda` processes for `01.em` instead of one MPI job — a change in
`easybfe/amber/workflow.py`, not a config flag. Not attempted here. All three
runs in §6 use `use_mpi: false` and completed.

Two traps this leaves behind, both of which look like a broken multi-GPU setup:

* An `nvidia-smi` snapshot early in a leg shows **only GPU 0 busy**. That is
  `01.em`. `run-test.sh` samples `nvidia-smi` for the whole run into
  `gpu-usage.csv` rather than trusting one snapshot.
* Every window's mdout reports `CUDA Device ID in use: 0`. That is the GPU-pinning
  shim: each rank gets `CUDA_VISIBLE_DEVICES` set to one physical card, so the
  device is always logical index 0 *within* the rank. Device IDs cannot be used
  to prove the ranks are spread; per-GPU utilisation and memory can (§6.3).

## 6. Measured: this image, three GPU types

One image (`easybfe-amber26:latest`, 12.8 GB, `sm_70/75/80/89`, NCCL linked), one
config (`e2e-abfe-test/inputs/config.yaml`), three nodes, all in us-central1-a,
all `exit=0` / `TEST PASSED`:

| | 4×A100 | 4×L4 | 4×T4 |
|---|---|---|---|
| machine | `a2-highgpu-4g` | `g2-standard-48` | `n1-standard-32` + 4×T4 |
| GPU memory | 40 GB | 23 GB | 15 GB |
| provisioning | SPOT | FLEX_START | FLEX_START |
| **total wall clock** | **19m 47s** | **24m 01s** | **38m 34s** |

### 6.1 Throughput, from `05.prod.out`

> **Corrected 2026-08-07.** This originally read `05.prod.info` (mdinfo), via
> `prod-rate.py`. `.info` is rewritten at every `bar_intervall` report
> (`nstlim` under REMD, `min(ntpr, nstlim)` otherwise — both work out to every
> 125 steps for this config, REMD or not), and its final snapshot before a
> stage ends is, empirically, unreliable to a degree that varies by
> leg/stage/GPU speed in a way not fully understood (working hypothesis: a
> small fixed synchronization/flush cost per rewrite, which is negligible
> against a slow leg's long 125-step wall-clock interval but not against a
> fast one's short interval — see
> `e2e-abfe-test/results/a100x4-nccl-mpi5-localssd/README.md` §4 for the
> `Elapsed(s)`/`Per Step(ms)` evidence this was diagnosed from). `.out`'s
> "all steps" block, printed once at the true end of the whole stage, does
> not show this effect and is what `prod-rate.py` now reads
> (`rate_from_mdout`). The tables below are regenerated from the same
> archived `mdinfo.tar.gz` (`gs://abfe-server-test-easybfe-results/{a100x4,
> l4x4,t4x4}/`) this section originally used — same runs, same hardware,
> corrected reader. **The A100-vs-L4 solvent conclusion two paragraphs down
> flips as a result** — kept below with a correction note rather than
> silently rewritten, because that is this document's own stated policy (its opening line: "the
> reasoning, including the parts that turned out to be wrong").

pmemd's own `ns/day` per λ window, read from the `Average timings for all steps`
block of each window's mdout. 24 windows per leg, 6 sharing each GPU via MPS.

**`05.prod` (H-REMD, the production stage):**

| leg | 4×A100 | 4×L4 | 4×T4 |
|---|---:|---:|---:|
| solvent | **343.4** | 332.3 | 155.7 |
| complex | **126.1** | 56.2 | 23.6 |
| restraint | **125.2** | 56.5 | 23.7 |

**`04.pre_prod` (plain MD, no exchanges):**

| leg | 4×A100 | 4×L4 | 4×T4 |
|---|---:|---:|---:|
| solvent | **372.2** | 366.6 | 182.0 |
| complex | **140.3** | 60.0 | 27.2 |
| restraint | **138.2** | 59.2 | 27.2 |

Aggregated over the 24 windows, the node sustains (µs/day, `04.pre_prod`):
solvent 8.93 / 8.80 / 4.37 and complex 3.37 / 1.44 / 0.65 for A100 / L4 / T4.

complex/restraint barely moved from the original `.info`-sourced numbers
(within 0.2 ns/day on `05.prod`) — **except A100 `04.pre_prod`, which moved a
lot** (124.6→140.3, 122.9→138.2, +12-13%). L4 and T4 `04.pre_prod`
complex/restraint did not move. Why the artifact hits A100-`04.pre_prod`-
complex/restraint but not A100-`05.prod`-complex/restraint, despite both
having the same 125-step report interval and near-identical per-step cost,
is not resolved — flagged as unresolved rather than guessed at.

**The hardware ranking still depends on the system, but the previous claim
about which way is now wrong.** On the complex and restraint legs A100 is
still clearly faster than L4 (2.2-2.4×) and T4 (5.3-5.9×) — that part holds.
~~On the *solvent* leg L4 **beats** A100 on `04.pre_prod` (373.5 vs
316.9)~~ **was the `.info` bug**: corrected, A100 (372.2) edges out L4
(366.6) on `04.pre_prod` solvent too, by a hair. ~~Buying A100s for a
solvent-dominated workload would be close to wasted money~~ **do not follow
this advice** — it was based on the bug above. A100 leads or ties L4 on
every leg/stage measured here once read correctly; T4 is behind everywhere.
The "six solvent windows don't saturate an A100" observation itself may
still be true (A100's *margin* over L4 is much smaller on solvent than on
complex/restraint), just not to the point of losing.

`05.prod` is 10-13% slower than `04.pre_prod` on A100 for the big legs
(126.1 vs 140.3, 125.2 vs 138.2) and T4 (23.6 vs 27.2, both ~13%) — the
H-REMD exchange every 125 steps does cost something at this size, more
consistently than the original numbers suggested (original A100 numbers put
`05.prod` complex *faster* than `04.pre_prod`, an anomaly now attributable
to the `04.pre_prod` `.info` undercount rather than a real effect). Solvent:
A100 `05.prod` (343.4) is also slower than `04.pre_prod` (372.2), ~8%,
consistent with the same exchange cost showing up there too.

### 6.2 The ΔG spread across hardware is the real error bar

| | 4×A100 | 4×L4 | 4×T4 |
|---|---:|---:|---:|
| complex | 363.891 ± 0.417 | 364.612 ± 0.438 | 365.270 ± 0.438 |
| solvent | 339.950 ± 0.424 | 339.824 ± 0.423 | 338.958 ± 0.432 |
| restraint | −0.699 ± 0.008 | −0.680 ± 0.008 | −1.224 ± 0.015 |
| boresch | 8.285 | 8.288 | 8.318 |
| **total** | **−16.355 ± 0.595** | **−17.180 ± 0.609** | **−19.218 ± 0.615** |

Three runs of identical software on identical inputs land **2.9 kcal/mol apart**,
while each reports a statistical error of ±0.6. The runs are not bitwise
comparable — `ig = -1` seeds from the wall clock, so each is an independent
sample — and that is the point: at this sampling length the MBAR error bar
understates the true uncertainty by roughly 5×. The decomposition is
self-consistent across all three (complex and solvent agree to ~1 kcal/mol), which
is what the test is for. **None of these is a converged affinity.**

### 6.3 All four GPUs are driven, and memory is not the constraint

Per-GPU utilisation over the whole run (`gpu-usage.csv`, 60 s sampling) and peak
memory:

| | GPU 0 | GPU 1 | GPU 2 | GPU 3 | peak memory / GPU |
|---|---:|---:|---:|---:|---:|
| A100 | 32% | 14% | 14% | 15% | 6.4 GB |
| L4 | 48% | 36% | 36% | 35% | 4.8 GB |
| T4 | 56% | 49% | 49% | 50% | 4.1 GB |

The means are low because the sampled window includes `01.em` (serial, GPU 0
only), ligand parameterization, the Boresch MD, and MBAR — all of which are CPU
work or single-GPU. During the MPI stages themselves every GPU reaches >90%: on
T4, 17–19 of 39 samples exceed 90% on each card. GPU 0 always runs hotter because
it also hosts the CUDA MPS server.

Peak memory tracks the card, not the workload — the same six windows take 6.4 GB
on A100 and 4.1 GB on T4, because CUDA sizes its pools against available memory.
**T4's 15 GB is not a limit for this system**, which I expected to be the failure
mode and it was not: 4.1 GB of 15 GB used, no OOM anywhere.

### 6.4 Not re-measured on this image

The MPS on/off comparison (6.1× per window), the windows-per-GPU packing sweep,
the GPU-pinning A/B, and the NUMA / clock / seccomp / CPU-starvation eliminations
were measured in an earlier round on the previous image, with a bench harness
that required a bundle of pre-equilibrated systems which was never checked in.
The harness has been removed rather than left as scripts nobody can run; the raw
CSVs stay in `results/archive-prev-image/` as the evidence behind those claims,
separated from this round's numbers so the two cannot be confused. The operationally
important one is worth repeating because it is easy to lose by accident:

> **Without CUDA MPS the six ranks per GPU time-slice one context — 61.1 vs
> 371.5 ns/day, a 6.1× loss.** If a run is inexplicably ~4× slow, look for
> `CUDA MPS: unavailable` in the leg log before anything else.

## 7. NCCL at runtime: verified, and one hypothesis killed

`selfcheck.sh` on every node confirmed the linkage end to end —
`pmemd build: linked`, `libnccl.so.2 => /lib/x86_64-linux-gnu/libnccl.so.2`,
`this GPU is sm_89, which is compiled in`.

Linkage is not execution, so the NCCL path was exercised directly on the A100
node: one simulation across four GPUs, which is the only shape that reaches it
(`bSingleNode && nGpus > 2`).

```
mpirun -np 4 pmemd.cuda.MPI -O -i 04.pre_prod.in -p ../../system.prmtop \
       -c ../03.pres/03.pres.rst7 ...
```

AMBER's own banner:

```
|---------------- GPU PEER TO PEER INFO -----------------
|   Peer to Peer support: DISABLED
|   NCCL support: ENABLED
```

All four GPUs sat at 100% with ~1.17 GB each, so the decomposition is real.
**NCCL is engaged at runtime**, not merely linked.

**A hypothesis I had recorded as plausible, now disproved.** I suspected the
GPU-pinning shim disabled peer-to-peer, because it leaves each rank one visible
device. Re-running with `EASYBFE_DISABLE_GPU_PIN=1`, all four devices visible and
each rank on a distinct one:

| | devices detected per rank | device in use | P2P | NCCL |
|---|---|---|---|---|
| pinned (default) | 1 | 0, 0, 0, 0 | DISABLED | ENABLED |
| unpinned | 4 | 0, 1, 2, 3 | **DISABLED** | ENABLED |

P2P is disabled either way, so the shim is not the cause: AMBER uses NCCL
*instead of* direct peer-to-peer when NCCL is compiled in. The shim is innocent.

**It is still not something to use for this workload.** The 50 ps that one GPU
finishes in ~12 s had not completed after 10 minutes across four — a small
solvent box decomposed four ways is entirely communication-bound. That is exactly
why EasyBFE gives each λ window its own GPU, and why NCCL is a capability here
rather than an optimisation.

Still unverified, and stated so rather than left to inference:

* **NCCL × MPS.** They are used by disjoint configurations — MPS packs many ranks
  onto one GPU, NCCL wants one rank per GPU inside a communicator — so they never
  met in these runs. The combination to watch is `-ng 1` with more ranks than
  GPUs, which puts two ranks of one communicator on one card; NCCL rejects that
  with "Duplicate GPU detected".
* **V100 (`sm_70`)** is compiled in and gated by `cuobjdump`, but no V100 has run
  this image.
* **Multi-node.** Everything here is single-node; `bSingleNode` is a precondition
  for AMBER's NCCL path in the first place.

## 8. Cloud notes

### 8.1 Provisioning: FLEX_START earns its place

| target | FLEX_START | STANDARD | outcome |
|---|---|---|---|
| 4×T4 | us-central1-a, first try | — | FLEX_START |
| 4×L4 | us-central1-a, first try | — | FLEX_START |
| 4×A100 | stockout in a, b, c | **quota-blocked** | fell back to SPOT |

Two things worth keeping:

* **FLEX_START (Dynamic Workload Scheduler) succeeded where spot and on-demand
  had both failed.** An hour before these runs, `g2-standard-48` + 4×L4 returned
  `ZONE_RESOURCE_POOL_EXHAUSTED` in *every* US zone tried, on spot **and** on
  demand. FLEX_START took it on the first attempt. For a batch job that does not
  care when it starts, it should be the first thing tried, not the last resort.
* **Quota and capacity fail differently, and only one is worth retrying.**
  On-demand 4×A100 is impossible in this project — `NVIDIA_A100_GPUS = 2` against
  a request for 4 — and no amount of retrying or zone-hopping changes that. A
  capacity failure says `ZONE_RESOURCE_POOL_EXHAUSTED`; a quota failure names a
  metric and a limit. `run-gpu-test.sh` retries the first and reports the second.

FLEX_START also needs `--maintenance-policy=TERMINATE` alongside
`--instance-termination-action=DELETE` and `--max-run-duration`. Omitting it
produces a self-contradictory error that costs real time to decode:

```
Invalid value for field 'resource.scheduling.onHostMaintenance': 'TERMINATE'.
Scheduling must have onHostMaintenance be one of the following valid types:
[TERMINATE]. But was MIGRATE
```

It means "you did not set it". `--max-run-duration` doubles as a spend cap: the
node deletes itself, so a hung run or a forgotten node cannot bill indefinitely.

Zone hints inside stockout errors are stale and should not be trusted:
us-central1-a suggested b and c, b suggested a and c, while all three refused.

### 8.2 Build

Build and run are separated: the build needs 32 CPUs and no GPU, and running it
on an A100 node would burn GPU-hours at ~30× the price. A cold build of four SM
targets is ~50 min of nvcc plus ~15 min of conda solve; with the serial-CUDA
layer cached it is **29 min**.

BuildKit cache behaviour that cost a full recompile this round: `COPY` keys on
file *contents*, so moving `pmemd26.tar.bz2` to a new path stayed cached — but a
one-line **comment** edit to `limit-cuda-arch.sh` invalidated that step and
every step after it, including both pmemd compiles. Put the files you are likely
to touch as late in the Dockerfile as their dependencies allow.

Other things that bite on a fresh project:

* Deep Learning VM images ship the NVIDIA driver but **no Docker**. Ubuntu's
  `docker.io` has no buildx, which BuildKit needs; install Docker CE from
  Docker's own repository. Replacing `docker.io` leaves `docker.socket` disabled,
  after which dockerd dies with "no sockets found via socket activation". All
  three are handled in `provision-gce-docker.sh` and the node startup script.
* The default compute service account may have **no IAM roles**, so nodes cannot
  pull from Artifact Registry or read the staging bucket. Grant
  `roles/artifactregistry.reader` and `roles/storage.objectViewer` explicitly.

### 8.3 Cost

This round: ≈ $12. The 4×A100 spot node dominates at ~$6.4 for ~50 min; L4 and
T4 on FLEX_START were ~$1.5–2 each for 24 and 39 minutes; the CPU build node
~$2.3. Running all three GPU nodes concurrently rather than in sequence cost
nothing extra — they bill by the minute — and cut the wall clock to that of the
slowest.

Artefacts from every run are in
`gs://abfe-server-test-easybfe-results/{a100x4,l4x4,t4x4}/`: `prod-rate.csv`,
`gpu-usage.csv`, `result.json`, the leg logs, and every window's mdinfo/mdout.
The trajectories are not kept — they are reproducible, the timings are the
deliverable.
