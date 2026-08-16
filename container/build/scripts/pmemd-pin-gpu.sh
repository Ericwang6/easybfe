#!/usr/bin/env bash
# Shim installed as /opt/easybfe/bin/pmemd.cuda.MPI, which sits ahead of
# /opt/amber/pmemd26/bin on PATH. Each MPI rank runs this, so it can pin the
# rank to one GPU before handing over to the real binary.
#
# Why
# ---
# Left alone, every one of the 24 ranks initialises all four GPUs and then picks
# device (rank mod 4) for itself. Restricting each rank to a single device
# instead measured ~3% faster on the REMD stage but, more usefully, roughly 8-10%
# less wall clock -- because it removes the stragglers. Unpinned, per-window
# rates on a stage spread over about 1.2x (e.g. 358-425 ns/day on the solvent
# pre-production stage); pinned, they land within 1.02x. A stage finishes only
# when its slowest rank does, so tightening that spread is worth more than the
# change in the mean.
#
# This deliberately lives on PATH rather than in EasyBFE's generated run.sh, so
# no change to easybfe/amber/workflow.py is needed and any launcher benefits.
# Set EASYBFE_DISABLE_GPU_PIN=1 to bypass it.
set -euo pipefail

REAL="${PMEMDHOME:-/opt/amber/pmemd26}/bin/pmemd.cuda.MPI"

# Refuse to exec ourselves. Only reachable if PMEMDHOME is pointed at this
# script's own directory, but the failure mode would be a fork bomb, so it is
# worth two lines to make it a clear error instead.
if [ "$(readlink -f "${REAL}" 2>/dev/null)" = "$(readlink -f "$0" 2>/dev/null)" ]; then
    echo "pmemd.cuda.MPI shim: PMEMDHOME=${PMEMDHOME:-} resolves back to this shim" >&2
    exit 1
fi

if [ "${EASYBFE_DISABLE_GPU_PIN:-0}" = "1" ] || [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
    # Already pinned by the caller (or explicitly disabled): do not second-guess it.
    exec "${REAL}" "$@"
fi

rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${PMI_LOCAL_RANK:-${SLURM_LOCALID:-}}}"
if [ -z "${rank}" ]; then
    exec "${REAL}" "$@"            # not launched under a recognised MPI runtime
fi

ngpu=$(nvidia-smi --list-gpus 2>/dev/null | wc -l | tr -d '[:space:]')
if [ -n "${ngpu}" ] && [ "${ngpu}" -gt 0 ] 2>/dev/null; then
    # Same mapping pmemd would have chosen for itself, made explicit so the rank
    # never opens a context on the other three devices.
    export CUDA_VISIBLE_DEVICES=$(( rank % ngpu ))
fi

exec "${REAL}" "$@"
