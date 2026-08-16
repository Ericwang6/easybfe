#!/usr/bin/env bash
# Container entrypoint: activate the conda environment (so that packages with
# activation hooks -- AmberTools, OpenBabel -- are configured), decide how
# OpenMPI should move shared-memory messages, then exec the requested command.
set -euo pipefail

eval "$(micromamba shell hook --shell bash)"
micromamba activate easybfe

# pmemd26 is installed outside the conda env; keep it (and the system mpirun
# shim in /opt/easybfe/bin) ahead of anything activation prepended.
export PATH="/opt/easybfe/bin:/opt/amber/pmemd26/bin:${PATH}"
# AmberTools' activation script points AMBERHOME at the conda env, which is what
# tleap/antechamber need for their data files. pmemd is found via PATH instead.
export AMBERHOME="${AMBERHOME:-/opt/conda/envs/easybfe}"

# --------------------------------------------------------------------------- #
# OpenMPI shared-memory transport                                              #
# --------------------------------------------------------------------------- #
# Ranks on this node talk through OpenMPI's vader/sm BTL. Its fast path is CMA
# (process_vm_readv): one copy, straight between two processes' address spaces.
# Docker's *default* seccomp profile blocks that syscall, and OpenMPI handles
# the resulting EPERM badly, so this image used to hard-disable CMA:
#
#     OMPI_MCA_btl_vader_single_copy_mechanism=none
#
# That is safe everywhere but costs throughput -- every message is then copied
# twice, through a shared bounce buffer. REMD is the workload that notices: one
# global exchange every `nstlim` steps across all lambda windows.
#
# So probe rather than assume. Started with `--security-opt seccomp=unconfined`
# (or `--cap-add SYS_PTRACE`), CMA works and is left on; otherwise the safe
# override goes back. EASYBFE_FORCE_VADER_COPY overrides the probe either way.
if [ -n "${EASYBFE_FORCE_VADER_COPY:-}" ]; then
    export OMPI_MCA_btl_vader_single_copy_mechanism="${EASYBFE_FORCE_VADER_COPY}"
elif [ -z "${OMPI_MCA_btl_vader_single_copy_mechanism:-}" ]; then
    if ! python -c '
import ctypes, ctypes.util, os, sys

libc = ctypes.CDLL(ctypes.util.find_library("c"), use_errno=True)

class IOVec(ctypes.Structure):
    _fields_ = [("iov_base", ctypes.c_void_p), ("iov_len", ctypes.c_size_t)]

# Read from our own address space: succeeds iff the syscall is permitted.
src = ctypes.create_string_buffer(b"probe")
dst = ctypes.create_string_buffer(len(src.raw))
local = IOVec(ctypes.cast(dst, ctypes.c_void_p), len(dst.raw))
remote = IOVec(ctypes.cast(src, ctypes.c_void_p), len(src.raw))
n = libc.process_vm_readv(os.getpid(), ctypes.byref(local), 1,
                          ctypes.byref(remote), 1, 0)
sys.exit(0 if n == len(src.raw) else 1)
' 2>/dev/null; then
        export OMPI_MCA_btl_vader_single_copy_mechanism=none
    fi
fi

exec "$@"
