#!/usr/bin/env bash
# Report what the image actually resolves at runtime: GPUs, MPI, pmemd,
# AmberTools and EasyBFE. Run it before paying for a long job.
#
#     docker run --rm --gpus all <image> selfcheck.sh
set -uo pipefail

status=0
section() { printf '\n=== %s ===\n' "$1"; }
require() {  # require <label> <command...>
    local label="$1"; shift
    if "$@" > /tmp/selfcheck.out 2>&1; then
        printf '  [ ok ] %-22s %s\n' "$label" "$(head -1 /tmp/selfcheck.out)"
    else
        printf '  [FAIL] %-22s %s\n' "$label" "$(head -1 /tmp/selfcheck.out)"
        status=1
    fi
}

section "GPUs"
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv || status=1
else
    echo "  [FAIL] nvidia-smi not found (run with --gpus all)"
    status=1
fi

section "CUDA MPS (lets the lambda-window ranks share each GPU)"
if command -v nvidia-cuda-mps-control > /dev/null 2>&1; then
    echo "  [ ok ] nvidia-cuda-mps-control: $(command -v nvidia-cuda-mps-control)"
else
    echo "  [warn] nvidia-cuda-mps-control not found -- ranks will time-slice the"
    echo "         GPUs instead of running concurrently (slower, still correct)."
fi

section "Executables"
for exe in pmemd.cuda pmemd.cuda.MPI mpirun tleap antechamber parmchk2 ambpdb easybfe; do
    path=$(command -v "$exe" 2>/dev/null) \
        && printf '  [ ok ] %-22s %s\n' "$exe" "$path" \
        || { printf '  [FAIL] %-22s not found\n' "$exe"; status=1; }
done

section "Versions"
require "mpirun"   bash -c "mpirun --version"
require "easybfe"  bash -c "easybfe --version"
require "python"   bash -c "python -c 'import sys; print(sys.version)'"
require "imports"  bash -c "python -c 'import openmm, rdkit, parmed, MDAnalysis, alchemlyb; print(\"openmm\", openmm.__version__)'"
require "AMBERHOME" bash -c "echo \$AMBERHOME"

section "MPI build"
if [ -f /opt/openmpi/.version ]; then
    echo "  source-built OpenMPI $(cat /opt/openmpi/.version), $(command -v mpirun)"
else
    echo "  apt OpenMPI, $(command -v mpirun)"
fi

section "MPI launch smoke test (no GPU needed -- catches a wrong OMPI_MCA_/PRTE_MCA_ var or run-as-root config on a build host, not a GPU node)"
NPROC=$(nproc 2>/dev/null || echo 1)
# Deliberately *without* --oversubscribe, at exactly nproc (hardware threads,
# what `nproc` reports on a hyperthreaded cloud vCPU). OpenMPI's slot count
# defaults to physical cores; without hwthreads-as-cpus this launch fails with
# "not enough slots" on any machine with hyperthreading/SMT enabled -- which is
# the actual failure this image hit before that MCA var was added. If it's
# spelled wrong for this OpenMPI's runtime layer (OMPI_MCA_ vs PRTE_MCA_), this
# is where that shows up, not 24 ranks in on a GPU node.
require "mpirun -np \$(nproc) true (no --oversubscribe)" \
    bash -c "mpirun -np ${NPROC} true"

section "MPI shared-memory path (entrypoint probes this at start-up)"
copy_mech="${OMPI_MCA_btl_vader_single_copy_mechanism:-}"
if [ -z "${copy_mech}" ]; then
    echo "  [ ok ] CMA enabled -- OpenMPI copies messages once, directly"
else
    echo "  [warn] single_copy_mechanism=${copy_mech}: process_vm_readv is blocked,"
    echo "         so every intra-node message is copied twice. Re-run with"
    echo "         --security-opt seccomp=unconfined to get the fast path."
fi

section "GPU targets compiled into pmemd"
# 70=V100, 75=T4, 80=A100, 89=L4. If the GPU above is not in this list, pmemd
# has no native code for it.
echo "  requested at build: $(cat /opt/amber/pmemd26/.sm_archs 2>/dev/null || echo unknown)"
if command -v nvidia-smi > /dev/null 2>&1; then
    cc=$(nvidia-smi --query-gpu=compute_cap --format=csv,noheader 2>/dev/null | head -1 | tr -d '.')
    if [ -n "${cc}" ]; then
        case " $(cat /opt/amber/pmemd26/.sm_archs 2>/dev/null) " in
            *" ${cc} "*) echo "  [ ok ] this GPU is sm_${cc}, which is compiled in" ;;
            *) echo "  [warn] this GPU is sm_${cc}, which is NOT compiled in"; status=1 ;;
        esac
    fi
fi

section "NCCL"
# Linked into pmemd.cuda.MPI. Note that AMBER only reaches its NCCL path when
# ONE simulation is split across >2 GPUs; EasyBFE runs one rank per lambda
# window, so the ABFE pipeline itself never exercises it -- container/DEVELOP.md.
echo "  pmemd build: $(cat /opt/amber/pmemd26/.nccl 2>/dev/null || echo unknown)"
ldd "${PMEMDHOME:-/opt/amber/pmemd26}/bin/pmemd.cuda.MPI" 2>/dev/null \
    | grep nccl || echo "  (no libnccl in pmemd.cuda.MPI)"

section "pmemd linkage"
# ldd the real ELF binaries, not whatever PATH resolves to: pmemd.cuda.MPI is
# shadowed by the GPU-pinning shim in /opt/easybfe/bin, which is a shell script.
for exe in pmemd.cuda pmemd.cuda.MPI; do
    real="${PMEMDHOME:-/opt/amber/pmemd26}/bin/$exe"
    if [ ! -x "${real}" ]; then
        echo "  [FAIL] ${real} missing"; status=1; continue
    fi
    if ldd "${real}" 2>/dev/null | grep -q "not found"; then
        echo "  [FAIL] $exe has unresolved libraries:"
        ldd "${real}" | grep "not found"
        status=1
    else
        echo "  [ ok ] $exe links cleanly"
    fi
done

section "GPU pinning shim"
if [ -x /opt/easybfe/bin/pmemd.cuda.MPI ] \
   && head -1 /opt/easybfe/bin/pmemd.cuda.MPI | grep -q '^#!'; then
    echo "  [ ok ] active (each MPI rank gets one GPU; EASYBFE_DISABLE_GPU_PIN=1 to skip)"
else
    echo "  [warn] not installed -- ranks will each initialise all visible GPUs"
fi

printf '\n'
if [ "$status" -eq 0 ]; then
    echo "selfcheck: PASS"
else
    echo "selfcheck: FAIL"
fi
exit "$status"
