#!/usr/bin/env bash
# Run the end-to-end ABFE job inside the EasyBFE + AMBER 26 image on all local
# GPUs. Meant to be run on a multi-GPU node (4x A100 / 4x L4).
#
#   ./container/e2e-abfe-test/run-test.sh
#   ./container/e2e-abfe-test/run-test.sh --image <registry>/easybfe-amber26:latest
#   ./container/e2e-abfe-test/run-test.sh --selfcheck-only     # just probe the image
#
# Inputs come from container/e2e-abfe-test/inputs and are **mounted read-only** at
# /work/inputs. The image carries no test data, so a different system or a
# changed config is a different --inputs directory, never a rebuild.
#
# 24 lambda windows x 3 legs run as 24 MPI ranks sharing the node's GPUs (via
# CUDA MPS when available).
set -euo pipefail

E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

IMAGE="${IMAGE:-easybfe-amber26:latest}"
CONFIG="${CONFIG:-config.yaml}"
PROTEIN="${PROTEIN:-5USZ_jh2_wt_protein_fixed.pdb}"
LIGAND="${LIGAND:-5USZ_ligand_fixed.sdf}"
OUTDIR="${OUTDIR:-$PWD/abfe-test-$(date +%Y%m%d-%H%M%S)}"
INPUTS="${INPUTS:-${E2E_DIR}/inputs}"
GPUS="${GPUS:-all}"
ARCHIVE="${ARCHIVE:-}"            # gs:// prefix to upload the finished run to
SELFCHECK_ONLY=0
EXTRA_ENV=()                      # extra -e KEY=VAL passed to docker run

usage() {
    sed -n '2,14p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --image IMAGE        image to run                  (default: ${IMAGE})
  --outdir DIR         host output directory         (default: ./abfe-test-<timestamp>)
  --config NAME        config file inside the inputs (default: ${CONFIG})
  --inputs DIR         directory holding the config, protein and ligand
                       (default: ${INPUTS}); mounted read-only at /work/inputs
  --gpus SPEC          docker --gpus value           (default: ${GPUS})
  --archive gs://B/P   tar the finished run and upload it there before exiting.
                       Use this on an ephemeral GPU node -- the outputs die with
                       the VM otherwise.
  --env KEY=VAL        extra environment variable for the container (repeatable).
                       e.g. OMPI_MCA_rmaps_base_oversubscribe=1 when there are
                       more lambda windows than the machine has hardware threads.
  --selfcheck-only     run selfcheck.sh and exit
  -h, --help           this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --image)          IMAGE="$2"; shift 2 ;;
        --outdir)         OUTDIR="$2"; shift 2 ;;
        --config)         CONFIG="$2"; shift 2 ;;
        --inputs)         INPUTS="$2"; shift 2 ;;
        --gpus)           GPUS="$2"; shift 2 ;;
        --archive)        ARCHIVE="$2"; shift 2 ;;
        --env)            EXTRA_ENV+=(-e "$2"); shift 2 ;;
        --selfcheck-only) SELFCHECK_ONLY=1; shift ;;
        -h|--help)        usage; exit 0 ;;
        *)                echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

command -v docker > /dev/null || { echo "error: docker not found" >&2; exit 1; }

echo "==> GPUs on this host"
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,memory.total --format=csv
    NGPU=$(nvidia-smi --list-gpus | wc -l)
    [ "${NGPU}" -ge 4 ] || echo "    note: ${NGPU} GPU(s) found; the job is sized for 4x A100."
else
    echo "    warning: nvidia-smi not on the host PATH"
fi

DOCKER_RUN=(docker run --rm --gpus "${GPUS}"
    --ipc=host --shm-size=8g
    --ulimit memlock=-1 --ulimit stack=67108864
    ${EXTRA_ENV[@]+"${EXTRA_ENV[@]}"})

echo "==> Image self-check"
"${DOCKER_RUN[@]}" "${IMAGE}" selfcheck.sh
[ "${SELFCHECK_ONLY}" -eq 1 ] && exit 0

mkdir -p "${OUTDIR}"
OUTDIR="$(cd "${OUTDIR}" && pwd)"
LOG="${OUTDIR}/test.log"

# Inputs are always a host directory, mounted read-only. The container writes
# only under /work.
INPUTS="$(cd "${INPUTS}" && pwd)"
for f in "${CONFIG}" "${PROTEIN}" "${LIGAND}"; do
    [ -f "${INPUTS}/${f}" ] || { echo "error: ${INPUTS}/${f} not found" >&2; exit 1; }
done
INPUT_MOUNT=(-v "${INPUTS}:/work/inputs:ro")

echo "==> Running ABFE pipeline"
echo "    image  : ${IMAGE}"
echo "    inputs : ${INPUTS}"
echo "    config : ${CONFIG}"
echo "    outdir : ${OUTDIR}"
echo "    log    : ${LOG}"

# Sample GPU utilisation on the host so the run leaves evidence that all GPUs
# were driven, not just GPU 0.
GPU_CSV="${OUTDIR}/gpu-usage.csv"
if command -v nvidia-smi > /dev/null 2>&1; then
    nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used \
        --format=csv -l 60 > "${GPU_CSV}" 2>/dev/null &
    SAMPLER_PID=$!
    trap 'kill "${SAMPLER_PID}" 2>/dev/null || true' EXIT
fi

start=$(date +%s)
set +e
"${DOCKER_RUN[@]}" \
    -v "${OUTDIR}:/work" "${INPUT_MOUNT[@]}" \
    "${IMAGE}" bash -c "
        set -euo pipefail
        cd /work
        which pmemd.cuda.MPI mpirun
        easybfe abfe pipeline /work/inputs/${CONFIG} \
            -p /work/inputs/${PROTEIN} \
            -l /work/inputs/${LIGAND} \
            -o /work/run
    " 2>&1 | tee "${LOG}"
rc=${PIPESTATUS[0]}
set -e
elapsed=$(( $(date +%s) - start ))

echo
echo "==> Finished in $((elapsed / 3600))h $(((elapsed % 3600) / 60))m $((elapsed % 60))s (exit ${rc})"

# Ephemeral GPU nodes (spot / flex-start) take their disks with them. Get the
# whole run off the machine before that happens -- including on failure, where
# the outputs are the only way to work out what went wrong.
if [ -n "${ARCHIVE}" ]; then
    tarball="/tmp/$(basename "${OUTDIR}").tar.gz"
    echo "==> Archiving $(du -sh "${OUTDIR}" | cut -f1) to ${ARCHIVE}"
    tar czf "${tarball}" -C "$(dirname "${OUTDIR}")" "$(basename "${OUTDIR}")"
    gcloud storage cp "${tarball}" "${ARCHIVE%/}/" && \
        echo "==> Archived: ${ARCHIVE%/}/$(basename "${tarball}")"
    rm -f "${tarball}"
fi

RESULT="${OUTDIR}/run/abfe/result.json"
if [ -f "${RESULT}" ]; then
    echo "==> ${RESULT}"
    cat "${RESULT}"
    echo
    echo "TEST PASSED"
    exit 0
fi

echo "==> No result.json produced. Look at:"
echo "    ${LOG}"
echo "    ${OUTDIR}/run/abfe.log"
echo "    ${OUTDIR}/run/abfe/{solvent,complex,restraint}/pipeline_run.log"
echo "TEST FAILED"
exit "${rc:-1}"
