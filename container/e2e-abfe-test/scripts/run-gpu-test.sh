#!/usr/bin/env bash
# Create a GPU node, run the end-to-end ABFE job on it unattended, and collect
# the timings.
#
#   run-gpu-test.sh --label a100x4 --machine a2-highgpu-4g
#   run-gpu-test.sh --label l4x4   --machine g2-standard-48
#   run-gpu-test.sh --label t4x4   --machine n1-standard-32 \
#                   --accelerator type=nvidia-tesla-t4,count=4
#   run-gpu-test.sh --label a100x4-ssd --machine a2-highgpu-4g --local-ssd 1
#   run-gpu-test.sh --label pin-off --env EASYBFE_DISABLE_GPU_PIN=1
#
# --env KEY=VAL (repeatable) is passed straight through to run-test.sh's own
# --env, i.e. becomes a `docker run -e KEY=VAL` on the node. For runtime
# knobs (EASYBFE_DISABLE_GPU_PIN, CUDA_MPS_ACTIVE_THREAD_PERCENTAGE, ...) that
# don't need a rebuild to test.
#
# --local-ssd N attaches N local NVMe SSD(s) (375 GB each) and has the node
# script mount them as scratch and run the job's --outdir there instead of the
# pd-balanced boot disk -- local SSD is directly attached, no network hop, so
# it answers whether the 05.prod REMD stage's per-exchange mdout/MBAR writes
# (bar_intervall collapses to nstlim under icfe=1 -- every exchange, not every
# ntwx) are network-disk-latency-bound at 24 ranks writing concurrently.
#
# It tars container/e2e-abfe-test out of the working tree, uploads it, and boots
# a node with gpu-test-node-startup.sh: Docker + NVIDIA runtime, image pull,
# run-test.sh, then pmemd's own ns/day from every window's 05.prod.out (not
# .info -- see prod-rate.py's docstring), all uploaded to GCS. Nothing here
# has to stay connected while a GPU node bills.
#
# Uploading the tree rather than baking it into the image is the point: the
# software is versioned by the image tag, the test case by this bundle, and
# changing the test case costs an upload rather than a rebuild.
#
# Capacity: 4-GPU shapes are frequently unavailable. The script tries each
# provisioning model against each zone in turn and takes the first that lands.
# FLEX_START (Dynamic Workload Scheduler) first -- it is priced below on-demand
# and is the model meant for exactly this -- then STANDARD. Add SPOT explicitly
# with --models if the cheapest possible node matters more than getting one.
#
# The node is NOT deleted when the job finishes unless --delete-on-finish is
# given: on a failure its disk is the only record. --max-run-duration is the
# backstop, so a forgotten node cannot bill indefinitely.
set -euo pipefail

E2E_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

LABEL="${LABEL:-gputest}"
MACHINE="${MACHINE:-a2-highgpu-4g}"
ACCELERATOR="${ACCELERATOR:-}"     # only for shapes without built-in GPUs (T4)
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c}"
MODELS="${MODELS:-FLEX_START STANDARD}"
IMAGE="${IMAGE:-us-central1-docker.pkg.dev/abfe-server-test/easybfe/easybfe-amber26:latest}"
CONFIG="${CONFIG:-config.yaml}"
RESULTS="${RESULTS:-gs://abfe-server-test-easybfe-results}"
STAGING="${STAGING:-gs://abfe-server-test-easybfe-build}"
DISK_GB="${DISK_GB:-300}"
LOCAL_SSD_COUNT="${LOCAL_SSD_COUNT:-0}"   # >0: attach N local NVMe SSDs, mount
                                           # as RAID0 scratch at /mnt/disks/scratch,
                                           # and point the run's --outdir there
                                           # instead of the pd-balanced boot disk.
MAX_RUN="${MAX_RUN:-4h}"
RETRIES="${RETRIES:-1}"        # whole model x zone matrix, retried this many times
RETRY_WAIT="${RETRY_WAIT:-300}"
DELETE=0
WAIT=1
EXTRA_ENV=()    # KEY=VAL pairs to hand to run-test.sh's --env, e.g. for
                # CUDA_MPS_ACTIVE_THREAD_PERCENTAGE or EASYBFE_DISABLE_GPU_PIN

usage() { sed -n '2,27p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [ $# -gt 0 ]; do
    case "$1" in
        --label)       LABEL="$2"; shift 2 ;;
        --machine)     MACHINE="$2"; shift 2 ;;
        --accelerator) ACCELERATOR="$2"; shift 2 ;;
        --zones)       ZONES="$2"; shift 2 ;;
        --models)      MODELS="$2"; shift 2 ;;
        --image)       IMAGE="$2"; shift 2 ;;
        --config)      CONFIG="$2"; shift 2 ;;
        --results)     RESULTS="$2"; shift 2 ;;
        --staging)     STAGING="$2"; shift 2 ;;
        --disk-gb)     DISK_GB="$2"; shift 2 ;;
        --local-ssd)   LOCAL_SSD_COUNT="$2"; shift 2 ;;
        --max-run)     MAX_RUN="$2"; shift 2 ;;
        --retries)     RETRIES="$2"; shift 2 ;;
        --retry-wait)  RETRY_WAIT="$2"; shift 2 ;;
        --delete-on-finish) DELETE=1; shift ;;
        --no-wait)     WAIT=0; shift ;;
        --env)         EXTRA_ENV+=("$2"); shift 2 ;;
        -h|--help)     usage; exit 0 ;;
        *) echo "unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

NAME="easybfe-${LABEL}"
BUNDLE="${STAGING%/}/e2e-abfe-test-bundle-${LABEL}.tar.gz"

# Ship the test case as it stands in the working tree. results/ is excluded --
# it is output, and the node produces its own.
echo "==> Uploading test bundle to ${BUNDLE}"
tarball="$(mktemp -t e2e-bundle-XXXXXX).tar.gz"
tar czf "${tarball}" -C "$(dirname "${E2E_DIR}")" \
    --exclude='results' --exclude='__pycache__' --exclude='.DS_Store' \
    "$(basename "${E2E_DIR}")"
gcloud storage cp "${tarball}" "${BUNDLE}"
rm -f "${tarball}"

# Joined with ';' rather than ',' -- the --metadata flag below already uses
# ',' to separate different keys, and a KEY=VAL pair itself never contains
# ';', so this is unambiguous to split back apart in gpu-test-node-startup.sh.
EXTRA_ENV_JOINED=""
if [ "${#EXTRA_ENV[@]}" -gt 0 ]; then
    EXTRA_ENV_JOINED="$(IFS=';'; echo "${EXTRA_ENV[*]}")"
fi

COMMON=(
    --machine-type="${MACHINE}"
    --image-family=common-cu129-ubuntu-2204-nvidia-580
    --image-project=deeplearning-platform-release
    --boot-disk-size="${DISK_GB}GB" --boot-disk-type=pd-balanced
    --scopes=cloud-platform
    --metadata="run-label=${LABEL},image=${IMAGE},config=${CONFIG},results-gs=${RESULTS},inputs-gs=${BUNDLE},local-ssd-count=${LOCAL_SSD_COUNT},extra-env=${EXTRA_ENV_JOINED}"
    --metadata-from-file="startup-script=${E2E_DIR}/scripts/gpu-test-node-startup.sh"
)
[ -n "${ACCELERATOR}" ] && COMMON+=(--accelerator="${ACCELERATOR}")
# Each --local-ssd flag attaches one 375 GB NVMe device; repeat for count > 1.
# interface=NVME (not SCSI) -- NVMe is the higher-throughput of the two attach
# paths GCE offers and is what current guest kernels are tuned for.
if [ "${LOCAL_SSD_COUNT}" -gt 0 ] 2>/dev/null; then
    for _ in $(seq 1 "${LOCAL_SSD_COUNT}"); do
        COMMON+=(--local-ssd=interface=NVME)
    done
fi

ZONE=""
for attempt in $(seq 1 "${RETRIES}"); do
  [ "${attempt}" -gt 1 ] && { echo "==> retry ${attempt}/${RETRIES} in ${RETRY_WAIT}s"; sleep "${RETRY_WAIT}"; }
  for model in ${MODELS}; do
    case "${model}" in
        # FLEX_START and SPOT both require an explicit termination action.
        # max-run-duration doubles as a spend cap: the node deletes itself.
        # FLEX_START additionally insists on an explicit maintenance policy.
        # Without it the API returns the self-contradictory
        #   Invalid value for field 'scheduling.onHostMaintenance': 'TERMINATE'.
        #   Scheduling must have onHostMaintenance be one of [TERMINATE].
        #   But was MIGRATE
        # which is really "you did not set it".
        FLEX_START) MODEL_ARGS=(--provisioning-model=FLEX_START
                                --instance-termination-action=DELETE
                                --maintenance-policy=TERMINATE
                                --max-run-duration="${MAX_RUN}") ;;
        SPOT)       MODEL_ARGS=(--provisioning-model=SPOT
                                --instance-termination-action=DELETE) ;;
        STANDARD)   MODEL_ARGS=(--maintenance-policy=TERMINATE
                                --instance-termination-action=DELETE
                                --max-run-duration="${MAX_RUN}") ;;
        *) echo "unknown provisioning model: ${model}" >&2; exit 2 ;;
    esac
    for z in ${ZONES}; do
        printf '==> %-11s %-18s ' "${model}" "${z}"
        if out=$(gcloud compute instances create "${NAME}" --zone="${z}" \
                    "${MODEL_ARGS[@]}" "${COMMON[@]}" 2>&1); then
            echo "created"
            ZONE="${z}"; break 3
        fi
        # Capacity and quota fail differently; only capacity is worth retrying.
        if echo "${out}" | grep -qiE "ZONE_RESOURCE_POOL_EXHAUSTED|currently unavailable|does not have enough resources"; then
            echo "no capacity"
        elif echo "${out}" | grep -qi "already exists"; then
            echo "instance already exists -- stopping"; exit 1
        else
            echo "failed"; echo "${out}" | tail -5 >&2
        fi
    done
  done
done

[ -n "${ZONE}" ] || { echo "==> No capacity for ${MACHINE} in any zone/model tried" >&2; exit 1; }
echo "==> ${NAME} running in ${ZONE} (auto-deletes after ${MAX_RUN})"

[ "${WAIT}" -eq 1 ] || { echo "==> Not waiting (--no-wait)"; exit 0; }

echo "==> Waiting for /var/log/node-done (setup + pull + run)"
start=$(date +%s)
while true; do
    if gcloud compute ssh "${NAME}" --zone="${ZONE}" --quiet \
            --command "sudo test -f /var/log/node-done" > /dev/null 2>&1; then
        break
    fi
    # SPOT can reclaim the node mid-run; instance-termination-action=DELETE
    # means it is simply gone, not stopped. Without this check the loop above
    # keeps SSH-failing and sleeping forever -- indistinguishable from "still
    # setting up" -- so a preempted run looks identical to a slow one until
    # someone notices the wall clock is wrong. Fail fast and say why instead.
    if ! gcloud compute instances describe "${NAME}" --zone="${ZONE}" > /dev/null 2>&1; then
        echo "==> ${NAME} no longer exists (SPOT preemption? check: gcloud compute operations list --filter=\"targetLink~${NAME}\")" >&2
        echo "TEST FAILED (node preempted/deleted before finishing)"
        exit 1
    fi
    sleep 60
    printf '    %s min elapsed\n' "$(( ($(date +%s) - start) / 60 ))"
done

echo "==> Node finished after $(( ($(date +%s) - start) / 60 )) min"
gcloud compute ssh "${NAME}" --zone="${ZONE}" --quiet \
    --command "sudo cat /var/log/node-done; sudo cat /opt/run/abfe-${LABEL}/prod-rate.txt 2>/dev/null"

echo "==> Artefacts: ${RESULTS%/}/${LABEL}/"
gcloud storage ls "${RESULTS%/}/${LABEL}/" || true

if [ "${DELETE}" -eq 1 ]; then
    echo "==> Deleting ${NAME}"
    gcloud compute instances delete "${NAME}" --zone="${ZONE}" --quiet
else
    echo "==> ${NAME} left running. Delete it when done:"
    echo "    gcloud compute instances delete ${NAME} --zone=${ZONE} --quiet"
fi
