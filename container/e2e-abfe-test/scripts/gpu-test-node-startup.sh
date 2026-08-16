#!/usr/bin/env bash
# GCE startup script for a GPU test node (4xA100 or 4xL4).
#
# Runs unattended at boot: installs Docker + the NVIDIA container runtime, pulls
# the image, runs the shortened ABFE job on all local GPUs, extracts pmemd's own
# ns/day from every window's 05.prod.out (not .info -- see prod-rate.py's
# docstring), and uploads the small artefacts to GCS. Nobody has to be at a
# prompt while a spot GPU node is billing.
#
# Metadata keys it reads (all optional):
#   image        container image to test
#   run-label    tag for the results directory, e.g. a100x4 / l4x4
#   config       config file inside the mounted inputs directory
#   inputs-gs    gs:// URL of the e2e-abfe-test bundle (run-gpu-test.sh uploads it)
#   results-gs   gs:// prefix for the uploaded artefacts
#   local-ssd-count  >0: mount N local NVMe SSD(s) as scratch, see below
#   extra-env    ';'-joined KEY=VAL pairs, passed to the container as --env
#
# Progress: /var/log/node-setup.log. Finished: /var/log/node-done.
set -uxo pipefail

meta() {  # meta <key> <default>
    curl -fsH "Metadata-Flavor: Google" \
        "http://metadata.google.internal/computeMetadata/v1/instance/attributes/$1" \
        2>/dev/null || echo "$2"
}

IMAGE=$(meta image "us-central1-docker.pkg.dev/abfe-server-test/easybfe/easybfe-amber26:latest")
LABEL=$(meta run-label "gpu")
CONFIG=$(meta config "config.yaml")
RESULTS=$(meta results-gs "gs://abfe-server-test-easybfe-results")
BUNDLE=$(meta inputs-gs "gs://abfe-server-test-easybfe-build/e2e-abfe-test-bundle.tar.gz")
LOCAL_SSD_COUNT=$(meta local-ssd-count "0")
EXTRA_ENV=$(meta extra-env "")
WORK=/opt/run

exec > >(tee -a /var/log/node-setup.log) 2>&1
echo "=== startup $(date -Is) label=${LABEL} image=${IMAGE} ==="

# Docker CE (not docker.io: no buildx plugin) ---------------------------------
if ! docker buildx version > /dev/null 2>&1; then
    apt-get remove -y -qq docker.io docker-doc docker-compose podman-docker containerd runc 2>/dev/null
    install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg | gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        > /etc/apt/sources.list.d/docker.list
    apt-get update -qq
    apt-get install -y -qq docker-ce docker-ce-cli containerd.io docker-buildx-plugin
fi

if ! command -v nvidia-ctk > /dev/null 2>&1; then
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        > /etc/apt/sources.list.d/nvidia-container-toolkit.list
    apt-get update -qq
    apt-get install -y -qq nvidia-container-toolkit
    nvidia-ctk runtime configure --runtime=docker
fi

systemctl unmask docker.socket docker.service
systemctl enable --now docker.socket
systemctl restart docker.service

# Persistence mode keeps the driver resident: without it the first CUDA context
# on an idle GPU pays a multi-second initialisation, which every rank pays.
nvidia-smi -pm 1 || true
nvidia-smi

# Local SSD scratch -------------------------------------------------------
# Directly-attached NVMe, no network hop -- unlike the pd-balanced boot disk
# WORK otherwise lives on. Ephemeral by construction (gone when the node is
# deleted), which is fine: run-test.sh already treats the run directory as
# throwaway and uploads only the small artefacts.
if [ "${LOCAL_SSD_COUNT}" -gt 0 ] 2>/dev/null; then
    mapfile -t SSDS < <(ls /dev/disk/by-id/google-local-nvme-ssd-* 2>/dev/null | sort)
    if [ "${#SSDS[@]}" -eq 0 ]; then
        echo "WARNING: local-ssd-count=${LOCAL_SSD_COUNT} but no /dev/disk/by-id/google-local-nvme-ssd-* found; falling back to the boot disk"
    else
        echo "==> ${#SSDS[@]} local NVMe SSD(s): ${SSDS[*]}"
        mkdir -p /mnt/disks/scratch
        if [ "${#SSDS[@]}" -eq 1 ]; then
            TARGET="${SSDS[0]}"
        else
            # RAID0: scratch data only, replaceability is not a concern, and
            # striping is what turns N devices' throughput into N*.
            apt-get update -qq && apt-get install -y -qq mdadm
            mdadm --create /dev/md0 --level=0 --raid-devices="${#SSDS[@]}" "${SSDS[@]}" --force
            TARGET=/dev/md0
        fi
        mkfs.ext4 -F -m 0 -E lazy_itable_init=0,lazy_journal_init=0,discard "${TARGET}"
        mount -o defaults "${TARGET}" /mnt/disks/scratch
        chmod 1777 /mnt/disks/scratch
        WORK=/mnt/disks/scratch
        echo "==> WORK=${WORK} ($(df -h /mnt/disks/scratch | tail -1))"
    fi
fi

# Payload ---------------------------------------------------------------------
# Only container/e2e-abfe-test comes down: the software is in the image, this is the
# test case. Uploaded by run-gpu-test.sh from the working tree, so what runs
# here is what is checked out there.
mkdir -p "${WORK}" && cd "${WORK}"
gcloud storage cp "${BUNDLE}" ./e2e-abfe-test-bundle.tar.gz && tar xzf e2e-abfe-test-bundle.tar.gz
chmod +x e2e-abfe-test/run-test.sh

gcloud auth configure-docker us-central1-docker.pkg.dev --quiet
docker pull "${IMAGE}" || { echo "PULL FAILED"; echo "pull-failed" > /var/log/node-done; exit 1; }

docker run --rm --gpus all "${IMAGE}" selfcheck.sh

OUT="${WORK}/abfe-${LABEL}"
# extra-env metadata is ';'-joined KEY=VAL pairs (run-gpu-test.sh's --env);
# turn each back into its own --env flag for run-test.sh.
ENV_ARGS=()
if [ -n "${EXTRA_ENV}" ]; then
    IFS=';' read -ra _pairs <<< "${EXTRA_ENV}"
    for kv in "${_pairs[@]}"; do
        ENV_ARGS+=(--env "${kv}")
    done
fi
./e2e-abfe-test/run-test.sh --image "${IMAGE}" --config "${CONFIG}" \
    --inputs "${WORK}/e2e-abfe-test/inputs" --outdir "${OUT}" \
    "${ENV_ARGS[@]}"
rc=$?

# pmemd's own rate, per lambda window, from each window's mdinfo.
python3 e2e-abfe-test/scripts/prod-rate.py "${OUT}/run" --label "${LABEL}" \
    --csv "${OUT}/prod-rate.csv" | tee "${OUT}/prod-rate.txt"
python3 e2e-abfe-test/scripts/prod-rate.py "${OUT}/run" --label "${LABEL}" \
    --stage 04.pre_prod --csv "${OUT}/pre-prod-rate.csv" | tee -a "${OUT}/prod-rate.txt"

# Upload the small artefacts (the trajectories stay on the disk that dies with
# the VM -- they are reproducible, the timings are the deliverable).
STAGE="/tmp/artifacts-${LABEL}"
mkdir -p "${STAGE}"
cp -f "${OUT}"/prod-rate.* "${OUT}/test.log" "${STAGE}/" 2>/dev/null
cp -f "${OUT}/gpu-usage.csv" "${STAGE}/" 2>/dev/null
cp -f "${OUT}/run/abfe/result.json" "${STAGE}/" 2>/dev/null
cp -f "${OUT}/run/abfe.log" "${STAGE}/" 2>/dev/null
nvidia-smi --query-gpu=index,name,memory.total,driver_version --format=csv > "${STAGE}/gpus.csv"
# The per-window mdinfo/mdout files: small, and the evidence behind every rate
# quoted above.
tar czf "${STAGE}/mdinfo.tar.gz" -C "${OUT}/run" \
    --exclude='*.mdcrd' --exclude='*.rst7' \
    $(cd "${OUT}/run" && find . \( -name '*.info' -o -name '*.out' \
                                   -o -name 'pipeline_run.log' \) -print)
gcloud storage cp -r "${STAGE}"/* "${RESULTS%/}/${LABEL}/"

echo "exit=${rc}" > /var/log/node-done
echo "=== done $(date -Is) exit=${rc} ===" | tee -a /var/log/node-done
