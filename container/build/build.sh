#!/usr/bin/env bash
# Build the EasyBFE + AMBER 26 image.
#
#   ./container/build/build.sh                          # build easybfe-amber26:latest
#   ./container/build/build.sh --sm-archs "80"          # A100 only (much faster build)
#   ./container/build/build.sh --no-nccl                # build pmemd.cuda.MPI without NCCL
#   ./container/build/build.sh --ompi-source 5.0.6       # link against OpenMPI 5.x (source build)
#   ./container/build/build.sh --cache remote            # share the layer cache via Artifact Registry
#
# Defaults target V100 (sm_70), T4 (sm_75), A100 (sm_80) and L4 (sm_89) in one
# image, with CUDA + MPI + NCCL.
#
# Requires container/build/pmemd26.tar.bz2 (licensed AMBER 26 source, not in git).
# The image contains AMBER 26 -- only push it to a PRIVATE registry (see push_gcp.sh).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/container/build"

# Defaults (override via flags or environment).
#
# The environment overrides are EASYBFE_-prefixed on purpose. A bare IMAGE_NAME
# is not this script's to claim: GCP's Deep Learning VM images export exactly
# that from /etc/profile.d/env.sh to name the *VM image* (e.g.
# common-cu129-ubuntu-2204-nvidia-580-stage), so honouring it tagged the build
# after the host OS image instead of easybfe-amber26.
IMAGE_NAME="${EASYBFE_IMAGE_NAME:-easybfe-amber26}"
TAG="${EASYBFE_TAG:-latest}"
SM_ARCHS="${SM_ARCHS:-70 75 80 89}"      # V100, T4, A100, L4
CUDA_VERSION="${CUDA_VERSION:-12.6.3}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"
NCCL="${NCCL:-TRUE}"
OMPI_MODE="${OMPI_MODE:-apt}"          # apt (4.1.2) | source
OMPI_VERSION="${OMPI_VERSION:-5.0.6}"  # only used when OMPI_MODE=source
PROGRESS="${PROGRESS:-auto}"

# Layer cache: local (this machine's BuildKit cache) or remote (a registry the
# whole fleet can read). CACHE_REGISTRY is host + project; the ref is completed
# with the repo and image name below.
CACHE_MODE="${CACHE_MODE:-local}"                                    # local | remote
CACHE_REGISTRY="${CACHE_REGISTRY:-us-docker.pkg.dev/abfe-server-test}"
CACHE_REPO="${CACHE_REPO:-easybfe}"
CACHE_REF="${CACHE_REF:-}"             # full override; derived from the above if empty
BUILDER="${BUILDER:-easybfe-builder}"  # buildx builder, remote cache only

usage() {
    sed -n '2,12p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --tag TAG            image tag                       (default: ${TAG})
  --name NAME          image name                      (default: ${IMAGE_NAME})
  --sm-archs "A B"     CUDA SM targets to compile      (default: ${SM_ARCHS})
                       70=V100, 75=T4, 80=A100, 89=L4. Each target is another
                       nvcc pass, so trim the list for a faster build.
  --cuda VERSION       CUDA base image version         (default: ${CUDA_VERSION})
  --no-nccl            build pmemd.cuda.MPI without NCCL. On by default; note
                       that AMBER only reaches its NCCL path when one
                       simulation spans >2 GPUs, which the EasyBFE ABFE
                       pipeline never does (container/DEVELOP.md).
  --ompi-source [VER]  build OpenMPI from source (default version: ${OMPI_VERSION})
                       and link pmemd.cuda.MPI against it, instead of Ubuntu's
                       apt OpenMPI 4.1.2                (default: apt)
  --cache local|remote where BuildKit reads/writes layers (default: ${CACHE_MODE})
                       local  -- this machine's builder cache only.
                       remote -- also pull from and push to
                                 ${CACHE_REGISTRY}/${CACHE_REPO}/${IMAGE_NAME}:buildcache
                                 so a fresh VM reuses a previous build's nvcc
                                 output. Needs gcloud auth and buildx; the
                                 cache holds AMBER object code, so the
                                 registry must stay PRIVATE.
  --cache-ref REF      full remote cache image ref, overriding the derived one
  -h, --help           this message

To push a built image to Artifact Registry, use push_gcp.sh.
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --tag)       TAG="$2"; shift 2 ;;
        --name)      IMAGE_NAME="$2"; shift 2 ;;
        --sm-archs)  SM_ARCHS="$2"; shift 2 ;;
        --cuda)      CUDA_VERSION="$2"; shift 2 ;;
        --nccl)      NCCL=TRUE; shift ;;
        --no-nccl)   NCCL=FALSE; shift ;;
        --ompi-source)
            OMPI_MODE=source; shift
            # Optional version argument: only consume it if it doesn't look
            # like the next flag.
            if [ $# -gt 0 ] && [ "${1#--}" = "$1" ]; then OMPI_VERSION="$1"; shift; fi
            ;;
        --cache)     CACHE_MODE="$2"; shift 2 ;;
        --cache-ref) CACHE_REF="$2"; CACHE_MODE=remote; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

case "${CACHE_MODE}" in
    local|remote) ;;
    *) echo "error: --cache must be local or remote (got '${CACHE_MODE}')" >&2; exit 2 ;;
esac

if [ ! -f "${BUILD_DIR}/pmemd26.tar.bz2" ]; then
    cat >&2 <<EOF
error: ${BUILD_DIR}/pmemd26.tar.bz2 not found.

The AMBER 26 pmemd source is licensed and is not kept in git. Download it from
https://ambermd.org/GetAmber.php with your licence and place it there, e.g.

    gsutil cp gs://<your-bucket>/pmemd26.tar.bz2 ${BUILD_DIR}/
EOF
    exit 1
fi

# setuptools_scm cannot see .git from inside the build context, so pin the
# version to the commit being built.
GIT_SHA="$(git -C "${REPO_ROOT}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
EASYBFE_VERSION="${EASYBFE_VERSION:-0.0.0+g${GIT_SHA}}"

LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"
[ -n "${CACHE_REF}" ] || CACHE_REF="${CACHE_REGISTRY}/${CACHE_REPO}/${IMAGE_NAME}:buildcache"

echo "==> Building ${LOCAL_IMAGE}"
echo "    context      : ${REPO_ROOT}"
echo "    CUDA         : ${CUDA_VERSION} (ubuntu ${UBUNTU_VERSION}, full -runtime base)"
echo "    SM targets   : ${SM_ARCHS}"
echo "    NCCL         : ${NCCL}"
echo "    OpenMPI      : ${OMPI_MODE}$( [ "${OMPI_MODE}" = source ] && echo " ${OMPI_VERSION}" )"
echo "    easybfe ver  : ${EASYBFE_VERSION}"
echo "    layer cache  : ${CACHE_MODE}$( [ "${CACHE_MODE}" = remote ] && echo " -> ${CACHE_REF}" )"

BUILD_ARGS=(
    --build-arg "CUDA_VERSION=${CUDA_VERSION}"
    --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}"
    --build-arg "SM_ARCHS=${SM_ARCHS}"
    --build-arg "NCCL=${NCCL}"
    --build-arg "OMPI_MODE=${OMPI_MODE}"
    --build-arg "OMPI_VERSION=${OMPI_VERSION}"
    --build-arg "EASYBFE_VERSION=${EASYBFE_VERSION}"
)

start=$(date +%s)
if [ "${CACHE_MODE}" = "local" ]; then
    DOCKER_BUILDKIT=1 docker build \
        --progress="${PROGRESS}" \
        -f "${BUILD_DIR}/Dockerfile" \
        -t "${LOCAL_IMAGE}" \
        "${BUILD_ARGS[@]}" \
        "${REPO_ROOT}"
else
    # `--cache-to type=registry` needs a docker-container builder: the default
    # "docker" driver that plain `docker build` uses cannot export a registry
    # cache (it fails with "cache export feature is currently not supported").
    # Hence buildx with a dedicated builder, and --load to put the result back
    # in the local image store the way `docker build` would have.
    command -v docker >/dev/null 2>&1 || { echo "error: docker not found" >&2; exit 1; }
    docker buildx version >/dev/null 2>&1 || {
        echo "error: docker buildx is required for --cache remote" >&2; exit 1; }

    # Credentials for the cache registry, keyed by host.
    CACHE_HOST="${CACHE_REF%%/*}"
    if command -v gcloud >/dev/null 2>&1; then
        gcloud auth configure-docker "${CACHE_HOST}" --quiet
    else
        echo "    warn: gcloud not found; assuming ${CACHE_HOST} is already authenticated"
    fi

    docker buildx inspect "${BUILDER}" >/dev/null 2>&1 || {
        echo "==> Creating buildx builder ${BUILDER} (driver: docker-container)"
        docker buildx create --name "${BUILDER}" --driver docker-container --bootstrap >/dev/null
    }

    # mode=max caches every layer including stage 1's, which is the whole point
    # here -- the default (mode=min) would only cache the layers that survive
    # into the final image, i.e. not the pmemd compile.
    #
    # cache-from is best-effort: the first build on a new cache ref finds
    # nothing, which BuildKit reports and moves past rather than failing.
    docker buildx build \
        --builder "${BUILDER}" \
        --progress="${PROGRESS}" \
        -f "${BUILD_DIR}/Dockerfile" \
        -t "${LOCAL_IMAGE}" \
        "${BUILD_ARGS[@]}" \
        --cache-from "type=registry,ref=${CACHE_REF}" \
        --cache-to "type=registry,ref=${CACHE_REF},mode=max" \
        --load \
        "${REPO_ROOT}"
fi
echo "==> Built ${LOCAL_IMAGE} in $(( ($(date +%s) - start) / 60 )) min"
echo "    size: $(docker image inspect "${LOCAL_IMAGE}" --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1e9}')"

# selfcheck.sh reports on GPUs, and its GPU sections need the devices actually
# passed in -- without --gpus there is no nvidia-smi in the container and it
# fails on a host that has GPUs sitting right there. Probe rather than assume:
# `--gpus all` is itself an error on a daemon with no NVIDIA runtime.
echo "==> Self-check"
GPU_RUN_FLAGS=()
if command -v nvidia-smi >/dev/null 2>&1 && nvidia-smi -L >/dev/null 2>&1; then
    if docker run --rm --gpus all "${LOCAL_IMAGE}" true >/dev/null 2>&1; then
        GPU_RUN_FLAGS=(--gpus all)
        echo "    GPUs: $(nvidia-smi -L | wc -l) visible, passed through"
    else
        echo "    warn: host has GPUs but 'docker run --gpus all' failed --"
        echo "          install/configure the NVIDIA container toolkit"
        echo "          (nvidia-ctk runtime configure --runtime=docker) to include"
        echo "          the GPU sections. Running without them."
    fi
else
    echo "    no host GPUs detected; GPU sections will report FAIL"
fi

docker run --rm "${GPU_RUN_FLAGS[@]+"${GPU_RUN_FLAGS[@]}"}" "${LOCAL_IMAGE}" selfcheck.sh || {
    if [ ${#GPU_RUN_FLAGS[@]} -eq 0 ]; then
        echo "    (GPU sections are expected to fail without --gpus all)"
    else
        echo "    selfcheck reported a failure with GPUs attached -- read the output above"
    fi
}
