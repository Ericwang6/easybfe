#!/usr/bin/env bash
# Build the EasyBFE + AMBER 26 image.
#
#   ./container/build/build.sh                          # build easybfe-amber26:latest
#   ./container/build/build.sh --push                   # ... and push to Artifact Registry
#   ./container/build/build.sh --sm-archs "80"          # A100 only (much faster build)
#   ./container/build/build.sh --no-nccl                # build pmemd.cuda.MPI without NCCL
#   ./container/build/build.sh --ompi-source 5.0.6       # link against OpenMPI 5.x (source build)
#
# Defaults target V100 (sm_70), T4 (sm_75), A100 (sm_80) and L4 (sm_89) in one
# image, with CUDA + MPI + NCCL.
#
# Requires container/build/pmemd26.tar.bz2 (licensed AMBER 26 source, not in git).
# The image contains AMBER 26 -- push it to a PRIVATE registry only.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
BUILD_DIR="${REPO_ROOT}/container/build"

# Defaults (override via flags or environment)
IMAGE_NAME="${IMAGE_NAME:-easybfe-amber26}"
TAG="${TAG:-latest}"
SM_ARCHS="${SM_ARCHS:-70 75 80 89}"      # V100, T4, A100, L4
CUDA_VERSION="${CUDA_VERSION:-12.6.3}"
UBUNTU_VERSION="${UBUNTU_VERSION:-22.04}"
NCCL="${NCCL:-TRUE}"
OMPI_MODE="${OMPI_MODE:-apt}"          # apt (4.1.2) | source
OMPI_VERSION="${OMPI_VERSION:-5.0.6}"  # only used when OMPI_MODE=source
GCP_PROJECT="${GCP_PROJECT:-abfe-server-test}"
GCP_REGION="${GCP_REGION:-us-central1}"
AR_REPO="${AR_REPO:-easybfe}"
PUSH=0
PROGRESS="${PROGRESS:-auto}"

usage() {
    sed -n '2,10p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
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
  --push               push to Artifact Registry
  --project ID         GCP project for --push          (default: ${GCP_PROJECT})
  --region REGION      Artifact Registry region        (default: ${GCP_REGION})
  --ar-repo NAME       Artifact Registry repository    (default: ${AR_REPO})
  -h, --help           this message
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
        --push)      PUSH=1; shift ;;
        --project)   GCP_PROJECT="$2"; shift 2 ;;
        --region)    GCP_REGION="$2"; shift 2 ;;
        --ar-repo)   AR_REPO="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

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
REMOTE_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/${IMAGE_NAME}:${TAG}"

echo "==> Building ${LOCAL_IMAGE}"
echo "    context      : ${REPO_ROOT}"
echo "    CUDA         : ${CUDA_VERSION} (ubuntu ${UBUNTU_VERSION}, full -runtime base)"
echo "    SM targets   : ${SM_ARCHS}"
echo "    NCCL         : ${NCCL}"
echo "    OpenMPI      : ${OMPI_MODE}$( [ "${OMPI_MODE}" = source ] && echo " ${OMPI_VERSION}" )"
echo "    easybfe ver  : ${EASYBFE_VERSION}"

start=$(date +%s)
DOCKER_BUILDKIT=1 docker build \
    --progress="${PROGRESS}" \
    -f "${BUILD_DIR}/Dockerfile" \
    -t "${LOCAL_IMAGE}" \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
    --build-arg "SM_ARCHS=${SM_ARCHS}" \
    --build-arg "NCCL=${NCCL}" \
    --build-arg "OMPI_MODE=${OMPI_MODE}" \
    --build-arg "OMPI_VERSION=${OMPI_VERSION}" \
    --build-arg "EASYBFE_VERSION=${EASYBFE_VERSION}" \
    "${REPO_ROOT}"
echo "==> Built ${LOCAL_IMAGE} in $(( ($(date +%s) - start) / 60 )) min"
echo "    size: $(docker image inspect "${LOCAL_IMAGE}" --format '{{.Size}}' | awk '{printf "%.2f GB", $1/1e9}')"

echo "==> Self-check (no GPU required for the executable/library checks)"
docker run --rm "${LOCAL_IMAGE}" selfcheck.sh || \
    echo "    (GPU sections are expected to fail on a host without GPUs)"

if [ "${PUSH}" -eq 1 ]; then
    echo "==> Pushing ${REMOTE_IMAGE}"
    gcloud auth configure-docker "${GCP_REGION}-docker.pkg.dev" --quiet
    gcloud artifacts repositories describe "${AR_REPO}" \
        --location="${GCP_REGION}" --project="${GCP_PROJECT}" > /dev/null 2>&1 || \
    gcloud artifacts repositories create "${AR_REPO}" \
        --repository-format=docker --location="${GCP_REGION}" \
        --project="${GCP_PROJECT}" \
        --description="EasyBFE + AMBER 26 (licensed: keep private)"
    docker tag "${LOCAL_IMAGE}" "${REMOTE_IMAGE}"
    docker push "${REMOTE_IMAGE}"
    echo "==> Pushed ${REMOTE_IMAGE}"
fi
