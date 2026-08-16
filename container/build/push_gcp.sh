#!/usr/bin/env bash
# Push a locally-built EasyBFE + AMBER image to GCP Artifact Registry.
#
#   ./container/build/push_gcp.sh                          # push easybfe-amber26:latest
#   ./container/build/push_gcp.sh --name foo --tag bar      # push foo:bar
#   ./container/build/push_gcp.sh --project other-project   # push to a different GCP project
#
# The image contains licensed AMBER 26 -- push it to a PRIVATE registry only.
set -euo pipefail

# Defaults (override via flags or environment).
#
# EASYBFE_-prefixed for the same reason as build.sh: GCP Deep Learning VMs
# export a bare IMAGE_NAME (the VM's own OS image) from /etc/profile.d/env.sh,
# which would otherwise pick the name this pushes under.
IMAGE_NAME="${EASYBFE_IMAGE_NAME:-easybfe-amber26}"
TAG="${EASYBFE_TAG:-latest}"
GCP_PROJECT="${GCP_PROJECT:-abfe-server-test}"
GCP_REGION="${GCP_REGION:-us}"   # AR multi-region -> us-docker.pkg.dev
AR_REPO="${AR_REPO:-easybfe}"

usage() {
    sed -n '2,7p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    cat <<EOF

Options:
  --tag TAG            image tag                       (default: ${TAG})
  --name NAME          image name                      (default: ${IMAGE_NAME})
  --project ID         GCP project                      (default: ${GCP_PROJECT})
  --region REGION      Artifact Registry region        (default: ${GCP_REGION})
  --ar-repo NAME       Artifact Registry repository    (default: ${AR_REPO})
  -h, --help           this message
EOF
}

while [ $# -gt 0 ]; do
    case "$1" in
        --tag)       TAG="$2"; shift 2 ;;
        --name)      IMAGE_NAME="$2"; shift 2 ;;
        --project)   GCP_PROJECT="$2"; shift 2 ;;
        --region)    GCP_REGION="$2"; shift 2 ;;
        --ar-repo)   AR_REPO="$2"; shift 2 ;;
        -h|--help)   usage; exit 0 ;;
        *)           echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

LOCAL_IMAGE="${IMAGE_NAME}:${TAG}"
REMOTE_IMAGE="${GCP_REGION}-docker.pkg.dev/${GCP_PROJECT}/${AR_REPO}/${IMAGE_NAME}:${TAG}"

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
