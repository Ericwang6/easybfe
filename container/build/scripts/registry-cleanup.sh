#!/usr/bin/env bash
# Prune old easybfe-amber26 images from Artifact Registry.
#
#     ./registry-cleanup.sh            # dry run: list what would go
#     ./registry-cleanup.sh --delete   # actually delete
#
# Scope is deliberately ONE image name. The same repository also holds
# `easybfe-server`, whose commit-SHA tags look like disposable build output but
# are pinned by the live Cloud Run service (easybfe-api) and by Batch job
# definitions -- a repo-wide sweep would take production down with it. Check
# before widening this:
#
#     gcloud run services describe easybfe-api --region=us-central1 \
#         --format='value(spec.template.spec.containers[0].image)'
set -euo pipefail

PROJECT="${PROJECT:-abfe-server-test}"
REGION="${REGION:-us-central1}"
AR_REPO="${AR_REPO:-easybfe}"
IMAGE_NAME="${IMAGE_NAME:-easybfe-amber26}"
KEEP_TAGS="${KEEP_TAGS:-latest}"
DELETE=0

while [ $# -gt 0 ]; do
    case "$1" in
        --delete)  DELETE=1; shift ;;
        --image)   IMAGE_NAME="$2"; shift 2 ;;
        --keep)    KEEP_TAGS="$2"; shift 2 ;;
        -h|--help) sed -n '2,16p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
        *)         echo "Unknown option: $1" >&2; exit 2 ;;
    esac
done

REPO="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPO}/${IMAGE_NAME}"

echo "==> ${REPO}"
echo "    keeping tags: ${KEEP_TAGS}"
echo

# version = the digest; tags = comma-separated, empty for untagged layers left
# behind by earlier pushes. Read with a plain loop rather than mapfile, which
# bash 3.2 (what macOS ships) does not have.
listing=$(gcloud artifacts docker images list "${REPO}" \
    --include-tags --format="csv[no-heading](version,tags)" --project="${PROJECT}")

to_delete=""
while IFS= read -r row; do
    [ -n "${row}" ] || continue
    digest="${row%%,*}"
    tags="${row#*,}"
    [ "${tags}" = "${digest}" ] && tags=""     # no comma in the row: untagged
    keep=0
    for k in ${KEEP_TAGS}; do
        case ",${tags}," in *",${k},"*) keep=1 ;; esac
    done
    if [ "${keep}" = "1" ]; then
        printf '  KEEP    %s  [%s]\n' "$(echo "${digest}" | cut -c1-19)" "${tags}"
    else
        printf '  DELETE  %s  [%s]\n' "$(echo "${digest}" | cut -c1-19)" "${tags:-untagged}"
        to_delete="${to_delete}${digest} "
    fi
done <<EOF
${listing}
EOF

echo
count=$(echo ${to_delete} | wc -w | tr -d ' ')
if [ "${count}" -eq 0 ]; then
    echo "==> nothing to delete"; exit 0
fi

if [ "${DELETE}" -ne 1 ]; then
    echo "==> dry run: ${count} image(s) would be deleted. Re-run with --delete."
    exit 0
fi

for digest in ${to_delete}; do
    echo "==> deleting ${digest:0:19}"
    gcloud artifacts docker images delete "${REPO}@${digest}" \
        --project="${PROJECT}" --delete-tags --quiet
done
echo "==> done"
gcloud artifacts repositories describe "${AR_REPO}" --location="${REGION}" \
    --project="${PROJECT}" --format="value(sizeBytes)"
