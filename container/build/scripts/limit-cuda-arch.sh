#!/usr/bin/env bash
# Restrict the GPU targets AMBER's CUDA build emits.
#
# pmemd26_src/cmake/CudaConfig.cmake hardcodes one -gencode per supported SM
# version (SM5.0 ... SM9.0 for CUDA 11.8-12.7). Every extra target is another
# full nvcc pass over the CUDA kernels, which dominates the image build time for
# GPUs we never run on. Rather than editing the vendor's arch selection, this
# appends a filter right after it: drop the -gencode pairs it produced and add
# back only the requested ones, leaving the rest of CUDA_NVCC_FLAGS untouched.
#
# usage: limit-cuda-arch.sh <path/to/CudaConfig.cmake> "<sm archs>"
#        limit-cuda-arch.sh .../CudaConfig.cmake "75 80"
set -euo pipefail

config="${1:?usage: limit-cuda-arch.sh <CudaConfig.cmake> \"<sm archs>\"}"
archs="${2:?usage: limit-cuda-arch.sh <CudaConfig.cmake> \"<sm archs>\"}"

# First statement after the per-CUDA-version arch selection block.
anchor='set(CUDA_PROPAGATE_HOST_FLAGS FALSE)'
if ! grep -qF "$anchor" "$config"; then
    echo "limit-cuda-arch.sh: anchor '$anchor' not found in $config" >&2
    echo "The upstream file changed; re-check the arch selection block." >&2
    exit 1
fi

inject=$(mktemp)
{
    echo '# --- injected by container/build/scripts/limit-cuda-arch.sh ---'
    echo 'list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX "arch=compute_")'
    echo 'list(FILTER CUDA_NVCC_FLAGS EXCLUDE REGEX "^-gencode$")'
    for arch in $archs; do
        echo "list(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_${arch},code=sm_${arch})"
    done
    echo "message(STATUS \"Restricted CUDA targets to SM: ${archs}\")"
    echo '# --- end injection ---'
} > "$inject"

awk -v injfile="$inject" -v anchor="$anchor" '
    index($0, anchor) && !done {
        while ((getline line < injfile) > 0) print line
        done = 1
    }
    { print }
' "$config" > "$config.new"

mv "$config.new" "$config"
rm -f "$inject"
echo "limit-cuda-arch.sh: $config now targets SM ${archs}"
