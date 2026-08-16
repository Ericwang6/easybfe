#!/usr/bin/env bash
# Register NCCL with AMBER's third-party-library machinery so that -DNCCL=TRUE
# can actually succeed.
#
# Why this is needed
# ------------------
# `cmake -DNCCL=TRUE` fails in the pmemd26 source with
#
#     CMake Error at cmake/PMEMDCompilerFlags.cmake:285 (message):
#       NCCL is selected for inter-GPU communications but was not found.
#
# and it is not a missing header or library -- the CUDA devel image ships
# /usr/include/nccl.h and /usr/lib/x86_64-linux-gnu/libnccl.so, and pointing
# NCCL_HOME / NCCL_INCLUDE_DIR / NCCL_LIBRARY at them changes nothing.
#
# The find never runs. `cmake/3rdPartyTools.cmake` computes the NEED_<tool>
# flags with
#
#     foreach(TOOL ${3RDPARTY_TOOLS})
#         list(FIND NEEDED_3RDPARTY_TOOLS ${TOOL} TOOL_INDEX)
#         test(NEED_${TOOL} NOT "${TOOL_INDEX}" EQUAL -1)
#     endforeach()
#
# i.e. only tools present in the *master* list `3RDPARTY_TOOLS` ever get a
# NEED_ flag. In this tarball that master list is a reduced, pmemd-only one
# (blas, lapack, netcdf, netcdf-fortran, zlib, libbz2, kmmd, libm, mkl, plumed)
# and does not contain nccl -- even though the top-level CMakeLists.txt appends
# nccl to NEEDED_3RDPARTY_TOOLS for every CUDA build, and 3rdPartyTools.cmake
# itself has an `if(NEED_nccl) find_package(NCCL) ...` block ready to use it.
# So NEED_nccl is never defined, find_package(NCCL) is skipped, nccl_ENABLED
# stays unset, and PMEMDCompilerFlags.cmake reports it as "not found".
#
# Adding nccl to the master list (and a matching description to the parallel
# 3RDPARTY_TOOL_USES list, which is indexed against it for the build report)
# lets the existing machinery run as its authors intended.
#
# usage: enable-nccl-tool.sh <path/to/cmake/3rdPartyTools.cmake>
set -euo pipefail

config="${1:?usage: enable-nccl-tool.sh <3rdPartyTools.cmake>}"

if grep -qE '^nccl$' "$config"; then
    echo "enable-nccl-tool.sh: $config already lists nccl"
    exit 0
fi

for anchor in 'set(3RDPARTY_TOOLS' 'set(3RDPARTY_TOOL_USES'; do
    if ! grep -qF "$anchor" "$config"; then
        echo "enable-nccl-tool.sh: '$anchor' not found in $config" >&2
        echo "The upstream file changed; re-check the 3rd-party tool lists." >&2
        exit 1
    fi
done

# Both lists end with a line whose last character is the closing paren. Append
# one entry to each, keeping the two lists index-aligned.
awk '
    /^set\(3RDPARTY_TOOLS$/     { block = "tools"; print; next }
    /^set\(3RDPARTY_TOOL_USES$/ { block = "uses";  print; next }
    block != "" && /\)[[:space:]]*$/ {
        sub(/\)[[:space:]]*$/, "", $0)      # drop the closing paren
        print
        if (block == "tools") print "nccl)"
        else print "\"for inter-GPU communications\")"
        block = ""
        next
    }
    { print }
' "$config" > "$config.new"

mv "$config.new" "$config"
echo "enable-nccl-tool.sh: registered nccl as a 3rd-party tool in $config"
