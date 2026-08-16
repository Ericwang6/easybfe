# EasyBFE + AMBER 26 (pmemd.cuda / pmemd.cuda.MPI), multi-GPU, portable.
#
# Stage 1 compiles the licensed AMBER 26 pmemd sources (container/build/pmemd26.tar.bz2,
# not redistributable -- keep the resulting image in a private registry) in a
# single CUDA+MPI+NCCL pass that emits *both* pmemd.cuda and pmemd.cuda.MPI
# into /opt/amber/pmemd26.
# Stage 2 is the full CUDA *runtime* image carrying the complete CUDA library
# set (including NCCL), the installed pmemd tree, and the complete conda
# environment from the repository's environment.yml.
#
# Design choices, deliberately:
#   * `-runtime` base, not `-base` + hand-picked libraries. Every CUDA library
#     is present, so a binary built here cannot fail to load on a node for want
#     of a library nobody thought to list -- and NCCL comes with it.
#   * The conda environment is environment.yml *unmodified*: nothing is dropped,
#     nothing is swapped for a "-base" variant, no post-install pruning. The
#     only edit is additive -- five packages the ABFE code path imports that
#     environment.yml does not list.
#   * The one thing that is stripped out is the AMBER pmemd *source tree*: it is
#     licensed, it is 350 MB, and the runtime stage only ever needs the compiled
#     binaries. It never leaves stage 1.
#   * GPU targets cover the whole fleet this is expected to run on -- V100
#     (sm_70), T4 (sm_75), A100 (sm_80) and L4 (sm_89) -- so one image tag
#     works on all four without a PTX JIT fallback.
#
# Build context is the repository root:
#     docker build -f container/build/Dockerfile .   (see container/build/build.sh)

ARG CUDA_VERSION=12.6.3
ARG UBUNTU_VERSION=22.04

# --------------------------------------------------------------------------- #
# Stage 1: compile pmemd26                                                     #
# --------------------------------------------------------------------------- #
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu${UBUNTU_VERSION} AS pmemd-builder

# GPU targets to emit, as real SM code (not PTX):
#
#   70 = V100 (Volta)      75 = T4 (Turing)
#   80 = A100 (Ampere)     89 = L4 (Ada)
#
# AMBER's cmake otherwise builds every SM from 5.0 to 9.0 -- eleven nvcc passes
# over the CUDA kernels, and it does that once for the serial CUDA library and
# again for the MPI one. Four targets is roughly 2x the compile time of one and
# gives a binary that runs natively on all four cards; leaving an architecture
# out would make that card fall back to JIT-compiling PTX at every launch (or
# fail outright, since AMBER does not ship a PTX-only target).
ARG SM_ARCHS="70 75 80 89"

# Link NCCL into pmemd.cuda.MPI. On by default here.
#
# Scope note, so the capability is not mistaken for a speed-up in this
# pipeline: AMBER only reaches its NCCL path when a *single* simulation is
# decomposed across >2 GPUs (gpu.cpp: bNCCL requires bSingleNode && nGpus > 2,
# and gpu_allreduce returns early unless nGpus > 1). EasyBFE runs one rank per
# lambda window, so each simulation group owns exactly one GPU and NCCL is not
# exercised by `easybfe abfe pipeline`. It is linked so that multi-GPU-per-
# window runs (a plain `pmemd.cuda.MPI -ng 1` across several GPUs, or any other
# AMBER job driven from this image) have it available.
ARG NCCL=TRUE

# MPI implementation pmemd.cuda.MPI is compiled against.
#
#   OMPI_MODE=apt     (default) -- Ubuntu 22.04's OpenMPI 4.1.2 from apt.
#   OMPI_MODE=source  -- build OMPI_VERSION from source into /opt/openmpi and
#                        build pmemd against that instead. Exists to answer one
#                        question: does OpenMPI 5's rewritten runtime (PRRTE/
#                        PMIx replacing ORTE) move throughput on the REMD
#                        exchange step, which is many small collectives across
#                        24 ranks rather than a bandwidth-bound workload.
#                        --with-libevent=internal --with-hwloc=internal so the
#                        only new apt dependency is curl to fetch the tarball.
ARG OMPI_MODE=apt
ARG OMPI_VERSION=5.0.6

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bison \
        bzip2 \
        ca-certificates \
        cmake \
        curl \
        flex \
        g++ \
        gcc \
        gfortran \
        make \
        patch \
        tar \
        zlib1g-dev \
    && rm -rf /var/lib/apt/lists/*

# Always create /opt/openmpi so the runtime stage's COPY --from is unconditional
# (empty in OMPI_MODE=apt, a full self-contained prefix install in OMPI_MODE=source).
RUN mkdir -p /opt/openmpi
RUN set -eux; \
    if [ "${OMPI_MODE}" = "apt" ]; then \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            libopenmpi-dev openmpi-bin \
        && rm -rf /var/lib/apt/lists/*; \
    elif [ "${OMPI_MODE}" = "source" ]; then \
        major_minor="$(echo "${OMPI_VERSION}" | cut -d. -f1-2)"; \
        curl -fsSL "https://download.open-mpi.org/release/open-mpi/v${major_minor}/openmpi-${OMPI_VERSION}.tar.bz2" \
            -o /tmp/openmpi.tar.bz2; \
        mkdir -p /tmp/ompi-src && tar -xjf /tmp/openmpi.tar.bz2 -C /tmp/ompi-src --strip-components=1; \
        cd /tmp/ompi-src && ./configure --prefix=/opt/openmpi \
            --with-cuda=/usr/local/cuda \
            --with-libevent=internal --with-hwloc=internal --with-pmix=internal \
            > /tmp/ompi-configure.log 2>&1 \
            || { tail -80 /tmp/ompi-configure.log; exit 1; } \
        && make -j"$(nproc)" > /tmp/ompi-make.log 2>&1 \
            || { tail -80 /tmp/ompi-make.log; exit 1; } \
        && make install \
        && cd / && rm -rf /tmp/ompi-src /tmp/openmpi.tar.bz2 /tmp/ompi-*.log; \
        echo "${OMPI_VERSION}" > /opt/openmpi/.version; \
    else \
        echo "unknown OMPI_MODE=${OMPI_MODE} (want apt|source)"; exit 1; \
    fi

# The CUDA devel image already carries /usr/include/nccl.h and libnccl.so, and
# apt-marks them held -- so there is nothing to install, and trying to
# (`apt-get install libnccl-dev`) fails with "held broken packages".
#
# NCCL_HOME is what AMBER's cmake/FindNCCL.cmake reads: it searches
# `$ENV{NCCL_HOME}/include` and `$ENV{NCCL_HOME}/lib` *in addition to* CMake's
# default paths, which is where the multiarch libnccl.so actually is. The other
# two are inert (find_path/find_library look at cache variables, not the
# environment) and are kept only because they cost nothing.
#
# None of these was the reason `-DNCCL=TRUE` used to fail; see the
# enable-nccl-tool.sh step below for what actually blocks it.
ENV NCCL_HOME=/usr \
    NCCL_INCLUDE_DIR=/usr/include \
    NCCL_LIBRARY=/usr/lib/x86_64-linux-gnu/libnccl.so

WORKDIR /build

COPY container/build/pmemd26.tar.bz2 /build/pmemd26.tar.bz2
RUN tar -xjf pmemd26.tar.bz2 && rm pmemd26.tar.bz2

COPY container/build/scripts/limit-cuda-arch.sh /build/limit-cuda-arch.sh
RUN bash limit-cuda-arch.sh /build/pmemd26_src/cmake/CudaConfig.cmake "${SM_ARCHS}"

ENV AMBER_PREFIX=/opt/amber/pmemd26

# Shared cmake options: pmemd only, no python/perl/GUI, no update checks.
# (Mirrors pmemd26_src/build/run_cmake, minus the tests.)
ENV PMEMD_CMAKE_ARGS="-Wno-dev \
    -DCMAKE_INSTALL_PREFIX=/opt/amber/pmemd26 \
    -DCOMPILER=GNU \
    -DPGM=FALSE \
    -DPMEMD_ONLY=TRUE \
    -DCHECK_UPDATES=FALSE \
    -DINSTALL_TESTS=FALSE \
    -DDOWNLOAD_MINICONDA=FALSE \
    -DBUILD_PYTHON=FALSE \
    -DBUILD_PERL=FALSE \
    -DBUILD_GUI=FALSE"

# AMBER 26 cannot configure `-DNCCL=TRUE` out of the box: nccl is missing from
# the master 3RDPARTY_TOOLS list that drives the NEED_<tool> flags, so
# find_package(NCCL) is never called and cmake reports NCCL as "not found" with
# the header and library sitting in plain sight. enable-nccl-tool.sh registers
# it. See the script for the full trace.
COPY container/build/scripts/enable-nccl-tool.sh /build/enable-nccl-tool.sh
RUN bash /build/enable-nccl-tool.sh /build/pmemd26_src/cmake/3rdPartyTools.cmake

# Both binaries, from one configure:
#
#   pmemd.cuda      serial, one GPU -- every non-MPI step, and `01.em`, which
#                   AMBER refuses to run in parallel on GPUs at all
#                   ("Minimization is NOT supported in parallel on GPUs").
#   pmemd.cuda.MPI  one rank per lambda window, ranks share the node's GPUs;
#                   NCCL linked in for the multi-GPU-per-simulation case.
#
# -DMPI=TRUE is additive, not a mode switch. In pmemd/src/CMakeLists.txt the
# serial executable is added and installed unconditionally inside `if(CUDA)`
# (add_executable at :149, install at :165, and the pmemd.cuda_SPFP -> pmemd.cuda
# symlink at :202 sits outside every `if(MPI)`); `if(MPI)` only *adds* the
# .MPI target beside it (:167-183). cuda/CMakeLists.txt is the same shape one
# level down -- the serial pmemd_cuda_${PRECISION} library at :56 is
# unconditional and `if(MPI)` at :84 adds a second _mpi library.
#
# So a separate -DMPI=FALSE pass, which this used to do first, built exactly the
# subset this pass already builds and then had its install overwritten by this
# one. It was removed; nothing in the image changed except the ~1/3 of stage 1
# spent producing it.
#
# One real difference, from cuda/CMakeLists.txt:80-82: `if(NCCL)` links nccl
# into the serial library too, so pmemd.cuda now links libnccl where the old
# -DMPI=FALSE pass (built before the NCCL patch, without -DNCCL) did not.
# Harmless -- libnccl.so.2 ships in the CUDA runtime base -- and the runtime
# stage's ldd gate covers it.
#
# /opt/openmpi/bin first on PATH: empty/absent in OMPI_MODE=apt (falls through
# to the apt mpicc on PATH), the source-built OMPI_VERSION's mpicc/mpicxx in
# OMPI_MODE=source -- so cmake's FindMPI picks up whichever was actually built.
RUN mkdir -p /build/build-cuda-mpi && cd /build/build-cuda-mpi \
    && { PATH="/opt/openmpi/bin:${PATH}" \
         cmake /build/pmemd26_src ${PMEMD_CMAKE_ARGS} -DCUDA=TRUE -DMPI=TRUE -DNCCL=${NCCL} > cmake.log 2>&1 \
         || { tail -60 cmake.log; exit 1; }; } \
    && grep -E "CUDA version|Checking CUDA and GNU|Restricted CUDA targets|NCCL|MPI" cmake.log | head -20 \
    && grep -E "MPI_C_COMPILER|MPI_C_LIBRARIES" cmake.log || true \
    && make -j"$(nproc)" install

# The licensed source tree is not needed past this point and must not reach the
# runtime image. Removed here, in stage 1, where it is the only thing this layer
# writes -- the runtime stage copies /opt/amber/pmemd26 alone, so no layer
# anywhere in the shipped image ever contains AMBER source.
RUN rm -rf /build/pmemd26_src /build/limit-cuda-arch.sh /build/enable-nccl-tool.sh

# Verify the build did what the arguments asked, rather than silently falling
# back. Both are build-time gates: a wrong answer here is far cheaper to find
# than on a GPU node.
#
# This is also what pins the single-configure build above honest: the loop reads
# ${AMBER_PREFIX}/bin/pmemd.cuda as well as pmemd.cuda.MPI, so if -DMPI=TRUE ever
# stopped emitting the serial binary the build fails here rather than shipping an
# image whose `01.em` dies on a GPU node.
RUN set -eux; \
    echo "== SM targets actually in the binaries =="; \
    # `cuobjdump -lelf` lists the cubins inside the fatbin, one per
    # architecture ("ELF file 1: pmemd_obj.1.sm_80.cubin"). The binaries on
    # PATH are symlinks to the *_SPFP builds; cuobjdump follows them.
    for exe in pmemd.cuda pmemd.cuda.MPI; do \
        got=$(cuobjdump -lelf "${AMBER_PREFIX}/bin/${exe}" \
              | grep -oE 'sm_[0-9]+' | sort -u | tr '\n' ' '); \
        echo "  ${exe}: ${got}"; \
        for arch in ${SM_ARCHS}; do \
            case " ${got}" in *" sm_${arch} "*) ;; \
                *) echo "ERROR: ${exe} has no sm_${arch} code"; exit 1 ;; \
            esac; \
        done; \
    done; \
    echo "== NCCL =="; \
    if ldd "${AMBER_PREFIX}/bin/pmemd.cuda.MPI" | grep -q nccl; then \
        echo linked > "${AMBER_PREFIX}/.nccl"; \
    else \
        echo not-linked > "${AMBER_PREFIX}/.nccl"; \
    fi; \
    echo "  requested NCCL=${NCCL}, result: $(cat ${AMBER_PREFIX}/.nccl)"; \
    if [ "${NCCL}" = "TRUE" ] && [ "$(cat ${AMBER_PREFIX}/.nccl)" != "linked" ]; then \
        echo "ERROR: NCCL=TRUE was requested but pmemd.cuda.MPI does not link libnccl"; exit 1; \
    fi; \
    echo "${SM_ARCHS}" > "${AMBER_PREFIX}/.sm_archs"

# Fail the build here rather than on a $10/hour GPU node if the runtime stage
# would be missing a shared library. LD_LIBRARY_PATH picks up /opt/openmpi/lib
# for OMPI_MODE=source (libmpi there isn't on the default ldconfig search path);
# harmless no-op when it's empty/absent (OMPI_MODE=apt).
RUN export LD_LIBRARY_PATH="/opt/openmpi/lib:${LD_LIBRARY_PATH:-}"; \
    for exe in pmemd.cuda pmemd.cuda.MPI; do \
        echo "== ldd $exe =="; ldd "${AMBER_PREFIX}/bin/$exe"; \
        if ldd "${AMBER_PREFIX}/bin/$exe" | grep -q "not found"; then \
            echo "ERROR: unresolved shared libraries in $exe"; exit 1; \
        fi; \
    done

# --------------------------------------------------------------------------- #
# Stage 2: runtime                                                             #
# --------------------------------------------------------------------------- #
# The complete CUDA runtime image: cudart, cuBLAS, cuFFT, cuRAND, cuSOLVER,
# cuSPARSE, nvJitLink, NPP, nvJPEG, cuFile *and* NCCL. Larger than a hand-picked
# set, and chosen on purpose -- every library pmemd (or anything else run from
# this image) might dlopen is present, and libnccl.so.2 is here for the NCCL-
# linked pmemd.cuda.MPI.
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu${UBUNTU_VERSION} AS runtime

ARG OMPI_MODE=apt

LABEL org.opencontainers.image.title="easybfe-amber26" \
      org.opencontainers.image.description="EasyBFE with AMBER 26 pmemd.cuda(.MPI), CUDA+MPI+NCCL, sm_70/75/80/89 (V100/T4/A100/L4)" \
      org.opencontainers.image.licenses="MIT (EasyBFE); AMBER 26 is proprietary -- private registries only"

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
        libgfortran5 \
        libgomp1 \
        procps \
        tini \
    && rm -rf /var/lib/apt/lists/*

# OMPI_MODE=apt: the apt runtime libs matching what pmemd.cuda.MPI linked in
# stage 1. OMPI_MODE=source: the self-contained /opt/openmpi prefix built in
# stage 1 is copied whole instead -- installing apt's libopenmpi3 here would be
# the wrong ABI (4.1.x) for a pmemd built against OMPI_VERSION 5.x.
COPY --from=pmemd-builder /opt/openmpi /opt/openmpi
RUN set -eux; \
    if [ "${OMPI_MODE}" = "apt" ]; then \
        apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
            libopenmpi3 openmpi-bin \
        && rm -rf /var/lib/apt/lists/*; \
        rm -rf /opt/openmpi; \
    fi
ENV LD_LIBRARY_PATH=/opt/openmpi/lib:${LD_LIBRARY_PATH:-}

COPY --from=pmemd-builder /opt/amber/pmemd26 /opt/amber/pmemd26

# The runtime image is not the devel one pmemd was built in; make sure nothing
# it needs (libnccl included) was left behind.
RUN set -eux; \
    for exe in pmemd.cuda pmemd.cuda.MPI; do \
        echo "== ldd $exe =="; ldd "/opt/amber/pmemd26/bin/$exe"; \
        if ldd "/opt/amber/pmemd26/bin/$exe" | grep -q "not found"; then \
            echo "ERROR: $exe is missing libraries in the runtime stage"; exit 1; \
        fi; \
    done; \
    if [ "$(cat /opt/amber/pmemd26/.nccl)" = "linked" ] \
       && ! ldd /opt/amber/pmemd26/bin/pmemd.cuda.MPI | grep -q 'libnccl.*=> /'; then \
        echo "ERROR: pmemd.cuda.MPI links NCCL but libnccl is not resolvable here"; exit 1; \
    fi

# Conda environment with AmberTools (tleap/antechamber/ambpdb/...) and the
# EasyBFE python stack, from the repository's environment.yml *as written*.
# Nothing is removed and nothing is substituted -- the environment inside the
# image is the environment a developer gets locally, so anything that works on a
# workstation works here.
#
# The only change is additive: five packages the ABFE code path imports
# (easybfe/amber/workflow.py, easybfe/analysis/*) that environment.yml happens
# not to list, because a development checkout picks them up transitively.
ENV MAMBA_ROOT_PREFIX=/opt/conda
RUN curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest \
      | tar -xj -C /usr/local bin/micromamba

COPY environment.yml /tmp/environment.yml
# The additions go *before* the trailing `- pip:` block, so conda solves them
# alongside everything else rather than pip installing them afterwards.
RUN set -eux; \
    awk '/^  - pip:/ && !done { \
             print "  - mdanalysis"; print "  - joblib"; print "  - tqdm"; \
             print "  - pyyaml";     print "  - openmmforcefields"; done=1 \
         } \
         { print } \
         END { if (!done) { \
             print "  - mdanalysis"; print "  - joblib"; print "  - tqdm"; \
             print "  - pyyaml";     print "  - openmmforcefields" } }' \
        /tmp/environment.yml > /tmp/environment.container.yml; \
    cat /tmp/environment.container.yml; \
    micromamba create -y -n easybfe -f /tmp/environment.container.yml; \
    # Package cache only -- the environment itself is left exactly as solved.
    micromamba clean -aqy; \
    rm -rf /opt/conda/pkgs

# EasyBFE itself. Its dependencies all come from conda above, hence --no-deps.
# The build context has no .git, so setuptools_scm needs the version handed to
# it. Source is removed in the same layer as the install, so the copy of the
# tree does not linger in the image.
ARG EASYBFE_VERSION=0.0.0+container
COPY pyproject.toml README.md LICENSE /opt/easybfe/src/
COPY easybfe /opt/easybfe/src/easybfe
RUN SETUPTOOLS_SCM_PRETEND_VERSION="${EASYBFE_VERSION}" \
        /opt/conda/envs/easybfe/bin/pip install --no-deps --no-cache-dir /opt/easybfe/src \
    && rm -rf /opt/easybfe/src

# Import smoke test, in the build rather than after the push: a broken
# environment should fail here, not on a GPU node an hour later.
RUN set -eux; \
    /opt/conda/envs/easybfe/bin/python -c \
        "import numpy, scipy, scipy.sparse, openmm, rdkit, parmed, MDAnalysis, \
                alchemlyb, pymbar, openfe, easybfe; print('imports ok')"; \
    /opt/conda/envs/easybfe/bin/easybfe --version

# No test data is baked in, deliberately. The image is software only: pmemd,
# AmberTools, the conda environment, EasyBFE and these three scripts. Test
# systems and configs live in container/e2e-abfe-test/inputs and are mounted at run
# time, so swapping the test case -- or fixing a typo in a config -- never
# rebuilds or re-pushes the deliverable.
COPY container/build/scripts/entrypoint.sh container/build/scripts/selfcheck.sh \
     container/build/scripts/pmemd-pin-gpu.sh /opt/easybfe/bin/

# /opt/easybfe/bin comes first so that `mpirun` in EasyBFE's generated run.sh is
# the OpenMPI that pmemd.cuda.MPI was linked against, never a conda one.
# OMPI_MODE=apt: /usr/bin (apt package). OMPI_MODE=source: /opt/openmpi/bin
# (the prefix built and copied above) -- symlinking the apt path there would
# point at a binary that was never installed in this stage.
#
# The same PATH position is used to interpose on pmemd.cuda.MPI: the shim pins
# each rank to a single GPU before exec'ing the real binary. Measured +3% on the
# REMD stage and 5-17% less wall clock per stage, because it removes the
# stragglers rather than raising the ceiling. Attributed with the `pin-only`
# benchmark variant -- OpenMPI's CPU binding is left at its default, which was
# shown not to matter.
# OMPI_MODE=source gets a wrapper script instead of a plain symlink: PRRTE (the
# runtime OpenMPI 5 replaced ORTE with) has no MCA/env-var form of "count
# hardware threads as slots" -- `mpirun --help` on a "not enough slots" failure
# says so explicitly, pointing at the CLI-only `--use-hwthread-cpus` flag. Since
# EasyBFE's generated run.sh calls a bare `mpirun -np <n> ...` (workflow.py owns
# those flags, not this image), the container has to inject it here instead.
# OMPI 4's ORTE takes the env var (OMPI_MCA_hwloc_base_use_hwthreads_as_cpus,
# below), so OMPI_MODE=apt keeps the plain symlink.
RUN set -eux; \
    if [ "${OMPI_MODE}" = "apt" ]; then \
        ln -s /usr/bin/mpirun /opt/easybfe/bin/mpirun; \
        ln -s /usr/bin/mpiexec /opt/easybfe/bin/mpiexec; \
    else \
        printf '#!/usr/bin/env bash\nexec /opt/openmpi/bin/mpirun --use-hwthread-cpus "$@"\n' \
            > /opt/easybfe/bin/mpirun; \
        printf '#!/usr/bin/env bash\nexec /opt/openmpi/bin/mpiexec --use-hwthread-cpus "$@"\n' \
            > /opt/easybfe/bin/mpiexec; \
        chmod +x /opt/easybfe/bin/mpirun /opt/easybfe/bin/mpiexec; \
    fi; \
    mv /opt/easybfe/bin/pmemd-pin-gpu.sh /opt/easybfe/bin/pmemd.cuda.MPI; \
    chmod +x /opt/easybfe/bin/entrypoint.sh /opt/easybfe/bin/selfcheck.sh \
             /opt/easybfe/bin/pmemd.cuda.MPI

ENV PATH=/opt/easybfe/bin:/opt/amber/pmemd26/bin:/opt/openmpi/bin:/opt/conda/envs/easybfe/bin:$PATH \
    AMBERHOME=/opt/conda/envs/easybfe \
    PMEMDHOME=/opt/amber/pmemd26 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    MPLCONFIGDIR=/tmp/matplotlib

# OpenMPI refuses to launch as root, which containers usually are.
# Note: the CMA/shared-memory decision is NOT baked in here -- entrypoint.sh
# probes at start-up whether process_vm_readv is permitted and only disables
# OpenMPI's single-copy path when the seccomp profile actually blocks it.
#
# OpenMPI 5 replaced the ORTE runtime with PRRTE/PMIx, which reads its own
# PRTE_ALLOW_RUN_AS_ROOT{,_CONFIRM} rather than OMPI's -- confirmed by testing,
# unlike the hwthreads-as-cpus setting below, which PRRTE doesn't take as an
# env var at all (see the mpirun wrapper above). Both namespaces are set
# unconditionally -- OMPI 4 ignores the PRTE_* vars it doesn't know about, so
# this is one image working for either OMPI_MODE rather than two conditional
# ENV blocks.
ENV OMPI_ALLOW_RUN_AS_ROOT=1 \
    OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
    PRTE_ALLOW_RUN_AS_ROOT=1 \
    PRTE_ALLOW_RUN_AS_ROOT_CONFIRM=1

# EasyBFE launches one rank per lambda window (24 here). OpenMPI counts physical
# cores as "slots", but a cloud vCPU is a hyperthread -- so an n1-standard-32
# (32 vCPU / 16 cores) advertises only 16 slots and mpirun aborts with "not
# enough slots". Counting hardware threads matches what the VM actually sells
# you. Ranks are GPU-bound anyway, so sharing a core between two is not the
# bottleneck.
#
# OMPI_MODE=source (PRRTE) does NOT have an env-var/MCA form of this -- it's
# the `--use-hwthread-cpus` CLI flag the /opt/easybfe/bin/mpirun wrapper above
# injects. This env var is therefore live for OMPI_MODE=apt only; harmless and
# unused otherwise. selfcheck.sh's mpirun sanity test (`mpirun -np $(nproc)
# true`, no --oversubscribe) is what caught PRRTE ignoring this var in the
# first place -- on a build host, not $8/hr into a GPU node.
ENV OMPI_MCA_hwloc_base_use_hwthreads_as_cpus=1

WORKDIR /work
ENTRYPOINT ["/usr/bin/tini", "--", "/opt/easybfe/bin/entrypoint.sh"]
CMD ["bash"]
