#!/usr/bin/env bash
# Install Docker + the NVIDIA container runtime on a GCE Ubuntu VM.
#
# Google's Deep Learning VM images ship the NVIDIA driver but not Docker, so
# both the build node and the GPU node need this once:
#
#     gcloud compute ssh <vm> --zone <zone> --command "bash -s" < provision-gce-docker.sh
#
# Idempotent: safe to re-run.
set -euo pipefail

# Docker CE from Docker's own repository rather than Ubuntu's docker.io: the
# latter has no buildx plugin, and BuildKit (which build.sh uses, for the
# Dockerfile-scoped .dockerignore) refuses to run without it.
if ! docker buildx version > /dev/null 2>&1; then
    echo "==> Installing docker-ce + buildx"
    sudo DEBIAN_FRONTEND=noninteractive apt-get remove -y -qq docker.io docker-doc \
        docker-compose podman-docker containerd runc 2>/dev/null || true
    sudo install -m 0755 -d /etc/apt/keyrings
    curl -fsSL https://download.docker.com/linux/ubuntu/gpg \
        | sudo gpg --dearmor --yes -o /etc/apt/keyrings/docker.gpg
    echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
https://download.docker.com/linux/ubuntu $(. /etc/os-release && echo "$VERSION_CODENAME") stable" \
        | sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq \
        docker-ce docker-ce-cli containerd.io docker-buildx-plugin
else
    echo "==> docker already installed: $(docker --version)"
fi

if ! command -v nvidia-ctk > /dev/null 2>&1; then
    echo "==> Installing nvidia-container-toolkit"
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
        | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
        | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' \
        | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
    sudo DEBIAN_FRONTEND=noninteractive apt-get update -qq
    sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -qq nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
else
    echo "==> nvidia-container-toolkit already installed"
fi

# docker.socket first: replacing docker.io leaves it disabled, and dockerd then
# dies with "no sockets found via socket activation".
sudo systemctl unmask docker.socket docker.service 2> /dev/null || true
sudo systemctl enable --now docker.socket
sudo systemctl restart docker.service
sudo usermod -aG docker "$USER"

echo "==> Versions"
docker --version
sudo docker info --format '{{println "runtimes:"}}{{range $k, $v := .Runtimes}}  {{$k}}
{{end}}' 2>/dev/null || true

if ! command -v nvidia-smi > /dev/null 2>&1; then
    # CPU-only node (e.g. a build machine): nothing to verify, and the GPU test
    # below would fail for a reason that does not matter here.
    echo "==> No GPU on this host; skipping GPU passthrough check"
    exit 0
fi

echo "==> GPU visible from a container?"
if sudo docker run --rm --gpus all nvidia/cuda:12.6.3-base-ubuntu22.04 nvidia-smi -L; then
    echo "==> OK"
else
    echo "==> GPU passthrough FAILED -- check 'nvidia-smi' on the host and the docker runtime config" >&2
    exit 1
fi
