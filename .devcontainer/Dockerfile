FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8 \
    PATH=/opt/miniforge/bin:$PATH

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
        bzip2 \
        ca-certificates \
        curl \
        git \
        build-essential \
        sudo \
 && rm -rf /var/lib/apt/lists/*

# `ubuntu` (uid 1000) is preinstalled in the base image; grant it passwordless
# sudo for ad-hoc apt installs in the running container.
RUN echo 'ubuntu ALL=(ALL) NOPASSWD:ALL' > /etc/sudoers.d/ubuntu \
 && chmod 0440 /etc/sudoers.d/ubuntu

# Miniforge only. Conda env, project deps, and tools are set up at
# post-create time so they live alongside the bind-mounted source.
RUN arch="$(uname -m)" \
 && curl -fsSL "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-${arch}.sh" -o /tmp/miniforge.sh \
 && bash /tmp/miniforge.sh -b -p /opt/miniforge \
 && rm /tmp/miniforge.sh \
 && chown -R ubuntu:ubuntu /opt/miniforge

RUN echo 'source /opt/miniforge/etc/profile.d/conda.sh && conda activate pxt 2>/dev/null || true' >> /etc/bash.bashrc

USER ubuntu
WORKDIR /workspaces/pixeltable
