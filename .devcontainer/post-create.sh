#!/usr/bin/env bash
# One-time setup after the dev container is created.
# Creates the `pxt` conda env and syncs project deps directly with uv,
# bypassing `make install` (which pulls in doc-generation tools like
# quarto that aren't available on linux-aarch64).

set -euo pipefail

cd /workspaces/pixeltable

source /opt/miniforge/etc/profile.d/conda.sh

if ! conda env list | awk '{print $1}' | grep -qx pxt; then
    conda create -y -n pxt -c conda-forge python=3.10 'cmake>=3.22'
fi

conda activate pxt

# Runtime media deps (subset of what `make install` would install).
mamba install -q -y -c conda-forge libiconv 'ffmpeg==6.1.1=gpl*'

# Install uv at the version pinned in the Makefile.
python -m pip install -qU pip
python -m pip install -q uv==0.9.3

# Install project deps directly.
VIRTUAL_ENV="$CONDA_PREFIX" uv sync --active --group extra-dev

# Jupyter kernel.
python -m ipykernel install --user --name=pixeltable
