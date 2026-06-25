# nnInteractive inference server image.
#
# Two build targets share one base:
#
#   * `runtime` (a.k.a. the "lite" image): the package is installed from the
#     source in the build context, but NO model weights are baked in. Mount a
#     checkpoint folder at /model (or point NNINTERACTIVE_MODEL_DIR elsewhere)
#     at `docker run` time. Use this when you want to serve an arbitrary / newer
#     model without rebuilding the image — drop the weights into the mount.
#
#   * `baked`: builds on top of `runtime` and downloads a specific model from
#     the HuggingFace repo `nnInteractive/nnInteractive` at build time, so the
#     weights live inside the image. No mount needed — `docker pull` and run.
#     Ideal for quick single-model deployments. The baked model is selected by
#     the MODEL_NAME build-arg (see below); code version and model version are
#     therefore decoupled.
#
# Build examples:
#   docker build -f nnInteractive/inference/server/Dockerfile --target runtime -t nninteractive-server:lite .
#   docker build -f nnInteractive/inference/server/Dockerfile --target baked \
#       --build-arg MODEL_NAME=nnInteractive_v1.0 -t nninteractive-server:latest .
#
# The build context must be the repository root so the package source is copied
# in (installing from source pins the image to exactly the checked-out commit).
#
# NOTE ON THE MODEL LICENSE: the code in this repository is Apache-2.0, but the
# official nnInteractive checkpoint is CC BY-NC-SA 4.0 (non-commercial). Baking
# the weights into an image redistributes them, so only publish baked images in
# a manner compatible with that license.

# ---------------------------------------------------------------------------
# Base / runtime stage  (this IS the "lite" image, target: runtime)
# ---------------------------------------------------------------------------
# Official PyTorch CUDA runtime image: torch is already installed, so
# `pip install .` keeps it and only adds nnInteractive's other dependencies.
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS runtime

# torch.compile (inductor) is ON by default in the server and shells out to a
# C++ compiler when it warms up at startup, so g++ must be present at runtime,
# not just at build time. git is cheap insurance for any VCS-based pip deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

# Keep Python output unbuffered so server logs stream out of the container.
ENV PYTHONUNBUFFERED=1

WORKDIR /opt/nninteractive/src

# Copy the entrypoint first so it caches independently of the source tree.
COPY nnInteractive/inference/server/docker-entrypoint.sh /usr/local/bin/nninteractive-entrypoint
RUN chmod +x /usr/local/bin/nninteractive-entrypoint

# Install the package from source (image == this commit). A .dockerignore at the
# repo root keeps the context small (no .git, no supervoxel/sam2 tree, etc.).
COPY . .
RUN pip install --no-cache-dir .

# Server runtime defaults. The server itself binds 127.0.0.1:1527 by default;
# inside a container we must listen on all interfaces to be reachable.
# NNINTERACTIVE_MODEL_DIR defaults to the /model mount point for this lite
# image; the baked stage overrides it to the in-image weights path.
# NNINTERACTIVE_FOLD is left unset so the entrypoint omits --fold and the server
# auto-detects the single fold_* folder (the official model ships fold_0). Set
# it (e.g. to 0 or all) only for a model with multiple fold_* folders.
ENV NNINTERACTIVE_MODEL_DIR=/model \
    NNINTERACTIVE_HOST=0.0.0.0 \
    NNINTERACTIVE_PORT=1527

EXPOSE 1527

# Set an API key via -e NN_INTERACTIVE_API_KEY=... ; the server reads it from
# the environment on its own. Extra CLI flags passed to `docker run` are
# forwarded to nninteractive-server by the entrypoint.
ENTRYPOINT ["nninteractive-entrypoint"]

# ---------------------------------------------------------------------------
# Baked stage  (weights downloaded into the image, target: baked)
# ---------------------------------------------------------------------------
FROM runtime AS baked

# Which model to bake. Override per build (and in the publish workflow) to
# decouple the baked model version from the code version.
ARG MODEL_NAME=nnInteractive_v1.0
ENV NNINTERACTIVE_MODEL_NAME=${MODEL_NAME}

# huggingface_hub is not a declared dependency of the package, so install it
# just for this build-time download (left in place — it is tiny, and nnunetv2
# may depend on it anyway). snapshot_download lands the files at
# /opt/nninteractive/models/<MODEL_NAME>/... for the public repo (no token).
RUN pip install --no-cache-dir "huggingface_hub>=0.23" \
    && python -c "import os; from huggingface_hub import snapshot_download; m=os.environ['NNINTERACTIVE_MODEL_NAME']; snapshot_download(repo_id='nnInteractive/nnInteractive', allow_patterns=[f'{m}/*'], local_dir='/opt/nninteractive/models')"

# Point the server at the baked weights instead of the /model mount.
ENV NNINTERACTIVE_MODEL_DIR=/opt/nninteractive/models/${MODEL_NAME}
