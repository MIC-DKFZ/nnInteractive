# nnInteractive inference server image. Two targets:
#   runtime ("lite") — no weights baked in; mount a checkpoint or fetch by id at run time.
#   baked            — a model downloaded into the image at build time (MODEL_NAME).
#
# Build context must be the repo root so the package is installed from source.
#
# The official checkpoint is CC BY-NC-SA 4.0 (non-commercial); the code is
# Apache-2.0. Baking weights redistributes them — publish baked images accordingly.
#
#   docker build -f nnInteractive/inference/server/Dockerfile --target runtime -t nninteractive-server:lite .
#   docker build -f nnInteractive/inference/server/Dockerfile --target baked \
#       --build-arg MODEL_NAME=nnInteractive_v1.0 -t nninteractive-server:latest .

# --- runtime / lite -------------------------------------------------------
FROM pytorch/pytorch:2.6.0-cuda12.4-cudnn9-runtime AS runtime

# g++ for torch.compile's startup warmup; git for any VCS-based pip deps.
RUN apt-get update \
    && apt-get install -y --no-install-recommends build-essential git \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1
WORKDIR /opt/nninteractive/src

COPY nnInteractive/inference/server/docker-entrypoint.sh /usr/local/bin/nninteractive-entrypoint
RUN chmod +x /usr/local/bin/nninteractive-entrypoint

# The full package depends on nninteractive-client (under client/) via an exact
# pin. Install the bundled client first so the pin resolves from source, not PyPI.
COPY . .
RUN pip install --no-cache-dir ./client \
    && pip install --no-cache-dir .

# Defaults: bind all interfaces (reachable from outside the container). For lite,
# NNINTERACTIVE_MODEL_DIR is the /model mount; a mounted checkpoint needs
# NNINTERACTIVE_FOLD, or set NNINTERACTIVE_MODEL=<id> to fetch by id instead.
ENV NNINTERACTIVE_MODEL_DIR=/model \
    NNINTERACTIVE_HOST=0.0.0.0 \
    NNINTERACTIVE_PORT=1527

EXPOSE 1527

# Pass NN_INTERACTIVE_API_KEY via -e to enable auth. Extra args after the image
# name are forwarded to nninteractive-server.
ENTRYPOINT ["nninteractive-entrypoint"]

# --- baked ----------------------------------------------------------------
FROM runtime AS baked

ARG MODEL_NAME=nnInteractive_v1.0

# Download into the image via the package's downloader -> /opt/nninteractive/models/<MODEL_NAME>.
RUN nninteractive-download-model "${MODEL_NAME}" --model-dir /opt/nninteractive

# Serve it via the manifest flow (model root + id); loads offline, fold auto-detected.
ENV NNINTERACTIVE_MODEL_DIR=/opt/nninteractive \
    NNINTERACTIVE_MODEL=${MODEL_NAME}
