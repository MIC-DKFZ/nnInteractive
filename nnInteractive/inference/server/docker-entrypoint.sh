#!/usr/bin/env bash
# Entrypoint for the nnInteractive server container.
#
# Translates a few environment variables into nninteractive-server CLI flags so
# the common knobs can be set with `docker run -e ...` without remembering the
# argument names, then execs the server. Any extra arguments passed to the
# container (`docker run IMAGE --max-sessions 4 ...`) are appended verbatim and
# override these defaults, so the full CLI remains available.
#
# Environment variables (with their in-image defaults):
#   NNINTERACTIVE_MODEL_DIR   path to the model folder
#                             (lite image: /model mount; baked image: in-image weights)
#   NNINTERACTIVE_HOST        bind host (default 0.0.0.0 so the container is reachable)
#   NNINTERACTIVE_PORT        bind port (default 1527)
#   NNINTERACTIVE_FOLD        fold to load. Empty (the default) lets the server
#                             auto-detect when exactly one fold_* folder is
#                             present (e.g. the official model's fold_0). Set
#                             to 0 / 1 / all to force a specific fold_* folder.
#
# Authentication: the server reads NN_INTERACTIVE_API_KEY directly, so just pass
# `-e NN_INTERACTIVE_API_KEY=...` to enable bearer-token auth. Without it the
# server runs unauthenticated and logs a warning.
set -euo pipefail

MODEL_DIR="${NNINTERACTIVE_MODEL_DIR:-/model}"
HOST="${NNINTERACTIVE_HOST:-0.0.0.0}"
PORT="${NNINTERACTIVE_PORT:-1527}"
FOLD="${NNINTERACTIVE_FOLD:-}"

if [ ! -d "$MODEL_DIR" ]; then
    echo "nninteractive-entrypoint: model directory '$MODEL_DIR' not found." >&2
    echo "  - Lite image: mount your checkpoint folder there, e.g. -v /path/to/model:/model" >&2
    echo "  - Or set NNINTERACTIVE_MODEL_DIR to the correct path." >&2
    exit 1
fi

args=(--model-dir "$MODEL_DIR" --host "$HOST" --port "$PORT")
# An empty NNINTERACTIVE_FOLD means "omit --fold" and let the server auto-detect.
if [ -n "$FOLD" ]; then
    args+=(--fold "$FOLD")
fi

exec nninteractive-server "${args[@]}" "$@"
