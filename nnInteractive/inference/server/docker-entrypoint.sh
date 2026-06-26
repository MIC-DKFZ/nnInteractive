#!/usr/bin/env bash
# Entrypoint for the nnInteractive server container.
#
# Translates a few environment variables into nninteractive-server CLI flags so
# the common knobs can be set with `docker run -e ...` without remembering the
# argument names, then execs the server. Any extra arguments passed to the
# container (`docker run IMAGE --max-sessions 4 ...`) are appended verbatim and
# override these defaults, so the full CLI remains available.
#
# The server offers two mutually exclusive ways to choose a model, and this
# entrypoint maps onto both:
#
#   * Manifest flow (NNINTERACTIVE_MODEL set): pass `--model <id>`. The server
#     resolves/downloads the official model via its manifest, using
#     NNINTERACTIVE_MODEL_DIR as the model ROOT. Baked images pre-download into
#     that root, so this resolves fully offline. --fold is optional here (the
#     server auto-detects the single fold_* folder).
#
#   * Custom-folder flow (only NNINTERACTIVE_MODEL_DIR set): pass
#     `--model-dir <folder>` for a checkpoint folder you mounted into the
#     container. The server REQUIRES an explicit --fold in this mode, so
#     NNINTERACTIVE_FOLD must be set (the official checkpoint ships fold_0).
#
# Environment variables (with their in-image defaults):
#   NNINTERACTIVE_MODEL       official model id → manifest flow (lite: unset; baked: the baked id)
#   NNINTERACTIVE_MODEL_DIR   manifest flow: model root; custom flow: the mounted checkpoint folder
#   NNINTERACTIVE_HOST        bind host (default 0.0.0.0 so the container is reachable)
#   NNINTERACTIVE_PORT        bind port (default 1527)
#   NNINTERACTIVE_FOLD        fold to load. Required in the custom-folder flow; optional
#                             (auto-detected) in the manifest flow. Set to 0 / 1 / all.
#
# Authentication: the server reads NN_INTERACTIVE_API_KEY directly, so just pass
# `-e NN_INTERACTIVE_API_KEY=...` to enable bearer-token auth. Without it the
# server runs unauthenticated and logs a warning.
set -euo pipefail

MODEL="${NNINTERACTIVE_MODEL:-}"
MODEL_DIR="${NNINTERACTIVE_MODEL_DIR:-}"
FOLD="${NNINTERACTIVE_FOLD:-}"
HOST="${NNINTERACTIVE_HOST:-0.0.0.0}"
PORT="${NNINTERACTIVE_PORT:-1527}"

args=(--host "$HOST" --port "$PORT")

if [ -n "$MODEL" ]; then
    # Manifest flow. NNINTERACTIVE_MODEL_DIR (the model root) is read by the
    # server from the environment directly, so we pass only --model here; passing
    # --model-dir as well would be rejected as mutually exclusive.
    args+=(--model "$MODEL")
    # --fold is optional in this flow; forward it only if the user pinned one.
    if [ -n "$FOLD" ]; then
        args+=(--fold "$FOLD")
    fi
elif [ -n "$MODEL_DIR" ]; then
    # Custom-folder flow: a checkpoint folder mounted into the container.
    if [ ! -d "$MODEL_DIR" ]; then
        echo "nninteractive-entrypoint: model directory '$MODEL_DIR' not found." >&2
        echo "  - Lite image: mount your checkpoint folder there, e.g. -v /path/to/model:/model" >&2
        echo "  - Or set NNINTERACTIVE_MODEL=<official-id> to fetch a model from the manifest instead." >&2
        exit 1
    fi
    # The server requires an explicit fold for a custom --model-dir.
    if [ -z "$FOLD" ]; then
        echo "nninteractive-entrypoint: NNINTERACTIVE_FOLD is required when serving a mounted" >&2
        echo "  checkpoint folder (NNINTERACTIVE_MODEL_DIR='$MODEL_DIR'). Set it, e.g." >&2
        echo "  -e NNINTERACTIVE_FOLD=0 (the official checkpoint ships fold_0)," >&2
        echo "  or use -e NNINTERACTIVE_MODEL=<official-id> to use the manifest flow instead." >&2
        exit 1
    fi
    args+=(--model-dir "$MODEL_DIR" --fold "$FOLD")
else
    # Neither set: let the server load the manifest's default model (fetched on
    # first run). Forward a fold only if the user pinned one.
    if [ -n "$FOLD" ]; then
        args+=(--fold "$FOLD")
    fi
fi

exec nninteractive-server "${args[@]}" "$@"
