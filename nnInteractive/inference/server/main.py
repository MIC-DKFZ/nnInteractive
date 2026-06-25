"""CLI entry point: launch a long-running nnInteractive inference server.

The network is built and weights loaded once at startup; each concurrent
client gets its own session whose ``self.network`` is a reference to that
single ``nn.Module`` instance (no per-session copy of the network or its
weights). See ``SERVER_CLIENT.md`` for the multi-session model.
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from typing import Optional

import torch
import uvicorn

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.inference.server.app import make_app

logger = logging.getLogger("nninteractive.server")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="nninteractive-server",
        description="Run an nnInteractive inference server. The model is loaded once at startup.",
    )
    p.add_argument(
        "--model",
        default=None,
        help="Official model id to serve (e.g. nnInteractive_v1.0). Resolved via the model "
        "manifest and downloaded on first use into $NNINTERACTIVE_MODEL_DIR (default "
        "~/.nninteractive). Run 'nninteractive-available-models' to list ids. Mutually "
        "exclusive with --model-dir; if neither is given, the manifest's default model is used.",
    )
    p.add_argument(
        "--model-dir",
        default=None,
        help="Path to a custom trained model folder (contains fold_*/checkpoint_*.pth). "
        "Mutually exclusive with --model; requires --fold.",
    )
    p.add_argument("--fold", default=None, help="Fold to use (int, 'all', or omit to auto-detect)")
    p.add_argument(
        "--checkpoint",
        default="checkpoint_final.pth",
        help="Checkpoint filename inside the fold folder",
    )
    p.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind host (default 127.0.0.1; use 0.0.0.0 to listen on all interfaces)",
    )
    p.add_argument("--port", type=int, default=1527, help="Bind port (default 1527)")
    p.add_argument(
        "--device",
        default="cuda",
        help="Torch device string (e.g. 'cuda', 'cuda:0', 'cpu')",
    )
    p.add_argument("--torch-n-threads", type=int, default=8, help="Number of CPU threads for torch")
    p.add_argument(
        "--no-torch-compile",
        action="store_true",
        help="Disable compiling the network with torch.compile (default: enabled). With compile "
        "enabled the network is compiled once at startup via a dummy warmup forward pass, which "
        "makes startup slower but every prediction faster; the one-time cost is amortized across "
        "the long-lived process. Pass this flag to skip compilation (e.g. for faster startup or "
        "to work around a compile/backend issue).",
    )
    p.add_argument(
        "--interactions-storage",
        choices=["blosc2", "tensor", "auto"],
        default="auto",
        help="Storage backend for the interaction tensor (default: auto). 'blosc2': compact "
        "in-memory array (low RAM, pays (de)compression per read/write). 'tensor': dense pinned "
        "CPU float16 torch.Tensor (more RAM, lower per-access overhead). 'auto': per image, use "
        "'tensor' for images up to 512x512x1024 voxels and 'blosc2' for larger ones.",
    )
    p.add_argument(
        "--no-autozoom",
        action="store_true",
        help="Disable adaptive zoom-out (default: enabled)",
    )
    p.add_argument(
        "--max-sessions",
        type=int,
        default=3,
        help="Maximum number of concurrent client sessions. Each session holds its own image, "
        "target buffer, and interaction state; the network module (and therefore its weights) "
        "is shared by reference across all sessions — exactly one copy on the GPU. Predictions "
        "remain GPU-serialized across sessions. Default: 3.",
    )
    p.add_argument(
        "--idle-timeout-seconds",
        type=float,
        default=600.0,
        help="Inactivity timeout (seconds) after which a session is reaped because the user "
        "stopped interacting. Refreshed only by real user actions (set_image, add_*_interaction, "
        "etc.) — NOT by heartbeats — so a connected-but-idle client is still reaped here. "
        "Default: 600 (10 min).",
    )
    p.add_argument(
        "--liveness-timeout-seconds",
        type=float,
        default=60.0,
        help="Liveness timeout (seconds): a session is reaped if the server sees no request at "
        "all from the client (not even a heartbeat) for this long. Detects crashed/disconnected "
        "clients and frees their slot quickly. The client heartbeats automatically at half this "
        "interval. Should be well below --idle-timeout-seconds. Default: 60.",
    )
    p.add_argument(
        "--api-key",
        default=None,
        help="Bearer token required on every request. If omitted, falls back to NN_INTERACTIVE_API_KEY. "
        "If neither is set, the server runs unauthenticated.",
    )
    p.add_argument("--verbose", action="store_true", help="Verbose session logging")
    p.add_argument("--log-level", default="info", help="uvicorn log level")
    return p


def _resolve_fold(raw: Optional[str]):
    if raw is None:
        return None
    if raw == "all":
        return "all"
    try:
        return int(raw)
    except ValueError:
        raise SystemExit(f"--fold must be an integer or 'all', got {raw!r}")


def _resolve_model_source(args) -> str:
    """Return the local model directory to load, downloading an official model if needed.

    * ``--model-dir``: a custom checkpoint folder; the user must also pass ``--fold``.
    * ``--model``: an official model id, resolved/downloaded via the manifest.
    * neither: the manifest's default model.
    ``--model`` and ``--model-dir`` are mutually exclusive.
    """
    if args.model and args.model_dir:
        raise SystemExit("Pass either --model (official id) or --model-dir (custom path), not both.")

    if args.model_dir:
        if args.fold is None:
            raise SystemExit("--fold is required when --model-dir (a custom checkpoint folder) is given.")
        return args.model_dir

    # Official model by id (or the manifest default). Imported lazily so the import
    # cost / network only happens on this path.
    from nnInteractive.model_management import (
        ensure_model_available,
        get_default_model_id,
        get_model_root_dir,
    )

    try:
        model_id = args.model or get_default_model_id()
        logger.info("Resolving official model '%s' (model root: %s)", model_id, get_model_root_dir())
        return str(ensure_model_available(model_id))
    except (RuntimeError, ValueError) as exc:
        raise SystemExit(str(exc))


def _resolve_api_key(cli_value: Optional[str]) -> Optional[str]:
    if cli_value:
        return cli_value
    env_value = os.environ.get("NN_INTERACTIVE_API_KEY")
    if env_value:
        return env_value
    return None


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    device = torch.device(args.device)
    use_torch_compile = not args.no_torch_compile
    if use_torch_compile and device.type != "cuda":
        # See nnInteractiveInferenceSession.__init__: torch.compile is not worth it on a
        # convolution-bound network running on CPU. Disable it here too so we skip the
        # misleading "compiling..." log and the pointless warmup below, and so client
        # sessions don't each emit the session-level warning.
        logger.warning(
            "torch.compile provides little benefit on '%s' (this network is convolution-bound) "
            "while adding significant compile-time overhead; disabling it. Pass --no-torch-compile "
            "to silence this.",
            device.type,
        )
        use_torch_compile = False
    api_key = _resolve_api_key(args.api_key)
    if api_key is None:
        logger.warning(
            "Starting server WITHOUT authentication. Anyone who can reach %s:%s can drive inference. "
            "Pass --api-key or set NN_INTERACTIVE_API_KEY to enable bearer-token auth.",
            args.host,
            args.port,
        )

    if args.max_sessions < 1:
        raise SystemExit("--max-sessions must be >= 1")
    if args.idle_timeout_seconds <= 0:
        raise SystemExit("--idle-timeout-seconds must be > 0")
    if args.liveness_timeout_seconds <= 0:
        raise SystemExit("--liveness-timeout-seconds must be > 0")
    if args.liveness_timeout_seconds >= args.idle_timeout_seconds:
        logger.warning(
            "--liveness-timeout-seconds (%.0fs) is not below --idle-timeout-seconds (%.0fs); "
            "the liveness check will then dominate and connected-but-idle clients will be reaped "
            "at the liveness timeout instead of the idle timeout.",
            args.liveness_timeout_seconds,
            args.idle_timeout_seconds,
        )

    # Resolve which model folder to serve (downloads an official model if needed).
    model_dir = _resolve_model_source(args)

    # Load the model once into a "loader" session; we keep only the artifacts dict.
    loader = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=use_torch_compile,
        verbose=args.verbose,
        torch_n_threads=args.torch_n_threads,
        do_autozoom=not args.no_autozoom,
    )
    logger.info(
        "Loading checkpoint from %s (fold=%s, checkpoint=%s)",
        model_dir,
        args.fold,
        args.checkpoint,
    )
    artifacts = loader._load_model_artifacts_from_disk(
        model_training_output_dir=model_dir,
        use_fold=_resolve_fold(args.fold),
        checkpoint_name=args.checkpoint,
    )
    # The loader instance holds no per-session state worth keeping. Discard it;
    # the artifacts dict carries everything sibling sessions need.
    loader.executor.shutdown(wait=False)
    del loader

    if use_torch_compile:
        # Compile the network once and run a single dummy forward pass so the
        # lazy torch.compile compilation happens here at startup rather than on
        # the first client's first prediction. We promote the resulting single
        # OptimizedModule back into the shared artifacts dict, so every client
        # session references that same compiled module (and the warmed compile
        # cache) instead of re-wrapping the raw module.
        logger.info("torch.compile enabled; compiling network and warming up (the first compile is slow)...")
        warmup_session = nnInteractiveInferenceSession(
            device=device,
            use_torch_compile=True,
            verbose=args.verbose,
            torch_n_threads=args.torch_n_threads,
            do_autozoom=not args.no_autozoom,
        )
        warmup_session.initialize_from_loaded_artifacts(artifacts)
        artifacts["network"] = warmup_session.network
        t0 = time.monotonic()
        warmup_session.warmup()
        logger.info(
            "warmup forward pass complete in %.1fs; clients' first prediction will be fast",
            time.monotonic() - t0,
        )
        warmup_session.executor.shutdown(wait=False)
        del warmup_session

    logger.info(
        "Checkpoint loaded; serving on http://%s:%s (max_sessions=%d, idle_timeout=%.0fs, " "liveness_timeout=%.0fs)",
        args.host,
        args.port,
        args.max_sessions,
        args.idle_timeout_seconds,
        args.liveness_timeout_seconds,
    )

    app = make_app(
        artifacts=artifacts,
        device=device,
        max_sessions=args.max_sessions,
        idle_timeout_seconds=args.idle_timeout_seconds,
        liveness_timeout_seconds=args.liveness_timeout_seconds,
        torch_n_threads=args.torch_n_threads,
        do_autozoom=not args.no_autozoom,
        use_torch_compile=use_torch_compile,
        interactions_storage=args.interactions_storage,
        verbose=args.verbose,
        api_key=api_key,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
