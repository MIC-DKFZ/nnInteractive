"""CLI entry point: launch a long-running nnInteractive inference server."""

from __future__ import annotations

import argparse
import logging
import os
import sys
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
        "--model-dir", required=True, help="Path to the trained model folder (contains fold_*/checkpoint_*.pth)"
    )
    p.add_argument("--fold", default=None, help="Fold to use (int, 'all', or omit to auto-detect)")
    p.add_argument("--checkpoint", default="checkpoint_final.pth", help="Checkpoint filename inside the fold folder")
    p.add_argument(
        "--host", default="127.0.0.1", help="Bind host (default 127.0.0.1; use 0.0.0.0 to listen on all interfaces)"
    )
    p.add_argument("--port", type=int, default=1527, help="Bind port (default 1527)")
    p.add_argument("--device", default="cuda", help="Torch device string (e.g. 'cuda', 'cuda:0', 'cpu')")
    p.add_argument("--torch-n-threads", type=int, default=8, help="Number of CPU threads for torch")
    p.add_argument("--no-autozoom", action="store_true", help="Disable adaptive zoom-out (default: enabled)")
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
    api_key = _resolve_api_key(args.api_key)
    if api_key is None:
        logger.warning(
            "Starting server WITHOUT authentication. Anyone who can reach %s:%s can drive inference. "
            "Pass --api-key or set NN_INTERACTIVE_API_KEY to enable bearer-token auth.",
            args.host,
            args.port,
        )

    session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=args.verbose,
        torch_n_threads=args.torch_n_threads,
        do_autozoom=not args.no_autozoom,
    )
    logger.info("Loading checkpoint from %s (fold=%s, checkpoint=%s)", args.model_dir, args.fold, args.checkpoint)
    session.initialize_from_trained_model_folder(
        model_training_output_dir=args.model_dir,
        use_fold=_resolve_fold(args.fold),
        checkpoint_name=args.checkpoint,
    )
    logger.info("Checkpoint loaded; serving on http://%s:%s", args.host, args.port)

    app = make_app(session, api_key=api_key)
    uvicorn.run(app, host=args.host, port=args.port, log_level=args.log_level)
    return 0


if __name__ == "__main__":
    sys.exit(main())
