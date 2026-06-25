"""Model discovery and download for nnInteractive.

This is the single source of truth for *which* official models exist and *where*
they live on disk. GUIs (napari, 3D Slicer, ...) and the inference server all go
through here instead of talking to Hugging Face directly.

Design notes
------------
* The authoritative list of selectable models is a ``models.json`` manifest at the
  root of the Hugging Face repo (``MIC-DKFZ/nnInteractive``). Arbitrary repo folders
  are NOT offered as models; only manifest entries are.
* Manifest loading is *remote-first* with a local cache fallback: on startup we try
  to refresh ``models.json`` from Hugging Face and cache it under the model root; if
  that fails (offline / HF unreachable) we fall back to the cached copy. If neither is
  available we raise a clear error telling the user to go online or supply a custom
  model path.
* Models are stored under ``<model_root>/models/<model_id>/`` and downloaded with
  ``snapshot_download(local_dir=...)`` so each checkpoint exists exactly once on disk
  (no extra copy into a separate cache).

This module is intentionally torch-free and only imports ``huggingface_hub`` lazily,
inside the functions that actually need network access, so it can be imported by
lightweight / remote-only clients without pulling heavy dependencies.
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path
from typing import List, Optional

# Canonical Hugging Face location of the official checkpoints. The repo id can be
# overridden via NNINTERACTIVE_REPO_ID (handy for testing against a staging repo).
DEFAULT_REPO_ID = "MIC-DKFZ/nnInteractive"
REPO_TYPE = "model"
MANIFEST_FILENAME = "models.json"
# Model ids and manifest paths must use this prefix (see validation below).
MODEL_PREFIX = "nnInteractive_"


def get_repo_id() -> str:
    """Hugging Face repo id, overridable via ``NNINTERACTIVE_REPO_ID``."""
    return os.environ.get("NNINTERACTIVE_REPO_ID", DEFAULT_REPO_ID)


def get_model_root_dir() -> Path:
    """Root directory for the manifest and downloaded models.

    ``NNINTERACTIVE_MODEL_DIR`` overrides the default ``~/.nninteractive``.
    """
    env_dir = os.environ.get("NNINTERACTIVE_MODEL_DIR")
    if env_dir:
        return Path(env_dir).expanduser().resolve()
    return Path.home() / ".nninteractive"


def _models_dir(model_root: Path) -> Path:
    return model_root / "models"


def _downloads_dir(model_root: Path) -> Path:
    return model_root / "downloads"


# --------------------------------------------------------------------------- #
# Manifest handling
# --------------------------------------------------------------------------- #
def _validate_and_normalize_manifest(manifest: dict) -> dict:
    """Validate a raw manifest and return a normalized copy.

    Rules (see project discussion):
    * ``schema_version`` must exist and ``models`` must be a list.
    * Every entry must have string ``id``, ``display_name`` and ``path``.
    * ``id`` and ``path`` must start with ``nnInteractive_``.
    * Malformed entries are dropped, not fatal.
    * At most one entry may be ``default: true``; if none is, the first becomes default.
    """
    if not isinstance(manifest, dict):
        raise ValueError("models.json must be a JSON object")
    if "schema_version" not in manifest:
        raise ValueError("models.json is missing 'schema_version'")
    raw_models = manifest.get("models")
    if not isinstance(raw_models, list):
        raise ValueError("models.json 'models' must be a list")

    valid: List[dict] = []
    seen_ids = set()
    default_seen = False
    for entry in raw_models:
        if not isinstance(entry, dict):
            continue
        model_id = entry.get("id")
        display_name = entry.get("display_name")
        path = entry.get("path")
        if not (isinstance(model_id, str) and isinstance(display_name, str) and isinstance(path, str)):
            continue
        if not model_id.startswith(MODEL_PREFIX) or not path.startswith(MODEL_PREFIX):
            continue
        if model_id in seen_ids:
            continue
        seen_ids.add(model_id)

        norm = dict(entry)
        is_default = bool(entry.get("default", False))
        if is_default and default_seen:
            # Enforce "at most one default": keep the first, demote the rest.
            is_default = False
        norm["default"] = is_default
        default_seen = default_seen or is_default
        valid.append(norm)

    if not valid:
        raise ValueError("models.json contains no valid model entries")
    if not default_seen:
        valid[0]["default"] = True

    return {"schema_version": manifest["schema_version"], "models": valid}


def _load_remote_manifest(model_root: Path) -> dict:
    from huggingface_hub import hf_hub_download

    manifest_path = hf_hub_download(
        repo_id=get_repo_id(),
        repo_type=REPO_TYPE,
        filename=MANIFEST_FILENAME,
        cache_dir=str(_downloads_dir(model_root)),
    )
    with open(manifest_path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save_cached_manifest(manifest: dict, model_root: Path) -> None:
    model_root.mkdir(parents=True, exist_ok=True)
    with open(model_root / MANIFEST_FILENAME, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


def _load_cached_manifest(model_root: Path) -> dict:
    with open(model_root / MANIFEST_FILENAME, "r", encoding="utf-8") as f:
        return json.load(f)


def load_model_manifest(model_root: Optional[Path] = None) -> dict:
    """Load the (validated, normalized) model manifest.

    Remote-first: try to refresh from Hugging Face and cache the result locally.
    On failure, fall back to the cached manifest. If neither is available, raise a
    clear error pointing the user at internet access or a custom model path.
    """
    model_root = model_root or get_model_root_dir()
    try:
        manifest = _validate_and_normalize_manifest(_load_remote_manifest(model_root))
        _save_cached_manifest(manifest, model_root)
        return manifest
    except Exception as remote_exc:
        cached_path = model_root / MANIFEST_FILENAME
        if cached_path.exists():
            try:
                return _validate_and_normalize_manifest(_load_cached_manifest(model_root))
            except Exception:
                pass  # corrupt cache: fall through to the hard error
        raise RuntimeError(
            "Could not load the nnInteractive model manifest from Hugging Face "
            f"({get_repo_id()}), and no usable cached manifest was found at {cached_path}. "
            "Connect to the internet to download a model, or provide a custom model "
            "path/checkpoint folder instead."
        ) from remote_exc


def get_default_model_id(model_root: Optional[Path] = None, manifest: Optional[dict] = None) -> str:
    """Return the id of the manifest's default model."""
    manifest = manifest or load_model_manifest(model_root)
    for entry in manifest["models"]:
        if entry.get("default"):
            return entry["id"]
    return manifest["models"][0]["id"]


def _find_entry(manifest: dict, model_id: str) -> Optional[dict]:
    for entry in manifest["models"]:
        if entry["id"] == model_id:
            return entry
    return None


# --------------------------------------------------------------------------- #
# Local model presence / download
# --------------------------------------------------------------------------- #
def _download_unavailable_msg(model_id: str) -> str:
    return (
        f"Could not download model '{model_id}' from Hugging Face ({get_repo_id()}). "
        "It is not available locally and could not be downloaded (no internet connection?). "
        "Connect to the internet, select an already-downloaded model, or provide a "
        "custom model path."
    )


def is_model_downloaded(model_id: str, model_root: Optional[Path] = None) -> bool:
    """True if ``<model_root>/models/<model_id>/`` exists and is non-empty."""
    model_root = model_root or get_model_root_dir()
    local_model_dir = _models_dir(model_root) / model_id
    return local_model_dir.is_dir() and any(local_model_dir.iterdir())


def get_local_model_dir(model_id: str, model_root: Optional[Path] = None) -> Path:
    """Path where ``model_id`` lives (or would live) locally."""
    model_root = model_root or get_model_root_dir()
    return _models_dir(model_root) / model_id


def ensure_model_available(model_id: str, model_root: Optional[Path] = None) -> Path:
    """Make ``model_id`` available locally and return its directory.

    Reuses an already-downloaded model without re-downloading. If the model is
    missing and the download fails (e.g. offline), raises a clear error.
    """
    model_root = model_root or get_model_root_dir()
    local_model_dir = _models_dir(model_root) / model_id

    # Reuse if already present. Note: no manifest/network access needed in this path,
    # so already-downloaded models work fully offline.
    if is_model_downloaded(model_id, model_root):
        return local_model_dir

    manifest = load_model_manifest(model_root)
    entry = _find_entry(manifest, model_id)
    if entry is None:
        available = [e["id"] for e in manifest["models"]]
        raise ValueError(f"Unknown model id '{model_id}'. Available models: {available}")
    remote_path = entry["path"].strip("/")

    from huggingface_hub import snapshot_download

    models_dir = _models_dir(model_root)
    models_dir.mkdir(parents=True, exist_ok=True)
    try:
        # Download only the selected model folder, straight into the model root (no
        # second copy). `*` in allow_patterns matches across path separators, so this
        # picks up nested files too; `/**` is added for robustness.
        snapshot_download(
            repo_id=get_repo_id(),
            repo_type=REPO_TYPE,
            allow_patterns=[f"{remote_path}/*", f"{remote_path}/**"],
            local_dir=str(models_dir),
        )
    except Exception as exc:
        raise RuntimeError(_download_unavailable_msg(model_id)) from exc

    # snapshot_download mirrors the repo layout, so files land under <models_dir>/<path>.
    # Canonicalize to <models_dir>/<model_id> (a no-op rename when id == path).
    downloaded_dir = models_dir / remote_path
    if downloaded_dir != local_model_dir and downloaded_dir.is_dir() and any(downloaded_dir.iterdir()):
        if local_model_dir.exists():
            shutil.rmtree(local_model_dir)
        local_model_dir.parent.mkdir(parents=True, exist_ok=True)
        downloaded_dir.rename(local_model_dir)

    # NOTE: snapshot_download(local_dir=...) does NOT raise when the repo is
    # unreachable (offline / 404) — it logs a warning and returns. So a missing/empty
    # result here is our real "could not download" signal.
    if not is_model_downloaded(model_id, model_root):
        raise RuntimeError(_download_unavailable_msg(model_id))
    return local_model_dir


def list_models(model_root: Optional[Path] = None) -> List[dict]:
    """Return manifest entries enriched with local availability.

    Each dict carries the manifest fields plus ``downloaded`` (bool) and
    ``local_path`` (str or None). Frontends populate their dropdown from this and,
    when offline, may restrict immediate selection to entries where ``downloaded``.
    """
    model_root = model_root or get_model_root_dir()
    manifest = load_model_manifest(model_root)
    models: List[dict] = []
    for entry in manifest["models"]:
        downloaded = is_model_downloaded(entry["id"], model_root)
        models.append(
            {
                **entry,
                "downloaded": downloaded,
                "local_path": str(get_local_model_dir(entry["id"], model_root)) if downloaded else None,
            }
        )
    return models


# --------------------------------------------------------------------------- #
# CLI entry points (registered in pyproject.toml [project.scripts])
# --------------------------------------------------------------------------- #
def _resolve_root_arg(model_dir: Optional[str]) -> Path:
    return Path(model_dir).expanduser().resolve() if model_dir else get_model_root_dir()


def available_models_cli(argv=None) -> int:
    """`nninteractive-available-models`: print manifest models + download status."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="nninteractive-available-models",
        description="List nnInteractive models from the manifest and show which are downloaded.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model root directory (default: $NNINTERACTIVE_MODEL_DIR or ~/.nninteractive)",
    )
    args = parser.parse_args(argv)
    model_root = _resolve_root_arg(args.model_dir)

    try:
        models = list_models(model_root)
    except RuntimeError as exc:
        print(str(exc))
        return 1

    print(f"Repository: {get_repo_id()}")
    print(f"Model root: {model_root}")
    print()
    for model in models:
        flags = []
        if model.get("default"):
            flags.append("default")
        flags.append("downloaded" if model["downloaded"] else "not downloaded")
        print(f"  {model['id']:<26} {model.get('display_name', ''):<26} [{', '.join(flags)}]")
    return 0


def download_model_cli(argv=None) -> int:
    """`nninteractive-download-model`: download a model by id (default if omitted)."""
    import argparse

    parser = argparse.ArgumentParser(
        prog="nninteractive-download-model",
        description="Download an nnInteractive model by id into the local model directory.",
    )
    parser.add_argument(
        "model_id",
        nargs="?",
        default=None,
        help="Model id (e.g. nnInteractive_v1.0). Omit to download the default model.",
    )
    parser.add_argument(
        "--model-dir",
        default=None,
        help="Model root directory (default: $NNINTERACTIVE_MODEL_DIR or ~/.nninteractive)",
    )
    args = parser.parse_args(argv)
    model_root = _resolve_root_arg(args.model_dir)

    try:
        model_id = args.model_id or get_default_model_id(model_root)
        print(f"Ensuring model '{model_id}' is available in {model_root} ...")
        local_dir = ensure_model_available(model_id, model_root)
    except (RuntimeError, ValueError) as exc:
        print(str(exc))
        return 1

    print(f"Model '{model_id}' is ready at: {local_dir}")
    return 0
