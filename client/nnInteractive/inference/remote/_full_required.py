"""Friendly errors for full-package features when only the client is installed.

The lightweight ``nninteractive-client`` distribution ships only the torch-free
remote client (``nnInteractive.inference.remote``). Everything else — the local
in-process inference engine, the server, model management — lives in the full
``nnInteractive`` distribution. When code in a client-only environment imports
one of those modules, a bare ``ModuleNotFoundError: No module named
'nnInteractive.inference.inference_session'`` is unhelpful.

To fix that we install a *last-resort* meta-path finder (appended to the END of
``sys.meta_path``) that turns an import of a known full-only module into a clear
"install the full package" message. Because it is last, it is consulted only
after every real finder has failed — so when the full ``nnInteractive`` package
IS installed, those modules resolve normally and this finder never fires.

The finder is registered from ``nnInteractive.inference.remote.__init__`` (i.e.
as soon as the remote client is imported, which is the first thing any client
user does). Importing this guard module is cheap: it pulls in only ``sys`` and
``importlib.abc``, never numpy/httpx/blosc2/torch.
"""

from __future__ import annotations

import sys
from importlib.abc import MetaPathFinder

# Map of full-only import prefixes -> short, human-readable feature name used in
# the error message. A module matches if its fully-qualified name equals a
# prefix or starts with ``prefix + "."`` (so submodules are covered too).
_FULL_ONLY = {
    "nnInteractive.inference.inference_session": "local (in-process) inference",
    "nnInteractive.inference.server": "the inference server",
    "nnInteractive.inference.cvpr2025_challenge_baseline": "the CVPR2025 challenge baseline",
    "nnInteractive.model_management": "model discovery / download",
    "nnInteractive.interaction": "the full nnInteractive package",
    "nnInteractive.trainer": "the full nnInteractive package",
    "nnInteractive.utils": "the full nnInteractive package",
}


def _feature_for(fullname: str):
    for prefix, feature in _FULL_ONLY.items():
        if fullname == prefix or fullname.startswith(prefix + "."):
            return feature
    return None


class _FullPackageRequiredFinder(MetaPathFinder):
    """Last-resort finder that explains how to install the full package."""

    def find_spec(self, fullname, path=None, target=None):
        feature = _feature_for(fullname)
        if feature is None:
            # Not a known full-only module: stay out of the way and let the
            # normal ModuleNotFoundError propagate.
            return None
        raise ModuleNotFoundError(
            f"{fullname!r} requires the full nnInteractive package "
            f"({feature}), which is not installed.\n"
            f"This environment only has the lightweight 'nninteractive-client' "
            f"(remote client only). Install the full package with:\n\n"
            f"    pip install nnInteractive\n",
            name=fullname,
        )


def install_finder() -> None:
    """Append the finder to ``sys.meta_path`` exactly once (idempotent)."""
    for finder in sys.meta_path:
        if isinstance(finder, _FullPackageRequiredFinder):
            return
    sys.meta_path.append(_FullPackageRequiredFinder())
