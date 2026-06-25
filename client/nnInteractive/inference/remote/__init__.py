# Install the friendly "install the full package" finder as soon as the remote
# client is imported (the first thing any client user does). Done before the
# heavy imports below so it is in place regardless of what happens next. It is a
# no-op when the full nnInteractive package is installed.
from nnInteractive.inference.remote._full_required import install_finder as _install_finder

_install_finder()

from nnInteractive.inference.remote.remote_session import (
    ServerAtCapacityError,
    SessionExpiredError,
    nnInteractiveRemoteInferenceSession,
)

__all__ = [
    "nnInteractiveRemoteInferenceSession",
    "SessionExpiredError",
    "ServerAtCapacityError",
]
