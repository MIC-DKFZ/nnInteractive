"""Client-side stand-in for nnInteractiveInferenceSession backed by an HTTP server.

Public API matches the local session (see nnInteractive/inference/inference_session.py
and API_CHANGES_v2.md). All model state lives on the server; this object holds
only the user's target_buffer (mirrored from server responses) and the
capability metadata fetched at construction time.
"""

from __future__ import annotations

import json
import os
import threading
import warnings
from typing import List, Optional, Tuple, Union

import blosc2
import httpx
import numpy as np

try:
    import torch
except ImportError:
    # torch is an optional dependency for the lightweight client role
    # (`pip install nnInteractive[client]`). It is only needed when the caller
    # uses a torch.Tensor target buffer; numpy buffers work without it.
    torch = None

from nnInteractive.inference.remote._protocol import (
    CONTENT_TYPE_OCTET_STREAM,
    LEASE_HEADER,
    META_HEADER,
    PATH_ADD_BBOX,
    PATH_ADD_INITIAL_SEG,
    PATH_ADD_LASSO,
    PATH_ADD_POINT,
    PATH_ADD_SCRIBBLE,
    PATH_CAPABILITIES,
    PATH_CLAIM,
    PATH_HEALTHZ,
    PATH_HEARTBEAT,
    PATH_LEASE_STATUS,
    PATH_RELEASE,
    PATH_RESET_INTERACTIONS,
    PATH_SET_DO_AUTOZOOM,
    PATH_SET_IMAGE,
    PATH_SET_TARGET_BUFFER,
    PATH_UNDO,
)
from nnInteractive.inference.remote.serialization import pack_array, unpack_array


def _compression_threads() -> int:
    """blosc2 thread count for client-side upload compression.

    Full logical CPU count: blosc2 scales measurably onto SMT siblings, so use them all to
    minimize upload latency. Per-call only (passed to pack_array → compress2), so it never
    mutates blosc2's global nthreads.
    """
    return max(1, os.cpu_count() or 1)


class SessionExpiredError(RuntimeError):
    """Raised when the server reports the client's lease no longer exists.

    A GUI should catch this and either reconnect (construct a new
    ``nnInteractiveRemoteInferenceSession``) or surface a "session expired"
    dialog. The most common causes are exceeding the server's idle timeout
    (default 10 min) and a server restart.
    """


class ServerAtCapacityError(RuntimeError):
    """Raised when the server has already issued ``--max-sessions`` leases.

    Wait and retry, or ask the operator to bump the cap.
    """


def _raise_for_lease_errors(resp: httpx.Response) -> None:
    """Translate server-side lease errors into typed exceptions before httpx raises."""
    if resp.status_code == 410:
        raise SessionExpiredError(_extract_detail(resp) or "lease expired or unknown")
    if resp.status_code == 503:
        raise ServerAtCapacityError(_extract_detail(resp) or "server is at capacity")


def _extract_detail(resp: httpx.Response) -> Optional[str]:
    try:
        data = resp.json()
    except (json.JSONDecodeError, ValueError):
        return None
    if isinstance(data, dict):
        detail = data.get("detail")
        if isinstance(detail, str):
            return detail
    return None


def _buffer_dtype_str(target_buffer: Union[np.ndarray, "torch.Tensor"]) -> str:
    if torch is not None and isinstance(target_buffer, torch.Tensor):
        return str(target_buffer.dtype).replace("torch.", "")
    return str(np.dtype(target_buffer.dtype))


def _to_jsonable(obj):
    """Recursively coerce numpy arrays/scalars into JSON-serializable builtins.

    The local session accepts numpy values for things like ``image_properties['spacing']``;
    the remote session JSON-encodes that metadata, so we have to match the contract.
    """
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, dict):
        return {k: _to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_jsonable(v) for v in obj]
    return obj


class nnInteractiveRemoteInferenceSession:
    """Drop-in replacement for nnInteractiveInferenceSession that talks to a server.

    Parameters
    ----------
    server_url:
        Base URL of the running nninteractive-server, e.g. ``http://host:8000``.
    api_key:
        Optional bearer token. If omitted, falls back to the
        ``NN_INTERACTIVE_API_KEY`` environment variable. Pass ``None`` (and unset
        the env var) when the server was started without ``--api-key``.
    connect_timeout:
        Seconds to wait for the TCP / TLS handshake. Kept short so "server
        unreachable" is reported quickly. On expiry: ``httpx.ConnectTimeout``.
    read_timeout:
        Seconds to wait for the server to start sending a response after the
        request was sent. This caps how long a single prediction can run on
        the server before the client gives up. Default 60s matches observed
        prediction times (100ms..~10s) with headroom for slow links. On
        expiry: ``httpx.ReadTimeout``.
    set_image_read_timeout:
        Read timeout (seconds) used *only* for ``set_image``. After the volume
        is uploaded, the server decompresses and preprocesses the full image
        before responding, which can take far longer than a prediction on a
        large volume. ``set_image`` therefore gets its own generous read
        timeout instead of the much tighter ``read_timeout`` used for
        predictions. On expiry: ``httpx.ReadTimeout``.
    write_timeout:
        Seconds to finish uploading the request body. ``set_image`` uploads
        the full 4D volume so this is the longest-running upload. On expiry:
        ``httpx.WriteTimeout``.
    pool_timeout:
        Seconds to wait for an httpx connection from the pool.
    """

    def __init__(
        self,
        server_url: str,
        api_key: Optional[str] = None,
        connect_timeout: float = 10.0,
        read_timeout: float = 60.0,
        set_image_read_timeout: float = 600.0,
        write_timeout: float = 120.0,
        pool_timeout: float = 10.0,
    ):
        if api_key is None:
            api_key = os.environ.get("NN_INTERACTIVE_API_KEY")

        headers = {}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        self._http = httpx.Client(
            base_url=server_url.rstrip("/"),
            timeout=httpx.Timeout(
                connect=connect_timeout,
                read=read_timeout,
                write=write_timeout,
                pool=pool_timeout,
            ),
            headers=headers,
        )
        # Per-request timeout override for set_image: same connect/write/pool as
        # the client default, but a much longer read budget for server-side
        # decompression + preprocessing of the full volume.
        self._set_image_timeout = httpx.Timeout(
            connect=connect_timeout,
            read=set_image_read_timeout,
            write=write_timeout,
            pool=pool_timeout,
        )
        self._lease_token: Optional[str] = None

        # Claim a session on the server. The lease token is then attached to
        # every subsequent request via the shared httpx headers dict; the
        # caller never has to think about it.
        claim_resp = self._http.post(PATH_CLAIM)
        _raise_for_lease_errors(claim_resp)
        claim_resp.raise_for_status()
        claim_info = claim_resp.json()
        self._lease_token = claim_info["lease_token"]
        self.idle_timeout_seconds: float = float(claim_info.get("idle_timeout_seconds", 0.0))
        self.liveness_timeout_seconds: float = float(claim_info.get("liveness_timeout_seconds", 0.0))
        self._http.headers[LEASE_HEADER] = self._lease_token

        # Background liveness heartbeat bookkeeping. Defined before any code that
        # might raise so close()/__del__ can always reference them safely. The
        # thread itself is started at the end of __init__, once construction has
        # fully succeeded.
        self._stop_heartbeat = threading.Event()
        self._heartbeat_thread: Optional[threading.Thread] = None

        caps = self._get_json(PATH_CAPABILITIES)

        # Attributes that mirror the local session so the GUI can introspect them
        # without caring whether it's holding a local or remote session.
        self.supported_interactions: dict = caps["supported_interactions"]
        # JSON loses tuples; channel_mapping uses (pos, neg) tuples in the local session.
        self.channel_mapping: dict = {
            k: tuple(v) if isinstance(v, list) else v for k, v in caps["channel_mapping"].items()
        }
        self.num_interaction_channels: int = caps["num_interaction_channels"]
        self.supports_initial_label: bool = caps["supports_initial_label"]
        self.supports_zero_shot_label_refinement: bool = caps["supports_zero_shot_label_refinement"]
        self.preferred_scribble_thickness = caps["preferred_scribble_thickness"]
        self.interaction_decay = caps["interaction_decay"]
        self.INFERENCE_SESSION_VERSION = caps["inference_session_version"]
        # License of the model loaded on the server. Mirrors
        # nnInteractiveInferenceSession.license so a GUI can display it
        # regardless of whether it holds a local or remote session.
        # "!!MISSING!!" means the server could not determine the license.
        self.license: Optional[str] = caps.get("license")
        # Older servers predate the /undo endpoint and omit this flag; default False so the
        # GUI can disable undo instead of issuing requests that would 404.
        self.supports_undo: bool = bool(caps.get("supports_undo", False))

        self.original_image_shape: Optional[Tuple[int, ...]] = None
        self.target_buffer: Union[np.ndarray, "torch.Tensor", None] = None
        # Bbox (clipped, in target-buffer coordinates) of the region the most recent prediction
        # wrote into target_buffer, mirrored from the server response. None when nothing was
        # written. Mirrors the local session's API so callers get the changed region either way.
        self._last_paste_bbox: Optional[List[List[int]]] = None
        self.do_autozoom: bool = bool(caps.get("do_autozoom", True))

        # Construction succeeded — start auto-heartbeating to keep the server
        # from reaping us as a dead client. Beat at half the liveness timeout so
        # one dropped request still leaves margin. Daemon thread: it never blocks
        # interpreter exit, and close() joins it cleanly.
        if self.liveness_timeout_seconds > 0:
            self._heartbeat_interval = max(5.0, self.liveness_timeout_seconds / 2.0)
            self._heartbeat_thread = threading.Thread(
                target=self._heartbeat_loop, name="nnInteractive-heartbeat", daemon=True
            )
            self._heartbeat_thread.start()

    # ------------------------------------------------------------------ #
    #                       HTTP helpers (private)                       #
    # ------------------------------------------------------------------ #

    def _get_json(self, path: str) -> dict:
        resp = self._http.get(path)
        _raise_for_lease_errors(resp)
        resp.raise_for_status()
        return resp.json()

    def _post_json(self, path: str, body: dict) -> httpx.Response:
        # Pre-serialize so numpy values in `body` (e.g. spacing as np.ndarray)
        # don't hit httpx's default json encoder, which can't handle them.
        payload = json.dumps(_to_jsonable(body), separators=(",", ":"))
        resp = self._http.post(path, content=payload, headers={"Content-Type": "application/json"})
        _raise_for_lease_errors(resp)
        resp.raise_for_status()
        return resp

    def _post_binary(
        self,
        path: str,
        meta: dict,
        array_bytes: bytes,
        timeout: Union[httpx.Timeout, float, None] = None,
    ) -> httpx.Response:
        headers = {
            META_HEADER: json.dumps(_to_jsonable(meta), separators=(",", ":")),
            "Content-Type": CONTENT_TYPE_OCTET_STREAM,
        }
        # httpx treats timeout=None as "no override" only when the arg is
        # omitted; pass it through explicitly only when a caller supplied one.
        kwargs = {} if timeout is None else {"timeout": timeout}
        resp = self._http.post(path, content=array_bytes, headers=headers, **kwargs)
        _raise_for_lease_errors(resp)
        resp.raise_for_status()
        return resp

    def _apply_prediction_response(self, resp: httpx.Response) -> bool:
        """Update self.target_buffer from a server response carrying a bbox diff.

        Returns whether the server reported a prediction/undo actually ran, so callers (e.g.
        undo()) can tell the difference between "applied a change" and "nothing happened". A
        change can run with an empty bbox (no visible diff), so the return reflects the server's
        ran_prediction flag, not whether voxels were written.
        """
        self._last_paste_bbox = None
        meta_raw = resp.headers.get(META_HEADER)
        if meta_raw is None:
            return False
        meta = json.loads(meta_raw)
        ran = bool(meta.get("ran_prediction", False))
        bbox = meta.get("bbox")
        if not ran or bbox is None or self.target_buffer is None:
            return ran
        # Record the changed region so callers can copy just this sub-volume. The server already
        # clipped this bbox to the buffer bounds, so it is directly sliceable.
        self._last_paste_bbox = [list(b) for b in bbox]
        body = resp.content
        if len(body) == 0:
            return ran
        diff = unpack_array(body)
        self._write_bbox_into_target_buffer(diff, bbox)
        return ran

    def _write_bbox_into_target_buffer(self, diff: np.ndarray, bbox: List[List[int]]) -> None:
        slicer = tuple(slice(int(lb), int(ub)) for lb, ub in bbox)
        tb = self.target_buffer
        if torch is not None and isinstance(tb, torch.Tensor):
            t = torch.from_numpy(diff).to(device=tb.device, dtype=tb.dtype)
            tb[slicer] = t
        else:
            tb[slicer] = diff.astype(tb.dtype, copy=False)

    # ------------------------------------------------------------------ #
    #                         Public API                                 #
    # ------------------------------------------------------------------ #

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_fold=None,
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        """The server loaded its checkpoint at startup. This call is a no-op in v1."""
        warnings.warn(
            "nnInteractiveRemoteInferenceSession ignores initialize_from_trained_model_folder(): "
            "the server loaded its checkpoint at startup. Switch-by-name will be added in a "
            "future release.",
            RuntimeWarning,
            stacklevel=2,
        )

    def set_image(self, image: np.ndarray, image_properties: Optional[dict] = None) -> None:
        assert image.ndim == 4, f"expected a 4d image as input, got {image.ndim}d. Shape {image.shape}"
        meta = {"image_properties": image_properties or {}}
        resp = self._post_binary(
            PATH_SET_IMAGE,
            meta,
            pack_array(image, nthreads=_compression_threads()),
            timeout=self._set_image_timeout,
        )
        info = resp.json()
        self.original_image_shape = tuple(info["original_image_shape"])

    def set_target_buffer(self, target_buffer: Union[np.ndarray, "torch.Tensor"]) -> None:
        self.target_buffer = target_buffer
        self._post_json(
            PATH_SET_TARGET_BUFFER,
            {
                "shape": list(target_buffer.shape),
                "dtype": _buffer_dtype_str(target_buffer),
            },
        )

    def set_do_autozoom(self, do_autozoom: bool) -> None:
        self.do_autozoom = bool(do_autozoom)
        self._post_json(PATH_SET_DO_AUTOZOOM, {"do_autozoom": bool(do_autozoom)})

    def reset_interactions(self) -> None:
        if self.target_buffer is not None:
            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer.fill(0)
            elif torch is not None and isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer.zero_()
        self._post_json(PATH_RESET_INTERACTIONS, {})

    def undo(self) -> bool:
        """Revert the most recent interaction. Patches the local target buffer with the changed
        region returned by the server. Returns True if something was undone, False if there was
        nothing to undo (mirrors the local session's undo())."""
        resp = self._post_json(PATH_UNDO, {})
        return self._apply_prediction_response(resp)

    def add_bbox_interaction(
        self,
        bbox_coords,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ) -> Optional[List[List[int]]]:
        resp = self._post_json(
            PATH_ADD_BBOX,
            {
                "bbox_coords": [list(b) for b in bbox_coords],
                "include_interaction": bool(include_interaction),
                "run_prediction": bool(run_prediction),
                "override_capability_checks": bool(override_capability_checks),
            },
        )
        self._apply_prediction_response(resp)
        return self._last_paste_bbox

    def add_point_interaction(
        self,
        coordinates,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ) -> Optional[List[List[int]]]:
        resp = self._post_json(
            PATH_ADD_POINT,
            {
                "coordinates": list(coordinates),
                "include_interaction": bool(include_interaction),
                "run_prediction": bool(run_prediction),
                "override_capability_checks": bool(override_capability_checks),
            },
        )
        self._apply_prediction_response(resp)
        return self._last_paste_bbox

    def add_scribble_interaction(
        self,
        scribble_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
        interaction_bbox: Optional[List[List[int]]] = None,
    ) -> Optional[List[List[int]]]:
        return self._post_mask_interaction(
            PATH_ADD_SCRIBBLE,
            scribble_image,
            include_interaction,
            run_prediction,
            override_capability_checks,
            interaction_bbox,
        )

    def add_lasso_interaction(
        self,
        lasso_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
        interaction_bbox: Optional[List[List[int]]] = None,
    ) -> Optional[List[List[int]]]:
        return self._post_mask_interaction(
            PATH_ADD_LASSO,
            lasso_image,
            include_interaction,
            run_prediction,
            override_capability_checks,
            interaction_bbox,
        )

    def _post_mask_interaction(
        self,
        path: str,
        mask_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool,
        override_capability_checks: bool,
        interaction_bbox: Optional[List[List[int]]],
    ) -> Optional[List[List[int]]]:
        meta = {
            "include_interaction": bool(include_interaction),
            "run_prediction": bool(run_prediction),
            "override_capability_checks": bool(override_capability_checks),
            "interaction_bbox": ([list(b) for b in interaction_bbox] if interaction_bbox is not None else None),
        }
        # Interactions (scribble/lasso masks) compress best with NOFILTER; skip auto-selection.
        resp = self._post_binary(
            path, meta, pack_array(mask_image, filters=[blosc2.Filter.NOFILTER], nthreads=_compression_threads())
        )
        self._apply_prediction_response(resp)
        return self._last_paste_bbox

    def add_initial_seg_interaction(
        self,
        initial_seg: np.ndarray,
        run_prediction: bool = False,
        override_capability_checks: bool = False,
    ) -> Optional[List[List[int]]]:
        # Mirror the local session: target_buffer is overwritten with initial_seg
        # before any prediction runs. The server does this on its side; we mirror
        # it client-side so the user's buffer reflects the result immediately,
        # without needing to ship initial_seg back over the wire.
        if self.target_buffer is not None:
            if torch is not None and isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer[:] = torch.from_numpy(initial_seg).to(
                    device=self.target_buffer.device, dtype=self.target_buffer.dtype
                )
            else:
                self.target_buffer[:] = initial_seg.astype(self.target_buffer.dtype, copy=False)

        meta = {
            "run_prediction": bool(run_prediction),
            "override_capability_checks": bool(override_capability_checks),
        }
        # Segmentations compress best with NOFILTER; skip auto-selection.
        resp = self._post_binary(
            PATH_ADD_INITIAL_SEG,
            meta,
            pack_array(initial_seg, filters=[blosc2.Filter.NOFILTER], nthreads=_compression_threads()),
        )
        self._apply_prediction_response(resp)
        return self._last_paste_bbox

    # ------------------------------------------------------------------ #
    #                          Lifecycle                                 #
    # ------------------------------------------------------------------ #

    def ping(self, timeout: float = 5.0) -> bool:
        """Reachability probe for a "Test connection" UI.

        Sends ``GET /healthz`` with a tight per-call timeout. Returns ``True``
        if the server answered with a 2xx, ``False`` on any HTTP/network
        error (including timeout, refused connection, wrong auth, proxy
        interception). This is intentionally non-raising so it composes well
        with UI code that just wants a yes/no signal.
        """
        try:
            resp = self._http.get(PATH_HEALTHZ, timeout=timeout)
            return resp.status_code == 200
        except httpx.HTTPError:
            return False

    def heartbeat(self) -> float:
        """Tell the server this client is still alive. Returns remaining seconds
        until the *idle* timeout.

        This proves liveness only: it stops the server from reaping the session
        as a crashed/dead client, but it does NOT postpone the idle timeout —
        that is refreshed solely by real user actions (``set_image``,
        ``add_*_interaction``, …). A session left untouched will therefore still
        be reaped at the idle timeout even while heartbeats keep arriving.

        You normally never call this yourself: the session auto-heartbeats from
        a background thread for the lifetime of the object.
        """
        resp = self._http.post(PATH_HEARTBEAT)
        _raise_for_lease_errors(resp)
        resp.raise_for_status()
        return float(resp.json().get("remaining_seconds", 0.0))

    def _heartbeat_loop(self) -> None:
        """Background daemon: prove liveness every ``_heartbeat_interval`` seconds.

        Stops when the session is closed or once the lease is gone. Transient
        network errors are swallowed so a brief blip doesn't kill the heartbeat;
        the server's liveness timeout tolerates a few missed beats. Lease expiry
        (idle reap or server restart) is surfaced to the user on their next real
        call, not from this thread.
        """
        while not self._stop_heartbeat.wait(self._heartbeat_interval):
            try:
                self.heartbeat()
            except SessionExpiredError:
                break
            except httpx.HTTPError:
                continue
            except Exception:
                # Never let the daemon thread die noisily (e.g. client closing
                # concurrently). Bail out quietly.
                break

    def lease_status(self) -> dict:
        """Read-only probe: how much time is left before this session is reaped?

        Returns ``{"remaining_seconds": float, "idle_timeout_seconds": float}``.
        Does NOT extend the lease — use ``heartbeat()`` for that. Raises
        :class:`SessionExpiredError` if the lease is already gone.
        """
        resp = self._http.get(PATH_LEASE_STATUS)
        _raise_for_lease_errors(resp)
        resp.raise_for_status()
        return resp.json()

    def close(self) -> None:
        # Stop the heartbeat thread first so it can't use self._http after we
        # close it. join() with a short timeout: the thread spends almost all
        # its time in Event.wait(), which the set() interrupts immediately.
        self._stop_heartbeat.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=5.0)
            self._heartbeat_thread = None

        # Best-effort release so the server can free our slot for other users
        # without waiting for the idle reaper. Swallow errors: the server may
        # already be gone, our lease may already be expired, etc. close()
        # must remain idempotent.
        if self._lease_token is not None:
            try:
                self._http.post(PATH_RELEASE, timeout=httpx.Timeout(5.0))
            except httpx.HTTPError:
                pass
            self._lease_token = None
            self._http.headers.pop(LEASE_HEADER, None)
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
