"""FastAPI app factory for the multi-session nnInteractive inference server.

The server hosts up to ``max_sessions`` concurrent
:class:`nnInteractiveInferenceSession` instances, one per connected client. The
model artifacts (``nn.Module`` network with its weights and buffers, plans/
configuration managers, dataset json, label manager) are loaded once at startup;
each session's ``self.network`` is a plain Python reference to that single
module — there is exactly one network and one copy of the weights on the GPU
regardless of session count. Per-session state (image, target buffer,
interactions tensor) is isolated. Safety of sharing relies on (a) inference
running under ``@torch.inference_mode()`` and (b) a global ``gpu_lock``
serializing predict-capable endpoints, so no two sessions ever touch the
module concurrently and nothing mutates it after construction.

Each client identifies itself via a lease token issued by ``POST /claim``. The
token rides along on every subsequent request in the ``X-Lease-Token`` header.
Sessions are reaped automatically for either of two reasons: the user went idle
(no real interaction for longer than ``idle_timeout_seconds``) or the client
went dead (no request of any kind — not even a heartbeat — for longer than the
much shorter ``liveness_timeout_seconds``, which reclaims slots held by crashed
clients quickly). Subsequent requests bearing a reaped lease receive HTTP 410
Gone so the client can surface a "session expired" message.

Concurrency model:
  - Each session has its own ``threading.Lock`` that serializes the per-session
    mutating endpoints (so a single client can't tear its own state with
    parallel calls).
  - A single global ``gpu_lock`` serializes the predict-capable endpoints
    (``add_*_interaction``) across *all* sessions, because the GPU is one
    resource. Two clients can preprocess images concurrently but only one
    prediction runs at a time.
  - The acquisition order is always (session lock → gpu lock) so there is no
    deadlock potential.
  - The endpoints that carry large payloads (``set_image`` and the mask
    interactions) are ``async`` so they can ``await`` the upload, but their
    CPU-bound work (blosc2 decompression, image preprocessing, prediction,
    response compression) is dispatched to a worker thread via
    ``run_in_threadpool``. This keeps the event loop free during a long
    ``set_image``/predict so lightweight endpoints — ``/heartbeat``,
    ``/healthz`` — and the background reaper stay responsive, and so two
    clients can genuinely preprocess concurrently. Acquiring a session/gpu
    lock therefore also happens off the loop, never stalling it.
"""

from __future__ import annotations

import asyncio
import json
import logging
import numbers
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from typing import Optional

import blosc2
import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Header, Request, Response, status
from starlette.concurrency import run_in_threadpool

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
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

logger = logging.getLogger("nninteractive.server")

# Cap a single client's target buffer at 25% of total system RAM. Falls back to 32 GiB
# of headroom if the system RAM can't be determined.
try:
    total_ram = os.sysconf("SC_PHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
except (ValueError, OSError, AttributeError):
    total_ram = 32 * 1024**3
MAX_TARGET_BUFFER_BYTES = int(total_ram * 0.25)


class SessionEntry:
    """One client's session plus its bookkeeping."""

    def __init__(self, session: nnInteractiveInferenceSession) -> None:
        self.session = session
        self.lock = threading.Lock()
        self.created_at = time.monotonic()
        # Two independent clocks. ``last_active_at`` tracks real user actions
        # (drives the idle timeout); ``last_seen_at`` tracks any sign of life
        # including heartbeats (drives the much shorter liveness timeout used to
        # reclaim slots held by crashed/disconnected clients).
        self.last_active_at = self.created_at
        self.last_seen_at = self.created_at

    def mark_seen(self) -> None:
        """Record that the client is still alive (liveness only)."""
        self.last_seen_at = time.monotonic()

    def mark_active(self) -> None:
        """Record real user activity. Activity implies liveness, so this bumps
        both clocks."""
        now = time.monotonic()
        self.last_active_at = now
        self.last_seen_at = now

    def close(self) -> None:
        """Free the session's per-instance state. The shared network module and
        other model artifacts are NOT freed — they live in the registry and are
        reused by future sessions.

        Best-effort: any exception here is logged but not re-raised; cleanup must
        not block reaping or shutdown.
        """
        try:
            self.session._reset_session()
        except Exception:
            logger.exception("error during session._reset_session() in SessionEntry.close()")
        try:
            self.session.executor.shutdown(wait=False)
        except Exception:
            logger.exception("error during executor.shutdown() in SessionEntry.close()")


class SessionFull(Exception):
    """Raised by SessionRegistry.claim() when the server is at capacity."""


class SessionRegistry:
    """Threadsafe lease-keyed dict of :class:`SessionEntry`.

    The model artifacts loaded at server startup are stashed here and handed to
    every newly created session by reference (the ``nn.Module`` instance, its
    weights, and the plans/configuration/label-manager objects are not copied
    or re-instantiated per session).
    """

    def __init__(
        self,
        artifacts: dict,
        max_sessions: int,
        idle_timeout_seconds: float,
        liveness_timeout_seconds: float,
        device: torch.device,
        torch_n_threads: int,
        do_autozoom: bool,
        use_torch_compile: bool,
        interactions_storage: str,
        verbose: bool,
    ) -> None:
        self._artifacts = artifacts
        self._max_sessions = int(max_sessions)
        self._idle_timeout_seconds = float(idle_timeout_seconds)
        self._liveness_timeout_seconds = float(liveness_timeout_seconds)
        self._device = device
        self._torch_n_threads = torch_n_threads
        self._do_autozoom = do_autozoom
        self._use_torch_compile = use_torch_compile
        self._interactions_storage = interactions_storage
        self._verbose = verbose
        self._entries: dict[str, SessionEntry] = {}
        self._mu = threading.Lock()

    @property
    def max_sessions(self) -> int:
        return self._max_sessions

    @property
    def idle_timeout_seconds(self) -> float:
        return self._idle_timeout_seconds

    @property
    def liveness_timeout_seconds(self) -> float:
        return self._liveness_timeout_seconds

    def claim(self) -> str:
        """Create a new session and return its lease token. Raises SessionFull if at cap."""
        with self._mu:
            if len(self._entries) >= self._max_sessions:
                raise SessionFull()
            token = uuid.uuid4().hex
            session = nnInteractiveInferenceSession(
                device=self._device,
                use_torch_compile=self._use_torch_compile,
                verbose=self._verbose,
                torch_n_threads=self._torch_n_threads,
                do_autozoom=self._do_autozoom,
                interactions_storage=self._interactions_storage,
            )
            session.initialize_from_loaded_artifacts(self._artifacts)
            entry = SessionEntry(session)
            self._entries[token] = entry
            logger.info(
                "claimed session %s (%d/%d active)",
                token,
                len(self._entries),
                self._max_sessions,
            )
            return token

    def get(self, token: Optional[str]) -> SessionEntry:
        """Look up a session by lease token, marking it seen (liveness) on success.

        Note this only refreshes the liveness clock, not activity: a bare
        ``/heartbeat`` keeps the session from being reaped as dead but does not
        postpone the idle timeout. Endpoints that represent real user actions
        call ``entry.mark_active()`` explicitly (see the lock helpers)."""
        if not token:
            raise HTTPException(status.HTTP_410_GONE, detail="lease token missing")
        with self._mu:
            entry = self._entries.get(token)
            if entry is None:
                raise HTTPException(status.HTTP_410_GONE, detail="lease expired or unknown")
        entry.mark_seen()
        return entry

    def peek(self, token: Optional[str]) -> SessionEntry:
        """Look up without touching last_active_at. Used for /lease_status."""
        if not token:
            raise HTTPException(status.HTTP_410_GONE, detail="lease token missing")
        with self._mu:
            entry = self._entries.get(token)
            if entry is None:
                raise HTTPException(status.HTTP_410_GONE, detail="lease expired or unknown")
        return entry

    def release(self, token: Optional[str]) -> bool:
        """Release a session by token. Idempotent; returns True if a session was actually released."""
        if not token:
            return False
        with self._mu:
            entry = self._entries.pop(token, None)
        if entry is None:
            return False
        entry.close()
        logger.info(
            "released session %s (%d/%d active)",
            token,
            len(self._entries),
            self._max_sessions,
        )
        return True

    def sweep(self) -> int:
        """Drop sessions that have either gone idle or stopped showing signs of life.

        A session is reaped if it has seen no real user activity for longer than
        ``idle_timeout_seconds`` (the user walked away) OR if it has not been
        seen at all for longer than ``liveness_timeout_seconds`` (the client
        process crashed/disconnected and stopped heartbeating). Returns the
        number reaped."""
        now = time.monotonic()
        reaped: list[tuple[str, SessionEntry, str]] = []
        with self._mu:
            for token, entry in list(self._entries.items()):
                if (now - entry.last_seen_at) > self._liveness_timeout_seconds:
                    reason = "dead"
                elif (now - entry.last_active_at) > self._idle_timeout_seconds:
                    reason = "idle"
                else:
                    continue
                self._entries.pop(token, None)
                reaped.append((token, entry, reason))
        for token, entry, reason in reaped:
            entry.close()
            logger.info("reaped %s session %s", reason, token)
        return len(reaped)

    def close_all(self) -> None:
        with self._mu:
            entries = list(self._entries.items())
            self._entries.clear()
        for token, entry in entries:
            entry.close()
            logger.info("closed session %s during shutdown", token)

    def remaining_seconds(self, entry: SessionEntry) -> float:
        return max(0.0, self._idle_timeout_seconds - (time.monotonic() - entry.last_active_at))


def make_app(
    artifacts: dict,
    device: torch.device,
    max_sessions: int = 1,
    idle_timeout_seconds: float = 600.0,
    liveness_timeout_seconds: float = 60.0,
    torch_n_threads: int = 8,
    do_autozoom: bool = True,
    use_torch_compile: bool = False,
    interactions_storage: str = "auto",
    verbose: bool = False,
    api_key: Optional[str] = None,
    sweep_interval_seconds: float = 15.0,
) -> FastAPI:
    registry = SessionRegistry(
        artifacts=artifacts,
        max_sessions=max_sessions,
        idle_timeout_seconds=idle_timeout_seconds,
        liveness_timeout_seconds=liveness_timeout_seconds,
        device=device,
        torch_n_threads=torch_n_threads,
        do_autozoom=do_autozoom,
        use_torch_compile=use_torch_compile,
        interactions_storage=interactions_storage,
        verbose=verbose,
    )
    gpu_lock = threading.Lock()

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        stop = asyncio.Event()

        async def reaper():
            try:
                while not stop.is_set():
                    try:
                        await asyncio.sleep(sweep_interval_seconds)
                    except asyncio.CancelledError:
                        break
                    if stop.is_set():
                        break
                    try:
                        registry.sweep()
                    except Exception:
                        logger.exception("error in registry sweep()")
            except asyncio.CancelledError:
                pass

        task = asyncio.create_task(reaper())
        try:
            yield
        finally:
            stop.set()
            task.cancel()
            try:
                await task
            except (asyncio.CancelledError, Exception):
                pass
            registry.close_all()

    app = FastAPI(title="nnInteractive Inference Server", lifespan=lifespan)

    # Capability snapshot is computed once and never changes (the network
    # module is loaded once at startup and the same instance is referenced by
    # every session). We build it from a fresh session initialized off the
    # artifacts.
    # This throwaway session only reads static capability metadata and never runs
    # a forward pass, so there is no point compiling its network reference.
    _capability_session = nnInteractiveInferenceSession(
        device=device,
        use_torch_compile=False,
        verbose=False,
        torch_n_threads=torch_n_threads,
        do_autozoom=do_autozoom,
        interactions_storage=interactions_storage,
    )
    _capability_session.initialize_from_loaded_artifacts(artifacts)
    _capability_snapshot = _build_capability_snapshot(_capability_session)
    # We don't need the per-session state of _capability_session, just the snapshot.
    _capability_session._reset_session()
    _capability_session.executor.shutdown(wait=False)
    del _capability_session

    # ----------------------------- auth ----------------------------------- #

    def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
        if api_key is None:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
        if authorization[len("Bearer ") :] != api_key:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")

    auth = Depends(require_auth)

    def require_lease(x_lease_token: Optional[str] = Header(default=None, alias=LEASE_HEADER)) -> SessionEntry:
        return registry.get(x_lease_token)

    lease = Depends(require_lease)

    # --------------------------- helpers ---------------------------------- #

    def _parse_meta_header(meta_header: Optional[str]) -> dict:
        if meta_header is None:
            return {}
        try:
            return json.loads(meta_header)
        except json.JSONDecodeError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Bad {META_HEADER}: {e}")

    def _read_target_bbox(session: nnInteractiveInferenceSession, bbox: list[list[int]]) -> np.ndarray:
        """Return the (cropped) target_buffer region for the given bbox as a numpy array.

        The bbox must already be clipped to valid in-bounds indices — passing a
        negative lower bound here would invoke numpy's "from the end" slicing
        semantics and return the wrong region (silently empty in most cases).
        """
        slicer = tuple(slice(int(lb), int(ub)) for lb, ub in bbox)
        tb = session.target_buffer
        if isinstance(tb, torch.Tensor):
            return tb[slicer].detach().cpu().numpy()
        return np.ascontiguousarray(tb[slicer])

    def _clip_bbox_to_buffer(session: nnInteractiveInferenceSession, bbox: list[list[int]]) -> list[list[int]]:
        """Clip an inference-time bbox to the target buffer's spatial bounds.

        The local session stores ``_last_paste_bbox`` in unclipped form: when
        the prediction patch extends past an image edge (common with autozoom
        + click near a boundary), bounds can be negative or exceed shape. We
        clip so the bbox we slice with is valid and matches what was actually
        pasted into the target buffer.
        """
        tb_shape = session.target_buffer.shape
        return [[max(int(lb), 0), min(int(ub), int(tb_shape[i]))] for i, (lb, ub) in enumerate(bbox)]

    def _build_prediction_response(session: nnInteractiveInferenceSession, ran_prediction: bool) -> Response:
        bbox = session._last_paste_bbox if ran_prediction else None
        if bbox is None or session.target_buffer is None:
            meta = {"ran_prediction": bool(ran_prediction), "bbox": None}
            return Response(
                content=b"",
                media_type=CONTENT_TYPE_OCTET_STREAM,
                headers={META_HEADER: json.dumps(meta, separators=(",", ":"))},
            )
        clipped = _clip_bbox_to_buffer(session, bbox)
        if any(lb >= ub for lb, ub in clipped):
            # Bbox lies entirely outside the buffer — nothing to send.
            session._last_paste_bbox = None
            meta = {"ran_prediction": True, "bbox": None}
            return Response(
                content=b"",
                media_type=CONTENT_TYPE_OCTET_STREAM,
                headers={META_HEADER: json.dumps(meta, separators=(",", ":"))},
            )
        sub = _read_target_bbox(session, clipped)
        meta = {
            "ran_prediction": True,
            "bbox": clipped,
            "dtype": str(sub.dtype),
            "shape": list(sub.shape),
        }
        # Reset so a subsequent call without a prediction can't accidentally re-send a stale region.
        session._last_paste_bbox = None
        return Response(
            # Segmentations compress best with NOFILTER; skip auto-selection.
            content=pack_array(
                sub, filters=[blosc2.Filter.NOFILTER], nthreads=min(session.torch_n_threads, os.cpu_count())
            ),
            media_type=CONTENT_TYPE_OCTET_STREAM,
            headers={META_HEADER: json.dumps(meta, separators=(",", ":"))},
        )

    def _parse_target_buffer_request(payload: dict) -> tuple[tuple[int, ...], np.dtype]:
        if "shape" not in payload:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="missing required field: shape")
        if "dtype" not in payload:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="missing required field: dtype")

        raw_shape = payload["shape"]
        if not isinstance(raw_shape, list):
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="shape must be a list of positive integers")
        if len(raw_shape) != 3:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="shape must be 3D")

        shape = []
        for dim in raw_shape:
            if not isinstance(dim, numbers.Integral) or isinstance(dim, bool):
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="shape must contain only integers")
            if dim <= 0:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail="shape dimensions must be positive")
            shape.append(dim)

        try:
            dtype = np.dtype(payload["dtype"])
        except (TypeError, ValueError) as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"invalid dtype: {payload['dtype']!r}") from e
        # 'b' = bool, 'i' = signed int, 'u' = unsigned int.
        if dtype.kind not in ("b", "i", "u"):
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                detail=f"unsupported dtype {dtype}: target buffer must be bool or an integer type",
            )

        nbytes = int(np.prod(shape, dtype=np.uint64)) * dtype.itemsize
        if nbytes > MAX_TARGET_BUFFER_BYTES:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                detail=(f"target buffer would require {nbytes} bytes, " f"limit is {MAX_TARGET_BUFFER_BYTES} bytes"),
            )

        return tuple(shape), dtype

    def _under_session_lock(entry: SessionEntry, fn):
        """Run ``fn(session)`` under the session's lock, converting known errors to HTTP 400.

        Every endpoint routed through here represents a real user action, so we
        also refresh the activity clock (postponing the idle timeout)."""
        entry.mark_active()
        with entry.lock:
            try:
                return fn(entry.session)
            except (ValueError, AssertionError) as e:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))

    def _under_session_and_gpu_lock(entry: SessionEntry, fn):
        """Run ``fn(session)`` under session lock + global GPU lock. Acquisition order
        is always session-then-gpu to avoid deadlocks.

        Like ``_under_session_lock``, this marks real user activity."""
        entry.mark_active()
        with entry.lock:
            with gpu_lock:
                try:
                    return fn(entry.session)
                except (ValueError, AssertionError) as e:
                    raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))

    # ----------------------------- routes --------------------------------- #

    @app.get(PATH_HEALTHZ)
    def healthz() -> dict:
        return {"ok": True}

    @app.get(PATH_CAPABILITIES, dependencies=[auth])
    def capabilities() -> dict:
        return _capability_snapshot

    @app.post(PATH_CLAIM, dependencies=[auth])
    def claim() -> Response:
        try:
            token = registry.claim()
        except SessionFull:
            raise HTTPException(
                status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"server is at capacity ({registry.max_sessions} sessions)",
                headers={"Retry-After": "10"},
            )
        body = {
            "lease_token": token,
            "idle_timeout_seconds": registry.idle_timeout_seconds,
            "liveness_timeout_seconds": registry.liveness_timeout_seconds,
            "max_sessions": registry.max_sessions,
        }
        return Response(
            content=json.dumps(body),
            media_type="application/json",
        )

    @app.post(PATH_RELEASE, dependencies=[auth])
    def release(x_lease_token: Optional[str] = Header(default=None, alias=LEASE_HEADER)) -> dict:
        registry.release(x_lease_token)
        # Idempotent: succeed even if the lease was already gone.
        return {"released": True}

    @app.post(PATH_HEARTBEAT, dependencies=[auth])
    def heartbeat(entry: SessionEntry = lease) -> dict:
        # require_lease already touched last_active_at.
        return {"remaining_seconds": registry.remaining_seconds(entry)}

    @app.get(PATH_LEASE_STATUS, dependencies=[auth])
    def lease_status(x_lease_token: Optional[str] = Header(default=None, alias=LEASE_HEADER)) -> dict:
        # Read-only probe: does NOT touch last_active_at.
        entry = registry.peek(x_lease_token)
        return {
            "remaining_seconds": registry.remaining_seconds(entry),
            "idle_timeout_seconds": registry.idle_timeout_seconds,
        }

    @app.post(PATH_SET_IMAGE, dependencies=[auth])
    async def set_image(request: Request, entry: SessionEntry = lease) -> dict:
        meta = _parse_meta_header(request.headers.get(META_HEADER))
        body = await request.body()
        image_properties = meta.get("image_properties") or {}

        # Decompression + full-volume preprocessing are CPU-bound and can run
        # for many seconds on a large image. Run them in a worker thread so the
        # event loop keeps servicing heartbeats/healthz and the reaper.
        def _work():
            image = unpack_array(body)

            def _do(session):
                session.set_image(image, image_properties)
                # set_image preprocesses in a background thread; force completion
                # so subsequent calls can safely use original_image_shape.
                session._finish_preprocessing_and_initialize_interactions()
                return {"original_image_shape": list(session.original_image_shape)}

            return _under_session_lock(entry, _do)

        return await run_in_threadpool(_work)

    @app.post(PATH_SET_TARGET_BUFFER, dependencies=[auth])
    def set_target_buffer(payload: dict, entry: SessionEntry = lease) -> dict:
        shape, dtype = _parse_target_buffer_request(payload)
        buf = np.zeros(shape, dtype=dtype)

        def _do(session):
            session.set_target_buffer(buf)
            return {}

        return _under_session_lock(entry, _do)

    @app.post(PATH_SET_DO_AUTOZOOM, dependencies=[auth])
    def set_do_autozoom(payload: dict, entry: SessionEntry = lease) -> dict:
        do_autozoom = bool(payload["do_autozoom"])

        def _do(session):
            session.set_do_autozoom(do_autozoom)
            return {}

        return _under_session_lock(entry, _do)

    @app.post(PATH_RESET_INTERACTIONS, dependencies=[auth])
    def reset_interactions(entry: SessionEntry = lease) -> dict:
        def _do(session):
            session.reset_interactions()
            return {}

        return _under_session_lock(entry, _do)

    @app.post(PATH_UNDO, dependencies=[auth])
    def undo(entry: SessionEntry = lease) -> Response:
        # Undo does no GPU inference (only CPU decompress/copy), so it must not hold the GPU lock.
        def _do(session):
            ran = session.undo()
            return _build_prediction_response(session, ran_prediction=ran)

        return _under_session_lock(entry, _do)

    @app.post(PATH_ADD_BBOX, dependencies=[auth])
    def add_bbox_interaction(payload: dict, entry: SessionEntry = lease) -> Response:
        run_prediction = bool(payload.get("run_prediction", True))

        def _do(session):
            session.add_bbox_interaction(
                bbox_coords=[list(b) for b in payload["bbox_coords"]],
                include_interaction=bool(payload["include_interaction"]),
                run_prediction=run_prediction,
                override_capability_checks=bool(payload.get("override_capability_checks", False)),
            )
            return _build_prediction_response(session, run_prediction)

        return _under_session_and_gpu_lock(entry, _do)

    @app.post(PATH_ADD_POINT, dependencies=[auth])
    def add_point_interaction(payload: dict, entry: SessionEntry = lease) -> Response:
        run_prediction = bool(payload.get("run_prediction", True))

        def _do(session):
            session.add_point_interaction(
                coordinates=list(payload["coordinates"]),
                include_interaction=bool(payload["include_interaction"]),
                run_prediction=run_prediction,
                override_capability_checks=bool(payload.get("override_capability_checks", False)),
            )
            return _build_prediction_response(session, run_prediction)

        return _under_session_and_gpu_lock(entry, _do)

    @app.post(PATH_ADD_SCRIBBLE, dependencies=[auth])
    async def add_scribble_interaction(request: Request, entry: SessionEntry = lease) -> Response:
        return await _handle_mask_interaction(request, entry, "scribble")

    @app.post(PATH_ADD_LASSO, dependencies=[auth])
    async def add_lasso_interaction(request: Request, entry: SessionEntry = lease) -> Response:
        return await _handle_mask_interaction(request, entry, "lasso")

    async def _handle_mask_interaction(request: Request, entry: SessionEntry, kind: str) -> Response:
        meta = _parse_meta_header(request.headers.get(META_HEADER))
        body = await request.body()
        run_prediction = bool(meta.get("run_prediction", True))
        interaction_bbox = meta.get("interaction_bbox")
        if interaction_bbox is not None:
            interaction_bbox = [list(b) for b in interaction_bbox]

        # Decompression + prediction + response compression are CPU/GPU-bound;
        # run them off the event loop (see set_image).
        def _work():
            mask = unpack_array(body)

            def _do(session):
                method = session.add_scribble_interaction if kind == "scribble" else session.add_lasso_interaction
                method(
                    mask,
                    bool(meta["include_interaction"]),
                    run_prediction=run_prediction,
                    override_capability_checks=bool(meta.get("override_capability_checks", False)),
                    interaction_bbox=interaction_bbox,
                )
                return _build_prediction_response(session, run_prediction)

            return _under_session_and_gpu_lock(entry, _do)

        return await run_in_threadpool(_work)

    @app.post(PATH_ADD_INITIAL_SEG, dependencies=[auth])
    async def add_initial_seg_interaction(request: Request, entry: SessionEntry = lease) -> Response:
        meta = _parse_meta_header(request.headers.get(META_HEADER))
        body = await request.body()
        run_prediction = bool(meta.get("run_prediction", False))

        # Decompression + (optional) prediction are CPU/GPU-bound; run them off
        # the event loop (see set_image).
        def _work():
            initial_seg = unpack_array(body)

            def _do(session):
                session.add_initial_seg_interaction(
                    initial_seg=initial_seg,
                    run_prediction=run_prediction,
                    override_capability_checks=bool(meta.get("override_capability_checks", False)),
                )
                return _build_prediction_response(session, run_prediction)

            return _under_session_and_gpu_lock(entry, _do)

        return await run_in_threadpool(_work)

    return app


def _channel_mapping_serializable(mapping: dict) -> dict:
    # tuples (pos, neg) → lists for JSON; client re-tuples them on receipt.
    out = {}
    for k, v in mapping.items():
        out[k] = list(v) if isinstance(v, (tuple, list)) else v
    return out


def _build_capability_snapshot(session: nnInteractiveInferenceSession) -> dict:
    cfg = session.configuration_manager
    return {
        "supported_interactions": session.supported_interactions,
        "channel_mapping": _channel_mapping_serializable(session.channel_mapping),
        "num_interaction_channels": int(session.num_interaction_channels),
        "supports_initial_label": bool(session.supports_initial_label),
        "supports_zero_shot_label_refinement": bool(session.supports_zero_shot_label_refinement),
        "preferred_scribble_thickness": session.preferred_scribble_thickness,
        "interaction_decay": (float(session.interaction_decay) if session.interaction_decay is not None else None),
        "patch_size": list(cfg.patch_size) if cfg is not None else None,
        "do_autozoom": bool(session.do_autozoom),
        "inference_session_version": session.INFERENCE_SESSION_VERSION,
        "license": session.license,
        "supports_undo": True,
    }
