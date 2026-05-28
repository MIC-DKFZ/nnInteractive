"""FastAPI app factory wrapping a single nnInteractiveInferenceSession.

All session-mutating endpoints are serialized by a single ``threading.Lock`` —
the underlying session is stateful and the GPU is the bottleneck, so there is no
benefit to overlapping calls.
"""

from __future__ import annotations

import json
import logging
import threading
from typing import Optional

import numpy as np
import torch
from fastapi import Depends, FastAPI, HTTPException, Header, Request, Response, status

from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
from nnInteractive.inference.remote._protocol import (
    CONTENT_TYPE_OCTET_STREAM,
    META_HEADER,
    PATH_ADD_BBOX,
    PATH_ADD_INITIAL_SEG,
    PATH_ADD_LASSO,
    PATH_ADD_POINT,
    PATH_ADD_SCRIBBLE,
    PATH_CAPABILITIES,
    PATH_HEALTHZ,
    PATH_RESET_INTERACTIONS,
    PATH_SET_DO_AUTOZOOM,
    PATH_SET_IMAGE,
    PATH_SET_TARGET_BUFFER,
)
from nnInteractive.inference.remote.serialization import pack_array, unpack_array

logger = logging.getLogger("nninteractive.server")


def make_app(session: nnInteractiveInferenceSession, api_key: Optional[str] = None) -> FastAPI:
    app = FastAPI(title="nnInteractive Inference Server")
    lock = threading.Lock()

    # ----------------------------- auth ----------------------------------- #

    def require_auth(authorization: Optional[str] = Header(default=None)) -> None:
        if api_key is None:
            return
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token")
        if authorization[len("Bearer ") :] != api_key:
            raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token")

    auth = Depends(require_auth)

    # --------------------------- helpers ---------------------------------- #

    def _parse_meta_header(meta_header: Optional[str]) -> dict:
        if meta_header is None:
            return {}
        try:
            return json.loads(meta_header)
        except json.JSONDecodeError as e:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=f"Bad {META_HEADER}: {e}")

    def _channel_mapping_serializable(mapping: dict) -> dict:
        # tuples (pos, neg) → lists for JSON; client re-tuples them on receipt.
        out = {}
        for k, v in mapping.items():
            out[k] = list(v) if isinstance(v, (tuple, list)) else v
        return out

    def _read_target_bbox(bbox: list[list[int]]) -> np.ndarray:
        """Return the (cropped) target_buffer region for the given bbox as a numpy array."""
        slicer = tuple(slice(int(lb), int(ub)) for lb, ub in bbox)
        tb = session.target_buffer
        if isinstance(tb, torch.Tensor):
            return tb[slicer].detach().cpu().numpy()
        return np.ascontiguousarray(tb[slicer])

    def _build_prediction_response(ran_prediction: bool) -> Response:
        bbox = session._last_paste_bbox if ran_prediction else None
        if bbox is None or session.target_buffer is None:
            meta = {"ran_prediction": bool(ran_prediction), "bbox": None}
            return Response(
                content=b"",
                media_type=CONTENT_TYPE_OCTET_STREAM,
                headers={META_HEADER: json.dumps(meta, separators=(",", ":"))},
            )
        sub = _read_target_bbox(bbox)
        meta = {
            "ran_prediction": True,
            "bbox": [[int(lb), int(ub)] for lb, ub in bbox],
            "dtype": str(sub.dtype),
            "shape": list(sub.shape),
        }
        # Reset so a subsequent call without a prediction can't accidentally re-send a stale region.
        session._last_paste_bbox = None
        return Response(
            content=pack_array(sub),
            media_type=CONTENT_TYPE_OCTET_STREAM,
            headers={META_HEADER: json.dumps(meta, separators=(",", ":"))},
        )

    def _under_lock(fn):
        """Run ``fn()`` under the global lock, converting known errors to HTTP 400.

        The lock spans the full request body — including reading
        ``_last_paste_bbox`` and the target-buffer slice — so no other request
        can interleave session state in between.
        """
        with lock:
            try:
                return fn()
            except (ValueError, AssertionError) as e:
                raise HTTPException(status.HTTP_400_BAD_REQUEST, detail=str(e))

    # ----------------------------- routes --------------------------------- #

    @app.get(PATH_HEALTHZ)
    def healthz() -> dict:
        return {"ok": True}

    @app.get(PATH_CAPABILITIES, dependencies=[auth])
    def capabilities() -> dict:
        cfg = session.configuration_manager
        return {
            "supported_interactions": session.supported_interactions,
            "channel_mapping": _channel_mapping_serializable(session.channel_mapping),
            "num_interaction_channels": int(session.num_interaction_channels),
            "supports_initial_label": bool(session.supports_initial_label),
            "supports_zero_shot_label_refinement": bool(session.supports_zero_shot_label_refinement),
            "preferred_scribble_thickness": session.preferred_scribble_thickness,
            "interaction_decay": float(session.interaction_decay) if session.interaction_decay is not None else None,
            "patch_size": list(cfg.patch_size) if cfg is not None else None,
            "do_autozoom": bool(session.do_autozoom),
            "inference_session_version": session.INFERENCE_SESSION_VERSION,
        }

    @app.post(PATH_SET_IMAGE, dependencies=[auth])
    async def set_image(request: Request) -> dict:
        meta = _parse_meta_header(request.headers.get(META_HEADER))
        body = await request.body()
        image = unpack_array(body)
        image_properties = meta.get("image_properties") or {}

        def _do():
            session.set_image(image, image_properties)
            # set_image preprocesses in a background thread; force completion so
            # subsequent calls can safely use original_image_shape.
            session._finish_preprocessing_and_initialize_interactions()
            return {"original_image_shape": list(session.original_image_shape)}

        return _under_lock(_do)

    @app.post(PATH_SET_TARGET_BUFFER, dependencies=[auth])
    def set_target_buffer(payload: dict) -> dict:
        shape = tuple(int(x) for x in payload["shape"])
        dtype = np.dtype(payload["dtype"])
        buf = np.zeros(shape, dtype=dtype)

        def _do():
            session.set_target_buffer(buf)
            return {}

        return _under_lock(_do)

    @app.post(PATH_SET_DO_AUTOZOOM, dependencies=[auth])
    def set_do_autozoom(payload: dict) -> dict:
        do_autozoom = bool(payload["do_autozoom"])

        def _do():
            session.set_do_autozoom(do_autozoom)
            return {}

        return _under_lock(_do)

    @app.post(PATH_RESET_INTERACTIONS, dependencies=[auth])
    def reset_interactions() -> dict:
        def _do():
            session.reset_interactions()
            return {}

        return _under_lock(_do)

    @app.post(PATH_ADD_BBOX, dependencies=[auth])
    def add_bbox_interaction(payload: dict) -> Response:
        run_prediction = bool(payload.get("run_prediction", True))

        def _do():
            session.add_bbox_interaction(
                bbox_coords=[list(b) for b in payload["bbox_coords"]],
                include_interaction=bool(payload["include_interaction"]),
                run_prediction=run_prediction,
                override_capability_checks=bool(payload.get("override_capability_checks", False)),
            )
            return _build_prediction_response(run_prediction)

        return _under_lock(_do)

    @app.post(PATH_ADD_POINT, dependencies=[auth])
    def add_point_interaction(payload: dict) -> Response:
        run_prediction = bool(payload.get("run_prediction", True))

        def _do():
            session.add_point_interaction(
                coordinates=list(payload["coordinates"]),
                include_interaction=bool(payload["include_interaction"]),
                run_prediction=run_prediction,
                override_capability_checks=bool(payload.get("override_capability_checks", False)),
            )
            return _build_prediction_response(run_prediction)

        return _under_lock(_do)

    @app.post(PATH_ADD_SCRIBBLE, dependencies=[auth])
    async def add_scribble_interaction(request: Request) -> Response:
        return await _handle_mask_interaction(request, session.add_scribble_interaction)

    @app.post(PATH_ADD_LASSO, dependencies=[auth])
    async def add_lasso_interaction(request: Request) -> Response:
        return await _handle_mask_interaction(request, session.add_lasso_interaction)

    async def _handle_mask_interaction(request: Request, session_method) -> Response:
        meta = _parse_meta_header(request.headers.get(META_HEADER))
        body = await request.body()
        mask = unpack_array(body)
        run_prediction = bool(meta.get("run_prediction", True))
        interaction_bbox = meta.get("interaction_bbox")
        if interaction_bbox is not None:
            interaction_bbox = [list(b) for b in interaction_bbox]

        def _do():
            session_method(
                mask,
                bool(meta["include_interaction"]),
                run_prediction=run_prediction,
                override_capability_checks=bool(meta.get("override_capability_checks", False)),
                interaction_bbox=interaction_bbox,
            )
            return _build_prediction_response(run_prediction)

        return _under_lock(_do)

    @app.post(PATH_ADD_INITIAL_SEG, dependencies=[auth])
    async def add_initial_seg_interaction(request: Request) -> Response:
        meta = _parse_meta_header(request.headers.get(META_HEADER))
        body = await request.body()
        initial_seg = unpack_array(body)
        run_prediction = bool(meta.get("run_prediction", False))

        def _do():
            session.add_initial_seg_interaction(
                initial_seg=initial_seg,
                run_prediction=run_prediction,
                override_capability_checks=bool(meta.get("override_capability_checks", False)),
            )
            return _build_prediction_response(run_prediction)

        return _under_lock(_do)

    return app
