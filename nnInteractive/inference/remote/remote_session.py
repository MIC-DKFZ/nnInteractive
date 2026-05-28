"""Client-side stand-in for nnInteractiveInferenceSession backed by an HTTP server.

Public API matches the local session (see nnInteractive/inference/inference_session.py
and API_CHANGES_v2.md). All model state lives on the server; this object holds
only the user's target_buffer (mirrored from server responses) and the
capability metadata fetched at construction time.
"""

from __future__ import annotations

import json
import os
import warnings
from typing import List, Optional, Tuple, Union

import httpx
import numpy as np
import torch

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


def _buffer_dtype_str(target_buffer: Union[np.ndarray, torch.Tensor]) -> str:
    if isinstance(target_buffer, torch.Tensor):
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

        self.original_image_shape: Optional[Tuple[int, ...]] = None
        self.target_buffer: Union[np.ndarray, torch.Tensor, None] = None
        self.do_autozoom: bool = bool(caps.get("do_autozoom", True))

    # ------------------------------------------------------------------ #
    #                       HTTP helpers (private)                       #
    # ------------------------------------------------------------------ #

    def _get_json(self, path: str) -> dict:
        resp = self._http.get(path)
        resp.raise_for_status()
        return resp.json()

    def _post_json(self, path: str, body: dict) -> httpx.Response:
        # Pre-serialize so numpy values in `body` (e.g. spacing as np.ndarray)
        # don't hit httpx's default json encoder, which can't handle them.
        payload = json.dumps(_to_jsonable(body), separators=(",", ":"))
        resp = self._http.post(path, content=payload, headers={"Content-Type": "application/json"})
        resp.raise_for_status()
        return resp

    def _post_binary(self, path: str, meta: dict, array_bytes: bytes) -> httpx.Response:
        headers = {
            META_HEADER: json.dumps(_to_jsonable(meta), separators=(",", ":")),
            "Content-Type": CONTENT_TYPE_OCTET_STREAM,
        }
        resp = self._http.post(path, content=array_bytes, headers=headers)
        resp.raise_for_status()
        return resp

    def _apply_prediction_response(self, resp: httpx.Response) -> None:
        """Update self.target_buffer from a server response carrying a bbox diff."""
        meta_raw = resp.headers.get(META_HEADER)
        if meta_raw is None:
            return
        meta = json.loads(meta_raw)
        if not meta.get("ran_prediction", False):
            return
        bbox = meta.get("bbox")
        if bbox is None or self.target_buffer is None:
            return
        body = resp.content
        if len(body) == 0:
            return
        diff = unpack_array(body)
        self._write_bbox_into_target_buffer(diff, bbox)

    def _write_bbox_into_target_buffer(self, diff: np.ndarray, bbox: List[List[int]]) -> None:
        slicer = tuple(slice(int(lb), int(ub)) for lb, ub in bbox)
        tb = self.target_buffer
        if isinstance(tb, torch.Tensor):
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
        resp = self._post_binary(PATH_SET_IMAGE, meta, pack_array(image))
        info = resp.json()
        self.original_image_shape = tuple(info["original_image_shape"])

    def set_target_buffer(self, target_buffer: Union[np.ndarray, torch.Tensor]) -> None:
        self.target_buffer = target_buffer
        self._post_json(
            PATH_SET_TARGET_BUFFER,
            {"shape": list(target_buffer.shape), "dtype": _buffer_dtype_str(target_buffer)},
        )

    def set_do_autozoom(self, do_autozoom: bool) -> None:
        self.do_autozoom = bool(do_autozoom)
        self._post_json(PATH_SET_DO_AUTOZOOM, {"do_autozoom": bool(do_autozoom)})

    def reset_interactions(self) -> None:
        if self.target_buffer is not None:
            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer.fill(0)
            elif isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer.zero_()
        self._post_json(PATH_RESET_INTERACTIONS, {})

    def add_bbox_interaction(
        self,
        bbox_coords,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ) -> None:
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

    def add_point_interaction(
        self,
        coordinates,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ) -> None:
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

    def add_scribble_interaction(
        self,
        scribble_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
        interaction_bbox: Optional[List[List[int]]] = None,
    ) -> None:
        self._post_mask_interaction(
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
    ) -> None:
        self._post_mask_interaction(
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
    ) -> None:
        meta = {
            "include_interaction": bool(include_interaction),
            "run_prediction": bool(run_prediction),
            "override_capability_checks": bool(override_capability_checks),
            "interaction_bbox": ([list(b) for b in interaction_bbox] if interaction_bbox is not None else None),
        }
        resp = self._post_binary(path, meta, pack_array(mask_image))
        self._apply_prediction_response(resp)

    def add_initial_seg_interaction(
        self,
        initial_seg: np.ndarray,
        run_prediction: bool = False,
        override_capability_checks: bool = False,
    ) -> None:
        # Mirror the local session: target_buffer is overwritten with initial_seg
        # before any prediction runs. The server does this on its side; we mirror
        # it client-side so the user's buffer reflects the result immediately,
        # without needing to ship initial_seg back over the wire.
        if self.target_buffer is not None:
            if isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer[:] = torch.from_numpy(initial_seg).to(
                    device=self.target_buffer.device, dtype=self.target_buffer.dtype
                )
            else:
                self.target_buffer[:] = initial_seg.astype(self.target_buffer.dtype, copy=False)

        meta = {
            "run_prediction": bool(run_prediction),
            "override_capability_checks": bool(override_capability_checks),
        }
        resp = self._post_binary(PATH_ADD_INITIAL_SEG, meta, pack_array(initial_seg))
        self._apply_prediction_response(resp)

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

    def close(self) -> None:
        self._http.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()

    def __del__(self):
        try:
            self._http.close()
        except Exception:
            pass
