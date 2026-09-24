from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from importlib.metadata import version as _package_version
import os
import sys
from os import cpu_count
from time import time
from typing import Union, List, Tuple, Optional
import warnings

import blosc2

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice
from batchgenerators.utilities.file_and_folder_operations import load_json, join, subdirs, isfile
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.functional import interpolate

from nnInteractive.interaction.point import PointInteraction_stub
from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub
from nnInteractive.utils.bboxes import generate_bounding_boxes
from nnInteractive.utils.crop import crop_and_pad_into_buffer, paste_tensor, pad_cropped, crop_to_valid
from nnInteractive.utils.erosion_dilation import iterative_3x3_same_padding_pool3d
from nnInteractive.utils.inference_helpers import (
    infer_num_interaction_channels_from_mapping,
    parse_channel_pair,
    version_to_tuple,
)
from nnInteractive.utils.rounding import round_to_nearest_odd
from nnInteractive.utils.staging import get_stager


class nnInteractiveInferenceSession:
    # The session version IS the installed nnInteractive package version (used for
    # checkpoint compatibility checks and reported to remote clients). `nnInteractive`
    # is a PEP 420 namespace package (no __init__ to carry __version__), so read it
    # from the distribution metadata instead.
    INFERENCE_SESSION_VERSION = _package_version("nnInteractive")
    REFINEMENT_CACHE_GPU_HEADROOM_BYTES = 4 * 1024**3
    # Maximum adaptive zoom-out factor (see _predict). Also bounds the largest interaction crop,
    # which sizes the reusable blosc2 decompression buffer.
    MAX_AUTOZOOM_FACTOR = 4
    # 'auto' interaction storage threshold: images with at most this many spatial voxels
    # (512*512*1024) use the dense tensor backend; larger ones use blosc2 to bound RAM.
    AUTO_TENSOR_MAX_VOXELS = 2**27
    INTERACTIONS_STORAGE_OPTIONS = ("blosc2", "tensor", "auto")
    # Interactions implemented by this inference session.
    SUPPORTED_INTERACTION_KEYS = ("scribble", "lasso", "points", "bbox2d", "bbox3d")

    def __init__(
        self,
        device: torch.device = torch.device("cuda"),
        use_torch_compile: bool = False,
        verbose: bool = False,
        torch_n_threads: int = 8,
        do_autozoom: bool = True,
        interactions_storage: str = "auto",
        enable_undo: bool = True,
    ):
        """
        Only intended to work with nnInteractiveTrainerV2 and its derivatives

        ``use_torch_compile``: compile the network with ``torch.compile``. The
        first prediction after enabling this is slow (compilation happens lazily
        on the first forward pass), but every subsequent prediction is faster.
        This is recommended for the persistent inference server, where the
        process is long-lived so the one-time compile cost is paid only once and
        amortized across the whole session lifetime.

        ``enable_undo``: keep single-level undo of the last interaction available
        (default ``True``; see ``undo()``). Undo works by saving a compressed copy of
        every region the interaction overwrites (interaction channels and target
        buffer) just before it is written, which costs a few ms per prediction and
        the RAM of those compressed regions. Set to ``False`` when you know you will
        never call ``undo()`` to skip that overhead entirely; ``undo()`` then always
        returns ``False`` and ``supports_undo`` reports ``False``.

        ``interactions_storage``: storage backend for the interaction tensor, one of
        ``"blosc2"``, ``"tensor"`` or ``"auto"`` (default).
        ``"blosc2"`` keeps it as a compact blosc2 in-memory NDArray (low RAM, pays
        (de)compression on every read/write). ``"tensor"`` stores it as a dense CPU
        float16 ``torch.Tensor`` (more RAM, far lower per-access overhead; not pinned,
        transfers to the GPU go through a small fixed pinned staging buffer, see
        ``nnInteractive.utils.staging``). ``"auto"`` decides per image at initialization from the
        interaction tensor's voxel count: at most ``AUTO_TENSOR_MAX_VOXELS`` (512*512*1024)
        spatial voxels uses ``"tensor"``, larger uses ``"blosc2"``.
        """
        if interactions_storage not in self.INTERACTIONS_STORAGE_OPTIONS:
            raise ValueError(
                f"interactions_storage must be one of {self.INTERACTIONS_STORAGE_OPTIONS}, "
                f"got {interactions_storage!r}."
            )
        print("session initialized")

        self.network = None
        self.label_manager = None
        self.dataset_json = None
        self.trainer_name = None
        self.configuration_manager = None
        self.plans_manager = None
        self._interactions_shape = None
        self.device = device
        if device.type == "cuda":
            # Every forward pass this session runs has the same fixed input shape
            # ([1, num_input_channels + num_interaction_channels, *patch_size]; see warmup()),
            # so cuDNN benchmark mode autotunes the convolution algorithms once on the first
            # pass and reuses the fastest ones thereafter. For fixed input shapes this is pure
            # upside; warmup() pays that autotuning cost at startup instead of on the first
            # real prediction.
            torch.backends.cudnn.benchmark = True
        if use_torch_compile and sys.platform.startswith("win"):
            # torch.compile relies on triton, which is not available out of the box on Windows.
            warnings.warn(
                "torch.compile is not supported on Windows (triton is not available out of the "
                "box), forcing use_torch_compile=False."
            )
            use_torch_compile = False
        if use_torch_compile and device.type != "cuda":
            # This network is convolution-dominated, so on CPU almost all time is spent inside
            # oneDNN/MKLDNN conv kernels. torch.compile's wins (pointwise fusion, lower dispatch
            # overhead) are marginal there, while the compile itself runs on the same CPU cores
            # and adds substantial startup latency. Not worth it, so disable it.
            warnings.warn(
                f"torch.compile provides little benefit on '{device.type}' (this network is "
                "convolution-bound, so most time is spent in conv kernels rather than in fusable "
                "ops) while adding significant compile-time overhead, forcing "
                "use_torch_compile=False."
            )
            use_torch_compile = False
        self.use_torch_compile = use_torch_compile
        self.interactions_storage = interactions_storage
        # Concrete backend ("blosc2"/"tensor") resolved per image in _initialize_interactions.
        self._interactions_storage_resolved: Optional[str] = None
        self.interaction_decay = None
        self.current_interaction_intensity: float = 1.0
        self._fp16_max_value = float(torch.finfo(torch.float16).max)
        # Keep renormalized interaction magnitudes around 1/10 of fp16 max to preserve headroom.
        self._interaction_renorm_target = self._fp16_max_value / 10
        self.num_interaction_channels: int = None
        self.supported_interactions: dict = {}
        self.channel_mapping: dict = {}
        self.supports_initial_label: bool = True
        self.supports_zero_shot_label_refinement: bool = True
        # License of the loaded model checkpoint. Set when the model is loaded
        # (read from the LICENSE file in the checkpoint folder, or derived for
        # legacy checkpoints without one). Exposed so GUIs can display it once
        # the session is initialized. "!!MISSING!!" means the license is unknown.
        self.license: Optional[str] = None

        # image specific
        self.interactions = None  # blosc2.NDArray or dense torch.Tensor (see interactions_storage)
        # Reusable, pre-faulted float16 buffer to decompress blosc2 interaction crops into (Path B).
        # Allocated per image in _initialize_interactions; None for the dense-tensor backend.
        self._interactions_read_buffer = None
        self.preprocessed_image: torch.Tensor = None
        self.target_buffer: Union[np.ndarray, torch.Tensor] = None
        # Bbox (in original-image coordinates) of the most recent target_buffer write.
        # Captured inside _paste_prediction_to_target_buffer so remote callers can
        # fetch just the touched region without diffing.
        self._last_paste_bbox: Optional[List[List[int]]] = None

        # Single-level undo. When disabled (enable_undo=False) nothing is ever recorded, so undo()
        # always returns False and none of the RAM/CPU cost is paid.
        # _undo_log records the most recent interaction (started by _begin_undo_log at the top of every
        # add_*_interaction): the scalar state before it plus the pre-image of every region it overwrote,
        # in write order. undo() restores the pre-images in reverse. None means there is nothing to undo.
        # See _begin_undo_log / _record_interactions_region / undo.
        self.supports_undo: bool = enable_undo
        self._undo_log: Optional[dict] = None
        # Interaction channels that may be nonzero (written since the interactions were last zeroed).
        # Operations that rewrite whole channels (renormalization, initial seg) only need to save these.
        self._dirty_channels: set = set()

        # this will be set when loading the model (initialize_from_trained_model_folder)
        self.pad_mode_data = self.preferred_scribble_thickness = self.point_interaction = None

        self.verbose = verbose

        self.do_autozoom: bool = do_autozoom

        torch.set_num_threads(min(torch_n_threads, cpu_count()))
        self.torch_n_threads = torch_n_threads

        self.original_image_shape = None

        self.new_interaction_zoom_out_factors: List[float] = []
        self.new_interaction_centers = []
        # Create a thread pool executor for background tasks.
        # this only takes care of preprocessing and interaction memory initialization so there is no need to give it
        # more than 2 workers
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preprocess_future = None
        self.interactions_future = None

    @staticmethod
    def _is_official_checkpoint(plans: dict, checkpoint: dict) -> bool:
        return (
            plans.get("dataset_name") == "Dataset225_nnInteractiveV2"
            and checkpoint.get("init_args", {}).get("configuration") == "3d_fullres_ps192_bs24"
        )

    @classmethod
    def _load_license(cls, model_training_output_dir: str, plans: dict, checkpoint: dict) -> str:
        """Determine the license of the model being loaded.

        Reads the ``LICENSE`` file from the checkpoint folder if present.
        Expected format: the FIRST line is a short license identifier (e.g.
        ``CC BY-NC-SA 4.0``); any following lines (URL, full text, …) are for
        human readers and are ignored. Only the first non-empty line is
        returned, so ``self.license`` stays a short, displayable string.

        If the folder has no ``LICENSE`` file it is most likely a legacy model:
        the official v1 checkpoint is CC BY-NC-SA 4.0, anything else is reported
        as ``"!!MISSING!!"`` so callers (e.g. GUIs) can flag the unknown license.
        """
        license_file = join(model_training_output_dir, "LICENSE")
        if isfile(license_file):
            with open(license_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        return line
        if cls._is_official_checkpoint(plans, checkpoint):
            return "CC BY-NC-SA 4.0"
        return "!!MISSING!!"

    def _legacy_default_capability(self) -> dict:
        return {
            "supported_interactions": {
                "scribble": True,
                "lasso": True,
                "points": True,
                "bbox2d": True,
                "bbox3d": False,
            },
            "supports_initial_label": True,
            "supports_zero_shot_label_refinement": True,
            "interaction_channels": 6,
            "channel_mapping": {
                "prev_seg": 0,
                "bbox2d": (1, 2),
                "bbox3d": (1, 2),
                "lasso": (1, 2),
                "points": (3, 4),
                "scribble": (5, 6),
            },
        }

    def _to_positive_channel_index(self, idx: int) -> int:
        return idx if idx >= 0 else self.num_interaction_channels + idx

    def _resolve_channel_pair(self, channel_name: str, override_capability_checks: bool) -> Tuple[int, int]:
        if channel_name in self.channel_mapping:
            return parse_channel_pair(channel_name, self.channel_mapping[channel_name])
        if override_capability_checks:
            warnings.warn(
                f"Interaction '{channel_name}' was forced but no channel mapping exists in capability metadata.",
                RuntimeWarning,
            )
        raise ValueError(f"Interaction '{channel_name}' cannot be executed because no channel mapping was found.")

    def _is_interaction_supported(self, interaction_name: str) -> bool:
        if interaction_name in self.SUPPORTED_INTERACTION_KEYS:
            return bool(self.supported_interactions.get(interaction_name, False))
        if interaction_name == "initial_label":
            return bool(self.supports_initial_label)
        return False

    def _get_prev_seg_channel(self) -> int:
        return int(self.channel_mapping["prev_seg"])

    @staticmethod
    def _clip_bbox_to_shape(bbox: List[List[int]], spatial_shape: Tuple[int, ...]) -> Optional[List[List[int]]]:
        clipped = [[max(0, int(lb)), min(int(ub), int(s))] for (lb, ub), s in zip(bbox, spatial_shape)]
        if any(ub <= lb for lb, ub in clipped):
            return None
        return clipped

    @staticmethod
    def _bbox_size(bbox: List[List[int]]) -> List[int]:
        return [int(ub - lb) for lb, ub in bbox]

    @staticmethod
    def _union_bboxes(*bboxes: Optional[List[List[int]]]) -> Optional[List[List[int]]]:
        valid_bboxes = [bbox for bbox in bboxes if bbox is not None]
        if len(valid_bboxes) == 0:
            return None
        return [
            [min(bbox[dim][0] for bbox in valid_bboxes), max(bbox[dim][1] for bbox in valid_bboxes)]
            for dim in range(len(valid_bboxes[0]))
        ]

    @staticmethod
    def _offset_bboxes(local_bboxes: List[List[List[int]]], offset_bbox: List[List[int]]) -> List[List[List[int]]]:
        return [
            [[lb + offset_bbox[dim][0], ub + offset_bbox[dim][0]] for dim, (lb, ub) in enumerate(bbox)]
            for bbox in local_bboxes
        ]

    @staticmethod
    def _bbox_to_local(bbox: List[List[int]], frame_bbox: List[List[int]]) -> List[List[int]]:
        """Translate a global bbox into the local coordinates of ``frame_bbox`` (inverse of _offset_bboxes
        for a single bbox): subtract the frame's lower bound per axis."""
        return [[lb - f[0], ub - f[0]] for (lb, ub), f in zip(bbox, frame_bbox)]

    @staticmethod
    def _nonzero_spatial_bbox(
        tensor: torch.Tensor, sum_dtype: torch.dtype = torch.float32
    ) -> Optional[List[List[int]]]:
        """Half-open ``[lb, ub)`` bounding box of the nonzero region along the trailing 3 (spatial) axes.

        A spatial position counts as occupied when the sum over every other axis (any leading
        channel/batch dims plus the two remaining spatial axes) is non-zero. We project with per-axis
        sums instead of ``torch.nonzero`` / ``torch.where`` over the whole tensor: those materialize one
        index per nonzero voxel (or a full bool mask) and eat RAM/VRAM for breakfast on full-size volumes.

        The fp32 accumulator cannot overflow for realistic volumes (worst-case projection sum is ~30
        orders of magnitude below fp32's max) and benchmarks ~8x faster than fp64 for an identical result;
        the only theoretical risk is a slice whose nonzero voxels cancel to exactly 0.0, which does not
        happen for real images. Returns ``None`` if the tensor is entirely zero.
        """
        x_ax, y_ax, z_ax = tensor.ndim - 3, tensor.ndim - 2, tensor.ndim - 1
        leading = tuple(range(x_ax))  # channel/batch dims, reduced into every projection
        # Two full passes over the tensor instead of three: collapse the leading dims + X down to a small
        # [Y, Z] plane (~1 MB) and read the Y and Z marginals off it cheaply; X gets its own pass. Reducing
        # the outermost axes (for the plane) and the innermost block (for X) are torch's fast directions, so
        # this benchmarks ~1.8x faster than three independent marginal sums. The middle-axis Y/Z reductions
        # then run on the tiny plane rather than the full volume.
        yz_plane = tensor.sum(dim=(*leading, x_ax), dtype=sum_dtype)
        projections = {
            x_ax: tensor.sum(dim=(*leading, y_ax, z_ax), dtype=sum_dtype),
            y_ax: yz_plane.sum(dim=1),
            z_ax: yz_plane.sum(dim=0),
        }
        bbox = []
        for axis in (x_ax, y_ax, z_ax):
            nonzero = torch.where(projections[axis] != 0)[0]
            if nonzero.numel() == 0:
                return None
            bbox.append([int(nonzero.min()), int(nonzero.max()) + 1])
        return bbox

    def _compute_prev_seg_positive_bbox(self) -> Optional[List[List[int]]]:
        prev_seg_ch = self._get_prev_seg_channel()
        spatial_shape = tuple(int(i) for i in self.interactions.shape[1:])

        occupancy_x = np.zeros(spatial_shape[0], dtype=bool)
        occupancy_y = np.zeros(spatial_shape[1], dtype=bool)
        occupancy_z = np.zeros(spatial_shape[2], dtype=bool)
        chunk_depth = 64
        for d0 in range(0, spatial_shape[0], chunk_depth):
            d1 = min(spatial_shape[0], d0 + chunk_depth)
            slab = np.asarray(self.interactions[(prev_seg_ch, slice(d0, d1), slice(None), slice(None))]) > 0.5
            if not slab.any():
                continue
            occupancy_x[d0:d1] |= np.any(slab, axis=(1, 2))
            occupancy_y |= np.any(slab, axis=(0, 2))
            occupancy_z |= np.any(slab, axis=(0, 1))

        occupancies = (occupancy_x, occupancy_y, occupancy_z)
        bbox = []
        for occ in occupancies:
            indices = np.flatnonzero(occ)
            if len(indices) == 0:
                return None
            bbox.append([int(indices[0]), int(indices[-1]) + 1])
        return bbox

    def _get_dilation_channels_for_resample(self) -> List[int]:
        dilation_channels = set()
        # During zoom-out, point/scribble signals can disappear when area interpolation averages tiny sparse
        # structures away. We therefore dilate only these "thin prompt" channels before resampling.
        for key in ("points", "scribble"):
            if not self.supported_interactions.get(key, False):
                continue
            if key not in self.channel_mapping:
                continue
            pos_ch, neg_ch = parse_channel_pair(key, self.channel_mapping[key])
            dilation_channels.add(pos_ch)
            dilation_channels.add(neg_ch)
        # Use a sorted list so execution is deterministic and easier to reason about in debugging/logging.
        return sorted(dilation_channels)

    def _check_capability_or_warn(self, interaction_name: str, override_capability_checks: bool):
        if self._is_interaction_supported(interaction_name):
            return
        msg = f"Interaction '{interaction_name}' is not supported by this checkpoint capability metadata."
        if override_capability_checks:
            warnings.warn(f"{msg} Proceeding because override_capability_checks=True.", RuntimeWarning)
            return
        raise ValueError(msg)

    def _get_non_prev_seg_channels(self) -> List[int]:
        if self.interactions is None:
            return []
        prev_seg_channel = self._get_prev_seg_channel()
        channels = list(range(self.interactions.shape[0]))
        if prev_seg_channel in channels:
            channels.remove(prev_seg_channel)
        return channels

    def _renormalize_interactions_if_needed(self):
        """Rescale the stored interaction channels down before the growing intensity overflows fp16.

        ``current_interaction_intensity`` grows by ``1 / decay`` on every interaction (see
        ``_prepare_new_interaction_intensity``) and the channels are stored pre-multiplied by it, so left
        unchecked it would eventually exceed fp16's max (~65504) and saturate to inf. Once it crosses that
        threshold we divide every non-prev_seg channel by the current intensity and reset the running
        intensity to a safe target. All channels are scaled by the same factor, so their *relative*
        magnitudes — and thus the decay ordering — are preserved. This is the one decay-related operation
        that touches the whole interactions tensor, but it fires only rarely (on the order of every hundred
        interactions).
        """
        if self.interactions is None:
            return
        if self.current_interaction_intensity <= self._fp16_max_value:
            return
        channels_to_scale = self._get_non_prev_seg_channels()
        if len(channels_to_scale) == 0:
            self.current_interaction_intensity = min(
                self.current_interaction_intensity, self._interaction_renorm_target
            )
            return
        scale = self._interaction_renorm_target / self.current_interaction_intensity
        # fp16 scaling is not exactly invertible, so undo needs the channels' pre-image.
        self._record_interactions_channels(channels_to_scale)
        for ch in channels_to_scale:
            self.interactions[ch] *= scale
        self.current_interaction_intensity = self._interaction_renorm_target

    def _interactions_inplace_maximum(self, channel_idx: int, int_slicer, new_values) -> None:
        """In-place element-wise maximum for a subregion of a channel."""
        full_slicer = (channel_idx, *int_slicer)
        if isinstance(self.interactions, torch.Tensor):
            # Dense torch backend: operate in place without a numpy round-trip.
            self._record_interactions_region(channel_idx, int_slicer)
            if not isinstance(new_values, torch.Tensor):
                new_values = torch.as_tensor(new_values)
            view = self.interactions[full_slicer]
            torch.maximum(view, new_values.to(view.dtype), out=view)
            return
        if isinstance(new_values, torch.Tensor):
            new_values = new_values.cpu().numpy().astype(np.float16)
        current_sub = np.asarray(self.interactions[full_slicer])
        # the region is decompressed anyway for the read-modify-write: its pre-image is compressed from it
        self._record_interactions_region(channel_idx, int_slicer, current=current_sub)
        np.maximum(current_sub, new_values, out=current_sub)
        self.interactions[full_slicer] = current_sub

    def _write_interactions_channel(self, channel_idx: int, value) -> None:
        """Write a full channel. Handles torch→numpy for blosc2."""
        self._record_interactions_channel_overwrite(channel_idx)
        if isinstance(self.interactions, torch.Tensor):
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            self.interactions[channel_idx] = value.to(self.interactions.dtype)
            return
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy().astype(np.float16)
        self.interactions[channel_idx] = value

    def _to_device(self, tensor: torch.Tensor) -> torch.Tensor:
        """Copy a CPU tensor (typically a non-contiguous crop) to the compute device.

        Non-contiguous CPU crops go through the process-wide pinned staging buffer (utils/staging.py), which
        gathers and transfers them in fixed-size slabs; everything else is a plain ``.to``. NOTE: like ``.to``,
        this returns ``tensor`` itself (no copy) when it already lives on ``self.device`` (e.g. a CPU session), so
        never modify the result in place when ``tensor`` is a view of persistent state.
        """
        stager = get_stager(self.device) if tensor.device.type == "cpu" else None
        if stager is None:
            return tensor.to(self.device)
        return stager.copy_to(tensor)

    def _copy_into_device(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Write CPU tensor ``src`` into ``dst``, a (possibly strided) region of a tensor on the compute device.

        Goes through the pinned staging buffer slab by slab (PinnedStager.copy_into), so the device never holds a
        temporary copy of the whole region -- for multi-GB regions such as the refinement cache that temporary
        would otherwise be the VRAM peak. Falls back to a plain copy where pinned staging is unavailable."""
        stager = get_stager(self.device) if src.device.type == "cpu" else None
        if stager is None:
            dst.copy_(src.to(dst.device))
            return
        stager.copy_into(src, dst)

    def _blosc2_staging_available(self) -> bool:
        """blosc2 crops can be decompressed straight into the pinned staging buffer (CUDA, pinned memory usable,
        blosc2 build with decompress-into-buffer)."""
        return (
            not isinstance(self.interactions, torch.Tensor)
            and hasattr(self.interactions, "get_slice_numpy")
            and get_stager(self.device) is not None
        )

    def _blosc2_channel_to_device(
        self, channel: int, valid: list[list[int]], out: torch.Tensor | None = None
    ) -> torch.Tensor:
        """Decompress one channel of the blosc2 interactions over the in-image region ``valid`` ([[lb, ub]] * 3)
        directly into the pinned staging buffer and on to the device, slab by slab, overlapping decompression
        with the transfer (utils/staging.py, PinnedStager.fill_to). No host copy of the region is made. Returns a
        contiguous fp16 tensor of shape [ub - lb for each axis] on self.device, or writes into ``out`` if given
        (which may be a strided region of a larger tensor, e.g. the refinement cache)."""
        shape = [ub - lb for lb, ub in valid]
        if out is None:
            out = torch.empty(shape, dtype=torch.float16, device=self.device)
        lbs = [lb for lb, _ in valid]

        def fill(view: np.ndarray, lo: tuple, hi: tuple) -> None:
            # lo/hi: the block of ``out`` to produce, local to ``valid``
            start = (channel, *[lb + i for lb, i in zip(lbs, lo)])
            stop = (channel + 1, *[lb + i for lb, i in zip(lbs, hi)])
            self.interactions.get_slice_numpy(view[None], (start, stop))

        get_stager(self.device).fill_to(out, fill)
        return out

    def _interactions_region_into(self, channels: Tuple[int, int], valid: List[List[int]], dst: torch.Tensor) -> None:
        """Copy interaction channels ``channels[0]:channels[1]`` over the in-image region ``valid`` ([[lb, ub]] * 3)
        into ``dst`` (shape ``[n_channels, *region]``, on the compute device or the CPU, possibly a strided view
        such as the interior of a zero-padded buffer or a region of the refinement cache).

        The one place that reads interaction regions, for every backend:
        - dense tensor: the CPU view is copied into ``dst`` (through the pinned staging buffer for the device).
        - blosc2 to the device with pinned staging: each channel is decompressed straight into pinned memory and
          on to ``dst`` (_blosc2_channel_to_device), no host copy of the region.
        - blosc2 otherwise: one channel at a time, decompressed directly into ``dst`` where that is a contiguous
          CPU region, else into the reusable read buffer (when it fits) and copied from there.
        ``dst`` never aliases the stored interactions, so callers may modify it in place.
        """
        if dst.numel() == 0:
            return
        c0, c1 = channels
        spatial = bounding_box_to_slice(valid)
        to_compute_device = dst.device.type == self.device.type

        def copy(src: torch.Tensor, out: torch.Tensor) -> None:
            if to_compute_device:
                self._copy_into_device(src, out)
            else:
                out.copy_(src)

        if isinstance(self.interactions, torch.Tensor):
            copy(self.interactions[(slice(c0, c1), *spatial)], dst)
            return
        if to_compute_device and self._blosc2_staging_available():
            for i, c in enumerate(range(c0, c1)):
                self._blosc2_channel_to_device(c, valid, out=dst[i])
            return
        can_decompress_into = hasattr(self.interactions, "get_slice_numpy")
        shape = [ub - lb for lb, ub in valid]
        n = int(np.prod(shape, dtype=np.int64))
        for i, c in enumerate(range(c0, c1)):
            key = ((c, *[lb for lb, _ in valid]), (c + 1, *[ub for _, ub in valid]))
            out = dst[i]
            if can_decompress_into and out.device.type == "cpu" and out.is_contiguous():
                self.interactions.get_slice_numpy(out.numpy()[None], key)
                continue
            buffer = self._interactions_read_buffer
            if can_decompress_into and buffer is not None and n <= buffer.size:
                # a view into the shared read buffer: consumed by the (synchronous) copy before the next channel
                region = buffer[:n].reshape(1, *shape)
                self.interactions.get_slice_numpy(region, key)
                region = region[0]
            else:
                region = np.asarray(self.interactions[(c, *spatial)])
            copy(torch.from_numpy(region), out)

    def _read_interactions_region(self, channel: int, valid: List[List[int]]) -> torch.Tensor:
        """Interaction channel ``channel`` over the in-image region ``valid`` as a tensor on the compute device,
        for READ-ONLY use: for the dense backend on a CPU session it is a view of the stored interactions."""
        if isinstance(self.interactions, torch.Tensor) and self.interactions.device.type == self.device.type:
            return self.interactions[(channel, *bounding_box_to_slice(valid))]
        out = torch.empty([ub - lb for lb, ub in valid], dtype=torch.float16, device=self.device)
        self._interactions_region_into((channel, channel + 1), valid, out[None])
        return out

    def _paste_interactions(
        self, channel: int, source: torch.Tensor, bbox: List[List[int]], record_undo: bool = True
    ) -> None:
        """Paste ``source`` into interaction channel ``channel`` at ``bbox`` (may extend past the image; the
        in-image part is written). ``record_undo``: save the pre-image of the written region first. Only pass
        False when the caller has already recorded a region covering this write."""
        if record_undo:
            slicer = self._bbox_to_clipped_slicer(bbox, self.interactions.shape[1:])
            if slicer is not None:
                self._record_interactions_region(channel, slicer)
        paste_tensor(self.interactions, source, bbox, channel_idx=channel)

    def _paste_prediction_to_target_buffer(
        self, prediction: torch.Tensor, bbox: List[List[int]], record_undo: bool = True
    ) -> None:
        """Paste ``prediction`` into the target buffer at ``bbox`` (may extend past the buffer). ``record_undo``:
        save the pre-image of the written region first. Only pass False when the caller has already recorded a
        region covering this write."""
        if record_undo and self.target_buffer is not None:
            slicer = self._bbox_to_clipped_slicer(bbox, self.target_buffer.shape)
            if slicer is not None:
                self._record_target_region(slicer)
        # The target buffer shares the image's coordinate space (no cropping), so the bbox is used directly.
        if isinstance(self.target_buffer, torch.Tensor):
            pred_for_target = prediction.to(self.target_buffer.device)
        else:
            pred_for_target = prediction.to("cpu")
        paste_tensor(self.target_buffer, pred_for_target, bbox)
        self._last_paste_bbox = bbox

    def _clipped_last_paste_bbox(self) -> Optional[List[List[int]]]:
        """Return ``_last_paste_bbox`` clipped to the target buffer's spatial bounds, or None.

        ``_last_paste_bbox`` is stored unclipped: autozoom near an image edge can push the
        pasted region past the buffer bounds (and below 0). Callers that copy only the changed
        sub-region need valid, directly-sliceable indices, so we clip to ``[0, shape]`` per axis
        here without mutating the stored (unclipped) value the server relies on. Returns None if
        nothing was pasted or there is no target buffer.
        """
        bbox = self._last_paste_bbox
        if bbox is None or self.target_buffer is None:
            return None
        shape = self.target_buffer.shape
        return [[max(int(lb), 0), min(int(ub), int(shape[i]))] for i, (lb, ub) in enumerate(bbox)]

    def _estimate_refinement_cache_nbytes(self, cache_bbox: List[List[int]]) -> int:
        cache_voxels = int(np.prod(self._bbox_size(cache_bbox), dtype=np.int64))
        image_nbytes = cache_voxels * torch.empty((), dtype=self.preprocessed_image.dtype).element_size()
        interactions_nbytes = (
            cache_voxels * self.num_interaction_channels * torch.empty((), dtype=torch.float16).element_size()
        )
        return int(image_nbytes + interactions_nbytes)

    def _select_refinement_cache_device(self, cache_bbox: List[List[int]]) -> torch.device:
        if self.device.type != "cuda":
            return torch.device("cpu")

        cache_nbytes = self._estimate_refinement_cache_nbytes(cache_bbox)
        try:
            free_mem, _ = torch.cuda.mem_get_info(self.device)
            if free_mem - cache_nbytes >= self.REFINEMENT_CACHE_GPU_HEADROOM_BYTES:
                return self.device
        except Exception:
            pass

        return torch.device("cpu")

    @staticmethod
    def _zero_out_of_image_cache_border_(
        cache: torch.Tensor, cache_bbox: List[List[int]], spatial_shape: Tuple[int, ...]
    ) -> None:
        """Zero only the parts of an uninitialized (torch.empty) cache that lie outside the image.

        Refinement bboxes are not clipped to the image, so the cache can extend past the image
        bounds; those voxels act as zero-padding and are the only ones the in-image copies in
        _build_refinement_local_cache do not overwrite.
        Zeroing just this border instead of the whole cache saves a full memset over the
        (potentially multi-GB) cache.
        """
        lead = cache.ndim - len(cache_bbox)
        for d, (lb, ub) in enumerate(cache_bbox):
            size = int(ub - lb)
            left = min(max(int(-lb), 0), size)
            right = min(max(int(ub - spatial_shape[d]), 0), size)
            if left > 0:
                slicer = [slice(None)] * cache.ndim
                slicer[lead + d] = slice(0, left)
                cache[tuple(slicer)] = 0
            if right > 0:
                slicer = [slice(None)] * cache.ndim
                slicer[lead + d] = slice(size - right, size)
                cache[tuple(slicer)] = 0

    def _build_refinement_local_cache(self, bboxes_ordered: List[List[List[int]]]):
        cache_bbox = self._union_bboxes(*bboxes_ordered)
        cache_device = self._select_refinement_cache_device(cache_bbox)
        cache_shape = self._bbox_size(cache_bbox)

        # torch.empty + border-only zeroing: the in-image interior is fully overwritten by the
        # copies below. No pin_memory for a CPU cache: pinning multiple GB
        # costs seconds (cudaHostAlloc page-pinning) and would not even speed up the per-patch
        # reads, which are non-contiguous slices that cannot DMA directly from pinned memory --
        # they are staged through the small process-wide pinned buffer (utils/staging.py).
        cache_image = torch.empty(cache_shape, dtype=self.preprocessed_image.dtype, device=cache_device)
        cache_interactions = torch.empty(
            (self.num_interaction_channels, *cache_shape), dtype=torch.float16, device=cache_device
        )
        spatial_shape = tuple(int(i) for i in self.interactions.shape[1:])
        self._zero_out_of_image_cache_border_(cache_image, cache_bbox, spatial_shape)
        self._zero_out_of_image_cache_border_(cache_interactions, cache_bbox, spatial_shape)

        # A cache on the compute device is filled region by region straight from the pinned staging buffer
        # (copy_into): transferring the whole in-image region first and then copying it into the cache would hold
        # a second, equally large copy on the GPU (the cache-build VRAM peak).
        copy_into = self._copy_into_device if cache_device.type == self.device.type else None
        crop_and_pad_into_buffer(
            cache_image, cache_bbox, self.preprocessed_image[0], to_device=self._to_device, copy_into=copy_into
        )
        # The in-image region goes straight into the cache (for blosc2 one channel at a time; see
        # _interactions_region_into), so no temporary of the whole region is made on either side.
        valid = self._clip_bbox_to_shape(cache_bbox, spatial_shape)
        if valid is not None:
            tgt = bounding_box_to_slice(self._bbox_to_local(valid, cache_bbox))
            self._interactions_region_into(
                (0, self.num_interaction_channels), valid, cache_interactions[(slice(None), *tgt)]
            )
        # .type comparison: torch.device("cuda") != torch.device("cuda:0"), but both mean "the
        # compute device" here (_select_refinement_cache_device only returns self.device or cpu).
        # Must stay consistent with the same check in _refine_coarse_with_local_cache, which
        # normalizes per patch exactly when the cache is NOT normalized here.
        if cache_device.type == self.device.type:
            # Cheap on the compute device. A CPU cache is kept unnormalized instead -- in-place
            # fp16 division over the whole cache is slow on CPU -- and every patch is normalized
            # on the compute device after transfer (see _refine_coarse_with_local_cache).
            self._normalize_interaction_channels_for_network_(cache_interactions)
        return cache_bbox, cache_image, cache_interactions

    def _prepare_new_interaction_intensity(self):
        """Bump the intensity at which the *next* interaction is written — decay done cheaply.

        Older interactions should influence the prediction less than newer ones (``interaction_decay`` in
        ``(0, 1]``). Rather than multiply every existing interaction channel by ``decay`` on each new prompt
        — which would rewrite the whole (now full-size) interactions tensor every time — we do the inverse:
        keep a running ``current_interaction_intensity`` that grows by ``1 / decay`` per prompt and stamp
        each new interaction at that ever-larger value. The newest prompt therefore has the largest
        magnitude and older ones are relatively smaller, which is equivalent to decaying the old ones in
        place. The scaling is undone only when needed: ``_normalize_interaction_channels_for_network_``
        divides it back out just before the network sees a patch, and ``_renormalize_interactions_if_needed``
        rescales the stored channels before the unbounded intensity overflows fp16.
        """
        if self.interaction_decay is None:
            return
        if not (0 < self.interaction_decay <= 1):
            raise ValueError(f"interaction_decay must be in (0, 1], got {self.interaction_decay}.")
        if self.interaction_decay < 1:
            self.current_interaction_intensity *= 1 / self.interaction_decay
            self._renormalize_interactions_if_needed()

    def _normalize_interaction_channels_for_network_(self, interaction_tensor: torch.Tensor):
        """Undo the cumulative decay scaling on a *patch-sized* interaction crop before the network sees it.

        Interactions are stored pre-multiplied by the growing ``current_interaction_intensity`` (see
        ``_prepare_new_interaction_intensity``); dividing the non-prev_seg channels by it here maps the
        newest interaction back to ~1 and older ones to <1. Operates in place on the small per-patch crop,
        never on the full interactions tensor. prev_seg is excluded — it is a 0/1 mask, not a decayed
        interaction.
        """
        if interaction_tensor is None or self.current_interaction_intensity == 0:
            return
        if self.current_interaction_intensity == 1:
            return
        prev_seg_channel = self._get_prev_seg_channel()
        for ch in range(interaction_tensor.shape[0]):
            if ch != prev_seg_channel:
                interaction_tensor[ch] /= self.current_interaction_intensity

    def _load_capability_and_runtime_defaults(self, model_training_output_dir: str):
        capability_file = join(model_training_output_dir, "inference_info.json")
        legacy_file = join(model_training_output_dir, "inference_session_class.json")

        point_interaction_radius = 4
        preferred_scribble_thickness = [2, 2, 2]
        interaction_decay = 0.98
        pad_mode_data = "constant"
        capability_content = {}

        # Prefer modern capability metadata; fall back to legacy session metadata for older checkpoints.
        if isfile(capability_file):
            capability_content = load_json(capability_file)
            if not isinstance(capability_content, dict):
                raise RuntimeError(f"Invalid capability metadata in {capability_file}. Expected a JSON object.")
            self._validate_capability_version(capability_content)
            point_interaction_radius = capability_content.get("point_radius", point_interaction_radius)
            preferred_scribble_thickness = capability_content.get(
                "preferred_scribble_thickness", preferred_scribble_thickness
            )
            interaction_decay = capability_content.get("interaction_decay", interaction_decay)
            pad_mode_data = capability_content.get("pad_mode_image", pad_mode_data)
        elif isfile(legacy_file):
            legacy_content = load_json(legacy_file)
            if isinstance(legacy_content, str):
                interaction_decay = 0.9
            else:
                point_interaction_radius = legacy_content.get("point_radius", point_interaction_radius)
                preferred_scribble_thickness = legacy_content.get(
                    "preferred_scribble_thickness", preferred_scribble_thickness
                )
                interaction_decay = legacy_content.get("interaction_decay", interaction_decay)
                pad_mode_data = legacy_content.get("pad_mode_image", pad_mode_data)
        else:
            raise FileNotFoundError(
                f"Neither capability metadata ({capability_file}) nor legacy metadata ({legacy_file}) was found."
            )

        # Accept scalar thickness in metadata for backward compatibility.
        if not isinstance(preferred_scribble_thickness, (tuple, list)):
            preferred_scribble_thickness = [preferred_scribble_thickness] * 3

        return (
            capability_content,
            point_interaction_radius,
            preferred_scribble_thickness,
            interaction_decay,
            pad_mode_data,
        )

    def _apply_capability(self, capability: dict):
        default_capability = self._legacy_default_capability()
        default_supported = default_capability["supported_interactions"]
        default_mapping = default_capability["channel_mapping"]
        supported_keys = set(self.SUPPORTED_INTERACTION_KEYS)
        mapping_keys = set(self.SUPPORTED_INTERACTION_KEYS).union({"prev_seg"})

        raw_supported = capability.get("supported_interactions", {}) if isinstance(capability, dict) else {}
        unknown_supported = set(raw_supported.keys()) - supported_keys
        if len(unknown_supported) > 0:
            raise ValueError(
                f"Capability requests unsupported interactions: {sorted(unknown_supported)}. "
                f"Supported: {sorted(supported_keys)}"
            )
        filtered_supported = {k: bool(v) for k, v in raw_supported.items() if k in supported_keys}
        self.supported_interactions = {**default_supported, **filtered_supported}
        self.supports_initial_label = capability.get("supports_initial_label", True)
        self.supports_zero_shot_label_refinement = capability.get("supports_zero_shot_label_refinement", True)

        raw_mapping = capability.get("channel_mapping", {}) if isinstance(capability, dict) else {}
        unknown_mapping = set(raw_mapping.keys()) - mapping_keys
        if len(unknown_mapping) > 0:
            raise ValueError(
                f"Capability channel_mapping contains unsupported keys: {sorted(unknown_mapping)}. "
                f"Supported mapping keys: {sorted(mapping_keys)}"
            )
        self.channel_mapping = dict(default_mapping)
        for k, v in raw_mapping.items():
            if k == "prev_seg":
                self.channel_mapping[k] = int(v)
            else:
                self.channel_mapping[k] = parse_channel_pair(k, v)

        if "interaction_channels" in capability:
            self.num_interaction_channels = int(capability["interaction_channels"]) + 1
        else:
            self.num_interaction_channels = infer_num_interaction_channels_from_mapping(self.channel_mapping)

        # Normalize all channel indices to positive indexing once at load time so downstream code can
        # use direct indexing without handling negative-offset semantics repeatedly.
        self.channel_mapping["prev_seg"] = self._to_positive_channel_index(int(self.channel_mapping["prev_seg"]))
        for k, v in list(self.channel_mapping.items()):
            if k == "prev_seg":
                continue
            pos_ch, neg_ch = parse_channel_pair(k, v)
            self.channel_mapping[k] = (
                self._to_positive_channel_index(pos_ch),
                self._to_positive_channel_index(neg_ch),
            )

    def _validate_capability_version(self, capability: dict):
        current_class = self.__class__.__name__
        required_class = capability.get("inference_class", current_class)
        if required_class != current_class:
            raise RuntimeError(
                f"Checkpoint requires inference class '{required_class}', but current class is " f"'{current_class}'."
            )

        min_version = capability.get("inference_class_min_version")
        if min_version is None:
            return
        if version_to_tuple(min_version) > version_to_tuple(self.INFERENCE_SESSION_VERSION):
            raise RuntimeError(
                f"Checkpoint requires nnInteractiveInferenceSession>={min_version}, but current version is "
                f"{self.INFERENCE_SESSION_VERSION}. Please update nnInteractive."
            )

    def set_image(self, image: np.ndarray, image_properties: dict = None):
        """
        Image must be 4D to satisfy nnU-Net needs: [c, x, y, z]
        Offload the processing to a background thread.

        ``image_properties`` (e.g. spacing) is accepted for API compatibility but currently
        unused — nnInteractive operates purely in voxel space.
        """
        if image_properties is None:
            image_properties = {}
        self._reset_session()
        assert image.ndim == 4, f"expected a 4d image as input, got {image.ndim}d. Shape {image.shape}"
        if self.verbose:
            print(f"Initialize with raw image shape {image.shape}")

        # Offload all image preprocessing to a background thread.
        self.preprocess_future = self.executor.submit(self._background_set_image, image, image_properties)
        self.original_image_shape = image.shape

    def _finish_preprocessing_and_initialize_interactions(self):
        """
        Block until both the image preprocessing and the interactions tensor initialization
        are finished.
        """
        if self.preprocess_future is not None:
            # Wait for image preprocessing to complete.
            self.preprocess_future.result()
            self.preprocess_future = None

    def set_target_buffer(self, target_buffer: Union[np.ndarray, torch.Tensor]):
        """
        Must be 3d numpy array or torch.Tensor
        """
        if target_buffer.ndim != 3:
            raise ValueError(f"target_buffer must be 3D (shape [X, Y, Z]), got ndim={target_buffer.ndim}")
        self.target_buffer = target_buffer

    def set_do_autozoom(self, do_autozoom: bool):
        self.do_autozoom = do_autozoom

    def _reset_session(self):
        self.interactions_future = None
        self.preprocess_future = None

        self._undo_log = None
        self._dirty_channels = set()

        del self.preprocessed_image
        del self.target_buffer
        del self.interactions
        self.preprocessed_image = None
        self.target_buffer = None
        self.interactions = None
        self.current_interaction_intensity = 1.0
        empty_cache(self.device)
        self.original_image_shape = None
        self._last_paste_bbox = None

    def _resolve_interactions_storage(self, spatial_shape) -> str:
        """Resolve the configured storage to a concrete backend ("blosc2" or "tensor").

        For "auto", pick "tensor" for images with at most AUTO_TENSOR_MAX_VOXELS spatial voxels
        (lower per-access overhead) and "blosc2" for larger ones (to bound RAM).
        """
        if self.interactions_storage != "auto":
            return self.interactions_storage
        n_voxels = int(np.prod(spatial_shape, dtype=np.int64))
        return "blosc2" if n_voxels > self.AUTO_TENSOR_MAX_VOXELS else "tensor"

    def _new_interactions_array(self, shape, compression_nthreads: int):
        """Allocate a zeroed interaction array using the resolved backend.

        "tensor" selects a dense CPU float16 torch.Tensor (more RAM, lower per-access
        overhead); "blosc2" uses a compact blosc2 in-memory NDArray.
        """
        if self._interactions_storage_resolved == "tensor":
            # Deliberately NOT pinned. Only crops of this tensor ever go to the GPU, and those are
            # non-contiguous views that cannot DMA from pinned memory directly; they are staged through a
            # small fixed pinned buffer instead (utils/staging.py). Pinning the whole tensor bought no
            # speed but cost RAM: torch's caching host allocator rounds pinned blocks up to a power of two
            # and never returns freed blocks to the OS, so RAM grew with every image of a new size.
            return torch.zeros(shape, dtype=torch.float16, device="cpu")
        return blosc2.zeros(
            shape,
            dtype=np.float16,
            chunks=(1, *[min(64, s) for s in shape[1:]]),
            blocks=(1, *[min(32, s) for s in shape[1:]]),
            cparams=self._blosc2_cparams(compression_nthreads),
            # Decompression of this sparse interaction tensor is fastest single-threaded:
            # blosc2's per-chunk thread sync costs more than it saves here, badly so on
            # many-core/many-CCD servers (see benchmarks). Multithreading only hurts.
            dparams={"nthreads": 1},
        )

    def _initialize_interactions(self, image_torch: torch.Tensor):
        shape = (self.num_interaction_channels, *image_torch.shape[1:])
        self._interactions_storage_resolved = self._resolve_interactions_storage(shape[1:])
        via_auto = self.interactions_storage == "auto"
        if self.verbose or via_auto:
            backend = (
                "dense torch.Tensor"
                if self._interactions_storage_resolved == "tensor"
                else "blosc2 in-memory compression"
            )
            print(f"Initialize interactions with {backend}{' (auto)' if via_auto else ''}")
        self.interactions = self._new_interactions_array(shape, min(self.torch_n_threads, os.cpu_count()))
        self._dirty_channels = set()
        self._interactions_shape = shape
        self._interactions_read_buffer = self._new_interactions_read_buffer(shape)

    def _new_interactions_read_buffer(self, shape) -> Optional[np.ndarray]:
        """Pre-faulted buffer to decompress blosc2 interaction crops into (Path B), or None.

        Interaction crops are read one channel at a time (_interactions_region_into), so the buffer holds ONE
        channel of the largest network-input crop: the patch size scaled by the maximum autozoom factor, capped to
        the image size. Only allocated for the blosc2 backend that exposes the decompress-into-buffer method and
        cannot decompress straight into pinned staging memory; the dense-tensor backend returns views and needs
        no buffer.
        """
        if self._interactions_storage_resolved != "blosc2":
            return None
        if self._blosc2_staging_available():
            # blosc2 crops are decompressed straight into the pinned staging buffer (_blosc2_channel_to_device)
            return None
        if not hasattr(self.interactions, "get_slice_numpy"):
            print(
                "WARNING: this blosc2 build has no NDArray.get_slice_numpy; cannot reuse a "
                "decompression buffer for interaction crops. Falling back to a fresh allocation on "
                "every read (slower). Consider updating blosc2."
            )
            return None
        max_valid = [
            min(round(p * self.MAX_AUTOZOOM_FACTOR), s)
            for p, s in zip(self.configuration_manager.patch_size, shape[1:])
        ]
        n = int(np.prod(max_valid, dtype=np.int64))
        buffer = np.empty(n, dtype=np.float16)
        buffer[:] = 0  # first-touch the pages once, up front
        return buffer

    @torch.inference_mode()
    def _background_set_image(self, image: np.ndarray, image_properties: dict):
        # Convert to a float32 torch tensor with exactly one copy (the in-place normalization
        # below must never mutate the caller's array). ascontiguousarray converts dtype and
        # fixes layout in a single pass; only when it was a no-op (already contiguous float32)
        # is an explicit copy needed. The old `image.copy()` + `.float()` copied twice for
        # non-float32 inputs (e.g. int16 CT), a transient full-volume RAM spike.
        # Check memory overlap, not identity: for ndarray subclasses (np.memmap) or a
        # non-canonical float32 dtype (e.g. explicit '<f4' from MedVol/napari-nifti),
        # ascontiguousarray returns a *new view* of the same buffer, and normalizing that
        # in place would overwrite the caller's image (e.g. the napari layer turns grey).
        image_np = np.ascontiguousarray(image, dtype=np.float32)
        if np.may_share_memory(image_np, image):
            image_np = image_np.copy()
        image = torch.from_numpy(image_np)

        # The image is intentionally NOT cropped: interactions, predictions and the target buffer all
        # live in the original image's coordinate space (so the previously unreachable zero-valued border
        # region can be segmented too). We still locate the nonzero region, but only to compute the
        # normalization statistics over it, so that mean/std match what the model saw during training
        # (nnU-Net normalizes after cropping to nonzero).
        if self.verbose:
            print("Locating nonzero region for normalization statistics")
        # Sum-project to find the nonzero region rather than torch.where over the whole image (see
        # _nonzero_spatial_bbox; torch.where "eats RAM/VRAM for breakfast").
        spatial_bbox = self._nonzero_spatial_bbox(image)
        if spatial_bbox is None:
            raise ValueError("Input image is entirely zero; cannot determine normalization statistics.")
        # Channel slice fixed to channel 0: normalization statistics are taken from the first channel only.
        bbox = [[0, 1], *spatial_bbox]
        empty_cache(self.device)

        # Start initializing the interaction tensor (full image shape) in its own thread.
        self.interactions_future = self.executor.submit(self._initialize_interactions, image)

        # Compute normalization statistics over the nonzero region only, then normalize the FULL image
        # with them. Zero-valued border voxels become a constant (0 - mean) / std, like interior background.
        if self.verbose:
            print("Normalizing image using statistics from the nonzero region")
        slicer = bounding_box_to_slice(bbox)  # Assuming this returns a tuple of slices.
        crop = image[slicer]
        mean = crop.mean()
        std = crop.std()
        del crop
        image -= mean
        image /= std

        self.preprocessed_image = image.to("cpu")

        # we need to wait for this here I believe
        self.interactions_future.result()
        self.interactions_future = None

    def reset_interactions(self, _preserve_undo: bool = False):
        """
        Use this to reset all interactions and start from scratch for the current image. This includes the initial
        segmentation!

        _preserve_undo is an internal flag: add_initial_seg_interaction() resets interactions as part of
        applying the new seg, but the undo log it just started (holding the pre-reset state) must survive so
        that interaction remains undoable. Public callers must not set it.
        """
        if not _preserve_undo:
            self._undo_log = None
        if self.interactions is not None:
            if isinstance(self.interactions, torch.Tensor):
                # Same image -> same shape, so reuse the existing dense buffer and just zero it
                # instead of reallocating.
                self.interactions.zero_()
            else:
                del self.interactions
                self.interactions = self._new_interactions_array(self._interactions_shape, os.cpu_count())
        self._dirty_channels = set()
        self.current_interaction_intensity = 1.0

        if self.target_buffer is not None:
            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer.fill(0)
            elif isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer.zero_()
        self._last_paste_bbox = None
        empty_cache(self.device)

    def _blosc2_cparams(self, nthreads: Optional[int] = None) -> dict:
        """LZ4/NOFILTER compression params shared by the live interaction array
        (_new_interactions_array) and the undo pre-images (_compress_pre_image).
        Interactions compress better with NOFILTER, which is also faster than SHUFFLE."""
        return {
            "codec": blosc2.Codec.LZ4,
            # Level 1, not 5: the interactions tensor is mostly zeros, so the
            # compressed size is unchanged (0.4 MB either way on a 2.9 GB
            # 8x694x512x512 fp16 tensor, measured) while compression drops
            # 271 ms -> 80 ms.
            "clevel": 1,
            "filters": [blosc2.Filter.NOFILTER],
            "nthreads": min(self.torch_n_threads, os.cpu_count()) if nthreads is None else nthreads,
        }

    # ------------------------------- undo --------------------------------- #
    # Single-level undo via an undo log: every add_*_interaction starts a fresh log (_begin_undo_log), and
    # each write to the interactions or the target buffer first saves the pre-image of the region it is
    # about to overwrite. undo() restores those pre-images in reverse order, which reproduces the state
    # before the interaction exactly. Compared with snapshotting the whole state after every prediction,
    # the cost scales with the size of the change, not the image. Global operations (renormalization,
    # initial seg) save whole channels, but only the ones that can be nonzero (_dirty_channels).

    def _begin_undo_log(self) -> None:
        """Start the undo log of a new interaction, discarding the previous one. Called at the top of every
        add_*_interaction, before any state is mutated. No-op when undo is disabled."""
        if not self.supports_undo:
            # Also drop a log left over from before undo was disabled, so it neither grows nor stays undoable.
            self._undo_log = None
            return
        self._undo_log = {
            "current_interaction_intensity": self.current_interaction_intensity,
            "dirty_channels": set(self._dirty_channels),
            "queued": (list(self.new_interaction_centers), list(self.new_interaction_zoom_out_factors)),
            "records": [],
        }

    @contextmanager
    def _undoable_interaction(self):
        """Scope of one add_*_interaction (entered after its input validation): starts the interaction's undo log
        and makes the interaction atomic. If it raises (e.g. an OOM during the prediction), everything it
        already changed (including the prediction queue) is rolled back from its undo log and the previous
        interaction's undo log and changed-region bbox are reinstated, so the failed call leaves the session as it
        was. Without undo (enable_undo=False) nothing is recorded and a failed call is not rolled back."""
        previous_log = self._undo_log
        last_paste_bbox = self._last_paste_bbox
        self._begin_undo_log()
        try:
            yield
        except BaseException:
            if self._undo_log is not None:
                try:
                    self.undo()
                except BaseException:
                    # The rollback failed as well: the state is inconsistent, so there is nothing safe to undo.
                    self._undo_log = None
                    raise
                self._undo_log = previous_log
                self._last_paste_bbox = last_paste_bbox
            raise

    def _compress_pre_image(self, region):
        """Compressed copy (blosc2, in RAM) of an array region (numpy array or torch view) for the undo log. A CPU
        region is compressed straight from the (possibly non-contiguous) view,
        chunk by chunk, so no full-size temporary is made. A region on the GPU is copied to the host and
        compressed too: a pre-image held in VRAM would stay allocated until the next interaction and eat into the
        headroom of the next prediction."""
        if isinstance(region, torch.Tensor):
            region = region.cpu().numpy()
        return blosc2.asarray(np.asarray(region), cparams=self._blosc2_cparams(), dparams={"nthreads": 1})

    @staticmethod
    def _decompress_pre_image(data):
        return data[:]

    @staticmethod
    def _bbox_to_clipped_slicer(bbox: List[List[int]], spatial_shape) -> Optional[tuple]:
        """Slicer of ``bbox`` clipped to ``spatial_shape`` (the region paste_tensor writes), or None."""
        clipped = nnInteractiveInferenceSession._clip_bbox_to_shape(bbox, spatial_shape)
        return None if clipped is None else bounding_box_to_slice(clipped)

    def _record_interactions_region(self, channel: int, spatial_slicer: tuple, current=None) -> None:
        """Save the pre-image of ``interactions[channel][spatial_slicer]`` before it is written.
        ``spatial_slicer`` must lie inside the image (as the write sites guarantee). ``current``: the region's
        current values if the caller already holds them (a blosc2 read-modify-write), so they are not
        decompressed a second time."""
        self._dirty_channels.add(channel)
        if self._undo_log is None:
            return
        spatial_slicer = tuple(spatial_slicer)
        if any(len(range(*sl.indices(int(n)))) == 0 for sl, n in zip(spatial_slicer, self.interactions.shape[1:])):
            return  # nothing will be written
        key = (channel, *spatial_slicer)
        if current is not None:
            pre_image = self._compress_pre_image(current)
        elif isinstance(self.interactions, torch.Tensor):
            pre_image = self._compress_pre_image(self.interactions[key])
        else:
            # blosc2: NDArray.slice builds the compressed copy chunk by chunk (whole chunks are copied without
            # recompression), instead of decompressing the whole region into one host temporary first.
            pre_image = self.interactions.slice(key, cparams=self._blosc2_cparams(), dparams={"nthreads": 1})
        self._undo_log["records"].append(("interactions", channel, spatial_slicer, pre_image))

    def _record_interactions_channels(self, channels: List[int], array_replaced: bool = False) -> None:
        """Save the pre-image of whole interaction channels before a global operation rewrites them. Channels
        that cannot be nonzero are skipped. ``array_replaced``: the operation swaps in a new blosc2 array
        instead of writing into the current one (reset_interactions), so the current one can be kept as is."""
        if self._undo_log is None:
            return
        if not isinstance(self.interactions, torch.Tensor) and array_replaced:
            # blosc2, array about to be swapped out: keep the current one itself (free; covers all channels).
            self._undo_log["records"].append(("interactions_array", None, None, self.interactions))
            return
        channels = [c for c in channels if c in self._dirty_channels]
        if len(channels) == 0:
            return
        if not isinstance(self.interactions, torch.Tensor):
            # blosc2: a copy of the compressed array is far cheaper than decompressing channels.
            self._undo_log["records"].append(("interactions_array", None, None, self.interactions.copy()))
            return
        for c in channels:
            self._record_interactions_region(c, (slice(None),) * (self.interactions.ndim - 1))

    def _record_interactions_channel_overwrite(self, channel: int) -> None:
        """Save the pre-image of a whole channel before it is overwritten. A channel that cannot be nonzero
        gets a marker instead of a copy (undo zeroes it)."""
        if self._undo_log is not None and channel not in self._dirty_channels:
            self._dirty_channels.add(channel)
            self._undo_log["records"].append(("zero_channel", channel, None, None))
            return
        self._record_interactions_region(channel, (slice(None),) * (self.interactions.ndim - 1))

    def _record_target_region(self, spatial_slicer: tuple) -> None:
        """Save the pre-image of ``target_buffer[spatial_slicer]`` before it is written."""
        if self._undo_log is None or self.target_buffer is None:
            return
        spatial_slicer = tuple(spatial_slicer)
        region = self.target_buffer[spatial_slicer]
        if 0 in region.shape:
            return  # nothing will be written
        self._undo_log["records"].append(("target", None, spatial_slicer, self._compress_pre_image(region)))

    def _restore_undo_record(self, record) -> None:
        kind, channel, spatial_slicer, pre_image = record
        if kind == "interactions_array":
            self.interactions = pre_image
            return
        if kind == "zero_channel":
            if isinstance(self.interactions, torch.Tensor):
                self.interactions[channel].zero_()
            else:
                self.interactions[channel] = 0
            return
        if kind == "target":
            dst, slicer = self.target_buffer, spatial_slicer
        else:
            dst, slicer = self.interactions, (channel, *spatial_slicer)
        if not isinstance(dst, torch.Tensor):
            # numpy target buffer or blosc2 interactions
            dst[slicer] = self._decompress_pre_image(pre_image)
            return
        view = dst[slicer]
        if view.device.type == "cpu" and view.is_contiguous() and hasattr(pre_image, "get_slice_numpy"):
            # e.g. a whole dense channel: decompress straight into it, no temporary
            view_np = view.numpy()
            pre_image.get_slice_numpy(view_np, ((0,) * view_np.ndim, tuple(view_np.shape)))
        else:
            values = self._decompress_pre_image(pre_image)
            if not isinstance(values, torch.Tensor):
                values = torch.from_numpy(values)
            view.copy_(values)

    @staticmethod
    def _slicer_to_bbox(spatial_slicer: tuple, spatial_shape) -> List[List[int]]:
        return [list(sl.indices(int(s))[:2]) for sl, s in zip(spatial_slicer, spatial_shape)]

    def _diff_bbox(self, current, restored) -> Optional[List[List[int]]]:
        """Bounding box (local to the arrays) of voxels that differ between two equally shaped arrays, so
        undo can ship just the changed region. None if identical."""
        if isinstance(current, torch.Tensor):
            current = current.detach().cpu().numpy()
        if isinstance(restored, torch.Tensor):
            restored = restored.detach().cpu().numpy()
        diff = current != restored
        # Axis projections instead of np.where: np.where materializes 3 int64 index arrays with
        # one entry per differing voxel just to take min/max; np.any projections yield the same
        # bbox from three tiny 1D arrays (same trick as _nonzero_spatial_bbox).
        bbox = []
        for ax in range(diff.ndim):
            other_axes = tuple(i for i in range(diff.ndim) if i != ax)
            nz = np.flatnonzero(np.any(diff, axis=other_axes))
            if nz.size == 0:
                return None  # no differing voxels
            bbox.append([int(nz[0]), int(nz[-1]) + 1])
        return bbox

    def undo(self) -> bool:
        """Revert the most recent interaction, restoring the session to its prior state.

        Single level: only the last interaction can be undone. Returns True if something was undone,
        False if there was nothing to undo. After undo, the (now current) state becomes undoable again
        only once a new interaction is added. Always returns False when the session was created with
        enable_undo=False (nothing is recorded in that case).
        """
        # When undo is disabled, no log is ever started, so this check also covers that case.
        if self._undo_log is None:
            return False
        log = self._undo_log
        self._undo_log = None
        records = log["records"]

        # Remote callers fetch just the changed region via _last_paste_bbox: diff the target buffer before
        # and after the restore, over the hull of the restored target regions (nothing else changes).
        target_hull = None
        if self.target_buffer is not None:
            spatial_shape = self.target_buffer.shape
            target_hull = self._union_bboxes(
                *(self._slicer_to_bbox(r[2], spatial_shape) for r in records if r[0] == "target")
            )
        if target_hull is not None:
            hull_slicer = bounding_box_to_slice(target_hull)
            before = self.target_buffer[hull_slicer]
            before = before.clone() if isinstance(before, torch.Tensor) else before.copy()

        # A whole-array record restores every interaction channel, so the interaction records written after
        # the earliest one would only be overwritten again: skip them.
        first_array = next((i for i, r in enumerate(records) if r[0] == "interactions_array"), len(records))
        for i in reversed(range(len(records))):
            if i > first_array and records[i][0] != "target":
                continue
            self._restore_undo_record(records[i])

        diff_bbox = None
        if target_hull is not None:
            local_diff = self._diff_bbox(before, self.target_buffer[hull_slicer])
            del before
            if local_diff is not None:
                diff_bbox = self._offset_bboxes([local_diff], target_hull)[0]

        self.current_interaction_intensity = log["current_interaction_intensity"]
        # A superset is safe: restored channels may be nonzero again.
        self._dirty_channels |= log["dirty_channels"]
        # Predictions still pending before the undone interaction stay pending (their prompts are still there).
        centers, zoom_out_factors = log["queued"]
        self.new_interaction_centers, self.new_interaction_zoom_out_factors = list(centers), list(zoom_out_factors)
        self._last_paste_bbox = diff_bbox
        del log, records
        empty_cache(self.device)
        return True

    def add_bbox_interaction(
        self,
        bbox_coords,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ) -> Optional[List[List[int]]]:
        # sanity check
        raw_bbox_size = [i[1] - i[0] for i in bbox_coords]
        if any([i == 0 for i in raw_bbox_size]):
            raise ValueError(f"Given bounding box size is zero in at least one dimension: {bbox_coords}")

        # capability check
        dims_with_size_one = sum(i == 1 for i in raw_bbox_size)
        # if we do not support 3D bboxes we need to reject 3D bboxes!
        if not self._is_interaction_supported("bbox3d") and dims_with_size_one == 0:
            raise ValueError(
                f"The given bounding box {bbox_coords} has size {raw_bbox_size} indicating a 3D "
                f"bounding box. This is not supported by the loaded model checkpoint."
            )
        # a 2D bounding box is in principle a 3D box as well. Since 2D bboxes work better, we prefer to use a given
        # bbox as 2d if possible (sized 1 in at least one dim and bbox2d supported)
        bbox_kind = "bbox2d" if (dims_with_size_one >= 1 and self._is_interaction_supported("bbox2d")) else "bbox3d"
        self._check_capability_or_warn(bbox_kind, override_capability_checks)
        bbox_pos_channel, bbox_neg_channel = self._resolve_channel_pair(bbox_kind, override_capability_checks)

        self._finish_preprocessing_and_initialize_interactions()
        with self._undoable_interaction():
            return self._add_validated_bbox_interaction(
                bbox_coords, bbox_pos_channel if include_interaction else bbox_neg_channel, run_prediction
            )

    def _add_validated_bbox_interaction(
        self, bbox_coords, channel: int, run_prediction: bool
    ) -> Optional[List[List[int]]]:
        # Coordinates are already in the image's coordinate space (no cropping).
        transformed_bbox_coordinates = [[round(i[0]), round(i[1])] for i in bbox_coords]

        if self.verbose:
            print(f"Adding bounding box coordinates: {transformed_bbox_coordinates}")

        # Clip bbox to valid interaction volume and guarantee at least one voxel extent per axis.
        image_shape = self.preprocessed_image.shape  # Assuming shape is (C, H, W, D) or similar

        for dim in range(len(transformed_bbox_coordinates)):
            transformed_start, transformed_end = transformed_bbox_coordinates[dim]

            # Clip to image boundaries
            transformed_start = max(0, transformed_start)
            transformed_end = min(image_shape[dim + 1], transformed_end)  # +1 to skip channel dim

            # Ensure the bounding box does not collapse to a single point
            if transformed_end <= transformed_start:
                if transformed_start == 0:
                    transformed_end = min(1, image_shape[dim + 1])
                else:
                    transformed_start = max(transformed_start - 1, 0)

            transformed_bbox_coordinates[dim] = [transformed_start, transformed_end]

        if self.verbose:
            print(
                f"Bbox coordinates after clip to image boundaries and preventing dim collapse:\n"
                f"Bbox: {transformed_bbox_coordinates}\n"
                f"Internal image shape: {self.preprocessed_image.shape}"
            )

        self._add_patch_for_bbox_interaction(transformed_bbox_coordinates)

        self._prepare_new_interaction_intensity()

        # place bbox (clipped to the image above)
        slicer = bounding_box_to_slice(transformed_bbox_coordinates)
        self._record_interactions_region(channel, slicer)
        self.interactions[(channel, *slicer)] = self.current_interaction_intensity

        if run_prediction:
            return self._predict()
        return None

    def add_point_interaction(
        self,
        coordinates: Tuple[int, ...],
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ) -> Optional[List[List[int]]]:
        self._check_capability_or_warn("points", override_capability_checks)
        point_pos_channel, point_neg_channel = self._resolve_channel_pair("points", override_capability_checks)
        self._finish_preprocessing_and_initialize_interactions()
        with self._undoable_interaction():
            # Coordinates are already in the image's coordinate space (no cropping).
            rounded_coordinates = [round(i) for i in coordinates]

            self._add_patch_for_point_interaction(rounded_coordinates)

            self._prepare_new_interaction_intensity()

            interaction_channel = point_pos_channel if include_interaction else point_neg_channel
            self.point_interaction.place_point(
                rounded_coordinates,
                self.interactions,
                channel_idx=interaction_channel,
                intensity_scale=self.current_interaction_intensity,
                before_write=lambda target_slices, current: self._record_interactions_region(
                    target_slices[0], target_slices[1:], current=current
                ),
            )
            if run_prediction:
                return self._predict()
            return None

    def _add_image_interaction(
        self,
        image: np.ndarray,
        interaction_channel: int,
        run_prediction: bool,
        interaction_bbox: Optional[List[List[int]]],
    ) -> Optional[List[List[int]]]:
        if interaction_bbox is None:
            interaction_bbox = [[0, s] for s in self.original_image_shape[1:]]

        # User-input validation raises ValueError (not assert) so it survives python -O and
        # maps to a clean 400 on the server.
        if len(interaction_bbox) != 3:
            raise ValueError(f"interaction_bbox must have 3 dimensions, got {len(interaction_bbox)}")
        bbox_size = [ub - lb for lb, ub in interaction_bbox]
        if not all(s > 0 for s in bbox_size):
            raise ValueError("each dimension of interaction_bbox must have positive size")
        if list(image.shape) != bbox_size:
            raise ValueError(f"image shape {list(image.shape)} must match interaction_bbox size {bbox_size}")
        if not all(
            lb >= 0 and ub <= orig_dim for (lb, ub), orig_dim in zip(interaction_bbox, self.original_image_shape[1:])
        ):
            raise ValueError(
                f"interaction_bbox {interaction_bbox} exceeds original image bounds "
                f"{list(self.original_image_shape[1:])}"
            )

        self._finish_preprocessing_and_initialize_interactions()
        with self._undoable_interaction():
            # interaction_bbox is already in the image's coordinate space (no cropping), and the checks above
            # guarantee it lies fully within the interaction volume, so we write it directly at its bounds.
            lbs = [ib[0] for ib in interaction_bbox]

            image_t = torch.from_numpy(image)
            self._generic_add_patch_from_image(image_t, offset=lbs)

            self._prepare_new_interaction_intensity()

            int_slicer = bounding_box_to_slice(interaction_bbox)
            # Convert to fp16 before scaling: multiplying the (typically uint8) mask by a Python float
            # would promote to a full-volume float64 temporary. astype always copies, so the in-place
            # scale below never mutates the caller's array.
            new_values = image_t.numpy().astype(np.float16)
            if self.current_interaction_intensity != 1:
                new_values *= self.current_interaction_intensity
            self._interactions_inplace_maximum(interaction_channel, int_slicer, new_values)
            del new_values
            del image_t
            empty_cache(self.device)

            if run_prediction:
                return self._predict()
            return None

    def _add_mask_interaction(
        self,
        interaction_name: str,
        mask_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool,
        override_capability_checks: bool,
        interaction_bbox: Optional[List[List[int]]],
    ) -> Optional[List[List[int]]]:
        if self.verbose:
            print(f"Add new {interaction_name} of shape {mask_image.shape} and bbox {interaction_bbox}")
        self._check_capability_or_warn(interaction_name, override_capability_checks)
        pos_channel, neg_channel = self._resolve_channel_pair(interaction_name, override_capability_checks)
        return self._add_image_interaction(
            mask_image,
            pos_channel if include_interaction else neg_channel,
            run_prediction,
            interaction_bbox,
        )

    def add_scribble_interaction(
        self,
        scribble_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
        interaction_bbox: Optional[List[List[int]]] = None,
    ) -> Optional[List[List[int]]]:
        return self._add_mask_interaction(
            "scribble",
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
        return self._add_mask_interaction(
            "lasso", lasso_image, include_interaction, run_prediction, override_capability_checks, interaction_bbox
        )

    def add_initial_seg_interaction(
        self, initial_seg: np.ndarray, run_prediction: bool = False, override_capability_checks: bool = False
    ) -> Optional[List[List[int]]]:
        """
        WARNING THIS WILL RESET INTERACTIONS!

        Returns the bbox of the changed region when ``run_prediction`` is True; None otherwise.
        When ``run_prediction`` is False the *entire* target buffer is overwritten with
        ``initial_seg`` (the caller already holds the full mask), so no sub-region bbox applies.
        """
        self._check_capability_or_warn("initial_label", override_capability_checks)
        if not all(i == j for i, j in zip(self.original_image_shape[1:], initial_seg.shape)):
            raise ValueError(
                f"Given initial seg must match input image shape. Input image was: "
                f"{self.original_image_shape[1:]}, given: {initial_seg.shape}"
            )

        self._finish_preprocessing_and_initialize_interactions()
        with self._undoable_interaction():
            # This whole initial-seg op is one undoable step. It zeroes every interaction channel and overwrites
            # the entire target buffer, so their pre-images are saved whole (only channels that can be nonzero).
            self._record_interactions_channels(list(range(self.num_interaction_channels)), array_replaced=True)
            if self.target_buffer is not None:
                self._record_target_region((slice(None),) * self.target_buffer.ndim)
            self.reset_interactions(_preserve_undo=True)

            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer[:] = initial_seg

            initial_seg = torch.from_numpy(initial_seg)

            if isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer[:] = initial_seg

            # initial seg already matches the image's coordinate space (no cropping)
            # initial seg is written into initial seg buffer
            interaction_channel = self._get_prev_seg_channel()
            self._write_interactions_channel(interaction_channel, initial_seg)

            empty_cache(self.device)
            if run_prediction:
                self._generic_add_patch_from_image(initial_seg)
                del initial_seg
                return self._predict(force_full_refine=True)
            else:
                del initial_seg
                return None

    @torch.inference_mode()
    def warmup(self) -> None:
        """Run a single dummy forward pass to pay one-off first-pass costs up front.

        This serves two purposes:

        * **torch.compile**: with compilation enabled the network is compiled
          lazily on its first forward pass, which would otherwise make the user's
          *first* real prediction slow. The dummy pass triggers that compilation
          here instead.
        * **Device initialization (CUDA)**: even *without* ``torch.compile`` the
          first forward pass on a fresh CUDA device pays one-off costs — cuDNN
          autotunes/selects its convolution algorithms (``cudnn.benchmark`` is
          enabled for CUDA in ``__init__`` and fires here), the caching allocator
          grows its memory pool, and the CUDA context/kernels are loaded. Running
          the dummy pass at startup pays those costs here rather than on the user's
          first prediction. Note that cuDNN's benchmark cache is **thread-local**: call
          ``warmup()`` from the thread that will run the predictions, otherwise that
          thread re-pays the autotuning on its first pass (the server therefore runs both
          on one dedicated GPU thread).

        Every prediction path — the initial coarse pass, the zoom-out iterations,
        and the refinement patches — feeds the network an input of identical shape
        ``[1, num_input_channels + num_interaction_channels, *patch_size]``
        (``_build_network_input`` always resizes the crop to ``patch_size``, and
        refinement crops at exactly ``patch_size``). So a single dummy pass at that
        shape populates the compile cache and the cuDNN algorithm cache for every
        subsequent real prediction.

        Does nothing when there is nothing to gain: the network is neither compiled
        (no compile cache to populate) nor on a CUDA device (no cuDNN autotuning /
        allocator pool / context init to warm), so a dummy pass would not save the
        user any time. Mirrors ``_predict``'s autocast/inference-mode context and
        the float32 input dtype that ``torch.cat`` produces when concatenating the
        float32 image with the fp16 interactions.
        """
        if self.network is None or self.configuration_manager is None:
            raise RuntimeError("warmup() requires an initialized network; call initialize_* first")
        if not isinstance(self.network, OptimizedModule) and self.device.type != "cuda":
            return
        num_input_channels = (
            determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json)
            + self.num_interaction_channels
        )
        patch_size = self.configuration_manager.patch_size
        dummy = torch.zeros((1, num_input_channels, *patch_size), dtype=torch.float32, device=self.device)
        start = time()
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            self.network(dummy)
        del dummy
        empty_cache(self.device)
        print(f"warmup forward pass complete in {time() - start:.1f}s; the first prediction will be fast")

    @torch.inference_mode()
    def _predict(self, force_full_refine: bool = False) -> Optional[List[List[int]]]:
        """
        force_full_refine if True we run the refinement over the whole current prediction and not just the diff map.
        More effort but sometimes needed (refine initial seg)

        If it feels like we are excessively transferring tensors between CPU and GPU, this is deliberate.
        Our goal is to keep this tool usable even for people with smaller GPUs (8-10GB VRAM). In an ideal world
        everyone would have 24GB+ of VRAM and all tensors would like on GPU all the time.
        The amount of hours spent optimizing this function is substantial. Almost every line was turned and twisted
        multiple times. If something appears odd, it is probably so for a reason. Don't change things all willy nilly
        without first understanding what is going on. And don't make changes without verifying that the run time or
        VRAM consumption is not adversely affected.

        Returns:
            The bounding box (in target-buffer/image coordinates, clipped to the buffer bounds) of the
            region written to ``target_buffer`` by this prediction, as ``[[x1, x2], [y1, y2], [z1, z2]]``
            (half-open intervals, same axis convention as everywhere else in this package).
            GUI/clients that cannot share the underlying buffer can use this to copy only the changed
            sub-volume instead of the whole array. Returns None when no prediction ran (nothing queued).
        """
        if not isinstance(self.interactions, torch.Tensor):
            # cratio is a blosc2-only diagnostic; the dense tensor backend has no compression.
            print("Current cratio", self.interactions.cratio)

        assert self.pad_mode_data == "constant", "pad modes other than constant are not implemented here"
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)
        prev_seg_channel = self._get_prev_seg_channel()
        if len(self.new_interaction_centers) == 0:
            print("No patch queued for prediction. Nothing to do.")
            return None

        if len(self.new_interaction_centers) > 1:
            print(
                "It seems like more than one interaction was added since the last prediction. This is not "
                "recommended and may cause unexpected behavior or inefficient predictions\n"
                "!!!WE NO LONGER RUN ONE PREDICTION PER CENTER AND ONLY USE THE LAST ADDED INTERACTION AS CENTER!!!"
            )
        prediction_center, zoom_out_factor = self.new_interaction_centers[-1], self.new_interaction_zoom_out_factors[-1]
        zoom_out_factor = min(self.MAX_AUTOZOOM_FACTOR, zoom_out_factor)

        start_predict = time()
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            # make a prediction at zoom_out_factor, remember max_zoom_out_factor
            start_initial_pred = time()
            input_for_predict, scaled_patch_size, scaled_bbox, previous_prediction = self._build_network_input(
                prediction_center, zoom_out_factor
            )
            # .contiguous() is required for torch.compile: the input may be a non-contiguous
            # view (e.g. from the dense-tensor backend), and the compiled graph assumes contiguity.
            pred = self.network(input_for_predict[None].contiguous())[0].argmax(0).detach()
            del input_for_predict

            # detect changes at border. If there are, we enter autozoom
            has_change = self._detect_change_at_border(pred, previous_prediction)
            del previous_prediction
            empty_cache(self.device)

            print(
                f"Took {round(time() - start_initial_pred, 3)} s for initial prediction at zoom out factor {zoom_out_factor}"
            )

            # maybe do zoom out
            zoom_out_growth_factor = 1.5
            start_zoomout = time()
            while has_change and self.do_autozoom:
                print(f"AutoZoom zoom out factor {zoom_out_factor}")
                # we allow a max zoom out of MAX_AUTOZOOM_FACTOR
                if zoom_out_factor >= self.MAX_AUTOZOOM_FACTOR:
                    break
                else:
                    zoom_out_factor *= zoom_out_growth_factor
                    zoom_out_factor = min(self.MAX_AUTOZOOM_FACTOR, zoom_out_factor)

                input_for_predict, scaled_patch_size, scaled_bbox, previous_prediction_resized = (
                    self._build_network_input(prediction_center, zoom_out_factor)
                )
                # .contiguous() is required for torch.compile: the input may be a non-contiguous
                # view (e.g. from the dense-tensor backend), and the compiled graph assumes contiguity.
                pred = self.network(input_for_predict[None].contiguous())[0].argmax(0).detach()
                del input_for_predict
                empty_cache(self.device)

                has_change = self._detect_change_at_border(pred, previous_prediction_resized)

            if zoom_out_factor > 1:
                print(f"Zoom out took {round(time() - start_zoomout, 3)} s, max zoom out factor {zoom_out_factor}")
            else:
                print("No zoom out necessary")

            if zoom_out_factor == 1:
                # simply place pred in the prev_seg channel and target buffer
                self._paste_interactions(prev_seg_channel, pred.half(), scaled_bbox)
                self._paste_prediction_to_target_buffer(pred, scaled_bbox)
                print("No refinement necessary")
            else:
                # do refinement

                if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                    pred = (
                        interpolate(pred[None, None].to(torch.float32), scaled_patch_size, mode="trilinear")[0, 0]
                        >= 0.5
                    ).to(torch.uint8)

                refinement_bboxes = self._plan_refinement_bboxes(pred, scaled_bbox, force_full_refine)

                # NOTE: we deliberately do NOT write the coarse prediction into self.interactions here.
                # The refinement network needs the coarse segmentation as prev_seg *input context*, but that
                # context must stay confined to the local refinement cache (see _refine_coarse_with_local_cache).
                # Committing it to the persistent prev_seg channel would leave coarse data in the gaps between
                # refinement bboxes, poisoning the next prompt and (formerly) leaking into the target buffer.
                self._refine_coarse(refinement_bboxes, pred, scaled_bbox)

        print(f"Done. Total time {round(time() - start_predict, 3)}s")

        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        empty_cache(self.device)

        return self._clipped_last_paste_bbox()

    def _build_network_input(self, prediction_center, zoom_out_factor):
        scaled_patch_size = [round(i * zoom_out_factor) for i in self.configuration_manager.patch_size]
        scaled_bbox = [[c - p // 2, c + p // 2 + p % 2] for c, p in zip(prediction_center, scaled_patch_size)]
        prev_seg_channel = self._get_prev_seg_channel()

        # cropping happens on CPU, padding happens on GPU (later)
        crop_img, pad_image = crop_to_valid(self.preprocessed_image, scaled_bbox)
        zoomed = not all([i == j for i, j in zip(self.configuration_manager.patch_size, scaled_patch_size)])

        # Interactions are written straight into a buffer of the full scaled size (zero-filled where the crop extends
        # past the image) instead of transferring the in-image crop and then padding it into a second tensor, which
        # would make both coexist on the GPU. Same values as padding with zeros (pad_cropped). The buffer is
        # never a view of the stored interactions, so they can be normalized in place below.
        # The interactions share the image's spatial shape, so the padding is identical.
        pad_interaction = pad_image
        valid = [[max(lb, 0), min(ub, s)] for (lb, ub), s in zip(scaled_bbox, self.interactions.shape[1:])]
        interior = tuple(slice(pl, pl + ub - lb) for (pl, _), (lb, ub) in zip(pad_interaction, valid))
        needs_padding = any(x for pair in pad_interaction for x in pair)

        # resize input_for_predict (which may be larger than patch size) to patch size
        # this implementation may not seem straightforward but it does save VRAM which is crucial here
        if zoomed:
            patch_size = self.configuration_manager.patch_size
            max_pool_ks = round_to_nearest_odd(zoom_out_factor * 2 - 1)
            dilation_channels = set(self._get_dilation_channels_for_resample()) if max_pool_ks > 1 else set()
            # One buffer serves all channels: only its interior is ever written, so its zero border stays valid.
            padded = torch.zeros([1, *scaled_patch_size], dtype=torch.float16, device=self.device)

            def channel_to_padded(c: int) -> torch.Tensor:
                """Interaction channel c of the crop on the device, zero-padded to the scaled patch, [1, *scaled]."""
                self._interactions_region_into((c, c + 1), valid, padded[(slice(None), *interior)])
                return padded

            previous_prediction = interpolate(channel_to_padded(prev_seg_channel)[None], patch_size, mode="nearest")[
                0, 0
            ]

            # Process interaction channels one at a time to avoid materialising the full
            # [num_ch, scaled_patch_size³] tensor on GPU. Peak VRAM ≈ one channel at scaled size.
            num_interaction_ch = self.num_interaction_channels
            interactions_tensor = torch.empty(
                [num_interaction_ch, *patch_size], dtype=torch.float16, device=self.device
            )
            for i in range(num_interaction_ch):
                ch = channel_to_padded(i)
                if i in dilation_channels:
                    ch = iterative_3x3_same_padding_pool3d(ch[None], max_pool_ks)[0]
                interactions_tensor[i : i + 1] = interpolate(ch[None], patch_size, mode="area")[0]
                del ch
            del padded

            # Keep image and interaction tensors in identical spatial frames before concatenation.
            # Interactions use area downsampling (with selective dilation beforehand), image uses trilinear.
            # The image crop goes straight into a zero-padded buffer as well (the interactions share its shape).
            img_padded = torch.zeros([crop_img.shape[0], *scaled_patch_size], dtype=crop_img.dtype, device=self.device)
            self._copy_into_device(crop_img, img_padded[(slice(None), *interior)])
            crop_img = interpolate(img_padded[None], patch_size, mode="trilinear")[0]
            del img_padded

            empty_cache(self.device)
        else:
            # zoom_out_factor == 1: transfer both tensors to GPU, then pad if needed
            crop_img = self._to_device(crop_img)
            if needs_padding:
                crop_img = pad_cropped(crop_img, pad_image)
            alloc = torch.zeros if needs_padding else torch.empty
            interactions_tensor = alloc(
                [self.num_interaction_channels, *scaled_patch_size], dtype=torch.float16, device=self.device
            )
            self._interactions_region_into(
                (0, self.num_interaction_channels), valid, interactions_tensor[(slice(None), *interior)]
            )
            # previous_prediction is a channel of the interactions crop that was just transferred: copy it on
            # the device instead of sending the same data over PCIe a second time. A separate tensor (clone,
            # not a view) because the interaction channels are normalized in place below.
            previous_prediction = interactions_tensor[prev_seg_channel].clone()

        self._normalize_interaction_channels_for_network_(interactions_tensor)
        input_for_predict = torch.cat((crop_img, interactions_tensor))
        del crop_img, interactions_tensor
        empty_cache(self.device)
        return input_for_predict, scaled_patch_size, scaled_bbox, previous_prediction

    def _refine_coarse(
        self, bboxes_ordered: List[List[List[int]]], coarse_pred: torch.Tensor, coarse_bbox: List[List[int]]
    ):
        start_refinement = time()
        prev_seg_channel = self._get_prev_seg_channel()

        if self.verbose:
            print(f"Using {len(bboxes_ordered)} bounding boxes for refinement")

        self._refine_coarse_with_local_cache(bboxes_ordered, prev_seg_channel, coarse_pred, coarse_bbox)
        end_refinement = time()
        print(
            f"Took {round(end_refinement - start_refinement, 3)} s for refining the segmentation with {len(bboxes_ordered)} bounding boxes"
        )

    def _refine_coarse_with_local_cache(
        self,
        bboxes_ordered: List[List[List[int]]],
        prev_seg_channel: int,
        coarse_pred: torch.Tensor,
        coarse_bbox: List[List[int]],
    ) -> None:
        # The cache is cropped out of the *uncontaminated* self.interactions, so its prev_seg channel starts
        # as the previous refined segmentation.
        cache_bbox, cache_image, cache_interactions = self._build_refinement_local_cache(bboxes_ordered)

        # Inject the coarse prediction into the cache's prev_seg channel ONLY. This gives the refinement
        # network the coarse context it needs to sharpen, but the coarse data lives exactly as long as the
        # cache does -- it never reaches the persistent prev_seg channel or the target buffer.
        #
        # Restrict the injection to the in-image region. cache_bbox can extend past the image edge (refinement
        # bboxes are not clipped to image bounds), and those out-of-image cache voxels are zero-padding that a
        # border refinement patch also sees. Feeding coarse foreground into that padding would change the
        # refined border vs. how the coarse pass itself was fed (prev_seg is 0-padded in _build_network_input),
        # so we clip coarse_bbox to the image and slice coarse_pred to match, leaving out-of-image voxels at 0.
        spatial_shape = tuple(int(i) for i in self.interactions.shape[1:])
        inject_bbox = self._clip_bbox_to_shape(coarse_bbox, spatial_shape)
        if inject_bbox is not None:
            pred_slicer = bounding_box_to_slice(self._bbox_to_local(inject_bbox, coarse_bbox))
            inject_local_bbox = self._bbox_to_local(inject_bbox, cache_bbox)
            coarse_sub = coarse_pred[pred_slicer]
            if cache_interactions.device.type == "cpu":
                # Transfer at the narrow source dtype (uint8: half the D2H bytes of fp16) and let
                # the paste widen to the cache dtype on CPU.
                coarse_sub = coarse_sub.to("cpu")
            else:
                coarse_sub = coarse_sub.to(cache_interactions.device, dtype=cache_interactions.dtype)
            paste_tensor(cache_interactions, coarse_sub, inject_local_bbox, channel_idx=prev_seg_channel)
            del coarse_sub

        # A CPU cache pays a host->device transfer per patch. The patch slices are non-contiguous views; they
        # go through the process-wide pinned staging buffer (self._to_device, utils/staging.py), which turns
        # the copy into a true DMA without allocating per patch.

        for refinement_bbox in bboxes_ordered:
            local_bbox = self._bbox_to_local(refinement_bbox, cache_bbox)
            spatial_slicer = bounding_box_to_slice(local_bbox)
            image_patch = cache_image[spatial_slicer][None]
            interactions_patch = cache_interactions[(slice(None), *spatial_slicer)]
            # .type comparison, consistent with _build_refinement_local_cache: this branch must be
            # taken exactly when the cache was normalized at build. An == comparison would send a
            # cuda:0 cache down the else branch (torch.device("cuda") != torch.device("cuda:0")),
            # where .to() returns a *view* of the cache and the per-patch normalization would then
            # corrupt it in place (double-normalizing overlapping patches).
            if cache_image.device.type == self.device.type:
                # Cache lives on the compute device (already normalized at build): patches are
                # views, the cat is the only copy.
                patch = torch.cat((image_patch, interactions_patch), dim=0)
            else:
                image_gpu = self._to_device(image_patch)
                interactions_gpu = self._to_device(interactions_patch)
                # A CPU cache is stored unnormalized (see _build_refinement_local_cache): normalize
                # the per-patch copy on the compute device, where the fp16 division is cheap.
                self._normalize_interaction_channels_for_network_(interactions_gpu)
                patch = torch.cat((image_gpu, interactions_gpu), dim=0)
                del image_gpu, interactions_gpu

            # .contiguous(): see _predict — required for torch.compile with possibly non-contiguous input.
            pred = self.network(patch[None].contiguous())[0].argmax(0).detach()
            # Convert on the compute device before any transfer: fp16 is 4x smaller than the int64
            # argmax output.
            pred = pred.to(dtype=cache_interactions.dtype)
            paste_tensor(
                cache_interactions, pred.to(cache_interactions.device), local_bbox, channel_idx=prev_seg_channel
            )
            del image_patch, interactions_patch, patch
            del pred

        # Commit ONLY the refined bboxes into the persistent prev_seg channel and the target buffer. The gaps
        # of the rectangular union hull (cache_bbox) still hold coarse data in the cache, so pasting the whole
        # hull -- as we used to -- would leak that coarse prediction into both the persistent state (poisoning
        # the next prompt) and the target buffer (which then "appears coarse"). Writing each refined bbox
        # individually keeps every un-refined voxel at its previous full-resolution value; the coarse cache is
        # discarded below.
        final_prev_seg = cache_interactions[prev_seg_channel]
        # Undo: one pre-image of the (in-image part of the) hull covers all the overlapping pastes below, which
        # therefore skip their own (record_undo=False).
        hull_slicer = self._bbox_to_clipped_slicer(cache_bbox, spatial_shape)
        if hull_slicer is not None:
            self._record_interactions_region(prev_seg_channel, hull_slicer)
        if self.target_buffer is not None:
            hull_slicer = self._bbox_to_clipped_slicer(cache_bbox, self.target_buffer.shape)
            if hull_slicer is not None:
                self._record_target_region(hull_slicer)
        for refinement_bbox in bboxes_ordered:
            local_bbox = self._bbox_to_local(refinement_bbox, cache_bbox)
            local_slicer = bounding_box_to_slice(local_bbox)
            refined_patch = final_prev_seg[local_slicer]
            self._paste_interactions(prev_seg_channel, refined_patch, refinement_bbox, record_undo=False)
            self._paste_prediction_to_target_buffer(refined_patch, refinement_bbox, record_undo=False)

        # Report the full refined ROI (union hull of all bboxes) as the changed region so remote clients copy
        # every potentially-updated voxel in one shot; the per-bbox pastes above each left _last_paste_bbox at
        # their own (smaller) box.
        self._last_paste_bbox = cache_bbox

        del cache_image, cache_interactions, final_prev_seg
        empty_cache(self.device)

    def _detect_change_at_border(
        self,
        pred: torch.Tensor,
        prev_pred: torch.Tensor,
        abs_pxl_change_threshold=1500,
        rel_pxl_change_threshold=0.2,
        min_pxl_change_threshold=100,
    ):
        # Queue the statistics of all 6 border faces first, then fetch them with a SINGLE host
        # sync (.tolist()). The previous per-face variant paid several GPU syncs per face
        # (Python `if`/`max` on 0-dim CUDA tensors) plus an index_select copy and an H2D index
        # tensor per face; .select() returns a view. Sums accumulate in float32 (exact for face
        # sizes < 2**24 voxels; the old fp16 sums rounded above 2048). Views and fewer/smaller
        # temporaries: peak VRAM cannot grow.
        stats = []
        for dim in range(pred.ndim):
            for idx in (0, pred.shape[dim] - 1):
                slice_prev = prev_pred.select(dim, idx)
                slice_curr = pred.select(dim, idx).to(prev_pred.device)
                stats.append(torch.sum(slice_prev, dtype=torch.float32))
                stats.append(torch.sum(slice_curr, dtype=torch.float32))
                stats.append(torch.sum(slice_prev != slice_curr, dtype=torch.float32))
        stats = torch.stack(stats).tolist()  # one host sync for all faces

        # Same face order and thresholds as the old early-exit loop: return on the first trigger.
        for face_stats in (stats[i : i + 3] for i in range(0, len(stats), 3)):
            pixels_prev, pixels_current, pixels_diff = face_stats
            rel_change = max(pixels_prev, pixels_current) / max(min(pixels_prev, pixels_current), 1e-5) - 1
            if pixels_diff > abs_pxl_change_threshold:
                if self.verbose:
                    print(f"continue zooming because change at borders of {pixels_diff} > {abs_pxl_change_threshold}")
                return True
            if pixels_diff > min_pxl_change_threshold and rel_change > rel_pxl_change_threshold:
                if self.verbose:
                    print(
                        f"continue zooming because relative change of {rel_change} > {rel_pxl_change_threshold} and n_pixels {pixels_diff} > {min_pxl_change_threshold}"
                    )
                return True
        return False

    def _compute_local_diff_map(
        self, pred: torch.Tensor, scaled_bbox: List[List[int]], planning_bbox: List[List[int]]
    ) -> torch.Tensor:
        """
        Compute a local diff map inside planning_bbox only.

        pred is expected to be the coarse prediction resized to match scaled_bbox.
        planning_bbox is in global interaction coordinates and may be larger than scaled_bbox when
        force_full_refine expands the refinement planning ROI.
        """
        prev_seg_ch = self._get_prev_seg_channel()
        spatial_shape = tuple(int(i) for i in self.interactions.shape[1:])
        seen_bbox = self._clip_bbox_to_shape(scaled_bbox, spatial_shape)
        planning_bbox = self._clip_bbox_to_shape(planning_bbox, spatial_shape)
        if seen_bbox is None or planning_bbox is None:
            return torch.zeros((0, 0, 0), device=self.device, dtype=torch.uint8)

        local_shape = self._bbox_size(planning_bbox)
        diff_local = torch.zeros(local_shape, device=self.device, dtype=torch.float16)

        pred_bbox = self._bbox_to_local(seen_bbox, scaled_bbox)
        pred_bbox = [[max(0, lb), min(ub, int(pred.shape[dim]))] for dim, (lb, ub) in enumerate(pred_bbox)]
        local_seen_bbox = self._bbox_to_local(seen_bbox, planning_bbox)

        pred_slicer = bounding_box_to_slice(pred_bbox)
        local_slicer = bounding_box_to_slice(local_seen_bbox)

        prev_sub = self._read_interactions_region(prev_seg_ch, seen_bbox)

        diff_local[local_slicer] = (pred[pred_slicer] != prev_sub).to(diff_local.dtype)
        del prev_sub

        # Open/close the local difference map to reduce the number of refinement patches without materializing
        # a full-image planning tensor.
        diff_local[local_slicer] = iterative_3x3_same_padding_pool3d(
            diff_local[local_slicer][None, None], kernel_size=5, use_min_pool=True
        )[0, 0]
        diff_local[local_slicer] = iterative_3x3_same_padding_pool3d(
            diff_local[local_slicer][None, None], kernel_size=5, use_min_pool=False
        )[0, 0]

        return diff_local.to(torch.uint8)

    def _mark_prev_seg_in_local_diff(self, diff_local: torch.Tensor, planning_bbox: List[List[int]]) -> None:
        prev_seg_ch = self._get_prev_seg_channel()
        prev_sub = self._read_interactions_region(prev_seg_ch, planning_bbox)
        diff_local[prev_sub > 0.5] = 1
        del prev_sub

    def _plan_refinement_bboxes(
        self, pred: torch.Tensor, scaled_bbox: List[List[int]], force_full_refine: bool
    ) -> List[List[List[int]]]:
        def last_interaction_fallback_bbox() -> List[List[List[int]]]:
            # Single patch-sized bbox centered on the last interaction.
            center = self.new_interaction_centers[-1]
            return [
                [[ci - pi // 2, ci - pi // 2 + pi] for ci, pi in zip(center, self.configuration_manager.patch_size)]
            ]

        spatial_shape = tuple(int(i) for i in self.interactions.shape[1:])
        planning_bbox = self._clip_bbox_to_shape(scaled_bbox, spatial_shape)

        if force_full_refine:
            print("Forcing full refinement of entire structure")
            prev_seg_bbox = self._compute_prev_seg_positive_bbox()
            planning_bbox = self._union_bboxes(planning_bbox, prev_seg_bbox)

        if planning_bbox is None:
            return last_interaction_fallback_bbox()

        diff_local = self._compute_local_diff_map(pred, scaled_bbox, planning_bbox)
        if force_full_refine:
            self._mark_prev_seg_in_local_diff(diff_local, planning_bbox)

        local_bboxes = generate_bounding_boxes(
            diff_local, self.configuration_manager.patch_size, stride="auto", margin=(24, 24, 24), max_depth=3
        )
        del diff_local
        empty_cache(self.device)

        # If no bounding boxes are returned we basically have almost no changes. Still we should at least perform
        # refinement in the bounding box where the interaction was as the user evidently wanted something here.
        if len(local_bboxes) == 0:
            return last_interaction_fallback_bbox()

        return self._offset_bboxes(local_bboxes, planning_bbox)

    def _add_patch_for_point_interaction(self, coordinates):
        self.new_interaction_zoom_out_factors.append(1)
        self.new_interaction_centers.append(coordinates)
        print(f"Added new point interaction: center {coordinates}, zoom-out factor 1")

    def _add_patch_for_bbox_interaction(self, bbox):
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox]
        bbox_size = [i[1] - i[0] for i in bbox]
        # we want to see some context, so the crop we see for the initial prediction should be patch_size / 3 larger
        requested_size = [i + j // 3 for i, j in zip(bbox_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(
            max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)]))
        )
        self.new_interaction_centers.append(bbox_center)
        print(
            f"Added new bbox interaction: center {bbox_center}, "
            f"zoom-out factor {self.new_interaction_zoom_out_factors[-1]}"
        )

    def _generic_add_patch_from_image(self, image: torch.Tensor, offset: Optional[List[int]] = None):
        # _nonzero_spatial_bbox doubles as the emptiness check (returns None) and avoids materializing the
        # full torch.nonzero index list, which matters for full-image prompts (initial seg, full scribbles).
        local_bbox = self._nonzero_spatial_bbox(image)
        if local_bbox is None:
            print("Received empty image prompt. Cannot add patches for prediction")
            return
        if offset is None:
            offset = [0] * image.ndim
        roi = [[lb + off, ub + off] for (lb, ub), off in zip(local_bbox, offset)]
        roi_center = [round((i[0] + i[1]) / 2) for i in roi]
        roi_size = [i[1] - i[0] for i in roi]
        requested_size = [i + j // 3 for i, j in zip(roi_size, self.configuration_manager.patch_size)]
        zoom_out_factor = max(1, max(i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)))
        self.new_interaction_zoom_out_factors.append(zoom_out_factor)
        self.new_interaction_centers.append(roi_center)
        print(f"Added new image interaction: center {roi_center}, zoom-out factor {zoom_out_factor}")

    def initialize_from_trained_model_folder(
        self,
        model_training_output_dir: str,
        use_fold: Union[int, str] = None,
        checkpoint_name: str = "checkpoint_final.pth",
    ):
        """
        This is used when making predictions with a trained model
        """
        artifacts = self._load_model_artifacts_from_disk(model_training_output_dir, use_fold, checkpoint_name)
        self.initialize_from_loaded_artifacts(artifacts)
        # Pay the one-off cost of the first forward pass now, at initialization, rather than on
        # the user's first real prediction where it is far more noticeable. With torch.compile
        # this triggers the (slow, once) lazy compilation; without it, the dummy forward pass
        # still warms a fresh CUDA device (cuDNN algorithm selection / benchmark, allocator
        # memory pool, CUDA context). warmup() is a no-op when there is nothing to gain (not
        # compiled and not on CUDA). The server takes care of its own warmup explicitly (it
        # shares one network across sessions via initialize_from_loaded_artifacts), so we only
        # do this on the direct, local entry point.
        if self.use_torch_compile:
            print("torch.compile enabled; warming up (compiling) the network now (this is slow once)...")
        self.warmup()

    def _load_model_artifacts_from_disk(
        self,
        model_training_output_dir: str,
        use_fold: Union[int, str] = None,
        checkpoint_name: str = "checkpoint_final.pth",
    ) -> dict:
        """Read all model artifacts from disk and build the network on ``self.device``.

        Returns an artifact dict that can be applied to this or any other freshly
        constructed session via :meth:`initialize_from_loaded_artifacts`. The
        returned values are the actual objects (the ``nn.Module`` with its
        weights and buffers, the plans/configuration managers, the dataset
        json, the label manager) — not copies. Multiple sessions calling
        :meth:`initialize_from_loaded_artifacts` with the same dict will all
        end up with ``self.network`` pointing at the same module instance and
        the same weight tensors on the GPU. This is safe as long as callers
        treat these objects as read-only after construction; in the multi-
        session server that is enforced by running inference under
        ``@torch.inference_mode()`` and running every predict call on one
        dedicated GPU thread.

        Note: this also mutates ``self`` (applies capability, sets pad/decay/
        thickness) because ``num_interaction_channels`` is required to build the
        network. The caller should follow up with
        :meth:`initialize_from_loaded_artifacts` (this is what
        :meth:`initialize_from_trained_model_folder` does).
        """
        point_interaction_use_etd = True
        (
            capability_content,
            point_interaction_radius,
            self.preferred_scribble_thickness,
            self.interaction_decay,
            self.pad_mode_data,
        ) = self._load_capability_and_runtime_defaults(model_training_output_dir)

        self.point_interaction = PointInteraction_stub(point_interaction_radius, point_interaction_use_etd)
        self._apply_capability(capability_content)

        dataset_json = load_json(join(model_training_output_dir, "dataset.json"))
        plans = load_json(join(model_training_output_dir, "plans.json"))
        plans_manager = PlansManager(plans)

        if use_fold is not None:
            use_fold = int(use_fold) if use_fold != "all" else use_fold
            fold_folder = f"fold_{use_fold}"
        else:
            fldrs = subdirs(model_training_output_dir, prefix="fold_", join=False)
            assert len(fldrs) == 1, f"Attempted to infer fold but there is != 1 fold_ folders: {fldrs}"
            fold_folder = fldrs[0]

        checkpoint = torch.load(
            join(model_training_output_dir, fold_folder, checkpoint_name), map_location=self.device, weights_only=False
        )
        self.license = self._load_license(model_training_output_dir, plans, checkpoint)
        print("=" * 80)
        print("Model license:")
        print(self.license)
        print("=" * 80)
        trainer_name = checkpoint["trainer_name"]
        configuration_name = checkpoint["init_args"]["configuration"]

        parameters = checkpoint["network_weights"]

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = (
            determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
            + self.num_interaction_channels
        )
        # Locate the trainer dir via the trainer subpackage itself. `nnInteractive` is a PEP 420
        # namespace package, so `nnInteractive.__path__[0]` is order-dependent and may point at the
        # client distribution's portion (which has no trainer/); `nnInteractive.trainer` is a real
        # subpackage with a single, unambiguous path.
        import nnInteractive.trainer

        trainer_class = recursive_find_python_class(
            nnInteractive.trainer.__path__[0], trainer_name, "nnInteractive.trainer"
        )
        if trainer_class is None:
            # fall back to looking for the trainer in nnunetv2
            import nnunetv2

            trainer_class = recursive_find_python_class(
                join(nnunetv2.__path__[0], "training", "nnUNetTrainer"),
                trainer_name,
                "nnunetv2.training.nnUNetTrainer",
            )
        if trainer_class is None:
            print(
                f"Unable to locate trainer class {trainer_name} in nnInteractive.trainer. "
                f"Please place it there (in any .py file)!"
            )
            print(
                "Attempting to use default nnInteractiveTrainer_stub. If you encounter errors, this is where you need to look!"
            )
            trainer_class = nnInteractiveTrainer_stub

        # nnInteractive is always a binary problem (target object vs background): checkpoints are trained with
        # 2 output channels regardless of how many labels the training dataset.json carries, so we must
        # reconstruct the architecture with the same count. Using
        # num_segmentation_heads here would silently work for datasets that happen to have 2 labels but blows up for
        # checkpoints trained on aggregated datasets whose dataset.json lists thousands of object ids as labels (the
        # decoder channel sizes depend on the class count, so the state_dict would not load).
        num_network_output_channels = 2
        network = trainer_class.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            num_network_output_channels,
            enable_deep_supervision=False,
        ).to(self.device)
        network.load_state_dict(parameters)

        return {
            "capability_content": capability_content,
            "point_interaction": self.point_interaction,
            "preferred_scribble_thickness": self.preferred_scribble_thickness,
            "interaction_decay": self.interaction_decay,
            "pad_mode_data": self.pad_mode_data,
            "network": network,
            "plans_manager": plans_manager,
            "configuration_manager": configuration_manager,
            "dataset_json": dataset_json,
            "trainer_name": trainer_name,
            "label_manager": plans_manager.get_label_manager(dataset_json),
            "license": self.license,
        }

    def initialize_from_loaded_artifacts(self, artifacts: dict):
        """Apply pre-loaded artifacts to this session instance.

        ``artifacts`` is the dict returned by :meth:`_load_model_artifacts_from_disk`.
        Useful for spawning multiple sessions that share one loaded model (e.g.
        the multi-session inference server). All artifact entries — including
        ``self.network`` — are stored by reference; passing the same dict to
        multiple sessions does not duplicate the network or its weights in
        memory.
        """
        self.preferred_scribble_thickness = artifacts["preferred_scribble_thickness"]
        self.interaction_decay = artifacts["interaction_decay"]
        self.pad_mode_data = artifacts["pad_mode_data"]
        self.point_interaction = artifacts["point_interaction"]
        self._apply_capability(artifacts["capability_content"])
        self.plans_manager = artifacts["plans_manager"]
        self.configuration_manager = artifacts["configuration_manager"]
        self.network = artifacts["network"]
        self.dataset_json = artifacts["dataset_json"]
        self.trainer_name = artifacts["trainer_name"]
        self.label_manager = artifacts["label_manager"]
        self.license = artifacts["license"]
        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print("Using torch.compile")
            self.network = torch.compile(self.network)

    def manual_initialization(
        self,
        network: nn.Module,
        plans_manager: PlansManager,
        configuration_manager: ConfigurationManager,
        dataset_json: dict,
        trainer_name: str,
    ):
        """
        This is used by the nnUNetTrainer to initialize nnUNetPredictor for the final validation
        """
        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network.to(self.device)
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)

        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print("Using torch.compile")
            self.network = torch.compile(self.network)

        if not self.use_torch_compile and isinstance(self.network, OptimizedModule):
            self.network = self.network._orig_mod

    def __del__(self):
        # Be robust to a partially-constructed instance (e.g. __init__ raised on bad arguments):
        # these attributes may not exist yet.
        if hasattr(self, "preprocess_future"):
            self._finish_preprocessing_and_initialize_interactions()
        if hasattr(self, "executor"):
            self.executor.shutdown()
