from concurrent.futures import ThreadPoolExecutor
import os
from os import cpu_count
from time import time
from typing import Union, List, Tuple, Optional
import warnings

import blosc2

import numpy as np
import torch
from acvl_utils.cropping_and_padding.bounding_boxes import bounding_box_to_slice, crop_and_pad_nd
from batchgenerators.utilities.file_and_folder_operations import load_json, join, subdirs, isfile
from nnunetv2.utilities.find_class_by_name import recursive_find_python_class
from nnunetv2.utilities.helpers import dummy_context, empty_cache
from nnunetv2.utilities.label_handling.label_handling import determine_num_input_channels
from nnunetv2.utilities.plans_handling.plans_handler import PlansManager, ConfigurationManager
from torch import nn
from torch._dynamo import OptimizedModule
from torch.nn.functional import interpolate

import nnInteractive
from nnInteractive.interaction.point import PointInteraction_stub
from nnInteractive.trainer.nnInteractiveTrainer import nnInteractiveTrainer_stub
from nnInteractive.utils.bboxes import generate_bounding_boxes
from nnInteractive.utils.crop import crop_and_pad_into_buffer, paste_tensor, pad_cropped, crop_to_valid
from nnInteractive.utils.erosion_dilation import iterative_3x3_same_padding_pool3d
from nnInteractive.utils.inference_helpers import (
    infer_num_interaction_channels_from_mapping,
    parse_channel_pair,
    transform_coordinates_noresampling,
    version_to_tuple,
)
from nnInteractive.utils.os_shennanigans import is_linux_kernel_6_11
from nnInteractive.utils.rounding import round_to_nearest_odd


class nnInteractiveInferenceSession:
    INFERENCE_SESSION_VERSION = nnInteractive.__version__
    REFINEMENT_CACHE_GPU_HEADROOM_BYTES = 4 * 1024**3
    # Maximum adaptive zoom-out factor (see _predict). Also bounds the largest interaction crop,
    # which sizes the reusable blosc2 decompression buffer.
    MAX_AUTOZOOM_FACTOR = 4
    # 'auto' interaction storage threshold: images with at most this many spatial voxels
    # (512*512*1024) use the dense tensor backend; larger ones use blosc2 to bound RAM.
    AUTO_TENSOR_MAX_VOXELS = 2**28
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
    ):
        """
        Only intended to work with nnInteractiveTrainerV2 and its derivatives

        ``use_torch_compile``: compile the network with ``torch.compile``. The
        first prediction after enabling this is slow (compilation happens lazily
        on the first forward pass), but every subsequent prediction is faster.
        This is recommended for the persistent inference server, where the
        process is long-lived so the one-time compile cost is paid only once and
        amortized across the whole session lifetime.

        ``interactions_storage``: storage backend for the interaction tensor, one of
        ``"blosc2"``, ``"tensor"`` or ``"auto"`` (default).
        ``"blosc2"`` keeps it as a compact blosc2 in-memory NDArray (low RAM, pays
        (de)compression on every read/write). ``"tensor"`` stores it as a dense CPU
        float16 ``torch.Tensor`` (more RAM, far lower per-access overhead; pinned memory
        by default, skipped when ``device`` is not CUDA or on Linux kernel 6.11 where
        pinning is buggy). ``"auto"`` decides per image at initialization from the
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
        self.preprocessed_props = None
        self.target_buffer: Union[np.ndarray, torch.Tensor] = None
        # Bbox (in original-image coordinates) of the most recent target_buffer write.
        # Captured inside _paste_prediction_to_target_buffer so remote callers can
        # fetch just the touched region without diffing.
        self._last_paste_bbox: Optional[List[List[int]]] = None

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

    def _interaction_bbox_to_target_bbox(self, bbox: List[List[int]]) -> List[List[int]]:
        return [
            [i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in zip(bbox, self.preprocessed_props["bbox_used_for_cropping"])
        ]

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
        for ch in channels_to_scale:
            self.interactions[ch] *= scale
        self.current_interaction_intensity = self._interaction_renorm_target

    def _interactions_inplace_maximum(self, channel_idx: int, int_slicer, new_values) -> None:
        """In-place element-wise maximum for a subregion of a channel."""
        full_slicer = (channel_idx, *int_slicer)
        if isinstance(self.interactions, torch.Tensor):
            # Dense torch backend: operate in place without a numpy round-trip.
            if not isinstance(new_values, torch.Tensor):
                new_values = torch.as_tensor(new_values)
            view = self.interactions[full_slicer]
            torch.maximum(view, new_values.to(view.dtype), out=view)
            return
        if isinstance(new_values, torch.Tensor):
            new_values = new_values.cpu().numpy().astype(np.float16)
        current_sub = np.asarray(self.interactions[full_slicer])
        np.maximum(current_sub, new_values, out=current_sub)
        self.interactions[full_slicer] = current_sub

    def _write_interactions_channel(self, channel_idx: int, value) -> None:
        """Write a full channel. Handles torch→numpy for blosc2."""
        if isinstance(self.interactions, torch.Tensor):
            if not isinstance(value, torch.Tensor):
                value = torch.as_tensor(value)
            self.interactions[channel_idx] = value.to(self.interactions.dtype)
            return
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy().astype(np.float16)
        self.interactions[channel_idx] = value

    def _read_interactions_to_device(self, full_slicer, device) -> torch.Tensor:
        """Read an interaction subregion as a torch.Tensor on ``device``, regardless of backend."""
        sub = self.interactions[full_slicer]
        if isinstance(sub, torch.Tensor):
            return sub.to(device)
        return torch.from_numpy(np.asarray(sub)).to(device)

    def _paste_prediction_to_target_buffer(self, prediction: torch.Tensor, bbox: List[List[int]]) -> None:
        target_bbox = self._interaction_bbox_to_target_bbox(bbox)
        if isinstance(self.target_buffer, torch.Tensor):
            pred_for_target = prediction.to(self.target_buffer.device)
        else:
            pred_for_target = prediction.to("cpu")
        paste_tensor(self.target_buffer, pred_for_target, target_bbox)
        self._last_paste_bbox = target_bbox

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

    def _build_refinement_local_cache(self, bboxes_ordered: List[List[List[int]]]):
        cache_bbox = self._union_bboxes(*bboxes_ordered)
        cache_device = self._select_refinement_cache_device(cache_bbox)
        cache_shape = self._bbox_size(cache_bbox)
        pin_cache = cache_device.type == "cpu" and self.device.type == "cuda"

        cache_kwargs = {"device": cache_device}
        if pin_cache:
            cache_kwargs["pin_memory"] = True

        cache_image = torch.zeros(cache_shape, dtype=self.preprocessed_image.dtype, **cache_kwargs)
        cache_interactions = torch.zeros(
            (self.num_interaction_channels, *cache_shape), dtype=torch.float16, **cache_kwargs
        )

        crop_and_pad_into_buffer(cache_image, cache_bbox, self.preprocessed_image[0])
        crop_and_pad_into_buffer(cache_interactions, cache_bbox, self.interactions)
        self._normalize_interaction_channels_for_network_(cache_interactions)
        return cache_bbox, cache_image, cache_interactions

    def _prepare_new_interaction_intensity(self):
        if self.interaction_decay is None:
            return
        if not (0 < self.interaction_decay <= 1):
            raise ValueError(f"interaction_decay must be in (0, 1], got {self.interaction_decay}.")
        if self.interaction_decay < 1:
            self.current_interaction_intensity *= 1 / self.interaction_decay
            self._renormalize_interactions_if_needed()

    def _normalize_interaction_channels_for_network_(self, interaction_tensor: torch.Tensor):
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
            del self.preprocess_future
            self.preprocess_future = None

    def set_target_buffer(self, target_buffer: Union[np.ndarray, torch.Tensor]):
        """
        Must be 3d numpy array or torch.Tensor
        """
        self.target_buffer = target_buffer

    def set_do_autozoom(self, do_autozoom: bool):
        self.do_autozoom = do_autozoom

    def _reset_session(self):
        self.interactions_future = None
        self.preprocess_future = None

        del self.preprocessed_image
        del self.target_buffer
        del self.interactions
        del self.preprocessed_props
        self.preprocessed_image = None
        self.target_buffer = None
        self.interactions = None
        self.preprocessed_props = None
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
            # Pinning enables faster non-blocking host->device copies, but only helps for a
            # CUDA target and is buggy on Linux kernel 6.11 (see utils/os_shennanigans).
            pin = self.device.type == "cuda" and not is_linux_kernel_6_11()
            tensor = torch.zeros(shape, dtype=torch.float16, device="cpu")
            return tensor.pin_memory() if pin else tensor
        return blosc2.zeros(
            shape,
            dtype=np.float16,
            chunks=(1, *[min(64, s) for s in shape[1:]]),
            blocks=(1, *[min(32, s) for s in shape[1:]]),
            # Interactions compress better with NOFILTER, which is also faster than SHUFFLE.
            cparams={
                "codec": blosc2.Codec.LZ4,
                "clevel": 5,
                "filters": [blosc2.Filter.NOFILTER],
                "nthreads": compression_nthreads,
            },
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
            backend = "dense torch.Tensor" if self._interactions_storage_resolved == "tensor" else "blosc2 in-memory compression"
            print(f"Initialize interactions with {backend}{' (auto)' if via_auto else ''}")
        self.interactions = self._new_interactions_array(shape, min(self.torch_n_threads, os.cpu_count()))
        self._interactions_shape = shape
        self._interactions_read_buffer = self._new_interactions_read_buffer(shape)

    def _new_interactions_read_buffer(self, shape) -> Optional[np.ndarray]:
        """Pre-faulted buffer to decompress blosc2 interaction crops into (Path B), or None.

        Sized to the largest possible crop: the patch size scaled by the maximum autozoom factor,
        capped to the image size. Only allocated for the blosc2 backend that exposes the
        decompress-into-buffer method; the dense-tensor backend returns views and needs no buffer.
        """
        if self._interactions_storage_resolved != "blosc2":
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
        n = self.num_interaction_channels * int(np.prod(max_valid, dtype=np.int64))
        buffer = np.empty(n, dtype=np.float16)
        buffer[:] = 0  # first-touch the pages once, up front
        return buffer

    @torch.inference_mode()
    def _background_set_image(self, image: np.ndarray, image_properties: dict):
        # Convert and clone the image tensor.
        image = torch.from_numpy(image.copy())

        # Crop to nonzero region.
        if self.verbose:
            print("Cropping input image to nonzero region")
        # torch.where eats RAM / VRAM for breakfast. Avoid!!!
        # nonzero_idx = torch.where(image != 0)
        # # Create bounding box: for each dimension, get the min and max (plus one) of the nonzero indices.
        # bbox = [[i.min().item(), i.max().item() + 1] for i in nonzero_idx]
        # del nonzero_idx
        # instead we sum dimensions
        s_x = image.sum(axis=(0, 2, 3), dtype=torch.float)
        wh_x = torch.where(s_x != 0)[0]
        bbox_x = [wh_x.min().item(), wh_x.max().item() + 1]
        del s_x, wh_x
        s_y = image.sum(axis=(0, 1, 3), dtype=torch.float)
        wh_y = torch.where(s_y != 0)[0]
        bbox_y = [wh_y.min().item(), wh_y.max().item() + 1]
        del s_y, wh_y
        s_z = image.sum(axis=(0, 1, 2), dtype=torch.float)
        wh_z = torch.where(s_z != 0)[0]
        bbox_z = [wh_z.min().item(), wh_z.max().item() + 1]
        del s_z, wh_z
        bbox = [[0, 1], bbox_x, bbox_y, bbox_z]
        empty_cache(self.device)

        slicer = bounding_box_to_slice(bbox)  # Assuming this returns a tuple of slices.
        image = image[slicer].float()
        if self.verbose:
            print(f"Cropped image shape: {image.shape}")

        # As soon as we have the target shape, start initializing the interaction tensor in its own thread.
        self.interactions_future = self.executor.submit(self._initialize_interactions, image)

        # Normalize the cropped image.
        if self.verbose:
            print("Normalizing cropped image")
        image -= image.mean()
        image /= image.std()

        self.preprocessed_image = image.to("cpu")

        self.preprocessed_props = {"bbox_used_for_cropping": bbox[1:]}

        # we need to wait for this here I believe
        self.interactions_future.result()
        del self.interactions_future
        self.interactions_future = None

    def reset_interactions(self):
        """
        Use this to reset all interactions and start from scratch for the current image. This includes the initial
        segmentation!
        """
        if self.interactions is not None:
            del self.interactions
            self.interactions = self._new_interactions_array(self._interactions_shape, os.cpu_count())
        self.current_interaction_intensity = 1.0

        if self.target_buffer is not None:
            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer.fill(0)
            elif isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer.zero_()
        self._last_paste_bbox = None
        empty_cache(self.device)

    def add_bbox_interaction(
        self,
        bbox_coords,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ):
        self._finish_preprocessing_and_initialize_interactions()
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

        # Convert user-space coordinates (original image) to the cropped nnU-Net internal space.
        lbs_transformed = [
            round(i)
            for i in transform_coordinates_noresampling(
                [i[0] for i in bbox_coords], self.preprocessed_props["bbox_used_for_cropping"]
            )
        ]
        ubs_transformed = [
            round(i)
            for i in transform_coordinates_noresampling(
                [i[1] for i in bbox_coords], self.preprocessed_props["bbox_used_for_cropping"]
            )
        ]
        transformed_bbox_coordinates = [[i, j] for i, j in zip(lbs_transformed, ubs_transformed)]

        if self.verbose:
            print(
                f"Adding bounding box coordinates.\n"
                f"Raw: {bbox_coords}\n"
                f"Transformed: {transformed_bbox_coordinates}\n"
                f"Crop Bbox: {self.preprocessed_props['bbox_used_for_cropping']}"
            )

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

        # place bbox
        slicer = tuple([slice(*i) for i in transformed_bbox_coordinates])
        channel = bbox_pos_channel if include_interaction else bbox_neg_channel
        self.interactions[(channel, *slicer)] = self.current_interaction_intensity

        if run_prediction:
            self._predict()

    def add_point_interaction(
        self,
        coordinates: Tuple[int, ...],
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
    ):
        self._check_capability_or_warn("points", override_capability_checks)
        point_pos_channel, point_neg_channel = self._resolve_channel_pair("points", override_capability_checks)
        self._finish_preprocessing_and_initialize_interactions()

        transformed_coordinates = [
            round(i)
            for i in transform_coordinates_noresampling(coordinates, self.preprocessed_props["bbox_used_for_cropping"])
        ]

        self._add_patch_for_point_interaction(transformed_coordinates)

        self._prepare_new_interaction_intensity()

        interaction_channel = point_pos_channel if include_interaction else point_neg_channel
        self.point_interaction.place_point(
            transformed_coordinates,
            self.interactions,
            channel_idx=interaction_channel,
            intensity_scale=self.current_interaction_intensity,
        )
        if run_prediction:
            self._predict()

    def _add_image_interaction(
        self,
        image: np.ndarray,
        interaction_channel: int,
        run_prediction: bool,
        interaction_bbox: Optional[List[List[int]]],
        patch_fn,
    ):
        if interaction_bbox is None:
            interaction_bbox = [[0, s] for s in self.original_image_shape[1:]]

        assert len(interaction_bbox) == 3
        bbox_size = [ub - lb for lb, ub in interaction_bbox]
        assert all(s > 0 for s in bbox_size), "each dimension of interaction_bbox must have positive size"
        assert (
            list(image.shape) == bbox_size
        ), f"image shape {list(image.shape)} must match interaction_bbox size {bbox_size}"
        assert all(
            lb >= 0 and ub <= orig_dim for (lb, ub), orig_dim in zip(interaction_bbox, self.original_image_shape[1:])
        ), f"interaction_bbox {interaction_bbox} exceeds original image bounds {list(self.original_image_shape[1:])}"

        self._finish_preprocessing_and_initialize_interactions()

        lbs_internal = [
            round(i)
            for i in transform_coordinates_noresampling(
                [ib[0] for ib in interaction_bbox], self.preprocessed_props["bbox_used_for_cropping"]
            )
        ]
        ubs_internal = [
            round(i)
            for i in transform_coordinates_noresampling(
                [ib[1] for ib in interaction_bbox], self.preprocessed_props["bbox_used_for_cropping"]
            )
        ]

        image_t = torch.from_numpy(image)
        patch_fn(image_t, offset=lbs_internal)

        self._prepare_new_interaction_intensity()

        interaction_shape = self.interactions.shape[1:]
        # Map possibly out-of-bounds transformed bbox to overlapping source/target slices so we only
        # materialize and write the intersecting subregion.
        clipped_lb = [max(0, lb) for lb in lbs_internal]
        clipped_ub = [min(ub, s) for ub, s in zip(ubs_internal, interaction_shape)]
        src_lb = [cl - lb for cl, lb in zip(clipped_lb, lbs_internal)]
        src_ub = [src_lb[d] + (clipped_ub[d] - clipped_lb[d]) for d in range(3)]
        int_slicer = tuple(slice(a, b) for a, b in zip(clipped_lb, clipped_ub))
        src_slicer = tuple(slice(a, b) for a, b in zip(src_lb, src_ub))
        new_values = image_t[src_slicer].cpu().numpy()
        if self.current_interaction_intensity != 1:
            new_values = new_values * self.current_interaction_intensity
        new_values = new_values.astype(np.float16)
        self._interactions_inplace_maximum(interaction_channel, int_slicer, new_values)
        del new_values
        del image_t
        empty_cache(self.device)

        if run_prediction:
            self._predict()

    def _add_mask_interaction(
        self,
        interaction_name: str,
        mask_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool,
        override_capability_checks: bool,
        interaction_bbox: Optional[List[List[int]]],
    ) -> None:
        if self.verbose:
            print(f"Add new {interaction_name} of shape {mask_image.shape} and bbox {interaction_bbox}")
        self._check_capability_or_warn(interaction_name, override_capability_checks)
        pos_channel, neg_channel = self._resolve_channel_pair(interaction_name, override_capability_checks)
        self._add_image_interaction(
            mask_image,
            pos_channel if include_interaction else neg_channel,
            run_prediction,
            interaction_bbox,
            self._generic_add_patch_from_image,
        )

    def add_scribble_interaction(
        self,
        scribble_image: np.ndarray,
        include_interaction: bool,
        run_prediction: bool = True,
        override_capability_checks: bool = False,
        interaction_bbox: Optional[List[List[int]]] = None,
    ):
        self._add_mask_interaction(
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
    ):
        self._add_mask_interaction(
            "lasso", lasso_image, include_interaction, run_prediction, override_capability_checks, interaction_bbox
        )

    def add_initial_seg_interaction(
        self, initial_seg: np.ndarray, run_prediction: bool = False, override_capability_checks: bool = False
    ):
        """
        WARNING THIS WILL RESET INTERACTIONS!
        """
        self._check_capability_or_warn("initial_label", override_capability_checks)
        assert all(
            [i == j for i, j in zip(self.original_image_shape[1:], initial_seg.shape)]
        ), f"Given initial seg must match input image shape. Input image was: {self.original_image_shape[1:]}, given: {initial_seg.shape}"

        self._finish_preprocessing_and_initialize_interactions()

        self.reset_interactions()

        if isinstance(self.target_buffer, np.ndarray):
            self.target_buffer[:] = initial_seg

        initial_seg = torch.from_numpy(initial_seg)

        if isinstance(self.target_buffer, torch.Tensor):
            self.target_buffer[:] = initial_seg

        # crop (as in preprocessing)
        initial_seg = crop_and_pad_nd(initial_seg, self.preprocessed_props["bbox_used_for_cropping"])

        # initial seg is written into initial seg buffer
        interaction_channel = self._get_prev_seg_channel()
        self._write_interactions_channel(interaction_channel, initial_seg)

        empty_cache(self.device)
        if run_prediction:
            self._add_patch_for_initial_seg_interaction(initial_seg)
            del initial_seg
            self._predict(force_full_refine=True)
        else:
            del initial_seg

    @torch.inference_mode()
    def warmup(self) -> bool:
        """Run a single dummy forward pass to trigger lazy ``torch.compile`` compilation up front.

        With ``torch.compile`` enabled the network is compiled lazily on its first
        forward pass, which would otherwise make the user's *first* real prediction
        slow. Every prediction path — the initial coarse pass, the zoom-out
        iterations, and the refinement patches — feeds the network an input of
        identical shape ``[1, num_input_channels + num_interaction_channels,
        *patch_size]`` (``_build_network_input`` always resizes the crop to
        ``patch_size``, and refinement crops at exactly ``patch_size``). So a single
        dummy pass at that shape populates the compile cache and every subsequent
        real prediction is fast.

        Returns ``True`` if a warmup pass was run, ``False`` if it was a no-op
        (network not compiled — there is nothing to pre-compile, so a dummy pass
        would not save the user any time). Mirrors ``_predict``'s autocast/
        inference-mode context and the float32 input dtype that ``torch.cat``
        produces when concatenating the float32 image with the fp16 interactions.
        """
        if self.network is None or self.configuration_manager is None:
            raise RuntimeError("warmup() requires an initialized network; call initialize_* first")
        if not isinstance(self.network, OptimizedModule):
            return False
        num_input_channels = (
            determine_num_input_channels(self.plans_manager, self.configuration_manager, self.dataset_json)
            + self.num_interaction_channels
        )
        patch_size = self.configuration_manager.patch_size
        dummy = torch.zeros((1, num_input_channels, *patch_size), dtype=torch.float32, device=self.device)
        with torch.autocast(self.device.type, enabled=True) if self.device.type == "cuda" else dummy_context():
            self.network(dummy)
        del dummy
        empty_cache(self.device)
        return True

    @torch.inference_mode()
    def _predict(self, force_full_refine: bool = False):
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

        """
        if not isinstance(self.interactions, torch.Tensor):
            # cratio is a blosc2-only diagnostic; the dense tensor backend has no compression.
            print("Current cratio", self.interactions.cratio)

        assert self.pad_mode_data == "constant", "pad modes other than constant are not implemented here"
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)
        prev_seg_channel = self._get_prev_seg_channel()
        if len(self.new_interaction_centers) == 0:
            print("No patch queued for prediction. Nothing to do.")
            return

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
                paste_tensor(self.interactions, pred.half(), scaled_bbox, channel_idx=prev_seg_channel)
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

                # Place the coarse segmentation into prev_seg before refinement
                paste_tensor(self.interactions, pred, scaled_bbox, channel_idx=prev_seg_channel)

                self._refine_coarse(refinement_bboxes)

        print(f"Done. Total time {round(time() - start_predict, 3)}s")

        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        empty_cache(self.device)

    def _build_network_input(self, prediction_center, zoom_out_factor):
        scaled_patch_size = [round(i * zoom_out_factor) for i in self.configuration_manager.patch_size]
        scaled_bbox = [[c - p // 2, c + p // 2 + p % 2] for c, p in zip(prediction_center, scaled_patch_size)]
        prev_seg_channel = self._get_prev_seg_channel()

        # cropping happens on CPU, padding happens on GPU (later)
        crop_img, pad_image = crop_to_valid(self.preprocessed_image, scaled_bbox)
        interactions_tensor, pad_interaction = crop_to_valid(
            self.interactions, scaled_bbox, out=self._interactions_read_buffer
        )
        # For blosc2, crop_to_valid returns a numpy array; convert to torch (still on CPU).
        if not isinstance(interactions_tensor, torch.Tensor):
            interactions_tensor = torch.from_numpy(np.asarray(interactions_tensor))

        previous_prediction = interactions_tensor[prev_seg_channel : prev_seg_channel + 1]

        # resize input_for_predict (which may be larger than patch size) to patch size
        # this implementation may not seem straightforward but it does save VRAM which is crucial here
        if not all([i == j for i, j in zip(self.configuration_manager.patch_size, scaled_patch_size)]):
            patch_size = self.configuration_manager.patch_size
            max_pool_ks = round_to_nearest_odd(zoom_out_factor * 2 - 1)
            dilation_channels = set(self._get_dilation_channels_for_resample()) if max_pool_ks > 1 else set()
            needs_pad_interaction = any(x for pair in pad_interaction for x in pair)

            previous_prediction = previous_prediction.to(self.device, non_blocking=True)
            if needs_pad_interaction:
                previous_prediction = pad_cropped(previous_prediction, pad_interaction)
            previous_prediction = interpolate(previous_prediction[None], patch_size, mode="nearest")[0, 0]

            # Process interaction channels one at a time to avoid materialising the full
            # [num_ch, scaled_patch_size³] tensor on GPU. Peak VRAM ≈ one channel at scaled size.
            num_interaction_ch = interactions_tensor.shape[0]
            interactions_out = torch.empty(
                [num_interaction_ch, *patch_size], dtype=interactions_tensor.dtype, device=self.device
            )
            for i in range(num_interaction_ch):
                ch = interactions_tensor[i : i + 1].to(self.device, non_blocking=True)
                if needs_pad_interaction:
                    ch = pad_cropped(ch, pad_interaction)
                if i in dilation_channels:
                    ch = iterative_3x3_same_padding_pool3d(ch[None], max_pool_ks)[0]
                interactions_out[i : i + 1] = interpolate(ch[None], patch_size, mode="area")[0]
                del ch
            del interactions_tensor
            interactions_tensor = interactions_out

            # Keep image and interaction tensors in identical spatial frames before concatenation.
            # Interactions use area downsampling (with selective dilation beforehand), image uses trilinear.
            crop_img = crop_img.to(self.device, non_blocking=True)
            if any(x for pair in pad_image for x in pair):
                crop_img = pad_cropped(crop_img, pad_image)
            crop_img = interpolate(crop_img[None], patch_size, mode="trilinear")[0]

            empty_cache(self.device)
        else:
            # zoom_out_factor == 1: transfer both tensors to GPU, then pad if needed
            crop_img = crop_img.to(self.device, non_blocking=True)
            interactions_tensor = interactions_tensor.to(self.device, non_blocking=True)
            previous_prediction = previous_prediction.to(self.device, non_blocking=True)
            if any(x for pair in pad_image for x in pair):
                crop_img = pad_cropped(crop_img, pad_image)
            if any(x for pair in pad_interaction for x in pair):
                interactions_tensor = pad_cropped(interactions_tensor, pad_interaction)
                previous_prediction = pad_cropped(previous_prediction, pad_interaction)
            previous_prediction = previous_prediction[0]

        self._normalize_interaction_channels_for_network_(interactions_tensor)
        input_for_predict = torch.cat((crop_img, interactions_tensor))
        del crop_img, interactions_tensor
        empty_cache(self.device)
        return input_for_predict, scaled_patch_size, scaled_bbox, previous_prediction

    def _refine_coarse(self, bboxes_ordered: List[List[List[int]]]):
        start_refinement = time()
        prev_seg_channel = self._get_prev_seg_channel()

        if self.verbose:
            print(f"Using {len(bboxes_ordered)} bounding boxes for refinement")

        self._refine_coarse_with_local_cache(bboxes_ordered, prev_seg_channel)
        end_refinement = time()
        print(
            f"Took {round(end_refinement - start_refinement, 3)} s for refining the segmentation with {len(bboxes_ordered)} bounding boxes"
        )

    def _refine_coarse_with_local_cache(self, bboxes_ordered: List[List[List[int]]], prev_seg_channel: int) -> None:
        cache_bbox, cache_image, cache_interactions = self._build_refinement_local_cache(bboxes_ordered)

        for refinement_bbox in bboxes_ordered:
            local_bbox = [
                [lb - cache_dim[0], ub - cache_dim[0]] for (lb, ub), cache_dim in zip(refinement_bbox, cache_bbox)
            ]
            spatial_slicer = tuple(slice(lb, ub) for lb, ub in local_bbox)
            image_patch = cache_image[spatial_slicer][None]
            interactions_patch = cache_interactions[(slice(None), *spatial_slicer)]
            if cache_image.device == self.device:
                patch = torch.cat((image_patch, interactions_patch), dim=0)
            else:
                patch = torch.cat(
                    (
                        image_patch.to(self.device, non_blocking=(self.device.type == "cuda")),
                        interactions_patch.to(self.device, non_blocking=(self.device.type == "cuda")),
                    ),
                    dim=0,
                )

            # .contiguous(): see _predict — required for torch.compile with possibly non-contiguous input.
            pred = self.network(patch[None].contiguous())[0].argmax(0).detach()
            paste_tensor(
                cache_interactions,
                pred.to(cache_interactions.device, dtype=cache_interactions.dtype),
                local_bbox,
                channel_idx=prev_seg_channel,
            )
            del image_patch, interactions_patch, patch
            del pred

        final_prev_seg = cache_interactions[prev_seg_channel]
        paste_tensor(self.interactions, final_prev_seg, cache_bbox, channel_idx=prev_seg_channel)
        self._paste_prediction_to_target_buffer(final_prev_seg, cache_bbox)

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
        has_change: bool = False
        for dim in range(pred.ndim):
            if has_change:
                break
            for idx in [0, pred.shape[dim] - 1]:
                slice_prev = prev_pred.index_select(dim, torch.tensor(idx, device=prev_pred.device))
                slice_curr = pred.index_select(dim, torch.tensor(idx, device=self.device)).to(prev_pred.device)
                pixels_prev = torch.sum(slice_prev)
                pixels_current = torch.sum(slice_curr)
                pixels_diff = torch.sum(slice_prev != slice_curr)
                rel_change = max(pixels_prev, pixels_current) / max(min(pixels_prev, pixels_current), 1e-5) - 1
                if pixels_diff > abs_pxl_change_threshold:
                    has_change = True
                    if self.verbose:
                        print(
                            f"continue zooming because change at borders of {pixels_diff} > {abs_pxl_change_threshold}"
                        )
                    break
                if pixels_diff > min_pxl_change_threshold and rel_change > rel_pxl_change_threshold:
                    has_change = True
                    if self.verbose:
                        print(
                            f"continue zooming because relative change of {rel_change} > {rel_pxl_change_threshold} and n_pixels {pixels_diff} > {min_pxl_change_threshold}"
                        )
                    break
                del slice_prev, slice_curr, pixels_prev, pixels_current, pixels_diff
        return has_change

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

        pred_bbox = [
            [seen_dim[0] - scaled_dim[0], seen_dim[1] - scaled_dim[0]]
            for seen_dim, scaled_dim in zip(seen_bbox, scaled_bbox)
        ]
        pred_bbox = [[max(0, lb), min(ub, int(pred.shape[dim]))] for dim, (lb, ub) in enumerate(pred_bbox)]
        local_seen_bbox = [
            [seen_dim[0] - planning_dim[0], seen_dim[1] - planning_dim[0]]
            for seen_dim, planning_dim in zip(seen_bbox, planning_bbox)
        ]

        seen_slicer = tuple(slice(lb, ub) for lb, ub in seen_bbox)
        pred_slicer = tuple(slice(lb, ub) for lb, ub in pred_bbox)
        local_slicer = tuple(slice(lb, ub) for lb, ub in local_seen_bbox)

        prev_sub = self._read_interactions_to_device((prev_seg_ch, *seen_slicer), self.device)

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
        planning_slicer = tuple(slice(lb, ub) for lb, ub in planning_bbox)
        prev_sub = self._read_interactions_to_device((prev_seg_ch, *planning_slicer), self.device)
        diff_local[prev_sub > 0.5] = 1
        del prev_sub

    def _plan_refinement_bboxes(
        self, pred: torch.Tensor, scaled_bbox: List[List[int]], force_full_refine: bool
    ) -> List[List[List[int]]]:
        spatial_shape = tuple(int(i) for i in self.interactions.shape[1:])
        planning_bbox = self._clip_bbox_to_shape(scaled_bbox, spatial_shape)

        if force_full_refine:
            print("Forcing full refinement of entire structure")
            prev_seg_bbox = self._compute_prev_seg_positive_bbox()
            planning_bbox = self._union_bboxes(planning_bbox, prev_seg_bbox)

        if planning_bbox is None:
            center = self.new_interaction_centers[-1]
            return [
                [[ci - pi // 2, ci - pi // 2 + pi] for ci, pi in zip(center, self.configuration_manager.patch_size)]
            ]

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
            center = self.new_interaction_centers[-1]
            return [
                [[ci - pi // 2, ci - pi // 2 + pi] for ci, pi in zip(center, self.configuration_manager.patch_size)]
            ]

        return self._offset_bboxes(local_bboxes, planning_bbox)

    def _add_patch_for_point_interaction(self, coordinates):
        self.new_interaction_zoom_out_factors.append(1)
        self.new_interaction_centers.append(coordinates)
        print(
            f"Added new point interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}"
        )

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
            f"Added new bbox interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}"
        )

    def _add_patch_for_initial_seg_interaction(self, initial_seg):
        return self._generic_add_patch_from_image(initial_seg)

    def _generic_add_patch_from_image(self, image: torch.Tensor, offset: Optional[List[int]] = None):
        if not torch.any(image):
            print("Received empty image prompt. Cannot add patches for prediction")
            return
        if offset is None:
            offset = [0] * image.ndim
        nonzero_indices = torch.nonzero(image, as_tuple=False)
        mn = torch.min(nonzero_indices, dim=0)[0]
        mx = torch.max(nonzero_indices, dim=0)[0]
        roi = [[i.item() + off, x.item() + off + 1] for i, x, off in zip(mn, mx, offset)]
        roi_center = [round((i[0] + i[1]) / 2) for i in roi]
        roi_size = [i[1] - i[0] for i in roi]
        requested_size = [i + j // 3 for i, j in zip(roi_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(
            max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)]))
        )
        self.new_interaction_centers.append(roi_center)
        print(
            f"Added new image interaction: scale {self.new_interaction_zoom_out_factors[-1]}, center {self.new_interaction_centers}"
        )

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
        # With torch.compile the network is compiled lazily on the first forward pass. For a
        # locally hosted model that lag would otherwise surface on the user's first real
        # prediction, where it is far more noticeable than during initialization. Trigger the
        # compilation now with a dummy forward pass so the cost is paid here instead. warmup()
        # is a no-op when the network is not compiled. The server takes care of its own warmup
        # explicitly (it shares one compiled network across sessions via
        # initialize_from_loaded_artifacts), so we only do this on the direct, local entry point.
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
        ``@torch.inference_mode()`` and serializing predict calls with a
        global GPU lock.

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
        trainer_class = recursive_find_python_class(
            join(nnInteractive.__path__[0], "trainer"), trainer_name, "nnInteractive.trainer"
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

        network = trainer_class.build_network_architecture(
            plans_manager,
            configuration_manager,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
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

        self.network = self.network.to(self.device)

    def __del__(self):
        # Be robust to a partially-constructed instance (e.g. __init__ raised on bad arguments):
        # these attributes may not exist yet.
        if hasattr(self, "preprocess_future"):
            self._finish_preprocessing_and_initialize_interactions()
        if hasattr(self, "executor"):
            self.executor.shutdown()


if __name__ == "__main__":
    a = torch.zeros((160, 160, 160), device="cpu")
    a.index_select(0, torch.tensor([0]))
