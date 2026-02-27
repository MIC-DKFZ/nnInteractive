from concurrent.futures import ThreadPoolExecutor
import os
from os import cpu_count
from time import time
from typing import Union, List, Tuple, Optional
import warnings
import re

try:
    import blosc2
    _BLOSC2_AVAILABLE = True
except ImportError:
    _BLOSC2_AVAILABLE = False

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
from nnInteractive.utils.os_shennanigans import is_linux_kernel_6_11
from nnInteractive.utils.rounding import round_to_nearest_odd


class nnInteractiveInferenceSession():
    INFERENCE_SESSION_VERSION = nnInteractive.__version__

    def __init__(self,
                 device: torch.device = torch.device('cuda'),
                 use_torch_compile: bool = False,
                 verbose: bool = False,
                 torch_n_threads: int = 8,
                 do_autozoom: bool = True,
                 use_pinned_memory: bool = True,
                 use_in_mem_compression: bool = False,
                 ):
        """
        Only intended to work with nnInteractiveTrainerV2 and its derivatives
        """
        if use_in_mem_compression:
            if not _BLOSC2_AVAILABLE:
                raise ImportError(
                    "blosc2 is required for use_in_mem_compression=True. "
                    "Install with: pip install blosc2"
                )
            use_pinned_memory = False  # blosc2 and pinned memory are incompatible

        # set as part of initialization
        assert use_torch_compile is False, ('This implementation places the preprocessed image and the interactions '
                                            'into pinned memory for speed reasons. This is incompatible with '
                                            'torch.compile because of inconsistent strides in the memory layout. '
                                            'Note to self: .contiguous() on GPU could be a solution. Unclear whether '
                                            'that will yield a benefit though.')
        self.network = None
        self.label_manager = None
        self.dataset_json = None
        self.trainer_name = None
        self.configuration_manager = None
        self.plans_manager = None
        self.use_pinned_memory = use_pinned_memory
        self.use_in_mem_compression = use_in_mem_compression
        self._interactions_blosc2_shape = None
        self.device = device
        self.use_torch_compile = use_torch_compile
        self.interaction_decay = None
        self.current_interaction_intensity: float = 1.0
        self._fp16_max_value = float(torch.finfo(torch.float16).max)
        # Keep renormalized interaction magnitudes around 1/10 of fp16 max to preserve headroom.
        self._interaction_renorm_target = self._fp16_max_value / 10
        self.num_interaction_channels: int = 7
        self.supported_interactions: dict = {}
        self.channel_mapping: dict = {}
        self.supports_initial_label: bool = True
        self.supports_zero_shot_label_refinement: bool = True

        # image specific
        self.interactions: torch.Tensor = None
        self.preprocessed_image: torch.Tensor = None
        self.preprocessed_props = None
        self.target_buffer: Union[np.ndarray, torch.Tensor] = None

        # this will be set when loading the model (initialize_from_trained_model_folder)
        self.pad_mode_data = self.preferred_scribble_thickness = self.point_interaction = None

        self.verbose = verbose

        self.do_autozoom: bool = do_autozoom

        torch.set_num_threads(min(torch_n_threads, cpu_count()))

        self.original_image_shape = None

        self.new_interaction_zoom_out_factors: List[float] = []
        self.new_interaction_centers = []
        self.has_positive_bbox = False

        # Create a thread pool executor for background tasks.
        # this only takes care of preprocessing and interaction memory initialization so there is no need to give it
        # more than 2 workers
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.preprocess_future = None
        self.interactions_future = None

    @staticmethod
    def _is_official_checkpoint(plans: dict, checkpoint: dict) -> bool:
        return plans.get('dataset_name') == 'Dataset225_nnInteractiveV2' and \
            checkpoint.get('init_args', {}).get('configuration') == '3d_fullres_ps192_bs24'

    @staticmethod
    def _version_to_tuple(version: str) -> Tuple[int, ...]:
        return tuple(int(i) for i in re.findall(r'\d+', version))

    def _legacy_default_capability(self) -> dict:
        return {
            'supported_interactions': {
                'scribble': True,
                'lasso': True,
                'points': True,
                'bbox2d': True,
                'bbox3d': False,
            },
            'supports_initial_label': True,
            'supports_zero_shot_label_refinement': True,
            'interaction_channels': 6,
            'channel_mapping': {
                'prev_seg': 0,
                'bbox2d': (1, 2),
                'bbox3d': (1, 2),
                'lasso': (1, 2),
                'points': (3, 4),
                'scribble': (5, 6),
            },
        }

    def _to_positive_channel_index(self, idx: int) -> int:
        return idx if idx >= 0 else self.num_interaction_channels + idx

    @staticmethod
    def _parse_channel_pair(channel_name: str, raw_channels) -> Tuple[int, int]:
        if not isinstance(raw_channels, (tuple, list)) or len(raw_channels) != 2:
            raise ValueError(
                f"Invalid channel mapping for '{channel_name}': expected a pair [pos, neg], got {raw_channels}."
            )
        return int(raw_channels[0]), int(raw_channels[1])

    def _resolve_channel_pair(self, channel_name: str, override_capability_checks: bool) -> Tuple[int, int]:
        if channel_name in self.channel_mapping:
            return self._parse_channel_pair(channel_name, self.channel_mapping[channel_name])
        if override_capability_checks:
            warnings.warn(
                f"Interaction '{channel_name}' was forced but no channel mapping exists in capability metadata.",
                RuntimeWarning
            )
        raise ValueError(
            f"Interaction '{channel_name}' cannot be executed because no channel mapping was found."
        )

    def _is_interaction_supported(self, interaction_name: str) -> bool:
        if interaction_name in ('lasso', 'scribble', 'points', 'bbox2d', 'bbox3d'):
            return bool(self.supported_interactions.get(interaction_name, False))
        if interaction_name == 'initial_label':
            return bool(self.supports_initial_label)
        return False

    def _get_prev_seg_channel(self) -> int:
        return int(self.channel_mapping['prev_seg'])

    def _get_dilation_channels_for_resample(self) -> List[int]:
        dilation_channels = set()
        # During zoom-out, point/scribble signals can disappear when area interpolation averages tiny sparse
        # structures away. We therefore dilate only these "thin prompt" channels before resampling.
        for key in ('points', 'scribble'):
            if not self.supported_interactions.get(key, False):
                continue
            if key not in self.channel_mapping:
                continue
            pos_ch, neg_ch = self._parse_channel_pair(key, self.channel_mapping[key])
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

    @staticmethod
    def _infer_num_interaction_channels_from_mapping(channel_mapping: dict) -> int:
        max_positive_index = -1
        max_negative_magnitude = 0

        for k, v in channel_mapping.items():
            if k == 'prev_seg':
                indices = [int(v)]
            else:
                pos_ch, neg_ch = nnInteractiveInferenceSession._parse_channel_pair(k, v)
                indices = [pos_ch, neg_ch]

            for idx in indices:
                if idx >= 0:
                    max_positive_index = max(max_positive_index, idx)
                else:
                    max_negative_magnitude = max(max_negative_magnitude, abs(idx))

        # Positive indexing is 0-based, while negative indexing is 1-based-from-end.
        return max(max_positive_index + 1, max_negative_magnitude, 1)

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
            self.current_interaction_intensity = min(self.current_interaction_intensity, self._interaction_renorm_target)
            return
        scale = self._interaction_renorm_target / self.current_interaction_intensity
        self.interactions[channels_to_scale] *= scale
        self.current_interaction_intensity = self._interaction_renorm_target

    def _crop_and_pad_interactions_channel0(self, bbox) -> torch.Tensor:
        """Read interactions[prev_seg_channel] at bbox with zero padding.
        For blosc2: decompresses only the needed chunks, not the full channel."""
        prev_seg_ch = self._get_prev_seg_channel()
        out_shape = tuple(int(i[1] - i[0]) for i in bbox)
        out = torch.zeros(out_shape, dtype=torch.float16)
        seen_bbox = [[max(0, i[0]), min(i[1], s)]
                     for i, s in zip(bbox, self.interactions.shape[1:])]
        if any(i[1] <= i[0] for i in seen_bbox):
            return out
        source_slices = tuple(slice(i[0], i[1]) for i in seen_bbox)
        target_slices = tuple(slice(i[0] - b[0], i[1] - b[0])
                              for i, b in zip(seen_bbox, bbox))
        if self.use_in_mem_compression:
            sub = np.asarray(self.interactions[(prev_seg_ch, *source_slices)])
            out[target_slices] = torch.from_numpy(sub)
        else:
            out[target_slices] = self.interactions[prev_seg_ch][source_slices].cpu()
        return out

    def _interactions_inplace_maximum(self, channel_idx: int, int_slicer, new_values) -> None:
        """In-place element-wise maximum for a subregion of a channel."""
        if self.use_in_mem_compression:
            if isinstance(new_values, torch.Tensor):
                new_values = new_values.cpu().numpy().astype(np.float16)
            full_slicer = (channel_idx, *int_slicer)
            current_sub = np.asarray(self.interactions[full_slicer])
            np.maximum(current_sub, new_values, out=current_sub)
            self.interactions[full_slicer] = current_sub
        else:
            torch.maximum(self.interactions[channel_idx][int_slicer], new_values,
                          out=self.interactions[channel_idx][int_slicer])

    def _mask_interaction_channel_with_prediction(self, ch: int, prediction_with_coarse) -> None:
        """Zero out channel ch where prediction_with_coarse <= 0.5."""
        if self.use_in_mem_compression:
            prev_seg_ch = self._get_prev_seg_channel()
            if prediction_with_coarse is None:
                # Read from blosc2 directly when no numpy copy is available.
                mask = np.asarray(self.interactions[prev_seg_ch]) > 0.5
            elif isinstance(prediction_with_coarse, torch.Tensor):
                mask = prediction_with_coarse.numpy() > 0.5
            else:
                mask = np.asarray(prediction_with_coarse) > 0.5
            ch_data = np.asarray(self.interactions[ch])
            ch_data[~mask] = 0
            self.interactions[ch] = ch_data.astype(np.float16)
        else:
            self.interactions[ch][(~(prediction_with_coarse > 0.5))] = 0

    def _write_interactions_channel(self, channel_idx: int, value) -> None:
        """Write a full channel. Handles torch→numpy for blosc2."""
        if self.use_in_mem_compression:
            if isinstance(value, torch.Tensor):
                value = value.cpu().numpy().astype(np.float16)
            self.interactions[channel_idx] = value
        else:
            if isinstance(value, np.ndarray):
                value = torch.from_numpy(value)
            self.interactions[channel_idx] = value.to(
                self.interactions.device, dtype=self.interactions.dtype)

    def _prepare_new_interaction_intensity(self):
        if self.interaction_decay is None:
            return
        if not (0 < self.interaction_decay <= 1):
            raise ValueError(f"interaction_decay must be in (0, 1], got {self.interaction_decay}.")
        if self.interaction_decay < 1:
            self.current_interaction_intensity *= (1 / self.interaction_decay)
            self._renormalize_interactions_if_needed()

    def _normalize_interaction_channels_for_network_(self, interaction_tensor: torch.Tensor):
        if interaction_tensor is None or self.current_interaction_intensity == 0:
            return
        if self.current_interaction_intensity == 1:
            return
        prev_seg_channel = self._get_prev_seg_channel()
        channels_to_normalize = [i for i in range(interaction_tensor.shape[0]) if i != prev_seg_channel]
        if len(channels_to_normalize) > 0:
            interaction_tensor[channels_to_normalize] /= self.current_interaction_intensity

    def _apply_capability(self, capability: dict):
        default_capability = self._legacy_default_capability()
        default_supported = default_capability['supported_interactions']
        default_mapping = default_capability['channel_mapping']
        supported_keys = set(default_supported.keys())
        mapping_keys = set(default_mapping.keys())

        raw_supported = capability.get('supported_interactions', {}) if isinstance(capability, dict) else {}
        filtered_supported = {k: bool(v) for k, v in raw_supported.items() if k in supported_keys}
        self.supported_interactions = {**default_supported, **filtered_supported}
        self.supports_initial_label = capability.get('supports_initial_label', True)
        self.supports_zero_shot_label_refinement = capability.get('supports_zero_shot_label_refinement', True)

        raw_mapping = capability.get('channel_mapping', {}) if isinstance(capability, dict) else {}
        self.channel_mapping = dict(default_mapping)
        for k, v in raw_mapping.items():
            if k not in mapping_keys:
                continue
            if k == 'prev_seg':
                self.channel_mapping[k] = int(v)
            else:
                self.channel_mapping[k] = self._parse_channel_pair(k, v)

        if 'interaction_channels' in capability:
            self.num_interaction_channels = int(capability['interaction_channels']) + 1
        else:
            self.num_interaction_channels = self._infer_num_interaction_channels_from_mapping(self.channel_mapping)

        # Normalize all channel indices to positive indexing once at load time.
        self.channel_mapping['prev_seg'] = self._to_positive_channel_index(int(self.channel_mapping['prev_seg']))
        for k, v in list(self.channel_mapping.items()):
            if k == 'prev_seg':
                continue
            pos_ch, neg_ch = self._parse_channel_pair(k, v)
            self.channel_mapping[k] = (
                self._to_positive_channel_index(pos_ch),
                self._to_positive_channel_index(neg_ch),
            )

    def _validate_capability_version(self, capability: dict):
        current_class = self.__class__.__name__
        required_class = capability.get('inference_class', current_class)
        if required_class != current_class:
            raise RuntimeError(
                f"Checkpoint requires inference class '{required_class}', but current class is "
                f"'{current_class}'."
            )

        min_version = capability.get('inference_class_min_version')
        if min_version is None:
            return
        if self._version_to_tuple(min_version) > self._version_to_tuple(self.INFERENCE_SESSION_VERSION):
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
        assert image.ndim == 4, f'expected a 4d image as input, got {image.ndim}d. Shape {image.shape}'
        if self.verbose:
            print(f'Initialize with raw image shape {image.shape}')

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
        self.has_positive_bbox = False

    def _initialize_interactions(self, image_torch: torch.Tensor):
        shape = (self.num_interaction_channels, *image_torch.shape[1:])
        if self.use_in_mem_compression:
            if self.verbose:
                print('Initialize interactions with blosc2 in-memory compression')
            self.interactions = blosc2.zeros(
                shape, dtype=np.float16,
                chunks=(1, *[min(64, s) for s in shape[1:]]),
                blocks=(1, *[min(32, s) for s in shape[1:]]),
                cparams={'codec': blosc2.Codec.LZ4, 'clevel': 5, 'nthreads': os.cpu_count()},
                dparams={'nthreads': 4}
            )
            self._interactions_blosc2_shape = shape
        else:
            # there is a bug in 6.11 that doesn't allow pinning large tensors
            use_pinned = not is_linux_kernel_6_11() and self.use_pinned_memory and self.device.type == 'cuda'
            if self.verbose:
                print(f'Initialize interactions. Pinned: {use_pinned}')
            # Create the interaction tensor based on the target shape.
            self.interactions = torch.zeros(
                shape,
                device='cpu',
                dtype=torch.float16,
                pin_memory=use_pinned
            )

    def _background_set_image(self, image: np.ndarray, image_properties: dict):
        # Convert and clone the image tensor.
        image = torch.from_numpy(image.copy())#.to(self.device)

        # Crop to nonzero region.
        if self.verbose:
            print('Cropping input image to nonzero region')
        nonzero_idx = torch.where(image != 0)
        # Create bounding box: for each dimension, get the min and max (plus one) of the nonzero indices.
        bbox = [[i.min().item(), i.max().item() + 1] for i in nonzero_idx]
        del nonzero_idx
        slicer = bounding_box_to_slice(bbox)  # Assuming this returns a tuple of slices.
        image = image[slicer].float()
        if self.verbose:
            print(f'Cropped image shape: {image.shape}')

        # As soon as we have the target shape, start initializing the interaction tensor in its own thread.
        self.interactions_future = self.executor.submit(self._initialize_interactions, image)

        # Normalize the cropped image.
        if self.verbose:
            print('Normalizing cropped image')
        image -= image.mean()
        image /= image.std()

        self.preprocessed_image = image

        self.preprocessed_props = {'bbox_used_for_cropping': bbox[1:]}

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
            if self.use_in_mem_compression:
                del self.interactions
                self.interactions = blosc2.zeros(
                    self._interactions_blosc2_shape, dtype=np.float16,
                    chunks=(1, *[min(64, s) for s in self._interactions_blosc2_shape[1:]]),
                    blocks=(1, *[min(32, s) for s in self._interactions_blosc2_shape[1:]]),
                    cparams={'codec': blosc2.Codec.LZ4, 'clevel': 5, 'nthreads': os.cpu_count()},
                    dparams={'nthreads': 4}
                )
            else:
                self.interactions.fill_(0)
        self.current_interaction_intensity = 1.0

        if self.target_buffer is not None:
            if isinstance(self.target_buffer, np.ndarray):
                self.target_buffer.fill(0)
            elif isinstance(self.target_buffer, torch.Tensor):
                self.target_buffer.zero_()
        empty_cache(self.device)
        self.has_positive_bbox = False

    def add_bbox_interaction(self, bbox_coords, include_interaction: bool, run_prediction: bool = True,
                             override_capability_checks: bool = False):
        # sanity check
        raw_bbox_size = [i[1] - i[0] for i in bbox_coords]
        if any([i == 0 for i in raw_bbox_size]):
            raise ValueError(f'Given bounding box size is zero in at least one dimension: {bbox_coords}')

        # capability check
        dims_with_size_one = sum(i == 1 for i in raw_bbox_size)
        # if we do not support 3D bboxes we need to reject 3D bboxes!
        if not self._is_interaction_supported('bbox3d') and dims_with_size_one == 0:
            raise ValueError(f"The given bounding box {bbox_coords} has size {raw_bbox_size} indicating a 3D "
                             f"bounding box. This is not supported by the loaded model checkpoint.")
        # a 2D bounding box is in principle a 3D box as well. Since 2D bboxes work better, we prefer to use a given
        # bbox as 2d if possible (sized 1 in at least one dim and bbox2d supported)
        bbox_kind = 'bbox2d' if (dims_with_size_one >= 1 and self._is_interaction_supported('bbox2d')) else 'bbox3d'
        self._check_capability_or_warn(bbox_kind, override_capability_checks)
        bbox_pos_channel, bbox_neg_channel = self._resolve_channel_pair(bbox_kind, override_capability_checks)

        lbs_transformed = [round(i) for i in transform_coordinates_noresampling([i[0] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        ubs_transformed = [round(i) for i in transform_coordinates_noresampling([i[1] for i in bbox_coords],
                                                             self.preprocessed_props['bbox_used_for_cropping'])]
        transformed_bbox_coordinates = [[i, j] for i, j in zip(lbs_transformed, ubs_transformed)]

        if self.verbose:
            print(f'Adding bounding box coordinates.\n'
                  f'Raw: {bbox_coords}\n'
                  f'Transformed: {transformed_bbox_coordinates}\n'
                  f"Crop Bbox: {self.preprocessed_props['bbox_used_for_cropping']}")

        # Prevent collapsed bounding boxes and clip to image shape
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
            print(f'Bbox coordinates after clip to image boundaries and preventing dim collapse:\n'
                  f'Bbox: {transformed_bbox_coordinates}\n'
                  f'Internal image shape: {self.preprocessed_image.shape}')

        if include_interaction:
            self.has_positive_bbox = True

        self._add_patch_for_bbox_interaction(transformed_bbox_coordinates)

        self._prepare_new_interaction_intensity()

        # place bbox
        slicer = tuple([slice(*i) for i in transformed_bbox_coordinates])
        channel = bbox_pos_channel if include_interaction else bbox_neg_channel
        self.interactions[(channel, *slicer)] = self.current_interaction_intensity

        # forward pass
        if run_prediction:
            self._predict()

    def add_point_interaction(self, coordinates: Tuple[int, ...], include_interaction: bool, run_prediction: bool = True,
                              override_capability_checks: bool = False):
        self._check_capability_or_warn('points', override_capability_checks)
        point_pos_channel, point_neg_channel = self._resolve_channel_pair('points', override_capability_checks)
        self._finish_preprocessing_and_initialize_interactions()

        transformed_coordinates = [round(i) for i in transform_coordinates_noresampling(coordinates,
                                                             self.preprocessed_props['bbox_used_for_cropping'])]

        self._add_patch_for_point_interaction(transformed_coordinates)

        self._prepare_new_interaction_intensity()

        interaction_channel = point_pos_channel if include_interaction else point_neg_channel
        if self.use_in_mem_compression:
            # place_point reads/writes only the structuring element subregion via channel_idx
            self.point_interaction.place_point(
                transformed_coordinates, self.interactions,
                channel_idx=interaction_channel,
                intensity_scale=self.current_interaction_intensity)
        else:
            self.interactions[interaction_channel] = self.point_interaction.place_point(
                transformed_coordinates, self.interactions[interaction_channel],
                intensity_scale=self.current_interaction_intensity)
        if run_prediction:
            self._predict()

    def _add_image_interaction(self, image: np.ndarray, interaction_channel: int, run_prediction: bool,
                               interaction_bbox: Optional[List[List[int]]], patch_fn):
        if interaction_bbox is None:
            interaction_bbox = [[0, s] for s in self.original_image_shape[1:]]

        assert len(interaction_bbox) == 3
        bbox_size = [ub - lb for lb, ub in interaction_bbox]
        assert all(s > 0 for s in bbox_size), \
            'each dimension of interaction_bbox must have positive size'
        assert list(image.shape) == bbox_size, \
            f'image shape {list(image.shape)} must match interaction_bbox size {bbox_size}'
        assert all(lb >= 0 and ub <= orig_dim
                   for (lb, ub), orig_dim in zip(interaction_bbox, self.original_image_shape[1:])), \
            f'interaction_bbox {interaction_bbox} exceeds original image bounds {list(self.original_image_shape[1:])}'

        self._finish_preprocessing_and_initialize_interactions()

        lbs_internal = [round(i) for i in transform_coordinates_noresampling(
            [ib[0] for ib in interaction_bbox], self.preprocessed_props['bbox_used_for_cropping'])]
        ubs_internal = [round(i) for i in transform_coordinates_noresampling(
            [ib[1] for ib in interaction_bbox], self.preprocessed_props['bbox_used_for_cropping'])]

        image_t = torch.from_numpy(image)
        patch_fn(image_t, offset=lbs_internal)

        self._prepare_new_interaction_intensity()

        interaction_shape = self.interactions.shape[1:]
        clipped_lb = [max(0, lb) for lb in lbs_internal]
        clipped_ub = [min(ub, s) for ub, s in zip(ubs_internal, interaction_shape)]
        src_lb = [cl - lb for cl, lb in zip(clipped_lb, lbs_internal)]
        src_ub = [src_lb[d] + (clipped_ub[d] - clipped_lb[d]) for d in range(3)]
        int_slicer = tuple(slice(a, b) for a, b in zip(clipped_lb, clipped_ub))
        src_slicer = tuple(slice(a, b) for a, b in zip(src_lb, src_ub))
        if self.use_in_mem_compression:
            new_values = image_t[src_slicer].cpu().numpy()
            if self.current_interaction_intensity != 1:
                new_values = new_values * self.current_interaction_intensity
            new_values = new_values.astype(np.float16)
        else:
            new_values = image_t[src_slicer].to(self.interactions.device, dtype=self.interactions.dtype)
            if self.current_interaction_intensity != 1:
                new_values = new_values * self.current_interaction_intensity
        self._interactions_inplace_maximum(interaction_channel, int_slicer, new_values)
        del new_values
        del image_t
        empty_cache(self.device)

        if run_prediction:
            self._predict()

    def add_scribble_interaction(self, scribble_image: np.ndarray, include_interaction: bool, run_prediction: bool = True,
                                 override_capability_checks: bool = False,
                                 interaction_bbox: Optional[List[List[int]]] = None):
        if True: #self.verbose:
            print(f'Add new scribble of shape {scribble_image.shape} and bbox {interaction_bbox}')
        self._check_capability_or_warn('scribble', override_capability_checks)
        pos_channel, neg_channel = self._resolve_channel_pair('scribble', override_capability_checks)
        self._add_image_interaction(scribble_image, pos_channel if include_interaction else neg_channel,
                                    run_prediction, interaction_bbox, self._add_patch_for_scribble_interaction)

    def add_lasso_interaction(self, lasso_image: np.ndarray, include_interaction: bool, run_prediction: bool = True,
                              override_capability_checks: bool = False,
                              interaction_bbox: Optional[List[List[int]]] = None):
        if True: #self.verbose:
            print(f'Add new lasso of shape {lasso_image.shape} and bbox {interaction_bbox}')
        self._check_capability_or_warn('lasso', override_capability_checks)
        pos_channel, neg_channel = self._resolve_channel_pair('lasso', override_capability_checks)
        self._add_image_interaction(lasso_image, pos_channel if include_interaction else neg_channel,
                                    run_prediction, interaction_bbox, self._add_patch_for_lasso_interaction)

    def add_initial_seg_interaction(self, initial_seg: np.ndarray, run_prediction: bool = False,
                                    override_capability_checks: bool = False):
        """
        WARNING THIS WILL RESET INTERACTIONS!
        """
        self._check_capability_or_warn('initial_label', override_capability_checks)
        assert all([i == j for i, j in zip(self.original_image_shape[1:], initial_seg.shape)]), f'Given initial seg must match input image shape. Input image was: {self.original_image_shape[1:]}, given: {initial_seg.shape}'

        self._finish_preprocessing_and_initialize_interactions()

        self.reset_interactions()

        if isinstance(self.target_buffer, np.ndarray):
            self.target_buffer[:] = initial_seg

        initial_seg = torch.from_numpy(initial_seg)

        if isinstance(self.target_buffer, torch.Tensor):
            self.target_buffer[:] = initial_seg

        # crop (as in preprocessing)
        initial_seg = crop_and_pad_nd(initial_seg, self.preprocessed_props['bbox_used_for_cropping'])

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
        if self.use_in_mem_compression:
            print('Current cratio', self.interactions.cratio)

        assert self.pad_mode_data == 'constant', 'pad modes other than constant are not implemented here'
        assert len(self.new_interaction_centers) == len(self.new_interaction_zoom_out_factors)
        prev_seg_channel = self._get_prev_seg_channel()
        if len(self.new_interaction_centers) == 0:
            print('No patch queued for prediction. Nothing to do.')
            return

        if len(self.new_interaction_centers) > 1:
            print('It seems like more than one interaction was added since the last prediction. This is not '
                  'recommended and may cause unexpected behavior or inefficient predictions\n'
                  '!!!WE NO LONGER RUN ONE PREDICTION PER CENTER AND ONLY USE THE LAST ADDED INTERACTION AS CENTER!!!')
        prediction_center, zoom_out_factor = self.new_interaction_centers[-1], self.new_interaction_zoom_out_factors[-1]
        zoom_out_factor = min(4, zoom_out_factor)

        start_predict = time()
        with torch.autocast(self.device.type, enabled=True) if self.device.type == 'cuda' else dummy_context():
            # make a prediction at zoom_out_factor, remember max_zoom_out_factor
            start_initial_pred = time()
            input_for_predict, scaled_patch_size, scaled_bbox = self._build_network_input(prediction_center, zoom_out_factor)
            pred = self.network(input_for_predict[None])[0].argmax(0).detach()
            del input_for_predict

            # detect changes at border. If there are, we enter autozoom
            previous_prediction = self._crop_and_pad_interactions_channel0(scaled_bbox)

            if not all([i == j for i, j in zip(pred.shape, previous_prediction.shape)]):
                previous_prediction = \
                interpolate(previous_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]

            has_change = self._detect_change_at_border(pred, previous_prediction)
            del previous_prediction

            print(f'Took {round(time() - start_initial_pred, 3)} s for initial prediction at zoom out factor {zoom_out_factor}')

            # maybe do zoom out
            zoom_out_growth_factor = 1.5
            start_zoomout = time()
            while has_change and self.do_autozoom:
                print(f'AutoZoom zoom out factor {zoom_out_factor}')
                # we allow a max zoom out of 4
                if zoom_out_factor >= 4:
                    break
                else:
                    zoom_out_factor *= zoom_out_growth_factor
                    zoom_out_factor = min(4, zoom_out_factor)

                input_for_predict, scaled_patch_size, scaled_bbox = self._build_network_input(prediction_center, zoom_out_factor)
                pred = self.network(input_for_predict[None])[0].argmax(0).detach()
                del input_for_predict

                # detect changes at border. If there are, we enter autozoom
                previous_prediction = self._crop_and_pad_interactions_channel0(scaled_bbox)

                if not all([i == j for i, j in zip(pred.shape, previous_prediction.shape)]):
                    previous_prediction_resized = \
                    interpolate(previous_prediction[None, None].to(float), pred.shape, mode='nearest')[0, 0]
                else:
                    previous_prediction_resized = previous_prediction

                has_change = self._detect_change_at_border(pred, previous_prediction_resized)

            if zoom_out_factor > 1:
                print(f'Zoom out took {round(time() - start_zoomout, 3)} s, max zoom out factor {zoom_out_factor}')
            else:
                print('No zoom out necessary')

            if zoom_out_factor == 1:
                # simply place pred in self.interactions[0] and target buffer
                if self.use_in_mem_compression:
                    paste_tensor(self.interactions, pred.half(), scaled_bbox, channel_idx=prev_seg_channel)
                else:
                    paste_tensor(self.interactions[prev_seg_channel], pred.half(), scaled_bbox)
                bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                        zip(scaled_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
                paste_tensor(self.target_buffer, pred.to(self.target_buffer.device) if isinstance(self.target_buffer, torch.Tensor) else pred.to('cpu'), bbox)
                print('No refinement necessary')
            else:
                # do refinement

                # we need to resize the prediction to the correct shape and place it in a copy of self.interactions[0]
                # we don't want to place it into self.interactions[0] because we will update self.interactions[0] as
                # part of the refinement. Updating it could cause areas that are not refined to become coarse.
                # For blosc2: prediction_with_coarse is set to None here; it will be read from blosc2 in
                # _refine_coarse as needed (bbox masking reads from blosc2 directly when None).
                if self.use_in_mem_compression:
                    prediction_with_coarse = None
                else:
                    prediction_with_coarse = self.interactions[prev_seg_channel]

                if not all([i == j for i, j in zip(pred.shape, scaled_patch_size)]):
                    pred = (interpolate(pred[None, None].to(float), scaled_patch_size, mode='trilinear')[
                                0, 0] >= 0.5).to(torch.uint8)

                # compute the difference map
                diff_map = self._compute_diff_map(
                    pred,
                    None if self.use_in_mem_compression else self.interactions[prev_seg_channel],
                    scaled_bbox, scaled_patch_size
                )

                if force_full_refine:
                    print('Forcing full refinement of entire structure')
                    if self.use_in_mem_compression:
                        chunk_depth = 64
                        for d0 in range(0, self.interactions.shape[1], chunk_depth):
                            d1 = min(self.interactions.shape[1], d0 + chunk_depth)
                            ch0_chunk = torch.from_numpy(np.asarray(
                                self.interactions[(prev_seg_channel, slice(d0, d1), slice(None), slice(None))]))
                            diff_map[d0:d1][ch0_chunk > 0] = 1
                    else:
                        diff_map[self.interactions[prev_seg_channel] > 0] = 1

                # place resized coarse segmentation into interactions channel 0. Needed for network input
                if self.use_in_mem_compression:
                    paste_tensor(self.interactions, pred, scaled_bbox, channel_idx=prev_seg_channel)
                else:
                    paste_tensor(prediction_with_coarse, pred, scaled_bbox)

                self._refine_coarse(diff_map, prediction_with_coarse)

                del prediction_with_coarse

        print(f'Done. Total time {round(time() - start_predict, 3)}s')

        self.new_interaction_centers = []
        self.new_interaction_zoom_out_factors = []
        empty_cache(self.device)

    def _build_network_input(self, prediction_center, zoom_out_factor):
        scaled_patch_size = [round(i * zoom_out_factor) for i in self.configuration_manager.patch_size]
        scaled_bbox = [[c - p // 2, c + p // 2 + p % 2] for c, p in zip(prediction_center, scaled_patch_size)]

        # cropping happens on CPU, padding happens on GPU (later)
        crop_img, pad_image = crop_to_valid(self.preprocessed_image, scaled_bbox)
        crop_interactions, pad_interaction = crop_to_valid(self.interactions, scaled_bbox)
        crop_img = crop_img.to(self.device, non_blocking=True)
        # For blosc2, crop_to_valid returns a numpy array; convert to torch first.
        if not isinstance(crop_interactions, torch.Tensor):
            crop_interactions = torch.from_numpy(np.asarray(crop_interactions))
        crop_interactions = crop_interactions.to(self.device, non_blocking=True)

        # resize input_for_predict (which may be larger than patch size) to patch size
        # this implementation may not seem straightforward but it does save VRAM which is crucial here
        if not all([i == j for i, j in zip(self.configuration_manager.patch_size, scaled_patch_size)]):
            if any([x for y in pad_interaction for x in y]):
                tmp = pad_cropped(crop_interactions, pad_interaction)
            else:
                tmp = crop_interactions
            del crop_interactions

            max_pool_ks = round_to_nearest_odd(zoom_out_factor * 2 - 1)
            # point+, point-, scribble+, scribble-
            if max_pool_ks > 1:
                # dilate to preserve interactions after downsampling
                for i in self._get_dilation_channels_for_resample():
                    if 0 <= i < tmp.shape[0]:
                        tmp[i:i+1] = iterative_3x3_same_padding_pool3d(tmp[None, i:i+1], max_pool_ks)[0]
            crop_interactions_resampled_gpu = interpolate(tmp[None], self.configuration_manager.patch_size, mode='area')[0]

            del tmp

            # crop_img is already on device
            crop_img = interpolate(
                pad_cropped(crop_img, pad_image)[None] if any([x for y in pad_interaction for x in y]) else crop_img[
                    None], self.configuration_manager.patch_size, mode='trilinear')[0]
            crop_interactions = crop_interactions_resampled_gpu

            del crop_interactions_resampled_gpu
            empty_cache(self.device)
        else:
            # crop_img is already on device
            crop_img = pad_cropped(crop_img, pad_image) if any([x for y in pad_interaction for x in y]) else crop_img
            crop_interactions = pad_cropped(crop_interactions, pad_interaction) if any([x for y in pad_interaction for x in y]) else crop_interactions

        self._normalize_interaction_channels_for_network_(crop_interactions)
        input_for_predict = torch.cat((crop_img, crop_interactions))
        del crop_img, crop_interactions
        empty_cache(self.device)
        return input_for_predict, scaled_patch_size, scaled_bbox

    def _refine_coarse(self, diff_map, prediction_with_coarse):
        start_refinement = time()
        prev_seg_channel = self._get_prev_seg_channel()

        if self.has_positive_bbox:
            # Only do bbox->pseudo-lasso masking if lasso is available, and only on channels that are shared
            # between bbox positive channels and the positive lasso channel.
            if self.supported_interactions.get('lasso', False) and 'lasso' in self.channel_mapping:
                lasso_positive_channel, _ = self._parse_channel_pair('lasso', self.channel_mapping['lasso'])
                positive_bbox_channels = set()
                for bbox_key in ('bbox2d', 'bbox3d'):
                    if bbox_key not in self.channel_mapping:
                        continue
                    pos_ch, _ = self._parse_channel_pair(bbox_key, self.channel_mapping[bbox_key])
                    positive_bbox_channels.add(pos_ch)

                for ch in positive_bbox_channels.intersection({lasso_positive_channel}):
                    self._mask_interaction_channel_with_prediction(ch, prediction_with_coarse)
            self.has_positive_bbox = False

        bboxes_ordered = generate_bounding_boxes(diff_map, self.configuration_manager.patch_size, stride='auto',
                                                 margin=(10, 10, 10), max_depth=3)
        # if no bounding boxes are returned we basically have almost no changes. Still we should at least perform
        # refinement in the bounding box where the interaction was as the user evidently wanted something here.
        if len(bboxes_ordered) == 0:
            # build one bbox around self.new_interaction_centers[-1]
            center = self.new_interaction_centers[-1]
            bboxes_ordered = [[[ci - pi // 2, ci - pi // 2 + pi] for ci, pi in zip(center, self.configuration_manager.patch_size)]]
            # print('Debug: built dummy bboxes_ordered due to empty diff map')

        del diff_map
        empty_cache(self.device)

        if self.verbose:
            print(f'Using {len(bboxes_ordered)} bounding boxes for refinement')

        preallocated_input = torch.zeros((1 + self.num_interaction_channels, *self.configuration_manager.patch_size), device=self.device,
                                         dtype=torch.float)
        for nref, refinement_bbox in enumerate(bboxes_ordered):
            assert self.pad_mode_data == 'constant'
            crop_and_pad_into_buffer(preallocated_input[0], refinement_bbox, self.preprocessed_image[0])
            if self.use_in_mem_compression:
                assert self._get_prev_seg_channel() == 0, "blosc2 path assumes prev_seg_channel == 0"
                # Channel 0 (coarse pred already written by _predict): subregion only
                ch0_crop = self._crop_and_pad_interactions_channel0(refinement_bbox)
                preallocated_input[1].copy_(ch0_crop.to(self.device))
                # Channels 1+: subregion only, skipping channel 0
                crop_and_pad_into_buffer(preallocated_input[2:], refinement_bbox, self.interactions,
                                         source_leading_slice=slice(1, None))
            else:
                crop_and_pad_into_buffer(preallocated_input[1:], refinement_bbox, self.interactions)
            self._normalize_interaction_channels_for_network_(preallocated_input[1:])

            pred = self.network(preallocated_input[None])[0].argmax(0).detach()

            if self.use_in_mem_compression:
                paste_tensor(self.interactions, pred, refinement_bbox, channel_idx=prev_seg_channel)
            else:
                paste_tensor(self.interactions[prev_seg_channel], pred, refinement_bbox)
            # place into target buffer
            bbox = [[i[0] + bbc[0], i[1] + bbc[0]] for i, bbc in
                    zip(refinement_bbox, self.preprocessed_props['bbox_used_for_cropping'])]
            paste_tensor(self.target_buffer, pred.to(self.target_buffer.device) if isinstance(self.target_buffer, torch.Tensor) else pred.to('cpu'), bbox)
            del pred
            preallocated_input.zero_()
        del preallocated_input
        empty_cache(self.device)
        end_refinement = time()
        print(
            f'Took {round(end_refinement - start_refinement, 3)} s for refining the segmentation with {len(bboxes_ordered)} bounding boxes')

    def _detect_change_at_border(self,
                                 pred: torch.Tensor,
                                 prev_pred: torch.Tensor,
                                 abs_pxl_change_threshold = 1500,
                                 rel_pxl_change_threshold = 0.2,
                                 min_pxl_change_threshold = 100):
        has_change: bool = False
        for dim in range(pred.ndim):
            if has_change:
                break
            for idx in [0, pred.shape[dim] - 1]:
                slice_prev = prev_pred.index_select(dim, torch.tensor(idx, device='cpu'))
                slice_curr = pred.index_select(dim, torch.tensor(idx, device=self.device)).to('cpu')
                pixels_prev = torch.sum(slice_prev)
                pixels_current = torch.sum(slice_curr)
                pixels_diff = torch.sum(slice_prev != slice_curr)
                rel_change = max(pixels_prev, pixels_current) / max(min(pixels_prev, pixels_current),
                                                                    1e-5) - 1
                if pixels_diff > abs_pxl_change_threshold:
                    has_change = True
                    if self.verbose:
                        print(
                            f'continue zooming because change at borders of {pixels_diff} > {abs_pxl_change_threshold}')
                    break
                if pixels_diff > min_pxl_change_threshold and rel_change > rel_pxl_change_threshold:
                    has_change = True
                    if self.verbose:
                        print(
                            f'continue zooming because relative change of {rel_change} > {rel_pxl_change_threshold} and n_pixels {pixels_diff} > {min_pxl_change_threshold}')
                    break
                del slice_prev, slice_curr, pixels_prev, pixels_current, pixels_diff
        return has_change

    def _compute_diff_map(self, pred, previous_prediction, scaled_bbox, scaled_patch_size):
        """
        pred is expected to have shape scaled_bbox, previous_prediction is expected to have shape of self.interactions
        (or None when use_in_mem_compression=True, in which case we read from blosc2 directly).

        pred is expected to be on device already

        diff map has the same shape as self.interactions[1:] (spatial dims) and will be on self.device

        Args:
            pred:
            previous_prediction: torch.Tensor (3D, shape of interactions spatial dims) or None for blosc2.
            scaled_bbox:
            scaled_patch_size:

        Returns:

        """
        prev_seg_ch = self._get_prev_seg_channel()
        if self.use_in_mem_compression:
            interactions_shape = self.interactions.shape[1:]
            seen_bbox = [[max(0, i[0]), min(i[1], s)]
                         for i, s in zip(scaled_bbox, interactions_shape)]
        else:
            previous_prediction = previous_prediction.to(self.device, non_blocking=True)
            seen_bbox = [[max(0, i[0]), min(i[1], s)]
                         for i, s in zip(scaled_bbox, previous_prediction.shape)]

        bbox_tmp = [[i[0] - s[0], i[1] - s[0]] for i, s in zip(seen_bbox, scaled_bbox)]
        bbox_tmp = [[max(0, i[0]), min(i[1], s)] for i, s in zip(bbox_tmp, scaled_patch_size)]
        slicer = bounding_box_to_slice(seen_bbox)
        slicer2 = bounding_box_to_slice(bbox_tmp)

        if self.use_in_mem_compression:
            prev_sub = torch.from_numpy(np.asarray(
                self.interactions[(prev_seg_ch, *[slice(sb[0], sb[1]) for sb in seen_bbox])]
            )).to(self.device)
            diff = pred[slicer2] != prev_sub
            diff_map = torch.zeros(interactions_shape, device=self.device, dtype=torch.float16)
        else:
            diff = pred[slicer2] != previous_prediction[slicer]
            diff_map = torch.zeros_like(previous_prediction, device=self.device)

        diff_map[slicer] = diff

        # open the difference map to keep computational load in check (fewer refinement boxes)
        # open distance map
        diff_map[slicer] = \
            iterative_3x3_same_padding_pool3d(diff_map[slicer][None, None], kernel_size=5, use_min_pool=True)[0, 0]
        diff_map[slicer] = \
            iterative_3x3_same_padding_pool3d(diff_map[slicer][None, None], kernel_size=5, use_min_pool=False)[0, 0]

        return diff_map.to(torch.uint8)

    def _add_patch_for_point_interaction(self, coordinates):
        self.new_interaction_zoom_out_factors.append(1)
        self.new_interaction_centers.append(coordinates)
        print(f'Added new point interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')

    def _add_patch_for_bbox_interaction(self, bbox):
        bbox_center = [round((i[0] + i[1]) / 2) for i in bbox]
        bbox_size = [i[1]-i[0] for i in bbox]
        # we want to see some context, so the crop we see for the initial prediction should be patch_size / 3 larger
        requested_size = [i + j // 3 for i, j in zip(bbox_size, self.configuration_manager.patch_size)]
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(bbox_center)
        print(f'Added new bbox interaction: center {self.new_interaction_zoom_out_factors[-1]}, scale {self.new_interaction_centers}')

    def _add_patch_for_scribble_interaction(self, scribble_image, offset=None):
        return self._generic_add_patch_from_image(scribble_image, offset=offset)

    def _add_patch_for_lasso_interaction(self, lasso_image, offset=None):
        return self._generic_add_patch_from_image(lasso_image, offset=offset)

    def _add_patch_for_initial_seg_interaction(self, initial_seg):
        return self._generic_add_patch_from_image(initial_seg)

    def _generic_add_patch_from_image(self, image: torch.Tensor, offset: Optional[List[int]] = None):
        if not torch.any(image):
            print('Received empty image prompt. Cannot add patches for prediction')
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
        self.new_interaction_zoom_out_factors.append(max(1, max([i / j for i, j in zip(requested_size, self.configuration_manager.patch_size)])))
        self.new_interaction_centers.append(roi_center)
        print(f'Added new image interaction: scale {self.new_interaction_zoom_out_factors[-1]}, center {self.new_interaction_centers}')

    def initialize_from_trained_model_folder(self, model_training_output_dir: str,
                                             use_fold: Union[int, str] = None,
                                             checkpoint_name: str = 'checkpoint_final.pth'):
        """
        This is used when making predictions with a trained model
        """
        capability_file = join(model_training_output_dir, 'inference_info.json')
        legacy_file = join(model_training_output_dir, 'inference_session_class.json')

        point_interaction_radius = 4
        point_interaction_use_etd = True
        self.preferred_scribble_thickness = [2, 2, 2]
        self.pad_mode_data = "constant"
        self.interaction_decay = 0.98
        capability_content = {}

        if isfile(capability_file):
            capability_content = load_json(capability_file)
            if not isinstance(capability_content, dict):
                raise RuntimeError(f"Invalid capability metadata in {capability_file}. Expected a JSON object.")
            self._validate_capability_version(capability_content)
            point_interaction_radius = capability_content.get('point_radius', point_interaction_radius)
            self.preferred_scribble_thickness = capability_content.get(
                'preferred_scribble_thickness', self.preferred_scribble_thickness
            )
            self.interaction_decay = capability_content.get('interaction_decay', self.interaction_decay)
            self.pad_mode_data = capability_content.get('pad_mode_image', self.pad_mode_data)
        elif isfile(legacy_file):
            legacy_content = load_json(legacy_file)
            if isinstance(legacy_content, str):
                self.interaction_decay = 0.9
            else:
                point_interaction_radius = legacy_content.get('point_radius', point_interaction_radius)
                self.preferred_scribble_thickness = legacy_content.get(
                    'preferred_scribble_thickness', self.preferred_scribble_thickness
                )
                self.interaction_decay = legacy_content.get('interaction_decay', self.interaction_decay)
                self.pad_mode_data = legacy_content.get('pad_mode_image', self.pad_mode_data)
        else:
            raise FileNotFoundError(
                f"Neither capability metadata ({capability_file}) nor legacy metadata ({legacy_file}) was found."
            )

        if not isinstance(self.preferred_scribble_thickness, (tuple, list)):
            self.preferred_scribble_thickness = [self.preferred_scribble_thickness] * 3

        self.point_interaction = PointInteraction_stub(point_interaction_radius, point_interaction_use_etd)
        self._apply_capability(capability_content)

        dataset_json = load_json(join(model_training_output_dir, 'dataset.json'))
        plans = load_json(join(model_training_output_dir, 'plans.json'))
        plans_manager = PlansManager(plans)

        if use_fold is not None:
            use_fold = int(use_fold) if use_fold != 'all' else use_fold
            fold_folder = f'fold_{use_fold}'
        else:
            fldrs = subdirs(model_training_output_dir, prefix='fold_', join=False)
            assert len(fldrs) == 1, f'Attempted to infer fold but there is != 1 fold_ folders: {fldrs}'
            fold_folder = fldrs[0]

        checkpoint = torch.load(join(model_training_output_dir, fold_folder, checkpoint_name),
                                map_location=self.device, weights_only=False)
        if self._is_official_checkpoint(plans, checkpoint):
            print(
                'License reminder: The official nnInteractive checkpoint is licensed under '
                'Creative Commons Attribution Non Commercial Share Alike 4.0 (CC BY-NC-SA 4.0). '
                'See the license note in readme.md (# License).'
            )
        trainer_name = checkpoint['trainer_name']
        configuration_name = checkpoint['init_args']['configuration']

        parameters = checkpoint['network_weights']

        configuration_manager = plans_manager.get_configuration(configuration_name)
        # restore network
        num_input_channels = determine_num_input_channels(plans_manager, configuration_manager, dataset_json)
        trainer_class = recursive_find_python_class(join(nnInteractive.__path__[0], "trainer"),
                                                    trainer_name, 'nnInteractive.trainer')
        if trainer_class is None:
            print(f'Unable to locate trainer class {trainer_name} in nnInteractive.trainer. '
                               f'Please place it there (in any .py file)!')
            print('Attempting to use default nnInteractiveTrainer_stub. If you encounter errors, this is where you need to look!')
            trainer_class = nnInteractiveTrainer_stub

        network = trainer_class.build_network_architecture(
            configuration_manager.network_arch_class_name,
            configuration_manager.network_arch_init_kwargs,
            configuration_manager.network_arch_init_kwargs_req_import,
            num_input_channels,
            plans_manager.get_label_manager(dataset_json).num_segmentation_heads,
            enable_deep_supervision=False
        ).to(self.device)
        network.load_state_dict(parameters)

        self.plans_manager = plans_manager
        self.configuration_manager = configuration_manager
        self.network = network
        self.dataset_json = dataset_json
        self.trainer_name = trainer_name
        self.label_manager = plans_manager.get_label_manager(dataset_json)
        if self.use_torch_compile and not isinstance(self.network, OptimizedModule):
            print('Using torch.compile')
            self.network = torch.compile(self.network)

    def manual_initialization(self, network: nn.Module, plans_manager: PlansManager,
                              configuration_manager: ConfigurationManager,
                              dataset_json: dict, trainer_name: str):
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
            print('Using torch.compile')
            self.network = torch.compile(self.network)

        if not self.use_torch_compile and isinstance(self.network, OptimizedModule):
            self.network = self.network._orig_mod

        self.network = self.network.to(self.device)


def transform_coordinates_noresampling(
        coords_orig: Union[List[int], Tuple[int, ...]],
        nnunet_preprocessing_crop_bbox: List[Tuple[int, int]]
) -> Tuple[int, ...]:
    """
    converts coordinates in the original uncropped image to the internal cropped representation. Man I really hate
    nnU-Net's crop to nonzero!
    """
    return tuple([coords_orig[d] - nnunet_preprocessing_crop_bbox[d][0] for d in range(len(coords_orig))])


if __name__ == '__main__':
    a = torch.zeros((160, 160, 160), device='cpu')
    a.index_select(0, torch.tensor([0]))
