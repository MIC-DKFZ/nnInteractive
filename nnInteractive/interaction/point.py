from functools import lru_cache
from typing import Callable, Tuple, Optional

import numpy as np
import torch
from batchgeneratorsv2.helpers.scalar_type import sample_scalar, RandomScalar
from scipy.ndimage import distance_transform_edt
from skimage.morphology import disk, ball


@lru_cache(maxsize=5)
def build_point(radii, use_distance_transform, binarize):
    max_radius = max(radii)
    ndim = len(radii)

    # Create a spherical (or circular) structuring element with max_radius
    if ndim == 2:
        structuring_element = disk(max_radius)
    elif ndim == 3:
        structuring_element = ball(max_radius)
    else:
        raise ValueError("Unsupported number of dimensions. Only 2D and 3D are supported.")

    # Convert the structuring element to a tensor
    structuring_element = torch.from_numpy(structuring_element.astype(np.float32))

    # Create the target shape based on the sampled radii
    target_shape = [round(2 * r + 1) for r in radii]

    if any([i != j for i, j in zip(target_shape, structuring_element.shape)]):
        structuring_element_resized = torch.nn.functional.interpolate(
            structuring_element.unsqueeze(0).unsqueeze(0),  # Add batch and channel dimensions for interpolation
            size=target_shape,
            mode="trilinear" if ndim == 3 else "bilinear",
            align_corners=False,
        )[
            0, 0
        ]  # Remove batch and channel dimensions after interpolation
    else:
        structuring_element_resized = structuring_element

    if use_distance_transform:
        # Convert the structuring element to a binary mask for distance transform computation
        binary_structuring_element = (structuring_element_resized >= 0.5).numpy()

        # Compute the Euclidean distance transform of the binary structuring element
        structuring_element_resized = distance_transform_edt(binary_structuring_element)

        # Normalize the distance transform to have values between 0 and 1
        structuring_element_resized /= structuring_element_resized.max()
        structuring_element_resized = torch.from_numpy(structuring_element_resized)

    if binarize and not use_distance_transform:
        # Normalize the resized structuring element to binary (values near 1 are treated as the point region)
        structuring_element_resized = (structuring_element_resized >= 0.5).float()
    return structuring_element_resized


class PointInteraction_stub:
    interaction_type = "point"

    def __init__(self, point_radius: RandomScalar, use_distance_transform: bool = False):
        """
        Initializes the PointInteraction object.

        Parameters:
        point_radius (RandomScalar): Specifies the radius for the interaction points.
        use_distance_transform (bool): Determines whether to use a distance transform for smooth interactions.
        """
        super().__init__()
        self.point_radius = point_radius
        self.use_distance_transform = use_distance_transform

    def place_point(
        self,
        position: Tuple[int, ...],
        interaction_map,
        binarize: bool = False,
        intensity_scale: float = 1.0,
        channel_idx: Optional[int] = None,
        before_write: Optional[Callable[[tuple], None]] = None,
    ) -> torch.Tensor:
        """
        Places a point on the interaction map around the specified position.

        Parameters:
        position (Tuple[int, ...]): The (x, y, z) coordinates where the point should be placed.
        interaction_map: A tensor (or blosc2 NDArray when channel_idx is provided) representing
                         the interaction map where the point should be placed.
        binarize (bool): If True, inserts a binary mask. If False, may insert smooth values based on distance.
        intensity_scale (float): Scale factor applied to the structuring element values.
        channel_idx (int, optional): If provided, interaction_map is treated as a 4D blosc2 NDArray
                                     and only the structuring element subregion is read/written for
                                     channel channel_idx. Avoids decompressing the full channel.
        before_write (callable, optional): channel_idx path only. Called with the target slices
                                     (channel_idx, *spatial slices) right before they are written, e.g. to
                                     save their previous contents for undo.

        Returns:
        The updated interaction map (torch.Tensor for the default path; blosc2 NDArray for channel_idx path).
        """
        # Geometry is identical for both paths; only the final write differs.
        # channel_idx path: interaction_map is the full 4D array (torch tensor or blosc2 NDArray).
        # Default path: interaction_map is a single-channel tensor.
        spatial_shape = tuple(interaction_map.shape[1:] if channel_idx is not None else interaction_map.shape)
        ndim = len(spatial_shape)

        radius = tuple(sample_scalar(self.point_radius, d, spatial_shape) for d in range(ndim))
        strel = build_point(radius, self.use_distance_transform, binarize)
        if intensity_scale != 1.0:
            strel = strel * intensity_scale

        bbox = [
            [position[i] - strel.shape[i] // 2, position[i] + strel.shape[i] // 2 + strel.shape[i] % 2]
            for i in range(ndim)
        ]
        # detect if bbox is completely outside interaction_map
        if any(i[1] < 0 for i in bbox) or any(i[0] > s for i, s in zip(bbox, spatial_shape)):
            print("Point is outside the interaction map! Ignoring")
            print(f"Position: {position}")
            print(f"Interaction map shape: {spatial_shape}")
            print(f"Point bbox would have been {bbox}")
            return interaction_map

        # Clip the point bbox to the map and compute the matching subregion of the structuring element.
        slices = tuple(slice(max(0, bbox[i][0]), min(spatial_shape[i], bbox[i][1])) for i in range(ndim))
        structuring_slices = tuple(
            slice(max(0, -bbox[i][0]), slices[i].stop - slices[i].start + max(0, -bbox[i][0])) for i in range(ndim)
        )

        if channel_idx is None:
            # Single-channel tensor: place the structuring element via in-place maximum.
            torch.maximum(
                interaction_map[slices],
                strel[structuring_slices].to(interaction_map.device),
                out=interaction_map[slices],
            )
            return interaction_map

        target_slices = (channel_idx, *slices)
        if before_write is not None:
            before_write(target_slices)
        if isinstance(interaction_map, torch.Tensor):
            # Dense torch backend: in-place maximum, no numpy round-trip.
            view = interaction_map[target_slices]
            torch.maximum(view, strel[structuring_slices].to(view.dtype), out=view)
            return interaction_map
        # blosc2 backend: read-modify-write only the structuring element subregion
        # (avoids decompressing the full channel).
        current_sub = np.asarray(interaction_map[target_slices])
        strel_np = strel[structuring_slices].numpy().astype(current_sub.dtype)
        np.maximum(current_sub, strel_np, out=current_sub)
        interaction_map[target_slices] = current_sub
        return interaction_map
