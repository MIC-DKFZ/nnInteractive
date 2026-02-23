from typing import Sequence, Union
import numpy as np
import torch
import torch.nn.functional as F

def crop_and_pad_into_buffer(target_tensor: torch.Tensor,
                             bbox: Sequence[Sequence[int]],
                             source_tensor,
                             source_leading_slice: slice = None) -> None:
    """
    Copies a sub-region from source_tensor into target_tensor based on a bounding box.

    Args:
        target_tensor (torch.Tensor): A preallocated tensor that will be updated.
        bbox (sequence of [int, int]): A bounding box for each dimension of the source tensor
            that is covered by the bbox. The bbox is defined as [start, end) (half-open interval)
            and may extend outside the source tensor. If source_tensor has more dimensions than
            len(bbox), the leading dimensions will be fully included.
        source_tensor: The tensor to copy data from (torch.Tensor, numpy array, or blosc2 NDArray).
        source_leading_slice (slice, optional): If provided, use this slice for the leading
            dimension(s) of source_tensor instead of slice(None). Useful for selecting a subset
            of channels from a blosc2 NDArray without decompressing all channels.

    Behavior:
        For each dimension that the bbox covers (i.e. the last len(bbox) dims of source_tensor):
            - Compute the overlapping region between the bbox and the source tensor.
            - Determine the corresponding indices in the target tensor where the data will be copied.
        For any extra leading dimensions (i.e. source_tensor.ndim > len(bbox)):
            - Use slice(None) (or source_leading_slice if given) to include the dimension.
        If source_tensor and target_tensor are on different devices, only the overlapping subregion
        is transferred to the device of target_tensor.
    """
    total_dims = source_tensor.ndim
    bbox_dims = len(bbox)
    # Compute the number of leading dims that are not covered by bbox.
    leading_dims = total_dims - bbox_dims

    source_slices = []
    target_slices = []

    # For the leading dimensions, include the entire dimension (or use source_leading_slice).
    for _ in range(leading_dims):
        source_slices.append(source_leading_slice if source_leading_slice is not None else slice(None))
        target_slices.append(slice(None))

    # Process the dimensions covered by the bbox.
    for d in range(bbox_dims):
        box_start, box_end = bbox[d]
        d_source = leading_dims + d
        source_size = source_tensor.shape[d_source]

        # Compute the overlapping region in source coordinates.
        copy_start_source = max(box_start, 0)
        copy_end_source = min(box_end, source_size)
        copy_size = copy_end_source - copy_start_source

        # Compute the corresponding indices in the target tensor.
        copy_start_target = max(0, -box_start)
        copy_end_target = copy_start_target + copy_size

        source_slices.append(slice(copy_start_source, copy_end_source))
        target_slices.append(slice(copy_start_target, copy_end_target))

    # Extract the overlapping region from the source.
    sub_source = source_tensor[tuple(source_slices)]
    # Convert to torch if source is numpy (e.g. from blosc2 NDArray slicing)
    if not isinstance(sub_source, torch.Tensor):
        sub_source = torch.from_numpy(np.asarray(sub_source))
    # Transfer only this subregion to the target tensor's device.
    sub_source = sub_source.to(target_tensor.device) if isinstance(target_tensor, torch.Tensor) else sub_source.cpu()
    # Write the data into the preallocated target_tensor.
    target_tensor[tuple(target_slices)] = sub_source


def paste_tensor(target: torch.Tensor, source: torch.Tensor, bbox):
    """
    Paste a source tensor into a target tensor using a given bounding box.

    Both tensors are assumed to be 3D.
    The bounding box is specified in the coordinate system of the target as:
      [[x1, x2], [y1, y2], [z1, z2]]
    and its size is assumed to be equal to the shape of the source tensor.
    The bbox may exceed the boundaries of the target tensor.

    The function computes the valid overlapping region between the bbox and the target,
    and then adjusts the corresponding region in the source tensor so that only the valid
    parts are pasted.

    Args:
        target (torch.Tensor): The target tensor of shape (T0, T1, T2).
        source (torch.Tensor): The source tensor of shape (S0, S1, S2). It must be the same size as
                               the bbox, i.e. S0 = x2 - x1, etc.
        bbox (list or tuple): List of intervals for each dimension: [[x1, x2], [y1, y2], [z1, z2]].

    Returns:
        torch.Tensor: The target tensor after pasting in the source.
    """
    target_shape = target.shape  # (T0, T1, T2)

    # For each dimension compute:
    #   - The valid region in the target: [t_start, t_end)
    #   - The corresponding region in the source: [s_start, s_end)
    target_indices = []
    source_indices = []

    for i, (b0, b1) in enumerate(bbox):
        # Determine valid region in target tensor:
        t_start = max(b0, 0)
        t_end = min(b1, target_shape[i])
        # If there's no overlap in any dimension, nothing gets pasted.
        if t_start >= t_end:
            return target

        # Determine corresponding indices in the source tensor.
        # The source's coordinate 0 corresponds to b0 in the target.
        s_start = t_start - b0
        s_end = s_start + (t_end - t_start)

        target_indices.append((t_start, t_end))
        source_indices.append((s_start, s_end))

    # Paste the corresponding region from source into target.
    if isinstance(target, torch.Tensor):
        target[target_indices[0][0]:target_indices[0][1],
        target_indices[1][0]:target_indices[1][1],
        target_indices[2][0]:target_indices[2][1]] = \
            source[source_indices[0][0]:source_indices[0][1],
            source_indices[1][0]:source_indices[1][1],
            source_indices[2][0]:source_indices[2][1]].to(target.device)
    else:
        src = source[source_indices[0][0]:source_indices[0][1],
                     source_indices[1][0]:source_indices[1][1],
                     source_indices[2][0]:source_indices[2][1]]
        if isinstance(src, torch.Tensor):
            src = src.cpu().numpy()
        target[target_indices[0][0]:target_indices[0][1],
        target_indices[1][0]:target_indices[1][1],
        target_indices[2][0]:target_indices[2][1]] = src

    return target



def crop_to_valid(img: torch.Tensor, bbox):
    """
    Crops the image to the part of the bounding box that lies within the image.
    Supports a 4D tensor of shape (C, X, Y, Z). The bounding box is specified as
    [[x1, x2], [y1, y2], [z1, z2]] with half-open intervals.

    Args:
        img (torch.Tensor): Input tensor of shape (C, X, Y, Z).
        bbox (list or tuple): Bounding box as a list of three intervals for spatial dims:
                              [[x1, x2], [y1, y2], [z1, z2]].

    Returns:
        cropped (torch.Tensor): Cropped tensor of shape (C, cropped_x, cropped_y, cropped_z).
        pad (list of tuples): A list [(pad_x_left, pad_x_right),
                                     (pad_y_left, pad_y_right),
                                     (pad_z_left, pad_z_right)]
                              indicating how much padding needs to be applied on each side.
    """
    # Only spatial dimensions (X, Y, Z) are cropped; channels are preserved.
    spatial_dims = img.shape[1:]  # (X, Y, Z)
    crop_indices = []
    pad = []  # for each spatial dimension

    for i, (start, end) in enumerate(bbox):
        dim_size = spatial_dims[i]
        # Clamp the indices to the valid range for cropping.
        crop_start = max(start, 0)
        crop_end = min(end, dim_size)
        crop_indices.append((crop_start, crop_end))
        # Calculate padding if the bbox goes out-of-bound.
        pad_left = -start if start < 0 else 0
        pad_right = end - dim_size if end > dim_size else 0
        pad.append((pad_left, pad_right))

    # Crop the image on spatial dimensions, leaving the channel dimension intact.
    cropped = img[:,
                  crop_indices[0][0]:crop_indices[0][1],
                  crop_indices[1][0]:crop_indices[1][1],
                  crop_indices[2][0]:crop_indices[2][1]]
    return cropped, pad


def pad_cropped(cropped: torch.Tensor, pad):
    """
    Pads the cropped image using the given pad amounts.
    Supports a 4D tensor of shape (C, X, Y, Z) and applies padding only on the spatial dimensions.
    For 3D (volumetric) padding, F.pad expects a 5D tensor with shape (N, C, X, Y, Z).
    Hence, we temporarily add a dummy batch dimension.

    Args:
        cropped (torch.Tensor): Cropped tensor of shape (C, X, Y, Z).
        pad (list of tuples): List of padding for each spatial dimension, in order (x, y, z):
                              [(pad_x_left, pad_x_right),
                               (pad_y_left, pad_y_right),
                               (pad_z_left, pad_z_right)].

    Returns:
        padded (torch.Tensor): Padded tensor of shape (C, desired_x, desired_y, desired_z),
                               where the spatial dimensions match the bbox size.
    """
    # F.pad for 3D data expects a 5D input (N, C, X, Y, Z) and a pad tuple of length 6:
    # (pad_z_left, pad_z_right, pad_y_left, pad_y_right, pad_x_left, pad_x_right)
    need_unsqueeze = (cropped.dim() == 4)
    if need_unsqueeze:
        cropped = cropped.unsqueeze(0)  # Now shape is (1, C, X, Y, Z)

    # Reverse the pad list (currently in order x, y, z) to match F.pad's expected order: z, y, x.
    pad_rev = pad[::-1]
    pad_flat = [p for pair in pad_rev for p in pair]
    padded = F.pad(cropped, pad_flat)

    if need_unsqueeze:
        padded = padded.squeeze(0)
    return padded