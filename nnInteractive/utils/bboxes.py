from typing import List, Union, Tuple

import numpy as np
import torch


def _nonzero_bbox_inclusive(mask: torch.Tensor) -> Tuple[List[int], List[int]]:
    """Inclusive per-axis [min, max] coordinates of the nonzero voxels of a 3D mask (which must contain some).

    Same result as ``torch.nonzero(mask).min(0)`` / ``.max(0)``, but without materializing one int64 index triple per
    nonzero voxel (24 bytes each: several GB of VRAM when a large first prompt on a big object changes most of the
    refinement-planning region). Instead two max-projections: collapsing axis 0 gives a small plane from which the
    extents along axes 1 and 2 follow, and axis 0 gets its own projection. The first/last nonzero entry of each 1D
    projection is found with argmax (which returns the first maximum) and a single host sync.
    """
    if mask.dtype == torch.bool:
        mask = mask.view(torch.uint8)  # zero-copy; amax on uint8
    plane = mask.amax(dim=0)  # (Y, Z): nonzero where any voxel along axis 0 is
    projections = [mask.amax(dim=(1, 2)) != 0, plane.amax(dim=1) != 0, plane.amax(dim=0) != 0]
    ends = []
    for p in projections:
        p = p.to(torch.uint8)
        ends += [torch.argmax(p), p.numel() - 1 - torch.argmax(p.flip(0))]
    ends = torch.stack(ends).tolist()  # one host sync
    return ends[0::2], ends[1::2]


def generate_bounding_boxes(
    mask,
    bbox_size=(192, 192, 192),
    stride: Union[List[int], Tuple[int, int, int], str] = (16, 16, 16),
    margin=(10, 10, 10),
    max_depth=5,
    current_depth=0,
):
    """
    Generate overlapping bounding boxes to cover a 3D binary segmentation mask using PyTorch tensors.

    Parameters:
    - mask: 3D PyTorch tensor with values 0 or 1 (binary mask)
    - bbox_size: Tuple or list of three integers specifying the size of bounding boxes per dimension (x, y, z)
    - stride: Tuple or list of three integers specifying the stride for subsampling centers per dimension
    - margin: Tuple or list of three integers specifying the margin to leave uncovered per dimension
    - max_depth: Maximum recursion depth to prevent infinite recursion
    - current_depth: Current recursion depth (used internally)

    Returns:
    - List of bboxes [[x1, x2], [y1, y2], [z1, z2]], each a half-open interval [lower, upper) per dimension.
    """
    if not torch.any(mask):
        return []

    # Prevent infinite recursion
    if current_depth > max_depth:
        # print('random fallback due to max recursion depth')
        return random_sampling_fallback(mask, bbox_size, margin, 25)

    # Ensure bbox_size, stride, and margin are lists
    bbox_size = list(bbox_size)
    margin = list(margin)

    # Compute half sizes for each dimension
    half_size = [bs // 2 for bs in bbox_size]
    # Adjust end offsets to ensure full bbox_size (handles odd sizes)
    end_offset = [bs - hs for bs, hs in zip(bbox_size, half_size)]  # e.g., 193 - 96 = 97

    # Steps 1 + 2: the object's bounding box, to limit potential centers (projections, not torch.nonzero: see
    # _nonzero_bbox_inclusive)
    min_coords, max_coords = _nonzero_bbox_inclusive(mask)

    if isinstance(stride, str) and stride == "auto":
        stride = [max(1, round((j - i) / 4)) for i, j in zip(min_coords, max_coords)]

    stride = list(stride)
    # print('stride', stride)
    # print('bbox', [[i, j] for i, j in zip(min_coords, max_coords)])

    # Step 3: Generate potential centers within the object's bounding box. Vectorized: the old
    # triple Python loop probed mask[x, y, z] one element at a time — one GPU sync per probe on
    # a CUDA mask. One advanced-indexing call keeps the same lexicographic candidate order.
    axis_ranges = [
        torch.arange(
            max(0, min_coords[d]),
            min(mask.shape[d], max_coords[d] + 1),
            stride[d],
            device=mask.device,
        )
        for d in range(3)
    ]
    grid = torch.cartesian_prod(*axis_ranges)  # (N, 3)
    potential_centers = grid[mask[grid[:, 0], grid[:, 1], grid[:, 2]] != 0]
    # print(f'got {len(potential_centers)} center candidates')

    if len(potential_centers) == 0:
        return generate_bounding_boxes(
            mask, bbox_size, [max(1, s // 2) for s in stride], margin, max_depth, current_depth + 1
        )

    # Step 4: Greedy set cover algorithm
    uncovered = mask.clone().byte()  # Use byte tensor for efficiency

    bboxes = []

    while len(potential_centers) > 0 and uncovered.any():
        # Pull the candidate coordinates to the host once, queue every candidate's coverage sum,
        # then fetch them all with a single host sync. The old loop paid several GPU syncs per
        # candidate per round (Python max()/min() on 0-dim tensors plus .item() on each sum).
        centers_list = potential_centers.tolist()
        bounds = []
        cover_sums = []
        for c_x, c_y, c_z in centers_list:
            x_start = max(0, c_x - half_size[0] + margin[0])
            x_end = min(mask.shape[0], c_x + end_offset[0] - margin[0])  # Use end_offset for odd sizes
            y_start = max(0, c_y - half_size[1] + margin[1])
            y_end = min(mask.shape[1], c_y + end_offset[1] - margin[1])
            z_start = max(0, c_z - half_size[2] + margin[2])
            z_end = min(mask.shape[2], c_z + end_offset[2] - margin[2])
            bounds.append((x_start, x_end, y_start, y_end, z_start, z_end))
            cover_sums.append(uncovered[x_start:x_end, y_start:y_end, z_start:z_end].sum())
        covered = torch.stack(cover_sums).tolist()  # one host sync for all candidates

        # Find the center that covers the most uncovered voxels (first max wins, like before)
        best_center = None
        best_covered = 0
        best_bounds = None
        for idx, num_covered in enumerate(covered):
            if num_covered > best_covered:
                best_covered = num_covered
                best_center = idx
                best_bounds = bounds[idx]

        # If no new voxels are covered, stop
        if best_covered == 0:
            break

        # Add the best bounding box
        c_x, c_y, c_z = centers_list[best_center]
        bboxes.append(
            [
                [c_x - half_size[0], c_x + end_offset[0]],
                [c_y - half_size[1], c_y + end_offset[1]],
                [c_z - half_size[2], c_z + end_offset[2]],
            ]
        )

        # Mark voxels as covered, respecting the margin
        x_s, x_e, y_s, y_e, z_s, z_e = best_bounds
        uncovered[
            x_s:x_e,
            y_s:y_e,
            z_s:z_e,
        ] = 0

        # Remove the used center from potential_centers
        potential_centers = potential_centers[uncovered[tuple(potential_centers.T)] > 0]

    # Step 5: Cover remaining voxels via random sampling
    if uncovered.any():
        bboxes.extend(random_sampling_fallback(uncovered, bbox_size, margin, 10))

    return bboxes


def random_sampling_fallback(mask: torch.Tensor, bbox_size=(192, 192, 192), margin=(10, 10, 10), n_samples: int = 25):
    half_size = [bs // 2 for bs in bbox_size]
    # Adjust end offsets to ensure full bbox_size (handles odd sizes)
    end_offset = [bs - hs for bs, hs in zip(bbox_size, half_size)]  # e.g., 193 - 96 = 97

    bboxes = []

    while mask.any():
        indices = torch.nonzero(mask)  # nx3

        best_center = None
        best_covered = 0
        best_bounds = None

        # Find the center that covers the most uncovered voxels
        for i in range(n_samples):
            idx = np.random.choice(len(indices))
            center = indices[idx]
            c_x, c_y, c_z = [int(i.item()) for i in center]
            x_start = max(0, c_x - half_size[0] + margin[0])
            x_end = min(mask.shape[0], c_x + end_offset[0] - margin[0])  # Use end_offset for odd sizes
            y_start = max(0, c_y - half_size[1] + margin[1])
            y_end = min(mask.shape[1], c_y + end_offset[1] - margin[1])
            z_start = max(0, c_z - half_size[2] + margin[2])
            z_end = min(mask.shape[2], c_z + end_offset[2] - margin[2])

            num_covered = mask[x_start:x_end, y_start:y_end, z_start:z_end].sum().item()
            if num_covered > best_covered:
                best_covered = num_covered
                best_center = center
                best_bounds = (x_start, x_end, y_start, y_end, z_start, z_end)

        # Add the best bounding box
        c_x, c_y, c_z = [int(i.item()) for i in best_center]
        bboxes.append(
            [
                [c_x - half_size[0], c_x + end_offset[0]],
                [c_y - half_size[1], c_y + end_offset[1]],
                [c_z - half_size[2], c_z + end_offset[2]],
            ]
        )

        # Mark voxels as covered, respecting the margin
        x_s, x_e, y_s, y_e, z_s, z_e = best_bounds
        mask[
            x_s:x_e,
            y_s:y_e,
            z_s:z_e,
        ] = 0
    return bboxes
