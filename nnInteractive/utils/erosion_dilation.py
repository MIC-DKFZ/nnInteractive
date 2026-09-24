import torch

from nnunetv2.utilities.helpers import empty_cache


def _window_extreme_(acc: torch.Tensor, axis: int, kernel_size: int, op) -> torch.Tensor:
    """Sliding-window max/min of width ``kernel_size`` along ``axis`` without padding ('valid'): the output is
    shorter by kernel_size - 1. Windows grow by doubling (1 -> 2 -> 4 -> ... -> kernel_size), so a width-k window
    costs ~log2(k) elementwise ops over shifted views instead of k."""
    width = 1
    while width < kernel_size:
        step = min(width, kernel_size - width)
        n = acc.shape[axis] - step
        acc = op(acc.narrow(axis, 0, n), acc.narrow(axis, step, n))
        width += step
    return acc


def _replicate_pad(x: torch.Tensor, axis: int, before: int, after: int) -> torch.Tensor:
    if before == 0 and after == 0:
        return x
    shape = lambda k: [k if d == axis else -1 for d in range(x.ndim)]
    parts = [x.narrow(axis, 0, 1).expand(*shape(before))] if before else []
    parts.append(x)
    if after:
        parts.append(x.narrow(axis, x.shape[axis] - 1, 1).expand(*shape(after)))
    return torch.cat(parts, dim=axis)


@torch.inference_mode()
def iterative_3x3_same_padding_pool3d(x, kernel_size: int, use_min_pool: bool = False, slab_depth: int = 64):
    """
    3D max (or min) pooling with a kernel_size^3 window, stride 1 and 'same' output size, where the volume is
    extended by replicating its border voxels.

    Args:
        x (Tensor): Input tensor of shape (N, C, D, H, W) or unbatched (C, D, H, W)
        kernel_size (int): Odd window size, the same for all three dimensions.
        use_min_pool (bool): Min instead of max pooling (erosion instead of dilation).
        slab_depth (int): Number of output slices along D computed at once (bounds the temporaries).

    Returns:
        Tensor: Output tensor with the same shape as the input.

    Defined (and originally implemented) as replicate padding by (kernel_size - 1) / 2 followed by
    (kernel_size - 1) / 2 iterations of 3x3x3 max pooling, which equals the max over the kernel_size^3 window
    clipped to the volume. Computed here separably (max/min are separable and exact in any order), slab by slab
    along D, with torch.maximum/minimum over shifted views: bitwise identical to the original, but without
    F.max_pool3d, which on CUDA always materializes int64 argmax indices (8 bytes per voxel, the VRAM peak of the
    zoom-4 AutoZoom input), and with temporaries of one slab instead of the whole (padded) volume.
    """
    assert kernel_size % 2 == 1, "Only works with odd kernels"
    if kernel_size == 1:
        return x.clone()
    if x.ndim == 4:
        # unbatched input, as accepted by F.max_pool3d
        return iterative_3x3_same_padding_pool3d(x[None], kernel_size, use_min_pool, slab_depth)[0]
    r = (kernel_size - 1) // 2
    op = torch.minimum if use_min_pool else torch.maximum
    depth = x.shape[2]
    out = torch.empty_like(x)
    for z0 in range(0, depth, slab_depth):
        z1 = min(depth, z0 + slab_depth)
        a, b = max(0, z0 - r), min(depth, z1 + r)
        # real neighbours as halo inside the volume, replicated border slices at its ends
        acc = _replicate_pad(x[:, :, a:b], 2, r - (z0 - a), r - (b - z1))
        acc = _window_extreme_(acc, 2, kernel_size, op)
        for axis in (3, 4):
            acc = _window_extreme_(_replicate_pad(acc, axis, r, r), axis, kernel_size, op)
        out[:, :, z0:z1] = acc
        del acc
    empty_cache(x.device)
    return out
