import threading
from collections.abc import Callable

import numpy as np
import torch

from nnInteractive.utils.os_shennanigans import is_linux_kernel_6_11


class PinnedStager:
    """Host -> device copies of CPU crops through a small, fixed, reusable pinned buffer.

    The session keeps its large CPU tensors (preprocessed image, interactions) in regular pageable memory and
    only ever sends crops of them to the GPU. Those crops are non-contiguous views, which torch copies by first
    gathering them into a freshly allocated pageable staging area -- slow, and it makes ``non_blocking`` a no-op.
    Here the gather goes into one of two halves of a pinned buffer instead and is followed by an asynchronous
    DMA, so gathering the next slab overlaps with transferring the previous one. Crops larger than a buffer half
    are streamed in slabs along their leading dimension(s), so the buffer size is fixed regardless of the
    AutoZoom crop size.

    Contiguous sources are copied directly (``.to(device)``): the driver handles those efficiently and an extra
    gather into the staging buffer would only add a memcpy.

    Reuse safety: before a half is overwritten, the host waits for the event recorded after the last DMA out of
    that half. The source may be modified as soon as ``copy_to`` returns, because the gather into pinned memory
    is synchronous. The destination is filled asynchronously on the current stream, so any subsequent work on
    that stream sees complete data.

    One instance per device and process (see ``get_stager``); ``copy_to`` holds a lock, so concurrent sessions
    in one process (the inference server) share the buffer safely.
    """

    DEFAULT_BUFFER_BYTES = 128 * 2**20

    def __init__(self, device: torch.device, buffer_bytes: int = DEFAULT_BUFFER_BYTES):
        self.device = device
        self.half_bytes = buffer_bytes // 2
        self._halves = None
        self._events = None
        self._next = 0
        self._lock = threading.Lock()

    def _lazy_init(self):
        # Allocated on first use only: sessions that never transfer a non-contiguous crop pay nothing.
        self._halves = [torch.empty(self.half_bytes, dtype=torch.uint8, pin_memory=True) for _ in range(2)]
        self._events = [torch.cuda.Event() for _ in range(2)]
        for e in self._events:
            e.record()

    def copy_to(self, src: torch.Tensor) -> torch.Tensor:
        """Return a copy of CPU tensor ``src`` on ``self.device`` (same shape and dtype, contiguous)."""
        if src.device.type != "cpu" or src.numel() == 0 or src.is_contiguous():
            return src.to(self.device)
        dst = torch.empty(src.shape, dtype=src.dtype, device=self.device)
        with self._lock, torch.cuda.device(self.device):
            if self._halves is None:
                self._lazy_init()
            self._copy_into(src, dst)
        return dst

    def copy_into(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        """Copy CPU tensor ``src`` into the existing device tensor ``dst`` (same shape; may be a strided view,
        e.g. a region of a larger buffer), slab by slab through the pinned buffer. Unlike ``dst.copy_(src)`` or
        ``dst[...] = src.to(device)``, no device temporary larger than one slab is created, whatever the size of
        the region -- also for contiguous sources."""
        if src.numel() == 0:
            return
        if src.ndim == 0:
            dst.copy_(src)
            return
        with self._lock, torch.cuda.device(self.device):
            if self._halves is None:
                self._lazy_init()
            self._copy_into(src, dst)

    def _copy_into(self, src: torch.Tensor, dst: torch.Tensor) -> None:
        # src is at least 1D.
        row_bytes = src[0].numel() * src.element_size()
        if src.ndim > 1 and row_bytes > self.half_bytes:
            # A single slice along dim 0 does not fit into one half: recurse one dimension deeper.
            for i in range(src.shape[0]):
                self._copy_into(src[i], dst[i])
            return
        rows_per_slab = max(1, self.half_bytes // row_bytes)
        n = src.shape[0]
        for start in range(0, n, rows_per_slab):
            end = min(n, start + rows_per_slab)
            b = self._next
            self._next = 1 - b
            self._events[b].synchronize()  # the previous DMA out of this half has finished
            nbytes = (end - start) * row_bytes
            staged = self._halves[b][:nbytes].view(src.dtype).view(end - start, *src.shape[1:])
            staged.copy_(src[start:end])  # synchronous (multithreaded) gather into pinned memory
            dst[start:end].copy_(staged, non_blocking=True)  # asynchronous DMA on the current stream
            self._events[b].record()

    def fill_to(self, dst: torch.Tensor, fill: Callable[[np.ndarray, int, int], None]) -> None:
        """Produce the content of device tensor ``dst`` block by block along dim 0 directly in pinned memory.

        ``fill(view, r0, r1)`` must write rows ``r0:r1`` of the result into ``view``, a contiguous numpy array of
        shape ``(r1 - r0, *dst.shape[1:])`` backed by one pinned half (e.g. a blosc2 decompression straight into
        it). Each block is DMA'd asynchronously while the next one is produced in the other half, so the
        producer (typically memory-bound decompression) overlaps with the transfer and no intermediate host copy
        of the whole region is ever made. ``dst`` must be on ``self.device``; one row along dim 0 must fit into
        one half. ``dst`` may be a strided view (e.g. a region of a larger buffer): each block then lands via a
        device temporary of one block, never of the whole region.
        """
        if dst.numel() == 0:
            return
        row_bytes = dst[0].numel() * dst.element_size()
        if row_bytes > self.half_bytes:
            raise ValueError(f"one row of {row_bytes} bytes exceeds the staging half of {self.half_bytes} bytes")
        rows_per_block = self.half_bytes // row_bytes
        with self._lock, torch.cuda.device(self.device):
            if self._halves is None:
                self._lazy_init()
            n = dst.shape[0]
            for start in range(0, n, rows_per_block):
                end = min(n, start + rows_per_block)
                b = self._next
                self._next = 1 - b
                self._events[b].synchronize()  # the previous DMA out of this half has finished
                staged = self._halves[b][: (end - start) * row_bytes].view(dst.dtype).view(end - start, *dst.shape[1:])
                fill(staged.numpy(), start, end)  # synchronous producer writes straight into pinned memory
                dst[start:end].copy_(staged, non_blocking=True)  # asynchronous DMA on the current stream
                self._events[b].record()


_STAGERS: dict[str, PinnedStager] = {}
_STAGERS_LOCK = threading.Lock()


def get_stager(device: torch.device) -> PinnedStager | None:
    """Process-wide stager for ``device``, or None where pinned staging does not apply (non-CUDA device, or
    Linux kernel 6.11, where pinned memory is buggy). Callers fall back to a plain ``.to(device)``."""
    if device.type != "cuda" or is_linux_kernel_6_11():
        return None
    key = str(torch.device(device.type, device.index if device.index is not None else torch.cuda.current_device()))
    with _STAGERS_LOCK:
        if key not in _STAGERS:
            _STAGERS[key] = PinnedStager(torch.device(key))
        return _STAGERS[key]
