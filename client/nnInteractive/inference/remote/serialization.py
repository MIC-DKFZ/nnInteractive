"""Compact array (de)serialization for the nnInteractive client/server protocol.

Wire format:

    magic(4)   | b"NNIA"
    version(1) | uint8, currently 1
    codec(1)   | uint8 (1 = blosc2.Codec.ZSTD, 2 = blosc2.Codec.LZ4)
    ndim(1)    | uint8
    dtype_len(1) | uint8 (length of the dtype string in bytes)
    dtype(dtype_len) | ascii (e.g. "float32", "uint8", "float16")
    shape(ndim * 8) | int64 little-endian per dim
    payload    | chunked blosc2-compressed bytes (see below)

Payload format:

    nchunks(4) | uint32 little-endian
    for each chunk:
        ulen(8)      | uint64 little-endian, uncompressed byte length
        clen(8)      | uint64 little-endian, compressed byte length
        cbytes(clen) | blosc2-compressed bytes

Each chunk's uncompressed length is at most _CHUNK_SIZE bytes. This works
around the ~2 GiB per-call input limit of blosc2.compress2() (its source
length is a C int32), which is hit by e.g. a 1024^3 float32 image (~4 GiB)
or a 1024^3 int16 image (~2 GiB).
"""

from __future__ import annotations

import struct
from typing import Optional

import blosc2
import numpy as np

MAGIC = b"NNIA"
VERSION = 1

# blosc2.compress2 takes its source length as a C int32, capping per-call
# input at 2 GiB - 1. Chunk at 1 GiB to leave plenty of headroom.
_CHUNK_SIZE = 1 << 30

_CODEC_ID = {
    blosc2.Codec.ZSTD: 1,
    blosc2.Codec.LZ4: 2,
}
_ID_CODEC = {v: k for k, v in _CODEC_ID.items()}


# Fraction of each axis used for the center crop that the filter heuristic compresses.
_SELECT_FILTER_CROP_FRACTION = 0.25


def _compress_all(
    raw: memoryview, total: int, codec: blosc2.Codec, clevel: int, filters: list, nthreads: Optional[int], typesize: int
) -> int:
    """Compressed byte length of ``raw`` under ``filters``, chunked exactly as pack_array does."""
    extra = {} if nthreads is None else {"nthreads": nthreads}
    size = 0
    nchunks = (total + _CHUNK_SIZE - 1) // _CHUNK_SIZE
    for i in range(nchunks):
        start = i * _CHUNK_SIZE
        end = min(start + _CHUNK_SIZE, total)
        size += len(
            blosc2.compress2(raw[start:end], typesize=typesize, codec=codec, clevel=clevel, filters=filters, **extra)
        )
    return size


def _select_filter(arr: np.ndarray, codec: blosc2.Codec, clevel: int, nthreads: Optional[int]) -> "blosc2.Filter":
    """Pick NOFILTER vs SHUFFLE for ``arr`` by trial-compressing a small centered crop.

    Uses ``compress2`` on the raw bytes — exactly the path pack_array takes — so the decision
    is consistent with how the whole array is actually compressed. The crop is
    ``_SELECT_FILTER_CROP_FRACTION`` of each axis (centered), keeping the trial cheap and
    representative (lands on foreground). Ties go to NOFILTER; any failure falls back to it.
    """
    try:
        crop_shape = [max(1, int(s * _SELECT_FILTER_CROP_FRACTION)) for s in arr.shape]
        slices = tuple(slice((s - cs) // 2, (s - cs) // 2 + cs) for s, cs in zip(arr.shape, crop_shape))
        crop = np.ascontiguousarray(arr[slices])
        raw = memoryview(crop).cast("B")
        total = raw.nbytes
        typesize = crop.dtype.itemsize

        best_filter, best_bytes = blosc2.Filter.NOFILTER, None
        for f in (blosc2.Filter.NOFILTER, blosc2.Filter.SHUFFLE):
            cb = _compress_all(raw, total, codec, clevel, [f], nthreads, typesize)
            if best_bytes is None or cb < best_bytes:
                best_bytes, best_filter = cb, f
        return best_filter
    except Exception as e:
        from warnings import warn

        warn(f"_select_filter failed ({e!r}); falling back to NOFILTER.")
        return blosc2.Filter.NOFILTER


def pack_array(
    arr: np.ndarray,
    codec: blosc2.Codec = blosc2.Codec.ZSTD,
    clevel: int = 3,
    filters: Optional[list] = None,
    nthreads: Optional[int] = None,
) -> bytes:
    """Serialize a numpy array to a self-describing compressed byte string.

    ``filters`` is the blosc2 filter pipeline to apply. If ``None`` (the default), the
    better of NOFILTER/SHUFFLE is auto-selected by trial-compressing a cheap, representative
    slab — appropriate for images, whose optimum depends on the data. Callers that already
    know the optimum (interactions and segmentations compress best with NOFILTER) should pass
    ``[blosc2.Filter.NOFILTER]`` to skip the selection. The chosen filter is self-describing
    inside the blosc2 frame, so unpack_array (decompress2) needs no changes.

    ``nthreads`` is the per-call blosc2 thread count for compression. ``None`` (the default)
    inherits blosc2's global ``nthreads`` (= core count). Passing an explicit value overrides
    it for this call only, without mutating global state.
    """
    arr = np.ascontiguousarray(arr)
    dtype_str = arr.dtype.str.lstrip("<>|=").encode("ascii")
    if arr.dtype.byteorder not in ("=", "|", "<"):
        # Force little-endian on the wire so the reader doesn't need to swap.
        arr = arr.astype(arr.dtype.newbyteorder("<"))
        dtype_str = arr.dtype.str.lstrip("<>|=").encode("ascii")

    # Use a stable, readable dtype string (e.g. "float32") rather than the
    # platform-dependent shorthand ("f4").
    name = np.dtype(arr.dtype).name.encode("ascii")
    if len(name) > 255:
        raise ValueError(f"dtype name too long: {name!r}")

    header = struct.pack(
        f"<4sBBBB{len(name)}s",
        MAGIC,
        VERSION,
        _CODEC_ID[codec],
        arr.ndim,
        len(name),
        name,
    )
    shape_bytes = struct.pack(f"<{arr.ndim}q", *arr.shape)

    # Zero-copy byte view over the (contiguous) array; sliced per chunk below.
    raw = memoryview(arr).cast("B")
    total = raw.nbytes
    nchunks = (total + _CHUNK_SIZE - 1) // _CHUNK_SIZE

    # Declare the real element size to blosc2. This matters for two reasons:
    #   1. SHUFFLE only helps when typesize matches the element size (it regroups byte-planes of
    #      each element), so itemsize gives the best ratio for multi-byte arrays (e.g. float32 images).
    #   2. It avoids a blosc2 bug: compress2() on an all-zeros buffer whose length is not a multiple
    #      of typesize emits a "zero special value" frame that decompress2() then cannot read
    #      (e.g. a 707658-byte all-zero uint8 diff with the default typesize=8). A contiguous array's
    #      byte length is always a multiple of itemsize, and _CHUNK_SIZE is a multiple of every
    #      power-of-two itemsize, so every chunk slice stays aligned and the broken path is never hit.
    # See unpack_array (decompress2 reads typesize from the frame; it needs no matching argument).
    typesize = arr.dtype.itemsize

    if filters is None:
        # Auto-select the better filter from a small centered crop, using the same
        # compress2 path as below for consistency.
        filters = [_select_filter(arr, codec, clevel, nthreads)]

    extra = {} if nthreads is None else {"nthreads": nthreads}
    parts = [header, shape_bytes, struct.pack("<I", nchunks)]
    for i in range(nchunks):
        start = i * _CHUNK_SIZE
        end = min(start + _CHUNK_SIZE, total)
        chunk = blosc2.compress2(
            raw[start:end], typesize=typesize, codec=codec, clevel=clevel, filters=filters, **extra
        )
        parts.append(struct.pack("<QQ", end - start, len(chunk)))
        parts.append(chunk)
    return b"".join(parts)


def unpack_array(buf: bytes) -> np.ndarray:
    """Inverse of pack_array. Raises ValueError on malformed input."""
    if len(buf) < 8:
        raise ValueError("packed array too short")
    magic, version, codec_id, ndim, name_len = struct.unpack_from("<4sBBBB", buf, 0)
    if magic != MAGIC:
        raise ValueError(f"bad magic: {magic!r}")
    if version != VERSION:
        raise ValueError(f"unsupported wire version {version}")
    if codec_id not in _ID_CODEC:
        raise ValueError(f"unsupported codec id {codec_id}")
    offset = 8
    name = buf[offset : offset + name_len].decode("ascii")
    offset += name_len
    shape = struct.unpack_from(f"<{ndim}q", buf, offset)
    offset += ndim * 8

    dtype = np.dtype(name)

    # Decompress each chunk straight into a preallocated output buffer so we
    # don't materialize the full uncompressed payload as a separate bytes
    # object before reshaping.
    (nchunks,) = struct.unpack_from("<I", buf, offset)
    offset += 4
    nelem = 1
    for d in shape:
        nelem *= d
    out = np.empty(nelem, dtype=dtype)
    out_view = memoryview(out).cast("B")
    written = 0
    for _ in range(nchunks):
        ulen, clen = struct.unpack_from("<QQ", buf, offset)
        offset += 16
        chunk = blosc2.decompress2(buf[offset : offset + clen])
        if len(chunk) != ulen:
            raise ValueError(f"chunk size mismatch: header says {ulen} bytes, decoded {len(chunk)}")
        if written + ulen > out_view.nbytes:
            raise ValueError("payload larger than declared array shape")
        out_view[written : written + ulen] = chunk
        written += ulen
        offset += clen
    if written != out_view.nbytes:
        raise ValueError(f"payload size mismatch: expected {out_view.nbytes} bytes, got {written}")
    return out.reshape(shape)


def empty_payload() -> bytes:
    """Return a placeholder payload used when no array is being shipped."""
    return b""


def is_empty_payload(buf: Optional[bytes]) -> bool:
    return buf is None or len(buf) == 0
