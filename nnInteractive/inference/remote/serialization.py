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


def pack_array(arr: np.ndarray, codec: blosc2.Codec = blosc2.Codec.ZSTD, clevel: int = 3) -> bytes:
    """Serialize a numpy array to a self-describing compressed byte string."""
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
    parts = [header, shape_bytes, struct.pack("<I", nchunks)]
    for i in range(nchunks):
        start = i * _CHUNK_SIZE
        end = min(start + _CHUNK_SIZE, total)
        chunk = blosc2.compress2(raw[start:end], codec=codec, clevel=clevel)
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
