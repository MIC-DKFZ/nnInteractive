"""Compact array (de)serialization for the nnInteractive client/server protocol.

Wire format:

    magic(4)   | b"NNIA"
    version(1) | uint8, currently 1
    codec(1)   | uint8 (1 = blosc2.Codec.ZSTD, 2 = blosc2.Codec.LZ4)
    ndim(1)    | uint8
    dtype_len(1) | uint8 (length of the dtype string in bytes)
    dtype(dtype_len) | ascii (e.g. "float32", "uint8", "float16")
    shape(ndim * 8) | int64 little-endian per dim
    payload    | blosc2-compressed bytes of arr.tobytes(order="C")

The header is tiny (single-digit bytes for the dtypes we use); the payload is
where compression matters. blosc2 is already a project dependency.
"""

from __future__ import annotations

import struct
from typing import Optional

import blosc2
import numpy as np

MAGIC = b"NNIA"
VERSION = 1

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
    payload = blosc2.compress2(arr.tobytes(order="C"), codec=codec, clevel=clevel)
    return header + shape_bytes + payload


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
    payload = buf[offset:]

    raw = blosc2.decompress2(payload)
    arr = np.frombuffer(raw, dtype=np.dtype(name)).reshape(shape)
    # frombuffer returns a read-only view; return a writable copy for safety.
    return np.array(arr, copy=True)


def empty_payload() -> bytes:
    """Return a placeholder payload used when no array is being shipped."""
    return b""


def is_empty_payload(buf: Optional[bytes]) -> bool:
    return buf is None or len(buf) == 0
