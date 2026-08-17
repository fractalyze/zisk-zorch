"""ZisK's ``proof.bin`` envelope — the vadcop-final proof as the native
CLI writes and verifies it.

The file is a bincode-v2 varint stream:

    0x00 | Vec<u64> flat proof | Vec<u64> verkey(4) | 0x00
    | String hash-family | Vec<u8> digest field (256) | Vec<u64> 4 | 0x00

Varints: a value <= 250 is its own byte; 0xfb/0xfc/0xfd prefix a u16/
u32/u64 little-endian. The three trailer fields after the proof are
key/program-level constants (verified identical across independent
native proves of different proofs): the vadcop_final verkey, a 32-byte
digest in a fixed 256-byte field, and a second 4-word vector — the
writer takes them as one opaque `trailer` blob captured from the key's
lineage, and only the proof vector varies per prove.
"""

from __future__ import annotations

import numpy as np

_MARKER = {0xFB: 2, 0xFC: 4, 0xFD: 8, 0xFE: 16}


def _decode_varint(buf: bytes, pos: int) -> tuple[int, int]:
    if pos >= len(buf):
        raise ValueError(f"truncated varint at offset {pos}")
    b = buf[pos]
    pos += 1
    if b <= 250:
        return b, pos
    if b not in _MARKER:
        raise ValueError(f"unknown varint marker {b:#04x} at offset {pos - 1}")
    n = _MARKER[b]
    # int.from_bytes accepts a short slice, so an unchecked read past the
    # end decodes to a plausible but wrong word instead of failing.
    if pos + n > len(buf):
        raise ValueError(f"truncated {n}-byte varint at offset {pos}")
    return int.from_bytes(buf[pos : pos + n], "little"), pos + n


def _encode_varint(v: int) -> bytes:
    if v <= 250:
        return bytes([v])
    if v < 1 << 16:
        return b"\xfb" + v.to_bytes(2, "little")
    if v < 1 << 32:
        return b"\xfc" + v.to_bytes(4, "little")
    return b"\xfd" + v.to_bytes(8, "little")


def decode(buf: bytes) -> tuple[np.ndarray, bytes]:
    """`proof.bin` bytes -> (flat proof words, the constant trailer)."""
    if not buf:
        raise ValueError("empty buffer")
    if buf[0] != 0:
        raise ValueError(f"unexpected lead byte {buf[0]:#04x}")
    count, pos = _decode_varint(buf, 1)
    # Every word costs at least one byte, so a count past the remaining
    # length is a corrupt header — reject it before allocating for it.
    if count > len(buf) - pos:
        raise ValueError(f"proof count {count} exceeds {len(buf) - pos} bytes left")
    words = np.empty(count, dtype=np.uint64)
    for i in range(count):
        words[i], pos = _decode_varint(buf, pos)
    return words, buf[pos:]


def encode(proof: np.ndarray, trailer: bytes) -> bytes:
    """(flat proof words, trailer) -> `proof.bin` bytes."""
    parts = [b"\x00", _encode_varint(int(proof.size))]
    parts.extend(_encode_varint(int(w)) for w in proof)
    parts.append(trailer)
    return b"".join(parts)
