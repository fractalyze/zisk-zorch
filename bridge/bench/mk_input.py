#!/usr/bin/env python3
"""Write a ZiskStdin input file holding one `u32` for a guest that does
`ziskos::io::read::<u32>()` (the sha-hasher example's iteration count).

Usage: mk_input.py <n> <out.bin>

The layout mirrors proofman's `ZiskStdin::write_slice`
(https://github.com/0xPolygonHermez/zisk/blob/v1.0.0-alpha/common/src/io/stdin/zisk_stdin.rs):
one frame per `write`, `[len: u64 LE][data][zero pad]` with the pad taking
`8 + len` up to a multiple of 8. `write` serializes with bincode 2's standard
config, whose integers are varints: one byte below 251, `0xfb` + u16 LE up to
u16::MAX, `0xfc` + u32 LE above. A wrong width reads back as a different `n`,
so the framing is pinned by `mk_input_test.py`.
"""
from __future__ import annotations

import struct
import sys


def varint_u32(n: int) -> bytes:
    if not 0 <= n <= 0xFFFFFFFF:
        raise ValueError(f"not a u32: {n}")
    if n < 251:
        return bytes([n])
    if n <= 0xFFFF:
        return b"\xfb" + struct.pack("<H", n)
    return b"\xfc" + struct.pack("<I", n)


def frame(data: bytes) -> bytes:
    pad = (8 - (8 + len(data)) % 8) % 8
    return struct.pack("<Q", len(data)) + data + b"\0" * pad


def main(argv: list[str]) -> int:
    if len(argv) != 3:
        print(__doc__, file=sys.stderr)
        return 2
    n, out = int(argv[1]), argv[2]
    with open(out, "wb") as f:
        f.write(frame(varint_u32(n)))
    print(f"{out}: u32 {n}")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
