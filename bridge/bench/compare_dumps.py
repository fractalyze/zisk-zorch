#!/usr/bin/env python3
"""Word-for-word comparison of two `ZZ_DUMP_PROOFS` directories (one raw
little-endian u64 file per basic instance, named by instance id): the
native ↔ bridge byte-gate. Usage: compare_dumps.py <native-dir> <bridge-dir>.
Exits non-zero unless every native dump has an identical counterpart."""
from __future__ import annotations

import pathlib
import sys

import numpy as np


def main(argv: list[str]) -> int:
    a, b = pathlib.Path(argv[1]), pathlib.Path(argv[2])
    native = sorted(a.glob("*.bin"), key=lambda p: int(p.stem))
    same = 0
    for f in native:
        g = b / f.name
        if not g.exists():
            print(f"{f.stem}: missing in {b}")
            continue
        x = np.fromfile(f, dtype=np.uint64)
        y = np.fromfile(g, dtype=np.uint64)
        if x.shape == y.shape and np.array_equal(x, y):
            same += 1
        elif x.shape != y.shape:
            print(f"{f.stem}: {len(x)} words native, {len(y)} bridge")
        else:
            print(
                f"{f.stem}: first differing word {int(np.argmax(x != y))} of {len(x)}"
            )
    print(f"{len(native)} native dumps, {same} identical")
    return 0 if native and same == len(native) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
