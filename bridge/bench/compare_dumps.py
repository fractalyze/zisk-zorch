#!/usr/bin/env python3
"""Word-for-word comparison of two `ZZ_DUMP_PROOFS` directories (one raw
little-endian u64 file per basic instance, named by instance id): the
native ↔ bridge byte-gate. Usage: compare_dumps.py <native-dir> <bridge-dir>.
Exits non-zero unless every native dump has an identical counterpart.

Instance ids are assigned per run, so two runs of different workloads (or the
same guest under different prove flags) put different airs on the same id and
comparing by filename would then compare unrelated proofs. Each dump directory
written by `run.sh` sits beside the `run.log` that produced it, and proofman
logs `GEN_PROOF_<instance> [<air group>:<air>]` once per basic instance in both
modes, so the two plans are checked against each other before any bytes are.
"""
from __future__ import annotations

import pathlib
import re
import sys

import numpy as np

GEN_PROOF = re.compile(r">>> GEN_PROOF_(\d+) \[(\d+):(\d+)\]")


def plan(dumps: pathlib.Path) -> dict[str, str]:
    """Instance id -> `<air group>:<air>`, from the run.log beside the dumps.
    Empty when there is no log to read, which only disables the check."""
    log = dumps.parent / "run.log"
    if not log.exists():
        return {}
    text = log.read_text(errors="replace")
    return {inst: f"{group}:{air}" for inst, group, air in GEN_PROOF.findall(text)}


def same_plan(a: pathlib.Path, b: pathlib.Path) -> bool:
    """Whether the two runs put the same air on every instance they share."""
    pa, pb = plan(a), plan(b)
    if not pa or not pb:
        missing = a if not pa else b
        print(f"note: no run.log beside {missing}; comparing by instance id alone")
        return True
    differing = {i: (pa[i], pb[i]) for i in pa.keys() & pb.keys() if pa[i] != pb[i]}
    if differing:
        runs = f"{a.parent.name} vs {b.parent.name}"
        print(f"the two runs ({runs}) planned different airs, by instance:")
        for i, (x, y) in sorted(differing.items(), key=lambda kv: int(kv[0])):
            print(f"  instance {i}: air {x} vs {y}")
        return False
    return True


def main(argv: list[str]) -> int:
    a, b = pathlib.Path(argv[1]), pathlib.Path(argv[2])
    if not same_plan(a, b):
        print("refusing to compare: same instance ids, different airs")
        return 2
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
