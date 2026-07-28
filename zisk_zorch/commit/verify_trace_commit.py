# Copyright 2026 The zisk-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""pil2 byte-match harness for the stage-1 trace commit -- a runnable.

Runs `commit_trace` (pil2-stark's `extendAndMerkelize`: coset LDE + linear-hash
leaves + k-ary Poseidon2 Merkle) on the exact trace a real pil2-proofman dump
committed, and byte-matches the commitment root against the dump's reference:

- `commit.json` -- the dump's dims/arity and the reference root the pil2-stark
  CUDA prover produced;
- `trace.bin`   -- the row-major Goldilocks trace pil2 committed.

One scalar seals the stage: the root is a Poseidon2 image of every extended
row, so a match here means the coset LDE, the linear-hash leaves, and the k-ary
fold all reproduced pil2's output byte-for-byte. This is the assembled-path gate
the `golden/` unit vectors (tiny synthetic cases) cannot reach; the reference is
captured from the real CUDA kernels ZisK runs (`golden/pil2_dump/`, recipe in
its README).

    bazel run //zisk_zorch/commit:verify_trace_commit          # committed fixture
    bazel run //zisk_zorch/commit:verify_trace_commit -- \\
        --dump=/path/to/real/stage1_dump                       # a real-dims dump

Exits non-zero on any mismatch.
"""

from __future__ import annotations

import pathlib
import sys

from absl import app, flags

from zisk_zorch.commit.trace_commit import commit_trace
from zisk_zorch.dump import check_match, commit_root, load_commit_dump

# The committed fixture: a real pil2-stark CUDA stage-1 dump at a small
# production-shaped instance (arity 4), so the runnable gates out of the box
# without a GPU pil2 build. Point --dump at a real-dims capture for the scale gate.
_DEFAULT_DUMP = pathlib.Path(__file__).parent / "testdata" / "dump" / "stage1"

_DUMP = flags.DEFINE_string(
    "dump",
    str(_DEFAULT_DUMP),
    "stage-1 commit dump directory (commit.json + trace.bin) from golden/pil2_dump.",
)


def main(argv) -> None:
    del argv
    dump_dir = pathlib.Path(_DUMP.value)
    meta, trace = load_commit_dump(dump_dir)
    print(
        f"dump: n_bits={meta['n_bits']} blowup_bits={meta['blowup_bits']} "
        f"n_cols={meta['n_cols']} arity={meta['arity']} ({dump_dir})"
    )

    commitment = commit_trace(
        trace, blowup=1 << meta["blowup_bits"], arity=meta["arity"]
    )
    ok = check_match("commitment root vs pil2 dump", commitment.root, commit_root(meta))

    if not ok:
        sys.exit(1)
    print("stage-1 trace commit byte-match: ALL OK")


if __name__ == "__main__":
    app.run(main)
