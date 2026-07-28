"""Shared plumbing for the per-stage `verify_*` byte-match runnables.

A `verify_*` runnable replays one prover stage on a *real* pil2-proofman
reference dump and byte-matches every derived value, exiting non-zero on a
mismatch — the assembled-path counterpart to the `golden/` unit vectors (which
pin each primitive in isolation on synthetic inputs). This module holds the
pieces every stage runnable reuses: the `check_match` reporter and the dump
loaders. New stages add their own loader here and a thin runnable in the stage
package. The dump format is written by the harnesses under `golden/pil2_dump/`.
"""

from __future__ import annotations

import json
import pathlib

import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F


def check_match(label: str, got: Array, want: Array) -> bool:
    """Exact field-equality check that prints its own OK/MISMATCH line and
    returns the verdict — never a tolerance (field elements match or they
    don't). The caller ANDs these and `sys.exit(1)`s if any is False, so one
    stage's every anchor is reported before the process fails."""
    ok = bool(fnp.array_equal(got, want))
    print(f"{'OK      ' if ok else 'MISMATCH'} {label}")
    if not ok:
        print(f"  got:  {np.asarray(got).astype(np.uint64).tolist()}")
        print(f"  want: {np.asarray(want).astype(np.uint64).tolist()}")
    return ok


def load_commit_dump(dump_dir: pathlib.Path) -> tuple[dict, Array]:
    """A stage-1 trace-commit dump written by `golden/pil2_dump/`: `commit.json`
    (dims + arity + the reference root) plus `trace.bin` (raw little-endian u64,
    row-major `N x n_cols`, the exact trace pil2 committed). Returns the metadata
    and the trace as a goldilocks matrix."""
    meta = json.loads((dump_dir / "commit.json").read_text())
    n = 1 << meta["n_bits"]
    n_cols = meta["n_cols"]
    raw = np.fromfile(dump_dir / "trace.bin", dtype="<u8")
    if raw.size != n * n_cols:
        raise ValueError(
            f"trace.bin has {raw.size} elements, expected {n * n_cols} "
            f"(n_bits={meta['n_bits']}, n_cols={n_cols}) in {dump_dir}"
        )
    trace = fnp.array(raw.reshape(n, n_cols), dtype=F)
    return meta, trace


def commit_root(meta: dict) -> Array:
    """The reference commitment root from a commit dump's metadata."""
    return fnp.array(np.array([int(v) for v in meta["root"]], dtype=np.uint64), dtype=F)
