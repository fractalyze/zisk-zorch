# Copyright 2026 The zisk-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""ZisK chip loading over ``rw_constraints``.

``rw_constraints.Chip`` already carries the AIR constraints and LogUp
interactions each ZisK chip needs for stage-2; this module only binds the
Goldilocks field dtype and the ZisK target/version. Unlike sp1-zorch's loader
there is no name-mapping seam — the rw manifest's ZisK chip names are already
the names stage-2 uses (``arith``, ``binary``, ``mem``, …), so a chip is
addressed by its manifest name directly.

The constraints are authored upstream in riscv-witness (``constraints/zisk/v1``,
rw#1524 row-local + rw#1745 lookup-bus) and shipped in the ``rw-constraints``
wheel; the byte-match against pil2-proofman is what verifies them
(fractalyze/zisk-zorch#1).
"""

from __future__ import annotations

import atexit
import functools
import os
import shutil
import tempfile
from pathlib import Path
from typing import Optional

from rw_constraints import Chip, ConstraintRegistry, bundled_constraints_dir
from zk_dtypes import goldilocks


@functools.cache
def _constraints_root() -> Path:
    """``rw_constraints``' bundled data as a plain directory tree.

    Bazel runfiles expose the wheel as a per-file symlink farm, so the
    registry's containment check — each chip file must ``resolve()`` inside
    the version dir — escapes into the store and raises ``Chip file escapes
    version dir``. Materializing one symlink-following copy per process
    restores a tree the registry accepts; plain pip installs skip the copy.
    The manifest probe is a proxy for the whole tree: bazel symlinks every
    file uniformly, so one resolved-in-place file means none are farmed.
    Drop once fractalyze/riscv-witness#1580 makes the check runfiles-safe.
    """
    src = bundled_constraints_dir()
    if src is None:
        raise FileNotFoundError(
            "rw-constraints is installed without its bundled constraint data"
        )
    probe = next(src.rglob("manifest.json"), None)
    if probe is not None and probe.resolve().is_relative_to(src.resolve()):
        return src
    dst = Path(
        tempfile.mkdtemp(prefix="rw-constraints-", dir=os.environ.get("TEST_TMPDIR"))
    )
    atexit.register(shutil.rmtree, dst, ignore_errors=True)
    shutil.copytree(src, dst / "constraints", symlinks=False)
    return dst / "constraints"


@functools.cache
def _registry() -> ConstraintRegistry:
    """One registry per process so its internal per-(target, version, dtype)
    chip cache survives across :func:`load_zisk_chips` calls — a full registry
    load execs every bundled chip module twice (~seconds)."""
    return ConstraintRegistry(_constraints_root())


def load_zisk_chips(
    version: str = "v1",
    chip_names: Optional[list[str]] = None,
) -> dict[str, Chip]:
    """Load the ZisK chip definitions with the Goldilocks field dtype bound.

    Both constraints and interactions get ``goldilocks``. The registry's
    ``interaction_field_dtype`` default is ``jnp.uint32`` — right for SP1, whose
    interaction code is bitwise, but wrong here: ZisK's exported ``*_interaction``
    functions are pure field arithmetic over ``FIELD_DTYPE`` (e.g.
    ``jnp.full(..., dtype=jnp.uint64).view(FIELD_DTYPE)`` + field adds), so the
    bus tuples are Goldilocks-valued and the field dtype must be bound for
    ``eval_interactions`` to produce them.
    """
    chips = _registry().load(
        "zisk",
        version,
        constraint_field_dtype=goldilocks,
        interaction_field_dtype=goldilocks,
    )
    if chip_names is not None:
        unknown = set(chip_names) - chips.keys()
        if unknown:
            raise ValueError(f"unknown ZisK chip names: {sorted(unknown)}")
        chips = {k: v for k, v in chips.items() if k in chip_names}
    return chips
