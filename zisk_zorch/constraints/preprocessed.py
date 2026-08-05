# Copyright 2026 The zisk-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""The proving key's preprocessed (fixed) columns — pil2's const pols.

The trace ``LogUpBus.eval_pair_col`` evaluates ``is_preprocessed`` terms against
(fractalyze/zisk-zorch#115).

``<Air>.const`` is a bare little-endian u64 matrix, row-major
``(N, nConstants)``, canonical (not Montgomery), on the **base** domain — the
extended LDE is ``<Air>.consttree``. pil2 ships no schema for this, so
:func:`load_preprocessed` re-checks the byte count on every load: a format
change has to surface as an error, not as a sheared trace.

``constPolsMap`` order is pil2's, not rw's preprocessed index space — pil2
synthesizes selectors rw never authored (``__L1__``), and names authored pols
``<Air>.<NAME>`` against rw's bare ``<NAME>``. So ``values`` must not be passed
to ``eval_pair_col`` as if rw-indexed; :func:`load_chip_preprocessed` is what
crosses the two index spaces.
"""

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
from typing import TYPE_CHECKING, Sequence

import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F

if TYPE_CHECKING:
    from rw_constraints import Chip

# The ziskup proving key, same default as `scripts/extract_cexp.py`.
_PROVING_KEY = Path(
    os.environ.get("ZISK_PROVING_KEY", Path.home() / ".zisk/provingKey/zisk/Zisk/airs")
)


@dataclasses.dataclass(frozen=True, eq=False)
class Preprocessed:
    """An AIR's fixed columns: ``values`` ``(N, len(names))`` over ``F``, with
    ``names[j]`` naming column ``j``.

    ``eq=False``: ``values`` is a device array, so a generated ``__eq__`` would
    return an array rather than a bool and its ``__hash__`` would raise.
    """

    values: Array
    names: tuple[str, ...]

    def column(self, name: str) -> Array:
        """The single fixed column ``name`` as an ``(N,)`` array."""
        try:
            return self.values[:, self.names.index(name)]
        except ValueError:
            raise KeyError(
                f"no preprocessed column {name!r}; have {list(self.names)}"
            ) from None


def load_preprocessed(
    air: str,
    root: Path | None = None,
    columns: Sequence[str] | None = None,
) -> Preprocessed:
    """Read ``<Air>.const`` into a preprocessed trace.

    ``root`` defaults to ``$ZISK_PROVING_KEY`` (else ``~/.zisk/...``), read as
    ``<root>/<air>/air/<air>.{const,starkinfo.json}``.

    ``columns`` selects by name *and* fixes the order — how a caller
    materializes a trace in its own index space rather than pil2's. Only the
    selected columns come off the memory map.
    """
    air_dir = (root if root is not None else _PROVING_KEY) / air / "air"
    starkinfo = json.loads((air_dir / f"{air}.starkinfo.json").read_text())
    const_map = starkinfo["constPolsMap"]

    n_cols = starkinfo["nConstants"]
    if n_cols != len(const_map):
        raise ValueError(
            f"{air}: starkinfo nConstants={n_cols} but constPolsMap has "
            f"{len(const_map)} entries"
        )
    wide = [c["name"] for c in const_map if c["dim"] != 1]
    if wide:
        # A dim-3 const pol takes three u64 slots per row, which would shear the
        # `(n_rows, n_cols)` map below.
        raise NotImplementedError(f"{air}: non-base-field const pols {wide}")

    n_rows = 1 << starkinfo["starkStruct"]["nBits"]
    path = air_dir / f"{air}.const"
    expected = n_rows * n_cols * 8
    actual = path.stat().st_size
    if actual != expected:
        raise ValueError(
            f"{path}: {actual} bytes, expected {expected} "
            f"({n_rows} rows x {n_cols} cols x 8). The proving key does not "
            "match its starkinfo, or pil2's const layout has changed."
        )

    available = tuple(c["name"] for c in const_map)
    raw = np.memmap(path, dtype="<u8", mode="r", shape=(n_rows, n_cols))
    if columns is None:
        names, selected = available, np.array(raw)
    else:
        missing = [c for c in columns if c not in available]
        if missing:
            raise KeyError(
                f"{air}: no preprocessed columns {missing}; have {list(available)}"
            )
        names = tuple(columns)
        selected = np.stack([raw[:, available.index(c)] for c in names], axis=1)
    return Preprocessed(values=fnp.array(selected, dtype=F), names=names)


# rw chip name -> pil2 AIR directory name under the proving key. Mirrors the
# oracle's CHIP_AIRS table (riscv-witness tools/zisk/zisk_constraint_eval);
# not derivable from `chip.name`, since pil2 keeps the native ZisK air names.
ZISK_CHIP_AIRS: dict[str, str] = {
    "add256": "Add256",
    "arith": "Arith",
    "arith_eq": "ArithEq",
    "arith_eq_384": "ArithEq384",
    "binary": "Binary",
    "binary_add": "BinaryAdd",
    "binary_extension": "BinaryExtension",
    "keccak": "Keccakf",
    "main": "Main",
    "mem": "Mem",
    "mem_align": "MemAlign",
    "mem_align_byte": "MemAlignByte",
    "mem_align_read_byte": "MemAlignReadByte",
    "mem_align_write_byte": "MemAlignWriteByte",
    "sha256": "Sha256f",
}

# Derived preprocessed columns: rw-side rolls of a proving-key fixed column,
# with NO const pol of their own — the schema comments in riscv-witness carry
# the same derivation, and its cross-reference gate pins the values against
# the native shifted-clock gates. name -> (base column, k_lo, k_hi), meaning
# sum_{k=k_lo..k_hi} base[(row - k) mod N]: shifted PIL references wrap the
# trace cyclically, and N is generally NOT a multiple of the clock period, so
# the roll of the genuine key column is required — the values near row 0
# cannot be reconstructed from `row mod period`.
DERIVED_PREPROCESSED: dict[str, tuple[str, int, int]] = {
    "SEL_LATCH_GATE": ("CLK_0", 1, 20),
    "CLK_0_BACK_23": ("CLK_0", 23, 23),
    "IN_USE_LATCHED": ("CLK_0", 1, 24),
    "IN_USE_ACTIVE": ("CLK_0", 1, 17),
}


def full_trace(
    chip: "Chip",
    main_trace: Array,
    air: str | None = None,
    root: Path | None = None,
) -> Array:
    """The ``[prep | main]`` trace ``eval_constraints`` expects for `chip`.

    A chip whose schema carries ``{preprocessed}`` columns exports its
    constraint fns over the combined row, so the main trace alone is the
    wrong width. The prefix values come from the proving key (and the
    derived-column rolls), which fixes the row count: the main trace must be
    the air's full height, aligned at row 0 — exactly what a pil2 dump or a
    generated witness is. A chip with no fixed columns passes through.
    """
    if main_trace.shape[1] != chip.num_main_cols:
        # Width is the axis that fails quietly: an exported fn slicing a trace
        # of the wrong width stays in range and returns a wrong answer instead
        # of raising (`arith_eq` at 45 against its 46 evaluates and disagrees).
        raise ValueError(
            f"{chip.name}: main trace is {main_trace.shape[1]} columns wide, "
            f"expected num_main_cols {chip.num_main_cols} — a combined row "
            "passed in here would be prefixed a second time"
        )
    prep = load_chip_preprocessed(
        chip, air if air is not None else ZISK_CHIP_AIRS[chip.name], root=root
    )
    if prep is None:
        return main_trace
    if main_trace.shape[0] != prep.shape[0]:
        raise ValueError(
            f"{chip.name}: main trace has {main_trace.shape[0]} rows but the "
            f"proving key's preprocessed columns span {prep.shape[0]} — the "
            "trace must be the air's full height, aligned at row 0"
        )
    return fnp.concatenate([prep, main_trace], axis=1)


def load_chip_preprocessed(
    chip: "Chip", air: str, root: Path | None = None
) -> Array | None:
    """The chip's fixed columns as a trace in **rw's** preprocessed index space.

    This is the ``preprocessed`` argument of
    :meth:`~zisk_zorch.logup.bus.LogUpBus.eval_pair_col`: column ``j`` is what a
    term flagged ``is_preprocessed`` at index ``j`` reads. ``None`` when the chip
    declares no fixed column — which is what that argument takes for a chip whose
    terms never set the flag, so every ZisK chip today. Columns come back in
    ``chip.preprocessed_cols`` order, which is where their names stay.

    The two sides key the same column differently, and neither ordering can be
    made to serve the other: rw's manifest identifies it by index into a space
    its schema owns, pil2's key by name in ``constPolsMap``. ``air`` names the
    pil2 AIR holding the values; it is not derivable from ``chip.name``, since
    pil2's AIR names are the native ZisK ones rather than rw's chip names.
    """
    names = _preprocessed_names(chip)
    if not names:
        return None
    # A derived column has no const pol; it is a roll-sum of a key-backed base
    # column, so the key load is the union of the directly-declared names and
    # the derived columns' bases.
    key_backed = [n for n in names if n not in DERIVED_PREPROCESSED]
    bases = [DERIVED_PREPROCESSED[n][0] for n in names if n in DERIVED_PREPROCESSED]
    load_names = list(dict.fromkeys(key_backed + bases))
    loaded = load_preprocessed(
        air, root=root, columns=[_const_pol_name(air, n) for n in load_names]
    ).values
    by_name = {n: loaded[:, i] for i, n in enumerate(load_names)}
    cols = []
    for name in names:
        if name in DERIVED_PREPROCESSED:
            base, k_lo, k_hi = DERIVED_PREPROCESSED[name]
            src = by_name[base]
            acc = fnp.roll(src, k_lo, axis=0)
            for k in range(k_lo + 1, k_hi + 1):
                acc = acc + fnp.roll(src, k, axis=0)
            cols.append(acc)
        else:
            cols.append(by_name[name])
    return fnp.stack(cols, axis=1)


def _preprocessed_names(chip: "Chip") -> tuple[str, ...]:
    """The chip's ``{preprocessed}`` column names, ordered by rw's flat index.

    The descriptors must tile ``[0, num_cols - num_main_cols)`` exactly. A gap
    means a name is missing, and because the result is positional every later
    column would then sit one slot off — a real fixed value under the wrong
    index, which is the failure mode this whole path exists to prevent.
    """
    n_prep = chip.num_cols - chip.num_main_cols
    names: list[str] = []
    end = 0
    for col in sorted(chip.preprocessed_cols, key=lambda c: c.index):
        if col.index != end:
            raise ValueError(
                f"{chip.name}: preprocessed descriptors do not tile the index "
                f"space — {col.name!r} starts at {col.index}, expected {end}"
            )
        if col.width != 1:
            # How pil2 names a tensor const pol's elements is unpinned, and no
            # ZisK schema declares one (this is SP1's `program_rom.pc`).
            raise NotImplementedError(
                f"{chip.name}: preprocessed column {col.name!r} spans "
                f"{col.width} indices; pil2's per-element const-pol name is "
                "unverified (fractalyze/zisk-zorch#115)"
            )
        names.append(col.name)
        end += col.width
    if end != n_prep:
        raise ValueError(
            f"{chip.name}: preprocessed descriptors cover {end} columns, but "
            f"num_cols {chip.num_cols} - num_main_cols {chip.num_main_cols} "
            f"= {n_prep}"
        )
    return tuple(names)


def _const_pol_name(air: str, name: str) -> str:
    """pil2's name for the const pol rw's schema calls ``name``.

    pil2 qualifies an authored const pol with its AIR — ``SpecifiedRanges.OPID``,
    and ZisK's own ``Main.SEGMENT_STEP`` — where rw's ``chip.col`` name is bare.
    Its synthesized selectors (``__L1__``) are unqualified instead, but rw never
    authored those and so never references one.
    """
    return f"{air}.{name}"
