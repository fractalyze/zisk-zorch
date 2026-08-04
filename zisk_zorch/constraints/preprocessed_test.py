# Copyright 2026 The zisk-zorch Authors. SPDX-License-Identifier: Apache-2.0
"""Preprocessed columns — the key reader, and the join onto a chip's index space.

The `.const` input is built by `const_fixture` rather than committed, so the
fixture is readable code instead of an opaque 6 KB blob. Not a ZisK AIR — no
ZisK proving key is checked in, and the four chips that now declare `CLK_0`
(riscv-witness#2189 Phase C) draw its values from one — but `.const` is
pil2-stark's format, not a per-key one.

`__L1__` decoding as the row-0 Lagrange basis pins the reader's canonical,
row-major reading of what it is handed.

What a generated input cannot pin is the correspondence to pil2 itself — both
the values and the on-disk byte order, since the reader only ever parses what
`const_fixture` wrote. Both belong to `scripts/extract_const_fixture.py
--check`, which diffs the generated payload against a real pil2 bundle.

The same fixture serves both halves. It is a real pil2 `constPolsMap`, so it
carries the naming split the join turns on — `SpecifiedRanges.OPID` authored and
qualified, `__L1__` synthesized and bare — with a stand-in chip, since no ZisK
chip is defined over this key. `RealProvingKeyTest` closes that last gap when a
ZisK key is on hand.

There is deliberately no LDE assertion here. `extend`'s agreement with pil2's
`extendPol` is already byte-matched against a pil2-derived golden in
`commit/trace_commit_test.py` (`lde.json`, from `tools/fixture-gen`), which is
the package that owns the transform; repeating it here would pin the same
function twice and needed a prover dump to do it.
"""

from __future__ import annotations

import json
import os
import pathlib
import tempfile
from types import SimpleNamespace
from unittest import mock

import frx

# rw chip code views uint64 as the field dtype; x64 must be on before any array op.
frx.config.update("jax_enable_x64", True)

import frx.numpy as fnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402
from rw_constraints import PreprocessedColumn  # noqa: E402
from zk_dtypes import goldilocks as F  # noqa: E402
from zk_dtypes import goldilocksx3 as F3  # noqa: E402

from zisk_zorch.constraints import const_fixture  # noqa: E402
from zisk_zorch.constraints.preprocessed import (  # noqa: E402
    DERIVED_PREPROCESSED,
    Preprocessed,
    full_trace,
    load_chip_preprocessed,
    load_preprocessed,
)
from zisk_zorch.logup.bus import LogUpBus  # noqa: E402

_AIR = "SpecifiedRanges"
_NAMES = const_fixture.COL_NAMES
_N = const_fixture.N_ROWS
# The fixture's two authored const pols, under the bare names an rw schema would
# declare them with.
_OPID, _VALS = "OPID", "VALS"


def _key_root(tmp: pathlib.Path, const: bytes | None = None) -> pathlib.Path:
    """A generated proving-key root for `_AIR`. `const` overrides the payload,
    for the corruption case."""
    air_dir = tmp / _AIR / "air"
    air_dir.mkdir(parents=True, exist_ok=True)
    (air_dir / f"{_AIR}.starkinfo.json").write_text(
        json.dumps(const_fixture.build_starkinfo())
    )
    payload = const_fixture.build_const_bytes() if const is None else const
    (air_dir / f"{_AIR}.const").write_bytes(payload)
    return tmp


class PreprocessedTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.key = _key_root(pathlib.Path(self._tmp.name))

    def test_reads_the_key_in_constpolsmap_order(self) -> None:
        prep = load_preprocessed(_AIR, root=self.key)
        self.assertEqual(prep.names, _NAMES)
        self.assertEqual(prep.values.shape, (_N, len(_NAMES)))
        self.assertEqual(prep.values.dtype, F)

    def test_l1_decodes_as_the_row_0_lagrange_basis(self) -> None:
        l1 = load_preprocessed(_AIR, root=self.key).column("__L1__")
        want = np.zeros(_N, dtype=np.uint64)
        want[0] = 1
        self.assertTrue(bool(fnp.array_equal(l1, fnp.array(want, dtype=F))))

    def test_columns_selects_and_reorders(self) -> None:
        full = load_preprocessed(_AIR, root=self.key)
        picked = load_preprocessed(_AIR, root=self.key, columns=["__L1__", _NAMES[0]])
        self.assertEqual(picked.names, ("__L1__", _NAMES[0]))
        self.assertEqual(picked.values.shape, (_N, 2))
        self.assertTrue(
            bool(fnp.array_equal(picked.column("__L1__"), full.column("__L1__")))
        )
        self.assertTrue(
            bool(fnp.array_equal(picked.column(_NAMES[0]), full.column(_NAMES[0])))
        )

    def test_unknown_column_names_raise(self) -> None:
        with self.assertRaises(KeyError):
            load_preprocessed(_AIR, root=self.key, columns=["CLK_0"])
        with self.assertRaises(KeyError):
            load_preprocessed(_AIR, root=self.key).column("CLK_0")

    def test_size_mismatch_raises(self) -> None:
        """A key that disagrees with its starkinfo must fail, not shear."""
        with tempfile.TemporaryDirectory() as tmp:
            truncated = const_fixture.build_const_bytes()[:-8]
            bogus = _key_root(pathlib.Path(tmp), const=truncated)
            with self.assertRaisesRegex(ValueError, "bytes, expected"):
                load_preprocessed(_AIR, root=bogus)

    def test_feeds_eval_pair_col_as_a_preprocessed_trace(self) -> None:
        """The loop this module closes: #116 raises on a flagged term with no
        preprocessed trace, and what `load_preprocessed` returns is the trace
        that satisfies it."""
        prep = load_preprocessed(_AIR, root=self.key)
        self.assertIsInstance(prep, Preprocessed)
        main = fnp.array(np.zeros((_N, 1), dtype=np.uint64), dtype=F)
        # `1 * prep[__L1__]`, the flag set — so a fallback to `main` could not
        # produce this: column 2 is out of range there.
        vpc = SimpleNamespace(
            constant="0",
            column_weights=[(_NAMES.index("__L1__"), True, "1")],
            column_products=[],
        )

        got = LogUpBus.eval_pair_col(vpc, main, prep.values)
        want = prep.column("__L1__").astype(F3)
        self.assertTrue(bool(fnp.array_equal(got, want)))

        with self.assertRaisesRegex(ValueError, "preprocessed"):
            LogUpBus.eval_pair_col(vpc, main)


def _chip(
    *cols: PreprocessedColumn,
    n_prep: int | None = None,
    num_main_cols: int = 4,
    name: str = "arith_eq",
) -> SimpleNamespace:
    """The four `Chip` attributes the join reads. A stand-in because the key
    here is fibonacci-square's, which no ZisK chip is defined over — the
    rw-side shape is `PreprocessedColumn`, which is the real class.

    `n_prep` overrides the width `num_cols` implies, for the disagreement case.
    """
    if n_prep is None:
        n_prep = sum(c.width for c in cols)
    return SimpleNamespace(
        name=name,
        preprocessed_cols=list(cols),
        num_main_cols=num_main_cols,
        num_cols=num_main_cols + n_prep,
    )


class ChipPreprocessedTest(absltest.TestCase):
    """The join: rw's preprocessed index space against pil2's named const pols.

    The rule under test is that pil2 qualifies an authored const pol with its
    AIR where rw's schema name is bare. `SpecifiedRanges` exercises it for real
    — `OPID` / `VALS` are authored and dotted, `__L1__` is pil2's own and bare —
    on a stand-in chip. `RealProvingKeyTest` runs the same join against a ZisK
    key's `ArithEq.CLK_0`.
    """

    def setUp(self) -> None:
        super().setUp()
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.key = _key_root(pathlib.Path(self._tmp.name))

    def test_a_chip_with_no_fixed_column_needs_no_key(self) -> None:
        """Every ZisK chip today. The `None` is `eval_pair_col`'s own default,
        and no key is read to produce it — hence the unreadable root."""
        chip = _chip()
        self.assertIsNone(load_chip_preprocessed(chip, _AIR, root=self.key / "nope"))

    def test_columns_land_in_rw_index_order_not_constpolsmap_order(self) -> None:
        """The two orders are independent, so the chip declares them reversed:
        reading the key's order instead would swap the columns silently."""
        chip = _chip(
            PreprocessedColumn(name=_VALS, index=0, width=1),
            PreprocessedColumn(name=_OPID, index=1, width=1),
        )
        key = load_preprocessed(_AIR, root=self.key)

        prep = load_chip_preprocessed(chip, _AIR, root=self.key)

        assert prep is not None  # a declared column means a trace, not `None`
        self.assertEqual(prep.shape, (_N, 2))
        self.assertTrue(
            bool(fnp.array_equal(prep[:, 0], key.column(f"{_AIR}.{_VALS}")))
        )
        self.assertTrue(
            bool(fnp.array_equal(prep[:, 1], key.column(f"{_AIR}.{_OPID}")))
        )

    def test_the_trace_reads_back_through_a_flagged_term(self) -> None:
        """End to end: a term flagged at rw index 1 gets the column the chip
        declared there, through the same path a bus denominator takes."""
        chip = _chip(
            PreprocessedColumn(name=_VALS, index=0, width=1),
            PreprocessedColumn(name=_OPID, index=1, width=1),
        )
        prep = load_chip_preprocessed(chip, _AIR, root=self.key)
        main = fnp.array(np.zeros((_N, chip.num_main_cols), dtype=np.uint64), dtype=F)
        vpc = SimpleNamespace(
            constant="0", column_weights=[(1, True, "1")], column_products=[]
        )

        got = LogUpBus.eval_pair_col(vpc, main, prep)

        want = load_preprocessed(_AIR, root=self.key).column(f"{_AIR}.{_OPID}")
        self.assertTrue(bool(fnp.array_equal(got, want.astype(F3))))

    def test_a_name_the_key_does_not_carry_raises(self) -> None:
        """The join is the unverified half of this contract, so a miss has to be
        an error and not a fallback: `CLK_0` is what ZisK will declare first."""
        chip = _chip(PreprocessedColumn(name="CLK_0", index=0, width=1))
        with self.assertRaisesRegex(KeyError, "CLK_0"):
            load_chip_preprocessed(chip, _AIR, root=self.key)

    def test_a_gap_in_the_index_space_raises(self) -> None:
        chip = _chip(
            PreprocessedColumn(name=_OPID, index=0, width=1),
            PreprocessedColumn(name=_VALS, index=2, width=1),
            n_prep=3,
        )
        with self.assertRaisesRegex(ValueError, "do not tile"):
            load_chip_preprocessed(chip, _AIR, root=self.key)

    def test_descriptors_disagreeing_with_the_declared_width_raise(self) -> None:
        """`num_cols - num_main_cols` is the width a flagged index may reach;
        descriptors covering less of it would leave the tail unnamed."""
        chip = _chip(PreprocessedColumn(name=_OPID, index=0, width=1), n_prep=2)
        with self.assertRaisesRegex(ValueError, "cover 1 columns"):
            load_chip_preprocessed(chip, _AIR, root=self.key)

    def test_a_tensor_column_raises(self) -> None:
        """SP1's `program_rom.pc` is `tensor<3x!F>`; nothing pins what pil2 calls
        its elements, so the reader must not guess."""
        chip = _chip(PreprocessedColumn(name="pc", index=0, width=3))
        with self.assertRaises(NotImplementedError):
            load_chip_preprocessed(chip, _AIR, root=self.key)

    def test_a_derived_column_is_the_roll_sum_of_its_base(self) -> None:
        """A derived gate has no const pol of its own: its values are the
        cyclic roll-sum of a key-backed base column, and near row 0 those
        wrap into the END of the trace — which is why the genuine column is
        rolled instead of reconstructing `row mod period`."""
        chip = _chip(
            PreprocessedColumn(name=_OPID, index=0, width=1),
            PreprocessedColumn(name="OPID_BACK_2", index=1, width=1),
        )
        with mock.patch.dict(
            DERIVED_PREPROCESSED, {"OPID_BACK_2": (_OPID, 1, 2)}, clear=False
        ):
            prep = load_chip_preprocessed(chip, _AIR, root=self.key)

        assert prep is not None
        base = np.asarray(
            load_preprocessed(_AIR, root=self.key).column(f"{_AIR}.{_OPID}"),
            dtype=np.uint64,
        )
        want = np.roll(base, 1) + np.roll(base, 2)
        self.assertTrue(bool(fnp.array_equal(prep[:, 1], fnp.array(want, dtype=F))))

    def test_full_trace_prepends_the_prefix(self) -> None:
        chip = _chip(PreprocessedColumn(name=_OPID, index=0, width=1))
        main = fnp.array(
            np.arange(_N * chip.num_main_cols, dtype=np.uint64).reshape(
                _N, chip.num_main_cols
            )
            % 7,
            dtype=F,
        )

        full = full_trace(chip, main, air=_AIR, root=self.key)

        self.assertEqual(full.shape, (_N, chip.num_cols))
        key_col = load_preprocessed(_AIR, root=self.key).column(f"{_AIR}.{_OPID}")
        self.assertTrue(bool(fnp.array_equal(full[:, 0], key_col)))
        self.assertTrue(bool(fnp.array_equal(full[:, 1:], main)))

    def test_full_trace_passes_a_prep_free_chip_through(self) -> None:
        chip = _chip()
        main = fnp.array(np.zeros((3, chip.num_main_cols), dtype=np.uint64), dtype=F)
        # An unreadable root proves no key is touched on this path.
        got = full_trace(chip, main, air=_AIR, root=self.key / "nope")
        self.assertIs(got, main)

    def test_full_trace_rejects_a_height_mismatch(self) -> None:
        """The key's columns span the air's full height; a shorter main trace
        cannot be aligned to them without guessing which rows it covers."""
        chip = _chip(PreprocessedColumn(name=_OPID, index=0, width=1))
        main = fnp.array(
            np.zeros((_N // 2, chip.num_main_cols), dtype=np.uint64), dtype=F
        )
        with self.assertRaisesRegex(ValueError, "full height"):
            full_trace(chip, main, air=_AIR, root=self.key)

    def test_full_trace_rejects_an_already_combined_row(self) -> None:
        """Width is the axis that fails quietly. Prefixing a combined row a
        second time yields a trace an exported fn still slices in range, so it
        would evaluate and disagree rather than raise — measured on `arith_eq`,
        where a 45-wide trace against its 46 returns different violations.
        """
        chip = _chip(PreprocessedColumn(name=_OPID, index=0, width=1))
        combined = fnp.array(np.zeros((_N, chip.num_cols), dtype=np.uint64), dtype=F)
        with self.assertRaisesRegex(ValueError, "columns wide"):
            full_trace(chip, combined, air=_AIR, root=self.key)


# `$ZISK_PROVING_KEY`, the same variable `load_preprocessed` defaults from. No
# ZisK key is checked in — it is a multi-GB download — so this is the one place
# the contract can be checked against the real thing when a key is at hand.
_ZISK_KEY = (
    pathlib.Path(os.environ["ZISK_PROVING_KEY"])
    if os.environ.get("ZISK_PROVING_KEY")
    else None
)


@absltest.skipUnless(
    _ZISK_KEY is not None and _ZISK_KEY.is_dir(),
    "set ZISK_PROVING_KEY to a ZisK provingKey .../Zisk/airs directory",
)
class RealProvingKeyTest(absltest.TestCase):
    """The join and the reader against a real ZisK key, not the fixture.

    Everything else in this file proves the code is self-consistent. This is
    what proves it agrees with pil2.
    """

    # rw chip name -> (pil2 air, rows one operation occupies). The periods are
    # the schemas' own CLOCKS; only arith_eq's divides a power-of-two height.
    _GATED = (
        ("arith_eq", "ArithEq", 16),
        ("arith_eq_384", "ArithEq384", 24),
        ("keccak", "Keccakf", 25),
        ("sha256", "Sha256f", 72),
    )

    def _clk0(self, air: str) -> np.ndarray:
        chip = _chip(PreprocessedColumn(name="CLK_0", index=0, width=1))
        values = load_chip_preprocessed(chip, air, root=_ZISK_KEY)
        assert values is not None
        return np.asarray(values.view(fnp.uint64)).reshape(-1)

    def test_the_join_resolves_clk0_on_every_gated_air(self) -> None:
        """`CLK_0` is bare in rw's schema and AIR-qualified in the key, so this
        fails outright if the prefix rule is wrong — there is no near-miss."""
        for _, air, period in self._GATED:
            with self.subTest(air=air):
                clk0 = self._clk0(air)
                self.assertEqual(set(np.unique(clk0).tolist()), {0, 1})
                # Row-major decoding is what puts the ones on the period; a
                # column-major misread of the same bytes would not.
                self.assertEqual(
                    np.flatnonzero(clk0)[:3].tolist(), [0, period, 2 * period]
                )
                self.assertTrue(np.all(np.flatnonzero(clk0) % period == 0))

    def test_recomputing_the_period_would_not_reproduce_the_key(self) -> None:
        """Why the values are read rather than generated (#115 rejected
        recomputation as inventing a second authority for a fixed column).

        The key's column is the naive `r % period == 0` *minus a tail*: where
        the period does not divide the height the last operation cannot be
        whole, and the key drops those clock starts. rw's own producer fills
        `r % period == 0` for every row, so the two agree only on `arith_eq`.
        A generated column would be wrong on those rows — silently, since it is
        the right dtype and shape.
        """
        for name, air, period in self._GATED:
            with self.subTest(air=air):
                clk0 = self._clk0(air)
                naive = np.zeros_like(clk0)
                naive[::period] = 1
                # The key never sets a row the formula does not.
                self.assertTrue(np.all(clk0 <= naive))
                divides = len(clk0) % period == 0
                self.assertEqual(name == "arith_eq", divides)
                if divides:
                    self.assertTrue(np.array_equal(clk0, naive))
                else:
                    self.assertLess(int(clk0.sum()), int(naive.sum()))

    def test_derived_gates_hold_their_defining_property_on_the_key(self) -> None:
        """`DERIVED_PREPROCESSED` landed ahead of the wheel that declares these
        columns, so nothing had checked its shifts against real values.

        Each gate sums `CLK_0` over shifts `k_lo..k_hi`, and every range is
        narrower than its period, so at most one clock start can fall in the
        window: the gate is a 0/1 indicator of "this row sits `k_lo..k_hi` rows
        into a real operation". Asserting that from the clock's own set
        positions checks the shift direction and the range together — a forward
        roll, or an off-by-one range, breaks it.
        """
        gates = (
            ("ArithEq384", 24, "SEL_LATCH_GATE"),
            ("ArithEq384", 24, "CLK_0_BACK_23"),
            ("Keccakf", 25, "IN_USE_LATCHED"),
            ("Sha256f", 72, "IN_USE_ACTIVE"),
        )
        for air, period, gate_name in gates:
            with self.subTest(gate=gate_name):
                base, k_lo, k_hi = DERIVED_PREPROCESSED[gate_name]
                self.assertEqual(base, "CLK_0")
                self.assertLess(k_hi, period, "a wider window would double-count")
                chip = _chip(
                    PreprocessedColumn(name="CLK_0", index=0, width=1),
                    PreprocessedColumn(name=gate_name, index=1, width=1),
                )
                cols = load_chip_preprocessed(chip, air, root=_ZISK_KEY)
                assert cols is not None
                values = np.asarray(cols.view(fnp.uint64))
                clk0, gate = values[:, 0], values[:, 1]

                # Row-wise, not row-by-start: the pairwise form is O(N * ops),
                # which is 45e9 cells on ArithEq384 and takes the box down.
                # Each row's only candidate start is the multiple of the period
                # at or below it, and the gate fires when that start is real and
                # the row sits `k_lo..k_hi` past it.
                rows = np.arange(len(clk0))
                phase = rows % period
                inside = (phase >= k_lo) & (phase <= k_hi) & (clk0[rows - phase] == 1)
                self.assertTrue(np.array_equal(gate, inside.astype(gate.dtype)))
                # The zeroed tail is what keeps the wrap clean: rolling a column
                # whose last operation ran to the final row would carry starts
                # back onto row 0's window.
                self.assertEqual(gate[0], 0)


if __name__ == "__main__":
    absltest.main()
