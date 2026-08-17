"""Byte-match the cExp reference evaluator against the golden Rust VM.

The golden's `cexp_eval_case` interprets MemAlignReadByte's real composite-
constraint SSA (the proving key's `expressionsCode[cExpId]`) over pil2's `fields`
crate; `cexp_ref.evaluate` interprets the same vendored op list over the zk_dtypes
cubic extension. Equality pins the SSA semantics — the operand model (cm/const
rotations, the std challenges, air(group)Values, `Zi`) and the cubic arithmetic —
across the two implementations. The stage-2 prover's re-authored quotient (from
ingested rw constraints + generated std_sum constraints) must in turn match this
golden `q`.
"""

from __future__ import annotations

import frx

# rw-exported field constants and the cubic embeds need 64-bit ints; set before
# any array op (see chip_loader_test).
frx.config.update("jax_enable_x64", True)

import pathlib  # noqa: E402

import frx.numpy as fnp  # noqa: E402
from absl.testing import absltest  # noqa: E402

from zisk_zorch.golden import load, u64x3  # noqa: E402
from zisk_zorch.quotient.cexp_ref import (  # noqa: E402
    evaluate,
    evaluate_from_constraints,
    live_bytes_per_row,
)

_TESTDATA = pathlib.Path(__file__).parent / "testdata"

# Each golden case names its AIR; load the matching vendored cExp fragment.
_FRAGMENTS = {
    "MemAlignReadByte": "memalign_readbyte_cexp.json",
    "Binary": "binary_cexp.json",
    "Arith": "arith_cexp.json",
}

# The proving key's individual constraints[] per AIR (the generic-fold input).
_CONSTRAINTS = {
    "MemAlignReadByte": "memalign_readbyte_constraints.json",
    "Binary": "binary_constraints.json",
    "Arith": "arith_constraints.json",
}


class CExpRefTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.fragments = {air: load(_TESTDATA / f) for air, f in _FRAGMENTS.items()}
        self.constraints = {
            air: load(_TESTDATA / f)["constraints"] for air, f in _CONSTRAINTS.items()
        }
        self.golden = load(_TESTDATA / "golden" / "cexp_eval.json")

    def test_reference_matches_golden_q(self) -> None:
        for case in self.golden["cases"]:
            with self.subTest(air=case["air"], n_bits=case["n_bits"]):
                got = evaluate(self.fragments[case["air"]], case)
                self.assertTrue(bool(fnp.array_equal(got, u64x3(case["q"]))))

    def test_generic_constraint_fold_matches_golden_q(self) -> None:
        # The AIR-agnostic constraints[] fold reassembles the same q as pil2's
        # pre-folded composite — across both Binary and MemAlignReadByte.
        for case in self.golden["cases"]:
            with self.subTest(air=case["air"], n_bits=case["n_bits"]):
                got = evaluate_from_constraints(self.constraints[case["air"]], case)
                self.assertTrue(bool(fnp.array_equal(got, u64x3(case["q"]))))


def _op(kind: str, a: dict, b: dict, dest: dict) -> dict:
    return {"op": kind, "src": [a, b], "dest": dest}


def _t(i: int) -> dict:
    return {"type": "tmp", "id": i}


_CM = {"type": "cm", "id": 0}
_Q = {"type": "q", "id": 0}


class LiveBytesPerRowTest(absltest.TestCase):
    """The derived row window's input. Both properties it reads off an operand
    — extension degree and whether the value varies per row — are invisible to
    the golden cases (those pin values, not footprints) and easy to get
    backwards, so they are pinned directly.

    Every block below ends by consuming its temporaries into `q`, the way a
    real cExp does: a temporary nothing reads again is dead on the spot."""

    def _dims(self, *dims: int):
        return lambda s: dims[s["id"]]

    def test_a_base_column_times_a_base_column_is_one_element(self):
        code = [
            _op("mul", _CM, {"type": "const", "id": 0}, _t(0)),
            _op("add", _t(0), _CM, _Q),
        ]
        self.assertEqual(live_bytes_per_row(code, self._dims(1)), 8)

    def test_an_extension_column_is_three_elements(self):
        code = [
            _op("mul", _CM, {"type": "const", "id": 0}, _t(0)),
            _op("add", _t(0), _CM, _Q),
        ]
        self.assertEqual(live_bytes_per_row(code, self._dims(3)), 24)

    def test_a_number_widens_a_base_column_to_cubic(self):
        # `_operand` embeds a literal through `golden.embed`, so the product is
        # F3 even though the literal's value is base. Reading the literal as
        # base under-reports this temporary threefold.
        code = [
            _op("mul", _CM, {"type": "number", "value": "7"}, _t(0)),
            _op("add", _t(0), _CM, _Q),
        ]
        self.assertEqual(live_bytes_per_row(code, self._dims(1)), 24)

    def test_a_public_widens_a_base_column_to_cubic(self):
        code = [
            _op("mul", _CM, {"type": "public", "id": 0}, _t(0)),
            _op("add", _t(0), _CM, _Q),
        ]
        self.assertEqual(live_bytes_per_row(code, self._dims(1)), 24)

    def test_a_scalar_only_fold_costs_nothing_per_row(self):
        # A Horner chain over the quotient challenge is one 0-d scalar however
        # many rows are evaluated — charging it per row would pin the window at
        # its ceiling for a working set that does not exist. Only t2 is a
        # column here, and it is the whole per-row cost.
        chal = {"type": "challenge", "id": 0}
        code = [
            _op("mul", chal, chal, _t(0)),
            _op("add", _t(0), {"type": "airvalue", "id": 0}, _t(1)),
            _op("mul", _CM, _CM, _t(2)),
            _op("mul", _t(1), _t(2), _Q),
        ]
        self.assertEqual(live_bytes_per_row(code, self._dims(1)), 8)

    def test_a_scalar_reaching_a_column_becomes_per_row(self):
        chal = {"type": "challenge", "id": 0}
        code = [
            _op("mul", chal, chal, _t(0)),
            _op("mul", _t(0), _CM, _t(1)),
            _op("add", _t(1), _CM, _Q),
        ]
        self.assertEqual(live_bytes_per_row(code, self._dims(1)), 24)

    def test_peak_counts_only_simultaneously_live_temporaries(self):
        # t0 dies at op 1; t1 and t2 overlap, so the peak is those two.
        code = [
            _op("mul", _CM, _CM, _t(0)),
            _op("mul", _t(0), _CM, _t(1)),
            _op("mul", _CM, _CM, _t(2)),
            _op("add", _t(1), _t(2), _Q),
        ]
        self.assertEqual(live_bytes_per_row(code, self._dims(1)), 16)

    def test_an_unknown_operand_class_is_not_silently_sized(self):
        code = [_op("mul", {"type": "wat", "id": 0}, _CM, _t(0))]
        with self.assertRaisesRegex(ValueError, "unhandled cExp operand type"):
            live_bytes_per_row(code, self._dims(1))


if __name__ == "__main__":
    absltest.main()
