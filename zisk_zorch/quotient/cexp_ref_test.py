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
from zisk_zorch.quotient.cexp_ref import evaluate, evaluate_from_constraints  # noqa: E402

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


if __name__ == "__main__":
    absltest.main()
