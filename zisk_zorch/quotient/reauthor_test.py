"""Byte-match the re-authored quotient against the cExp reference `q`.

`reauthor.reauthor_binary_quotient` assembles `q` from the Binary AIR's row-local
constraints + the ingested `std_sum` interactions (rw's typed `Interaction`s),
folded in proving-key order. Equality with the `cexp_eval` golden `q` (generated
by interpreting pil2's composite-constraint SSA) is the end-to-end byte-match that
verifies rw's authored interactions for Binary.

`ArithOperationBusTest` narrows that to the one non-affine tuple, which the
whole-quotient match pins only implicitly. It is the gate on `LogUpBus`'
per-interaction denominators, and lives here because the cExp reference and its
goldens do.
"""

from __future__ import annotations

import frx

frx.config.update("jax_enable_x64", True)

import pathlib  # noqa: E402

import frx.numpy as fnp  # noqa: E402
from absl.testing import absltest  # noqa: E402

from zisk_zorch.constraints.chip_loader import load_zisk_chips  # noqa: E402
from zisk_zorch.golden import base_trace, embed, load, u64x3  # noqa: E402
from zisk_zorch.logup.bus import STD_ALPHA, STD_GAMMA, LogUpBus  # noqa: E402
from zisk_zorch.quotient import cexp_ref  # noqa: E402
from zisk_zorch.quotient.reauthor import (  # noqa: E402
    reauthor_arith_quotient,
    reauthor_binary_quotient,
)

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "cexp_eval.json"
_ARITH_CONSTRAINTS = (
    pathlib.Path(__file__).parent / "testdata" / "arith_constraints.json"
)


class ReauthorBinaryTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.chip = load_zisk_chips("v1", ["binary"])["binary"]
        self.cases = [c for c in load(_GOLDEN)["cases"] if c["air"] == "Binary"]

    def test_reauthored_q_matches_cexp_reference(self) -> None:
        self.assertTrue(self.cases, "no Binary cases in the cexp_eval golden")
        for case in self.cases:
            with self.subTest(n_bits=case["n_bits"], blowup_bits=case["blowup_bits"]):
                got = reauthor_binary_quotient(self.chip, case)
                self.assertTrue(bool(fnp.array_equal(got, u64x3(case["q"]))))


class ReauthorArithTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.chip = load_zisk_chips("v1", ["arith"])["arith"]
        self.cases = [c for c in load(_GOLDEN)["cases"] if c["air"] == "Arith"]
        self.constraints = load(_ARITH_CONSTRAINTS)["constraints"]

    def test_reauthored_q_matches_cexp_reference(self) -> None:
        self.assertTrue(self.cases, "no Arith cases in the cexp_eval golden")
        for case in self.cases:
            with self.subTest(n_bits=case["n_bits"], blowup_bits=case["blowup_bits"]):
                got = reauthor_arith_quotient(self.chip, case, self.constraints)
                self.assertTrue(bool(fnp.array_equal(got, u64x3(case["q"]))))


class ArithOperationBusTest(absltest.TestCase):
    def test_operation_bus_denominator_reconstructs_pil2_gsum_e(self) -> None:
        # rw's arith `proves_operation` interaction is non-affine (kind 5000, with
        # `div·chunk` column_products). Reproducing pil2's cExp constraint 61 —
        # `im_single(cm57)·(gsum_e[0] + std_gamma) − multiplicity(cm41)` — from its
        # VirtualPairCol tuple is what verifies rw authored that interaction; the
        # per-chip CPU test can't, since interactions are CPU-erased there.
        chip = load_zisk_chips("v1", ["arith"])["arith"]
        op = chip.get_receives()[0].interaction
        self.assertEqual(op.kind, 5000)  # OPERATION_BUS
        self.assertTrue(
            any(v.column_products for v in op.values), "expected a non-affine tuple"
        )
        constraints = load(_ARITH_CONSTRAINTS)["constraints"]
        cases = [c for c in load(_GOLDEN)["cases"] if c["air"] == "Arith"]
        self.assertTrue(cases, "no Arith cases in the cexp_eval golden")
        for case in cases:
            with self.subTest(n_bits=case["n_bits"]):
                env = cexp_ref._load_inputs(case)
                cm = {
                    c["id"]: (embed if c["dim"] == 1 else u64x3)(c["values"])
                    for c in case["cm"]
                }
                bus = LogUpBus(
                    alpha=env["challenges"][STD_ALPHA],
                    gamma=env["challenges"][STD_GAMMA],
                    interactions=[op],
                )
                den = bus.denominators(base_trace(case, 44))[0]
                authored = cm[57] * den - cm[41]
                target = cexp_ref._run_block(
                    constraints[61]["code"], env, 1 << case["blowup_bits"]
                )
                self.assertTrue(bool(fnp.array_equal(authored, target)))


if __name__ == "__main__":
    absltest.main()
