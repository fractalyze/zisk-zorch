"""pil2 std_sum LogUp witness: bus denominator (alpha-Horner + gamma), the
prefix-sum grand-sum, and the `VirtualPairCol` evaluator — byte-matched against
the `fields`-crate reference and (for the operation bus) pil2's cExp."""

from __future__ import annotations

import frx

# rw chip code views uint64 as the field dtype; x64 must be on before any array op.
frx.config.update("jax_enable_x64", True)

import pathlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import frx.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402
from zk_dtypes import goldilocks as F  # noqa: E402

from zisk_zorch.constraints.chip_loader import load_zisk_chips  # noqa: E402
from zisk_zorch.golden import load, u64x3  # noqa: E402
from zisk_zorch.quotient import cexp_ref  # noqa: E402
from zisk_zorch.quotient.field_io import base_trace, embed, embed_base  # noqa: E402
from zisk_zorch.quotient.gsum import bus_denominator, eval_pair_col, grand_sum, gsum_e  # noqa: E402

_TESTDATA = pathlib.Path(__file__).parent / "testdata"
_GOLDEN = _TESTDATA / "golden"


class GsumTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.golden = load(_GOLDEN / "gsum.json")

    def test_bus_denominator_matches_pil2(self) -> None:
        for case in self.golden["denominator"]:
            with self.subTest(tuple_width=case["tuple_width"]):
                tup = u64x3(case["tuple"])  # (T,) cubic
                alpha = u64x3(case["alpha"]).reshape(())
                gamma = u64x3(case["gamma"]).reshape(())
                got = bus_denominator(tup, alpha, gamma)
                want = u64x3(case["den"]).reshape(())
                self.assertTrue(bool(jnp.array_equal(got, want)))

    def test_grand_sum_matches_pil2(self) -> None:
        for case in self.golden["grand_sum"]:
            with self.subTest(n=case["n"], n_interactions=case["n_interactions"]):
                n, i = case["n"], case["n_interactions"]
                num = u64x3(case["numerators"]).reshape(n, i)
                den = u64x3(case["denominators"]).reshape(n, i)
                got = grand_sum(num, den)
                want = u64x3(case["gsum"])  # (N,) cubic
                self.assertTrue(bool(jnp.array_equal(got, want)))

    def test_eval_pair_col_evaluates_column_products(self) -> None:
        # const + Σ wᵢ·colᵢ + Σ wₖ·colₐ·col_b, the bilinear part a non-affine bus
        # tuple (e.g. arith's operation bus) needs. Weights are rw's decimal strings.
        trace = jnp.array(np.array([[2, 3], [4, 5]], dtype=np.uint64), dtype=F)
        vpc = SimpleNamespace(
            constant="7",
            column_weights=[(0, False, "3")],  # 3·col0
            column_products=[(0, False, 1, False, "5")],  # 5·col0·col1
        )
        got = eval_pair_col(vpc, trace)
        # row0: 7 + 3·2 + 5·2·3 = 43 ; row1: 7 + 3·4 + 5·4·5 = 119
        want = embed_base(jnp.array(np.array([43, 119], dtype=np.uint64), dtype=F))
        self.assertTrue(bool(jnp.array_equal(got, want)))

    def test_arith_operation_bus_reconstructs_pil2_gsum_e(self) -> None:
        # rw's arith `proves_operation` interaction is non-affine — its operation-bus
        # tuple (kind 5000) carries `div·chunk` column_products. Reconstructing its
        # gsum_e from the VirtualPairCol tuple and plugging it into pil2's std_sum
        # single constraint must reproduce that constraint's value. This is the
        # byte-match that verifies rw's authored operation-bus interaction (the
        # per-chip CPU test can't — interactions are CPU-erased there).
        #
        # Arith's cExp uses gsum_e[0] = the operation bus in constraint 61:
        #   im_single(cm57)·(gsum_e[0] + std_gamma) − multiplicity(cm41).
        op = load_zisk_chips("v1", ["arith"])["arith"].get_receives()[0].interaction
        self.assertEqual(op.kind, 5000)  # OPERATION_BUS
        self.assertTrue(any(v.column_products for v in op.values), "expected a non-affine tuple")
        constraints = load(_TESTDATA / "arith_constraints.json")["constraints"]
        cases = [c for c in load(_GOLDEN / "cexp_eval.json")["cases"] if c["air"] == "Arith"]
        self.assertTrue(cases, "no Arith cases in the cexp_eval golden")
        for case in cases:
            with self.subTest(n_bits=case["n_bits"]):
                env = cexp_ref._load_inputs(case)
                alpha, gamma = env["challenges"][0], env["challenges"][1]
                cm = {c["id"]: (embed if c["dim"] == 1 else u64x3)(c["values"]) for c in case["cm"]}
                authored = cm[57] * (gsum_e(op, base_trace(case, 44), alpha) + gamma) - cm[41]
                target = cexp_ref._run_block(constraints[61]["code"], env, 1 << case["blowup_bits"])
                self.assertTrue(bool(jnp.array_equal(authored, target)))


if __name__ == "__main__":
    absltest.main()
