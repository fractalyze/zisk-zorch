"""pil2 std_sum LogUp bus, byte-matched against the `fields`-crate reference.

The per-interaction denominators (`LogUpBus.denominators`) are pinned instead
against pil2's cExp, next to the constraint fold that consumes them:
`quotient/reauthor_test.py`.
"""

from __future__ import annotations

import frx

# rw chip code views uint64 as the field dtype; x64 must be on before any array op.
frx.config.update("jax_enable_x64", True)

import pathlib  # noqa: E402
from types import SimpleNamespace  # noqa: E402

import frx.numpy as fnp  # noqa: E402
import numpy as np  # noqa: E402
from absl.testing import absltest  # noqa: E402
from zk_dtypes import goldilocks as F  # noqa: E402
from zk_dtypes import goldilocksx3 as F3  # noqa: E402

from zisk_zorch.golden import load, u64x3  # noqa: E402
from zisk_zorch.logup.bus import LogUpBus  # noqa: E402

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden"


class BusTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.golden = load(_GOLDEN / "gsum.json")

    def test_denominator_matches_pil2(self) -> None:
        for case in self.golden["denominator"]:
            with self.subTest(tuple_width=case["tuple_width"]):
                bus = LogUpBus(
                    alpha=u64x3(case["alpha"]).reshape(()),
                    gamma=u64x3(case["gamma"]).reshape(()),
                )
                got = bus.denominator(u64x3(case["tuple"]))  # (T,) cubic
                want = u64x3(case["den"]).reshape(())
                self.assertTrue(bool(fnp.array_equal(got, want)))

    def test_grand_sum_matches_pil2(self) -> None:
        for case in self.golden["grand_sum"]:
            with self.subTest(n=case["n"], n_interactions=case["n_interactions"]):
                n, i = case["n"], case["n_interactions"]
                num = u64x3(case["numerators"]).reshape(n, i).T
                den = u64x3(case["denominators"]).reshape(n, i).T
                got = LogUpBus.grand_sum(num, den)
                want = u64x3(case["gsum"])  # (N,) cubic
                self.assertTrue(bool(fnp.array_equal(got, want)))
                # The airgroup export is that column's last row.
                self.assertTrue(
                    bool(fnp.array_equal(LogUpBus.gsum_result(got), want[-1]))
                )

    def test_eval_pair_col_evaluates_column_products(self) -> None:
        # const + Σ wᵢ·colᵢ + Σ wₖ·colₐ·col_b, the bilinear part a non-affine bus
        # tuple (e.g. arith's operation bus) needs. Weights are rw's decimal strings.
        trace = fnp.array(np.array([[2, 3], [4, 5]], dtype=np.uint64), dtype=F)
        vpc = SimpleNamespace(
            constant="7",
            column_weights=[(0, False, "3")],  # 3·col0
            column_products=[(0, False, 1, False, "5")],  # 5·col0·col1
        )
        got = LogUpBus.eval_pair_col(vpc, trace)
        # row0: 7 + 3·2 + 5·2·3 = 43 ; row1: 7 + 3·4 + 5·4·5 = 119
        want = fnp.array(np.array([43, 119], dtype=np.uint64), dtype=F).astype(F3)
        self.assertTrue(bool(fnp.array_equal(got, want)))

    def test_eval_pair_col_reads_preprocessed_terms_from_the_prep_trace(self) -> None:
        # Both traces carry a distinguishable value at index 0, so a fallback to
        # `trace` would show up in the result.
        trace = fnp.array(np.array([[2], [4]], dtype=np.uint64), dtype=F)
        prep = fnp.array(np.array([[10], [20]], dtype=np.uint64), dtype=F)
        vpc = SimpleNamespace(
            constant="1",
            column_weights=[(0, False, "3"), (0, True, "5")],  # 3·main0 + 5·prep0
            column_products=[(0, True, 0, False, "2")],  # 2·prep0·main0
        )
        got = LogUpBus.eval_pair_col(vpc, trace, prep)
        # row0: 1 + 3·2 + 5·10 + 2·10·2 = 97 ; row1: 1 + 3·4 + 5·20 + 2·20·4 = 273
        want = fnp.array(np.array([97, 273], dtype=np.uint64), dtype=F).astype(F3)
        self.assertTrue(bool(fnp.array_equal(got, want)))

    def test_eval_pair_col_rejects_preprocessed_terms_without_a_prep_trace(
        self,
    ) -> None:
        # main[col] would be a real value of the right dtype and shape, so the
        # byte-match would fail far from the cause (fractalyze/zisk-zorch#115).
        trace = fnp.array(np.array([[2], [4]], dtype=np.uint64), dtype=F)
        weight_only = SimpleNamespace(
            constant="0", column_weights=[(0, True, "1")], column_products=[]
        )
        with self.assertRaisesRegex(ValueError, "preprocessed"):
            LogUpBus.eval_pair_col(weight_only, trace)

        product_only = SimpleNamespace(
            constant="0",
            column_weights=[],
            column_products=[(0, False, 0, True, "1")],
        )
        with self.assertRaisesRegex(ValueError, "preprocessed"):
            LogUpBus.eval_pair_col(product_only, trace)


if __name__ == "__main__":
    absltest.main()
