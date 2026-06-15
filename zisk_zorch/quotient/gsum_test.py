"""pil2 std_sum LogUp witness: bus denominator (alpha-Horner + gamma) and the
prefix-sum grand-sum, byte-matched against the `fields`-crate reference."""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
from absl.testing import absltest

from zisk_zorch.golden import load, u64x3
from zisk_zorch.quotient.gsum import bus_denominator, grand_sum

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"


class GsumTest(absltest.TestCase):
    def setUp(self) -> None:
        super().setUp()
        self.golden = load(_TESTDATA / "gsum.json")

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


if __name__ == "__main__":
    absltest.main()
