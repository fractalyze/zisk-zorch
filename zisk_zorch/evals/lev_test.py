"""Byte-match of computeLEv against pil2's geometric-series INTT.

The golden builds each opening point's geometric series and runs the reference
`intt_tiny`; the prover reproduces the coefficients via the IDFT closed form.
Cases cover a single opening point, the current+next pair, and a negative offset.
"""

from __future__ import annotations

import pathlib

import frx.numpy as jnp
from absl.testing import absltest

from zisk_zorch.evals.lev import compute_lev
from zisk_zorch.golden import load, u64x3

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"


class ComputeLevTest(absltest.TestCase):
    def test_matches_pil2_compute_lev(self) -> None:
        for case in load(_TESTDATA / "compute_lev.json")["cases"]:
            with self.subTest(n_bits=case["n_bits"], opening=case["opening_points"]):
                lev = compute_lev(
                    u64x3(case["xi"]).reshape(()),
                    case["opening_points"],
                    case["n_bits"],
                )
                want = u64x3(case["lev"])  # row-major (k*nOpen + i)
                self.assertTrue(bool(jnp.array_equal(lev.reshape(-1), want)))


if __name__ == "__main__":
    absltest.main()
