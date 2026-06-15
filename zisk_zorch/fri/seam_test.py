"""Pil2FriCode.check_final — pil2-stark's terminal FRI low-degree test.

Byte-matched to pil2's `stark_verify` low-degree check (the `fri_final` golden is
built from the reference `intt_tiny`): a genuine low-degree final polynomial is
accepted, and a final pol carrying one coefficient at the degree bound is
rejected. This is the soundness check the fold chain alone cannot make — a chain
is internally consistent for any codeword, low-degree or not.
"""

from __future__ import annotations

import pathlib

from absl.testing import absltest

from zisk_zorch.fri.seam import Pil2FriCode
from zisk_zorch.golden import load, u64x3

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"


class CheckFinalTest(absltest.TestCase):
    def test_accepts_low_degree_rejects_high_degree(self) -> None:
        for case in load(_TESTDATA / "fri_final.json")["cases"]:
            with self.subTest(steps=case["steps"], n_bits=case["n_bits"]):
                code = Pil2FriCode(tuple(case["steps"]))
                n_bits = case["n_bits"]
                self.assertTrue(
                    bool(code.check_final(u64x3(case["final_low"]), n_bits)),
                    msg="rejected a genuine low-degree final polynomial",
                )
                self.assertFalse(
                    bool(code.check_final(u64x3(case["final_high"]), n_bits)),
                    msg="accepted a final polynomial above the degree bound",
                )


if __name__ == "__main__":
    absltest.main()
