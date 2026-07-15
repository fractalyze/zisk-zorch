"""Byte-match of the FRI fold against pil2-proofman's `verify_fold`.

The golden folds a random cubic codeword one step per case, covering a first
step (no coset-shift squaring), a later step (shift squared), the smallest fold
(`n_x = 2`), a wide single fold (`n_x = 32`), and a two-step chain.
"""

from __future__ import annotations

import pathlib

import frx.numpy as jnp
from absl.testing import absltest

from zisk_zorch.fri.fold import fold, verify_fold
from zisk_zorch.golden import load, u64x3

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"


class FriFoldTest(absltest.TestCase):
    def test_matches_pil2_verify_fold(self) -> None:
        for case in load(_TESTDATA / "fri_fold.json")["cases"]:
            pol = u64x3(case["pol"])
            challenge = u64x3(case["challenge"]).reshape(())
            folded = fold(
                pol,
                challenge,
                n_bits_ext=case["n_bits_ext"],
                prev_bits=case["prev_bits"],
                current_bits=case["current_bits"],
            )
            self.assertTrue(
                bool(jnp.all(folded == u64x3(case["folded"]))),
                msg=(
                    f"nBitsExt {case['n_bits_ext']}, prevBits {case['prev_bits']}, "
                    f"currentBits {case['current_bits']}"
                ),
            )

    def test_verify_fold_matches_per_group(self) -> None:
        # The verifier's single-group fold must reproduce the full fold's output
        # group by group: verify_fold(pol's group g, ..., g) == fold(pol, ...)[g].
        for case in load(_TESTDATA / "fri_fold.json")["cases"]:
            prev_bits, current_bits = case["prev_bits"], case["current_bits"]
            cur_n, n_x = 1 << current_bits, 1 << (prev_bits - current_bits)
            challenge = u64x3(case["challenge"]).reshape(())
            groups = u64x3(case["pol"]).reshape(n_x, cur_n).T  # (cur_n, n_x)
            folded = u64x3(case["folded"])
            for g in range(cur_n):
                value = verify_fold(
                    groups[g], challenge, case["n_bits_ext"], prev_bits, current_bits, g
                )
                self.assertTrue(bool(jnp.all(value == folded[g])), msg=f"group {g}")


if __name__ == "__main__":
    absltest.main()
