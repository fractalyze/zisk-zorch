"""Byte-match of FRI query sampling + grinding against the pil2 `fields` crate.

Three checks:
- `query_sample.json` pins the position derivation (finalPol absorb -> challenge
  -> reseed with challenge++nonce -> getPermutations). The discarded
  `pre_challenge` squeeze is cross-checked so a wrong pre-derivation transcript
  state can't pass on the positions alone, and the squeezed `challenge` is
  cross-checked against the golden.
- `grinding.json` pins the standalone PoW search: the smallest nonce whose
  width-4 Poseidon2 image has `pow_bits` leading zeros, plus the image value.
- the self-derivation round-trip: `sample_query_positions` finds that same
  smallest nonce and the matching positions.
"""

from __future__ import annotations

import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest

from zisk_zorch.fri.queries import (
    _grind,
    _grind_images,
    grind_is_valid,
    grinding_seed_challenge,
    query_positions_for,
    sample_query_positions,
)
from zisk_zorch.golden import load, u64, u64x3
from zisk_zorch.transcript.transcript import Transcript

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"
_QUERY_SAMPLE = _TESTDATA / "query_sample.json"
_GRINDING = _TESTDATA / "grinding.json"


class QuerySampleTest(absltest.TestCase):
    def test_derivation_matches_pil2_reference(self) -> None:
        for case in load(_QUERY_SAMPLE)["cases"]:
            width = case["width"]
            t = Transcript(width)
            t.put(u64(case["seed_absorb"]))

            pre = t.get_field()  # fold-loop tail challenge, discarded by the prover
            self.assertTrue(
                bool(fnp.array_equal(pre, u64(case["pre_challenge"]))),
                msg=f"width {width} pre_challenge",
            )

            challenge = grinding_seed_challenge(t, u64x3(case["final_pol"]))
            self.assertTrue(
                bool(fnp.array_equal(challenge, u64(case["challenge"]))),
                msg=f"width {width} challenge",
            )

            positions = query_positions_for(
                challenge,
                width,
                int(case["nonce"]),
                n_queries=case["n_queries"],
                n_bits_ext=case["n_bits_ext"],
            )
            expected = np.array([int(v) for v in case["positions"]], dtype=np.uint64)
            self.assertTrue(
                bool(np.array_equal(positions, expected)),
                msg=f"width {width} positions",
            )

    def test_grinding_matches_pil2_reference(self) -> None:
        for case in load(_GRINDING)["cases"]:
            challenge = u64(case["challenge"])
            pow_bits = case["pow_bits"]
            expected_nonce = int(case["nonce"])

            nonce = _grind(challenge, pow_bits)
            self.assertEqual(nonce, expected_nonce, msg=f"pow_bits {pow_bits} nonce")
            self.assertTrue(grind_is_valid(challenge, nonce, pow_bits))

            image = int(_grind_images(challenge, np.array([nonce], dtype=np.uint64))[0])
            self.assertEqual(image, int(case["image"]), msg=f"pow_bits {pow_bits} image")
            self.assertLess(image, 1 << (64 - pow_bits))

    def test_self_derives_grinding_nonce(self) -> None:
        # The public API self-derives the same smallest nonce and positions that
        # the grinding-seed challenge + standalone search + reseed produce.
        seed = u64(["7", "11", "13", "17", "19"])
        final_pol = u64x3(["2", "3", "5", "8", "13", "21"])  # 2 cubic elements
        pow_bits, n_queries, n_bits_ext = 8, 4, 5

        t1 = Transcript(12)
        t1.put(seed)
        challenge = grinding_seed_challenge(t1, final_pol)
        want_nonce = _grind(challenge, pow_bits)
        want_pos = query_positions_for(
            challenge, 12, want_nonce, n_queries=n_queries, n_bits_ext=n_bits_ext
        )

        t2 = Transcript(12)
        t2.put(seed)
        positions, nonce = sample_query_positions(
            t2, final_pol, pow_bits=pow_bits, n_queries=n_queries, n_bits_ext=n_bits_ext
        )
        self.assertEqual(nonce, want_nonce)
        self.assertTrue(bool(np.array_equal(positions, want_pos)))
        self.assertTrue(grind_is_valid(challenge, nonce, pow_bits))


if __name__ == "__main__":
    absltest.main()
