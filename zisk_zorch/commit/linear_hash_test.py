"""Byte-match of the chained linear hash against pil2-proofman's
`linear_hash_seq`, across every regime the goldens probe (short single-block
rows, one block, partial block, multi-block chaining) and every tree width."""

from __future__ import annotations

import pathlib

import frx.numpy as fnp
from absl.testing import absltest

from zisk_zorch.commit.linear_hash import LinearHash
from zisk_zorch.golden import load, u64
from zisk_zorch.poseidon2.goldilocks import goldilocks_perm

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "linear_hash.json"


class LinearHashTest(absltest.TestCase):
    def test_matches_pil2_reference(self) -> None:
        for entry in load(_GOLDEN)["widths"]:
            hasher = LinearHash(goldilocks_perm(entry["width"]))
            self.assertEqual(hasher.rate, entry["rate"])
            for case in entry["cases"]:
                out = hasher.hash(u64(case["input"]))
                # The reference returns the full state; the digest (and what
                # the tree consumes) is its first 4 lanes.
                expected = u64(case["output"])[:4]
                self.assertTrue(
                    bool(fnp.array_equal(out, expected)),
                    msg=f"width {entry['width']}, len {len(case['input'])}",
                )

    def test_value_equality(self) -> None:
        # Fresh instances over the same permutation are one static jit-zone
        # key; different widths are distinct keys.
        a, b = LinearHash(goldilocks_perm(12)), LinearHash(goldilocks_perm(12))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, LinearHash(goldilocks_perm(16)))
        self.assertNotEqual(a, object())


if __name__ == "__main__":
    absltest.main()
