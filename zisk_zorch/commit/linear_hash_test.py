"""Byte-match of the chained linear hash against pil2-proofman's
`linear_hash_seq`, across every regime the goldens probe (short single-block
rows, one block, partial block, multi-block chaining) and every tree width."""

from __future__ import annotations

import pathlib

import frx
import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.linear_hash import LinearHash
from zisk_zorch.golden import load, u64
from zisk_zorch.poseidon1.goldilocks import goldilocks_perm as poseidon1_perm
from zisk_zorch.poseidon2.goldilocks import goldilocks_perm as poseidon2_perm

_GOLDEN = pathlib.Path(__file__).parent / "testdata" / "golden" / "linear_hash.json"


class LinearHashTest(absltest.TestCase):
    def test_matches_pil2_reference(self) -> None:
        for entry in load(_GOLDEN)["widths"]:
            hasher = LinearHash(poseidon2_perm(entry["width"]))
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

    def test_batched_rows_hash_like_single_rows(self) -> None:
        # A vmapped leaf hash must equal the row-by-row hash at every batch
        # size. On the CPU plugin the dedicated `sponge_hash` emitter got this
        # wrong when the batch equals the permutation width (16 rows at width
        # 16; the first row right, the rest not — fractalyze/xla#653), which
        # only a FRI layer of exactly 16 leaves reached before this test.
        # `goldilocks_perm` keeps the CPU on the generic marker for that
        # reason; this pins the bytes whichever route the backend takes.
        #
        # Both absorb regimes: one partial block (a FRI layer's digest rows)
        # and two with a tail (a trace width like Main's 38 columns), because
        # the multi-block chain carries state across the batched permutes and
        # is where a batching bug has somewhere to hide.
        rng = np.random.default_rng(16)
        for width in (16, 8):
            hasher = LinearHash(poseidon2_perm(width))
            for cols in (hasher.rate - 1, hasher.rate + 1):
                for rows in (width - 1, width, width + 1):
                    matrix = fnp.array(
                        rng.integers(0, 2**63, size=(rows, cols), dtype=np.uint64),
                        dtype=F,
                    )
                    batched = frx.jit(frx.vmap(hasher.hash))(matrix)
                    for i in range(rows):
                        self.assertTrue(
                            bool(fnp.array_equal(batched[i], hasher.hash(matrix[i]))),
                            msg=f"width {width}, {cols} cols, {rows} rows, row {i}",
                        )

    def test_value_equality(self) -> None:
        # Fresh instances over the same permutation are one static jit-zone
        # key; a different permutation is a distinct key. Both families run at
        # the same width now that only arity 4 is modelled, so the hash family
        # is what has to separate them.
        a, b = LinearHash(poseidon2_perm(16)), LinearHash(poseidon2_perm(16))
        self.assertEqual(a, b)
        self.assertEqual(hash(a), hash(b))
        self.assertNotEqual(a, LinearHash(poseidon1_perm(16)))
        self.assertNotEqual(a, object())


if __name__ == "__main__":
    absltest.main()
