"""Wire-level pinning for the zkin builders: u64 exactness and shape."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from zisk_zorch.harness.pil2 import MODULUS
from zisk_zorch.harness.zkin import (
    BasicArtifacts,
    compressed_leaf_zkin,
    leaf_zkin,
    node_zkin,
    subtree_head,
    wire_prefix,
)

# Above 2^53, so a float64 round-trip loses the low bits.
_BIG = np.uint64(12345678901234567890)


class ZkinTest(absltest.TestCase):
    def test_prefix_keeps_full_words_beside_int64_parts(self) -> None:
        # `np.array([...])` from a Python list is int64, and numpy promotes
        # a mixed int64/uint64 concatenate to float64 — the parts must be
        # cast BEFORE they are joined, not after.
        verkey = np.full(4, _BIG, dtype=np.uint64)
        out = wire_prefix(
            np.array([1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8]), verkey
        )
        self.assertEqual(out.dtype, np.uint64)
        self.assertTrue(np.array_equal(out[-4:], verkey))
        self.assertEqual(out.size, 2 + 3 + 3 + 4)

    def test_prefix_without_verkey(self) -> None:
        out = wire_prefix(
            np.array([1, 2]), np.array([3, 4, 5]), np.array([6, 7, 8]), None
        )
        self.assertEqual(out.size, 2 + 3 + 3)

    def test_leaf_and_compressed_leaf_layout(self) -> None:
        prefix = np.array([1, 2])
        proof = np.array([_BIG], dtype=np.uint64)
        leaf = leaf_zkin(prefix, proof)
        self.assertEqual(leaf.dtype, np.uint64)
        self.assertTrue(np.array_equal(leaf, [1, 2, _BIG]))
        # The compressor's zkin is the only one whose head precedes it.
        comp = compressed_leaf_zkin(np.array([9]), prefix, proof)
        self.assertTrue(np.array_equal(comp, [9, 1, 2, _BIG]))

    def test_node_zero_fills_absent_segment(self) -> None:
        head = np.array([1, 1])
        proof = np.array([_BIG], dtype=np.uint64)
        out = node_zkin(np.array([7]), [(head, proof), None])
        self.assertEqual(out.dtype, np.uint64)
        self.assertTrue(np.array_equal(out, [7, 1, 1, _BIG, 0, 0, 0]))

    def test_node_rejects_disagreeing_widths(self) -> None:
        # An assert would vanish under `python -O` and zero-fill the absent
        # segment to whichever width popped out of the set.
        one = (np.array([1, 1]), np.array([2], dtype=np.uint64))
        two = (np.array([1, 1]), np.array([2, 3], dtype=np.uint64))
        with self.assertRaises(ValueError):
            node_zkin(np.array([7]), [one, two, None])

    def test_node_rejects_all_absent_segments(self) -> None:
        with self.assertRaises(ValueError):
            node_zkin(np.array([7]), [None, None])

    def test_subtree_head_reduces_both_sums(self) -> None:
        # Both the airgroup value and the lattice wrap, so the head pins the
        # mod-p addition as well as the layout.
        def basic(agv0: int, c0: int) -> BasicArtifacts:
            return BasicArtifacts(
                np.array([agv0, 0, 0], dtype=np.uint64),
                np.array([c0, 1], dtype=np.uint64),
            )

        head = subtree_head([basic(MODULUS - 1, 1), basic(2, MODULUS - 1)], (7, 1))
        self.assertEqual(head.dtype, np.uint64)
        self.assertTrue(np.array_equal(head, [7, 1, 0, 1, 0, 0, 0, 2]))

    def test_subtree_head_rejects_empty_subtree(self) -> None:
        with self.assertRaises(ValueError):
            subtree_head([], (1, 0))


if __name__ == "__main__":
    absltest.main()
