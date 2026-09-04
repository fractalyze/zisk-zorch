"""The commit kernels the pinned toolchain actually emits, per hash family.

`fusion_path` (the poseidon1/poseidon2 routing tests) reads the marker hash-frx
CHOSE; whether the pinned frx plugin RECOGNIZES it shows only in the compiled
module. An unrecognized composite inlines to the same bytes (see
`zorch.testkit.fusion`), so every golden passes either way while the Merkle
leaf level runs as one loop fusion per round over the whole leaf set — #168:
the frx 0820 plugin had retired the marker spellings hash-frx 0817 emitted,
and Main's commit1 grew to 746 kernels at 4x the baseline's time without a
single test noticing. This test is the one that notices.

GPU-only (`tags = ["gpu"]`): Poseidon1 routes to the generic marker on the CPU
backend by design (hash-frx#147), so there is nothing to recognize there.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
from absl.testing import absltest, parameterized
from zk_dtypes import goldilocks as F
from zorch.testkit.fusion import custom_fusion_names

from zisk_zorch.commit.trace_commit import merkle_tree

# 16 leaves under arity 4: two compress levels (16 -> 4 -> 1). Small on purpose —
# the kernels compile identically at any leaf count, and the sparse-Poseidon
# emitter's compile is what this test pays for.
_LEAVES = 16
_LEVELS = 2
# 38 columns is Main's cm1 width: rate 12 blocks, with a partial tail.
_COLS = 38


class CommitFusionTest(parameterized.TestCase):
    @parameterized.named_parameters(
        ("poseidon1", "Poseidon1", "sparse_poseidon"),
        ("poseidon2", "Poseidon2", "poseidon2"),
    )
    def test_commit_lowers_to_one_kernel_per_level(
        self, hash_family: str, node_key: str
    ) -> None:
        tree = merkle_tree(4, hash_family)
        names = custom_fusion_names(tree.commit, fnp.zeros((_LEAVES, _COLS), F))
        # One `sponge_hash` for the whole leaf layer, one permute per level above
        # it, and nothing else custom: a level that fell back to the inlined
        # permute leaves a hole here, not a wrong root.
        self.assertEqual(names.count("sponge_hash"), 1, msg=names)
        self.assertEqual(names.count(node_key), _LEVELS, msg=names)
        self.assertLen(names, 1 + _LEVELS, msg=names)

    def test_gpu_backend(self) -> None:
        # The target is GPU-tagged; a CPU run here means the tag filter is off.
        self.assertEqual(frx.default_backend(), "gpu")


if __name__ == "__main__":
    absltest.main()
