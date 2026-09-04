"""Byte-match of the stage-1 commit pipeline under the Poseidon1 hash family.

Native ZisK commits with Poseidon1 by default (the installed proving key sets no
`hash`, so pil2's `DEFAULT_HASH_ID = "Poseidon1"` wins). These goldens pin the
Poseidon1 leaf hash and the full extend -> leaf-hash -> merkelize chain, both
generated from pil2-proofman's own `Poseidon1_*` widths (`tools/fixture-gen/`).
The LDE is family-independent, so only the tree's permutation differs from the
Poseidon2 path.

The leaf-hash goldens run on the permutation's generic marker, not the fused
sponge kernel. The fused kernel compiles for ~75 s per input length on the GPU
(fractalyze/xla#652: the linear-hash absorb emits the whole Poseidon1 permute body
twice, and every field multiply lowers with branches, so LLVM spends the time
on an 80k-line module), and ten lengths made this test the slowest in the
suite by an order of magnitude — for a golden whose bytes are the same on
either route. What the ten lengths pin is pil2's chaining convention (zero-pad
the partial block, chain the digest through the capacity lanes, permute every
block), and that convention lives in the sponge's Python body, which the
generic route runs as-is. The fused kernel's own bytes are pinned where its
shapes are production's: the stage-1 commits below (5 and 9 columns, partial
tails), `fullprogram_commit_test` (real trace widths), and one whole-block
length here (`_DEDICATED_LENGTHS`), the one tail case no other golden reaches.
"""

from __future__ import annotations

import pathlib

import frx.numpy as fnp
from absl.testing import absltest
from hash_frx.fusion import FUSED_REGION_MARKER, FusionPath
from hash_frx.poseidon.sparse import SparsePoseidon

from zisk_zorch.commit.linear_hash import LinearHash
from zisk_zorch.commit.trace_commit import commit_trace
from zisk_zorch.golden import load, u64
from zisk_zorch.poseidon1.goldilocks import goldilocks_params, goldilocks_perm

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"

# Lengths hashed through the fused kernel as well: a whole number of rate
# blocks with no tail, the branch of the emitter's absorb that every other
# fused golden (5, 9, 38, ... columns) leaves untaken.
_DEDICATED_LENGTHS = frozenset({24})


class _GenericRoute(SparsePoseidon):
    """The Poseidon1 permutation on the generic region marker: the same
    bytes as `goldilocks_perm`, without the dedicated emitter's compile."""

    def _select_fused_region_name(self, rows: object) -> str:
        return FUSED_REGION_MARKER


class Poseidon1LinearHashTest(absltest.TestCase):
    def test_matches_pil2_reference(self) -> None:
        for entry in load(_TESTDATA / "poseidon1_linear_hash.json")["widths"]:
            generic = _GenericRoute(goldilocks_params(entry["width"]))
            # The override is a private hook of hash-frx's SparsePoseidon; if
            # it stops applying, this fails here rather than silently paying
            # the fused compile per length again.
            self.assertIs(generic.fusion_path, FusionPath.GENERIC)
            hashers = {"generic": LinearHash(generic)}
            self.assertEqual(hashers["generic"].rate, entry["rate"])
            for case in entry["cases"]:
                row = u64(case["input"])
                expected = u64(case["output"])[:4]
                if len(row) in _DEDICATED_LENGTHS:
                    hashers["dedicated"] = LinearHash(goldilocks_perm(entry["width"]))
                for route, hasher in hashers.items():
                    self.assertTrue(
                        bool(fnp.array_equal(hasher.hash(row), expected)),
                        msg=f"width {entry['width']}, len {len(row)}, {route}",
                    )
                hashers.pop("dedicated", None)


class Poseidon1Stage1CommitTest(absltest.TestCase):
    def test_matches_pil2_extend_and_merkelize(self) -> None:
        for case in load(_TESTDATA / "poseidon1_stage1_commit.json")["cases"]:
            lde = case["lde"]
            n, n_cols = 1 << lde["n_bits"], lde["n_cols"]
            trace = u64(lde["evals"]).reshape(n, n_cols)
            commitment = commit_trace(
                trace,
                blowup=1 << lde["blowup_bits"],
                arity=case["arity"],
                hash_family="Poseidon1",
            )
            self.assertTrue(
                bool(fnp.array_equal(commitment.root, u64(case["root"]))),
                msg=f"root mismatch (arity {case['arity']})",
            )


if __name__ == "__main__":
    absltest.main()
