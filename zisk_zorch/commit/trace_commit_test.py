"""Byte-match of the stage-1 commit pipeline against pil2-proofman.

Three layers, each pinned by its own golden so a mismatch localizes:
the k-ary Merkle root over linear-hashed rows (`partial_merkle_tree`), the
coset-7 LDE (`extendPol` semantics: reference INTT + naive coset evaluation),
and the full extend -> leaf-hash -> merkelize chain.
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
from absl.testing import absltest

from zisk_zorch.commit.trace_commit import commit_trace, extend, merkle_tree
from zisk_zorch.golden import load, u64

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"


class MerkleRootTest(absltest.TestCase):
    def test_matches_pil2_partial_merkle_tree(self) -> None:
        for case in load(_TESTDATA / "merkle_root.json")["cases"]:
            if case["arity"] == 2 and case["height"] & (case["height"] - 1):
                # zorch's binary tree keeps its pad-free power-of-two-height
                # contract (the fold-PCS query layout); a stage-1 commit always
                # has 2^nBitsExt rows, so pil2's padded binary case is
                # unreachable on this path. Arity 3/4 still pin the padding.
                continue
            rows = u64(case["rows"]).reshape(case["height"], case["n_cols"])
            root, _ = merkle_tree(case["arity"]).commit(rows)
            self.assertTrue(
                bool(jnp.array_equal(root, u64(case["root"]))),
                msg=f"arity {case['arity']}, height {case['height']}",
            )


class LdeTest(absltest.TestCase):
    def test_matches_pil2_extend_pol(self) -> None:
        for case in load(_TESTDATA / "lde.json")["cases"]:
            n, n_cols = 1 << case["n_bits"], case["n_cols"]
            evals = u64(case["evals"]).reshape(n, n_cols)
            extended = extend(evals, blowup=1 << case["blowup_bits"])
            expected = u64(case["extended"]).reshape(-1, n_cols)
            self.assertTrue(
                bool(jnp.array_equal(extended, expected)),
                msg=f"n_bits {case['n_bits']}, blowup_bits {case['blowup_bits']}",
            )


class Stage1CommitTest(absltest.TestCase):
    def test_matches_pil2_extend_and_merkelize(self) -> None:
        for case in load(_TESTDATA / "stage1_commit.json")["cases"]:
            lde = case["lde"]
            n, n_cols = 1 << lde["n_bits"], lde["n_cols"]
            trace = u64(lde["evals"]).reshape(n, n_cols)
            commitment = commit_trace(
                trace, blowup=1 << lde["blowup_bits"], arity=case["arity"]
            )
            self.assertTrue(
                bool(
                    jnp.array_equal(
                        commitment.extended,
                        u64(lde["extended"]).reshape(-1, n_cols),
                    )
                ),
                msg=f"extended mismatch (arity {case['arity']})",
            )
            self.assertTrue(
                bool(jnp.array_equal(commitment.root, u64(case["root"]))),
                msg=f"root mismatch (arity {case['arity']})",
            )


if __name__ == "__main__":
    absltest.main()
