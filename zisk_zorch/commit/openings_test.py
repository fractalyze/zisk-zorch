"""Byte-match of the query-phase opening against pil2-proofman.

`group_proof` must reproduce `MerkleTreeGL::getGroupProof`'s flat layout
exactly, and `verify_group_proof` must accept every golden opening against the
golden root — and reject a tampered one.
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.openings import group_proof, verify_group_proof
from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.golden import load, u64

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"


def _cases() -> list[dict]:
    cases = load(_TESTDATA / "merkle_proof.json")["cases"]
    # zorch's binary tree keeps its pad-free power-of-two-height contract (the
    # fold-PCS query layout); a stage-1 commit always has 2^nBitsExt rows, so
    # pil2's padded binary case is unreachable on this path. Arity 3/4 still
    # pin the padding.
    return [
        c for c in cases if c["arity"] != 2 or not c["height"] & (c["height"] - 1)
    ]


class GroupProofTest(absltest.TestCase):
    def test_matches_pil2_get_group_proof(self) -> None:
        for case in _cases():
            rows = u64(case["rows"]).reshape(case["height"], case["n_cols"])
            tree = merkle_tree(case["arity"])
            root, digest_layers = tree.commit(rows)
            self.assertTrue(
                bool(jnp.array_equal(root, u64(case["root"]))),
                msg=f"root mismatch (arity {case['arity']}, height {case['height']})",
            )
            for query in case["queries"]:
                proof = group_proof(tree, rows, digest_layers, query["index"])
                self.assertTrue(
                    bool(jnp.array_equal(proof, u64(query["proof"]))),
                    msg=f"arity {case['arity']}, height {case['height']}, "
                    f"index {query['index']}",
                )

    def test_golden_proofs_reconstruct_the_root(self) -> None:
        for case in _cases():
            tree = merkle_tree(case["arity"])
            root = u64(case["root"])
            for query in case["queries"]:
                proof = u64(query["proof"])
                self.assertTrue(
                    verify_group_proof(
                        tree, root, query["index"], proof, case["n_cols"]
                    ),
                    msg=f"arity {case['arity']}, height {case['height']}, "
                    f"index {query['index']}",
                )
                tampered = proof.at[0].set(proof[0] + jnp.array(1, F))
                self.assertFalse(
                    verify_group_proof(
                        tree, root, query["index"], tampered, case["n_cols"]
                    ),
                    msg=f"tampered proof accepted (arity {case['arity']}, "
                    f"height {case['height']}, index {query['index']})",
                )


if __name__ == "__main__":
    absltest.main()
