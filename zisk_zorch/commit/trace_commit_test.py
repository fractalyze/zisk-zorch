"""Byte-match of the stage-1 commit pipeline against pil2-proofman.

Three layers, each pinned by its own golden so a mismatch localizes:
the k-ary Merkle root over linear-hashed rows (`partial_merkle_tree`), the
coset-7 LDE (`extendPol` semantics: reference INTT + naive coset evaluation),
and the full extend -> leaf-hash -> merkelize chain.
"""

from __future__ import annotations

import pathlib

import frx.numpy as fnp
from absl.testing import absltest, parameterized

from zisk_zorch.commit.trace_commit import (
    _block_cols,
    commit_trace,
    extend,
    merkle_tree,
    unextend,
)
from zisk_zorch.golden import load, u64

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"


class MerkleRootTest(absltest.TestCase):
    def test_matches_pil2_partial_merkle_tree(self) -> None:
        for case in load(_TESTDATA / "merkle_root.json")["cases"]:
            rows = u64(case["rows"]).reshape(case["height"], case["n_cols"])
            root, _ = merkle_tree(case["arity"]).commit(rows)
            self.assertTrue(
                bool(fnp.array_equal(root, u64(case["root"]))),
                msg=f"arity {case['arity']}, height {case['height']}",
            )


class LdeTest(absltest.TestCase):
    def test_unextend_inverts_extend(self) -> None:
        # The golden LDE cases double as unextend fixtures: recovering the
        # base evaluations from the coset image must be exact (zz#138 —
        # custom commits are dumped extended-only).
        for case in load(_TESTDATA / "lde.json")["cases"]:
            n, n_cols = 1 << case["n_bits"], case["n_cols"]
            evals = u64(case["evals"]).reshape(n, n_cols)
            blowup = 1 << case["blowup_bits"]
            recovered = unextend(extend(evals, blowup=blowup), blowup)
            self.assertTrue(
                bool(fnp.array_equal(recovered, evals)),
                msg=f"n_bits {case['n_bits']}, blowup_bits {case['blowup_bits']}",
            )

    def test_matches_pil2_extend_pol(self) -> None:
        for case in load(_TESTDATA / "lde.json")["cases"]:
            n, n_cols = 1 << case["n_bits"], case["n_cols"]
            evals = u64(case["evals"]).reshape(n, n_cols)
            extended = extend(evals, blowup=1 << case["blowup_bits"])
            expected = u64(case["extended"]).reshape(-1, n_cols)
            self.assertTrue(
                bool(fnp.array_equal(extended, expected)),
                msg=f"n_bits {case['n_bits']}, blowup_bits {case['blowup_bits']}",
            )


class BlockedLdeTest(parameterized.TestCase):
    """The block size is a memory knob, never a bytes knob: each column's LDE
    is independent, so however the columns are split the codeword is the
    golden one."""

    @parameterized.named_parameters(
        # Under one column's worth, so `_block_cols` floors at 1 and each
        # section extends in as many blocks as it has columns.
        ("one_column_a_block", 1),
        # Two of the first golden's 256-byte columns, so its three split
        # 2 + 1 — the ragged tail a block size dividing `n_cols` never
        # reaches and the default leaves (`Binary_n22`, 39 columns, 4 a
        # block).
        ("ragged_tail", 512),
    )
    def test_block_size_does_not_move_the_codeword(self, block_bytes: int) -> None:
        for case in load(_TESTDATA / "lde.json")["cases"]:
            n, n_cols = 1 << case["n_bits"], case["n_cols"]
            evals = u64(case["evals"]).reshape(n, n_cols)
            extended = extend(
                evals, blowup=1 << case["blowup_bits"], block_bytes=block_bytes
            )
            expected = u64(case["extended"]).reshape(-1, n_cols)
            self.assertTrue(
                bool(fnp.array_equal(extended, expected)),
                msg=f"n_bits {case['n_bits']}, n_cols {n_cols}",
            )

    def test_block_cols_stays_within_the_section(self) -> None:
        # 8 bytes an element: 1024 rows is 8 KiB a column.
        self.assertEqual(_block_cols(1024, 40, 32 << 10), 4)
        # A budget past the whole section extends it in one block — the path
        # every AIR narrow enough to fit takes.
        self.assertEqual(_block_cols(1024, 40, 1 << 30), 40)
        # A budget under one column still makes progress rather than looping
        # forever on a zero-column block.
        self.assertEqual(_block_cols(1024, 40, 0), 1)
        # `ragged_tail` above is only ragged while this holds: the first
        # golden is 3 columns of 32 extended rows, so 512 bytes buys 2 and
        # leaves a 1-wide tail. Regenerating that golden at a new shape
        # would otherwise drop the tail coverage silently.
        self.assertEqual(_block_cols(32, 3, 512), 2)


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
                    fnp.array_equal(
                        commitment.extended,
                        u64(lde["extended"]).reshape(-1, n_cols),
                    )
                ),
                msg=f"extended mismatch (arity {case['arity']})",
            )
            self.assertTrue(
                bool(fnp.array_equal(commitment.root, u64(case["root"]))),
                msg=f"root mismatch (arity {case['arity']})",
            )


class MerkleTreeJitKeyTest(absltest.TestCase):
    def test_fresh_trees_share_one_jit_key(self) -> None:
        # The spine builds a fresh tree per proof; zorch's commit kernels are
        # jitted with the tree static, so fresh trees must be value-equal or
        # every proof recompiles them.
        self.assertEqual(merkle_tree(4), merkle_tree(4))
        self.assertEqual(hash(merkle_tree(4)), hash(merkle_tree(4)))
        # Arity no longer varies, so the hash family is what must still key
        # apart — two families at the same arity are different kernels.
        self.assertNotEqual(merkle_tree(4, "Poseidon1"), merkle_tree(4, "Poseidon2"))


if __name__ == "__main__":
    absltest.main()
