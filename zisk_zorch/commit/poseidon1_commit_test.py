"""Byte-match of the stage-1 commit pipeline under the Poseidon1 hash family.

Native ZisK commits with Poseidon1 by default (the installed proving key sets no
`hash`, so pil2's `DEFAULT_HASH_ID = "Poseidon1"` wins). These goldens pin the
Poseidon1 leaf hash and the full extend -> leaf-hash -> merkelize chain, both
generated from pil2-proofman's own `Poseidon1_*` widths (`tools/fixture-gen/`).
The LDE is family-independent, so only the tree's permutation differs from the
Poseidon2 path.

The arity-2 cases (leaf/node width 8) are skipped: that permutation compiles and
then never returns from execution on the pinned frx stack
(fractalyze/zorch#565). Arities 3 and 4 cover the same pipeline.
"""

from __future__ import annotations

import pathlib

import frx.numpy as fnp
from absl.testing import absltest

from zisk_zorch.commit.linear_hash import LinearHash
from zisk_zorch.commit.trace_commit import commit_trace
from zisk_zorch.golden import load, u64
from zisk_zorch.poseidon1.goldilocks import goldilocks_perm

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"

# Width 8 (arity 2) hangs at run time on the pinned frx stack — see
# fractalyze/zorch#565 and the module docstring.
_UNRUNNABLE_WIDTH = 8
_UNRUNNABLE_ARITY = 2


class Poseidon1LinearHashTest(absltest.TestCase):
    def test_matches_pil2_reference(self) -> None:
        for entry in load(_TESTDATA / "poseidon1_linear_hash.json")["widths"]:
            if entry["width"] == _UNRUNNABLE_WIDTH:
                continue
            hasher = LinearHash(goldilocks_perm(entry["width"]))
            self.assertEqual(hasher.rate, entry["rate"])
            for case in entry["cases"]:
                out = hasher.hash(u64(case["input"]))
                expected = u64(case["output"])[:4]
                self.assertTrue(
                    bool(fnp.array_equal(out, expected)),
                    msg=f"width {entry['width']}, len {len(case['input'])}",
                )


class Poseidon1Stage1CommitTest(absltest.TestCase):
    def test_matches_pil2_extend_and_merkelize(self) -> None:
        for case in load(_TESTDATA / "poseidon1_stage1_commit.json")["cases"]:
            if case["arity"] == _UNRUNNABLE_ARITY:
                continue
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
