"""FRI verifier round-trip: the verifier accepts the prover's output and rejects
tampering.

The prover is byte-matched to pil2 (fri_prove golden) and the verifier mirrors
pil2's verify path, so a valid proof must verify; flipping the final polynomial,
a layer root, or an opened value must make a fold or Merkle check fail. Reuses
the fri_prove golden's inputs to build real proofs (no separate fixture).
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import goldilocksx3_mont as F3

from zisk_zorch.fri.prover import prove, prove_queries
from zisk_zorch.fri.verifier import verify
from zisk_zorch.golden import load, u64, u64x3
from zisk_zorch.transcript.transcript import Transcript

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"


class FriVerifierTest(absltest.TestCase):
    def _prove(self, case: dict):
        transcript = Transcript()
        transcript.put(u64(case["seed"]))
        proof = prove(
            u64x3(case["init_pol"]),
            case["steps"],
            arity=case["arity"],
            transcript=transcript,
        )
        queries = [q["query"] for q in case["queries"]]
        return proof, prove_queries(proof, queries), queries

    def _verify(self, case, roots, final_pol, openings, queries) -> bool:
        transcript = Transcript()
        transcript.put(u64(case["seed"]))
        return verify(
            roots,
            final_pol,
            openings,
            queries,
            steps=case["steps"],
            arity=case["arity"],
            transcript=transcript,
        )

    def test_accepts_valid_and_rejects_tampering(self) -> None:
        for case in load(_TESTDATA / "fri_prove.json")["cases"]:
            with self.subTest(steps=case["steps"], arity=case["arity"]):
                proof, openings, queries = self._prove(case)

                self.assertTrue(
                    self._verify(case, proof.roots, proof.final_pol, openings, queries),
                    msg="valid proof rejected",
                )

                # Tamper the final polynomial: cascades through the replayed
                # challenges and the last fold's consistency check.
                bad_final = proof.final_pol.at[0].set(proof.final_pol[0] + jnp.ones((), F3))
                self.assertFalse(
                    self._verify(case, proof.roots, bad_final, openings, queries),
                    msg="accepted a tampered final polynomial",
                )

                # Tamper an opened value: breaks that layer's Merkle opening.
                bad_openings = [list(per_q) for per_q in openings]
                first = bad_openings[0][0]
                bad_openings[0][0] = first.at[0].set(first[0] + jnp.ones((), first.dtype))
                self.assertFalse(
                    self._verify(case, proof.roots, proof.final_pol, bad_openings, queries),
                    msg="accepted a tampered Merkle opening",
                )


if __name__ == "__main__":
    absltest.main()
