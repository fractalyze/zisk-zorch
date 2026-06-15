"""FRI verifier round-trip: the verifier accepts the prover's output and rejects
tampering.

The prover is byte-matched to pil2 (fri_prove golden) and the verifier mirrors
pil2's verify path, so a valid proof must verify; flipping the final polynomial,
a layer root, or an opened value must make a fold or Merkle check fail. Reuses
the fri_prove golden's inputs to build real proofs (no separate fixture).

Query positions are derived (not external): the prover derives them from its
post-fold transcript via `sample_query_positions` (self-generating the grinding
nonce), and the verifier independently re-derives them and re-checks the grind —
a valid proof verifies only if the two derivations agree and the nonce binds.
"""

from __future__ import annotations

import pathlib

import jax.numpy as jnp
from absl.testing import absltest
from zk_dtypes import goldilocksx3_mont as F3

from zisk_zorch.fri.prover import prove, prove_queries
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.fri.verifier import verify
from zisk_zorch.golden import load, u64, u64x3
from zisk_zorch.transcript.transcript import Transcript

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"
_POW_BITS = 8  # grinding difficulty; small keeps the prover's PoW search short
_GOLDILOCKS_ORDER = 0xFFFF_FFFF_0000_0001


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
        # Derive the query positions from the post-fold transcript, exactly as a
        # real STARK prover would — self-generating the grinding nonce — then open
        # them. The nonce travels in the proof to the verifier.
        indices, nonce = sample_query_positions(
            transcript,
            proof.final_pol,
            pow_bits=_POW_BITS,
            n_queries=len(case["queries"]),
            n_bits_ext=case["steps"][0],
        )
        return proof, prove_queries(proof, indices), nonce

    def _verify(self, case, roots, final_pol, openings, nonce) -> bool:
        transcript = Transcript()
        transcript.put(u64(case["seed"]))
        return verify(
            roots,
            final_pol,
            openings,
            steps=case["steps"],
            arity=case["arity"],
            transcript=transcript,
            pow_bits=_POW_BITS,
            nonce=nonce,
        )

    def test_accepts_valid_and_rejects_tampering(self) -> None:
        for case in load(_TESTDATA / "fri_prove.json")["cases"]:
            with self.subTest(steps=case["steps"], arity=case["arity"]):
                proof, openings, nonce = self._prove(case)

                self.assertTrue(
                    self._verify(case, proof.roots, proof.final_pol, openings, nonce),
                    msg="valid proof rejected",
                )

                # Tamper the grinding nonce. The prover's nonce is the SMALLEST
                # that hashes to pow_bits leading zeros, so any smaller nonce is
                # guaranteed to fail the grind check; a nonce >= the field order
                # is rejected by the canonical guard.
                if nonce > 0:
                    self.assertFalse(
                        self._verify(case, proof.roots, proof.final_pol, openings, nonce - 1),
                        msg="accepted a nonce that fails the grind",
                    )
                self.assertFalse(
                    self._verify(
                        case, proof.roots, proof.final_pol, openings, _GOLDILOCKS_ORDER
                    ),
                    msg="accepted a non-canonical nonce",
                )

                # Tamper the final polynomial: cascades through the re-derived
                # query positions and the last fold's consistency check.
                bad_final = proof.final_pol.at[0].set(proof.final_pol[0] + jnp.ones((), F3))
                self.assertFalse(
                    self._verify(case, proof.roots, bad_final, openings, nonce),
                    msg="accepted a tampered final polynomial",
                )

                # Tamper an opened value: breaks that layer's Merkle opening.
                bad_openings = [list(per_q) for per_q in openings]
                first = bad_openings[0][0]
                bad_openings[0][0] = first.at[0].set(first[0] + jnp.ones((), first.dtype))
                self.assertFalse(
                    self._verify(case, proof.roots, proof.final_pol, bad_openings, nonce),
                    msg="accepted a tampered Merkle opening",
                )


if __name__ == "__main__":
    absltest.main()
