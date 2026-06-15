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
import numpy as np
from absl.testing import absltest
from jax import Array
from zk_dtypes import goldilocks_mont as F
from zk_dtypes import goldilocksx3_mont as F3

from zisk_zorch.fri.fold import _COSET_SHIFT, _GOLDILOCKS_P, _TWO_ADIC_ROOT
from zisk_zorch.fri.prover import prove, prove_queries
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.fri.verifier import verify
from zisk_zorch.golden import load, u64, u64x3
from zisk_zorch.transcript.transcript import Transcript

_TESTDATA = pathlib.Path(__file__).parent / "testdata" / "golden"
_POW_BITS = 8  # grinding difficulty; small keeps the prover's PoW search short
_GOLDILOCKS_ORDER = 0xFFFF_FFFF_0000_0001


def _low_degree_codeword(n_bits: int, n_bits_ext: int, seed: int) -> Array:
    """A genuine FRI polynomial `f`: a random degree-`< 2^n_bits` polynomial
    evaluated on the extended coset domain (shift 7, pil2 root `W[n_bits_ext]`) in
    pil2 domain order. Folding it down the chain keeps it low-degree, so the final
    pol passes `check_final` at this `n_bits` (and fails at a stricter one). Built
    by schoolbook Horner — the LDE ground truth — per cubic limb: the evaluation
    point is base-field, so the three limbs evaluate independently."""
    deg = 1 << n_bits
    n_ext = 1 << n_bits_ext
    w = pow(_TWO_ADIC_ROOT, 1 << (32 - n_bits_ext), _GOLDILOCKS_P)
    rng = np.random.default_rng(seed)
    coeffs = rng.integers(0, 1 << 32, size=(deg, 3)).astype(object)
    out = np.zeros((n_ext, 3), dtype=object)
    for i in range(n_ext):
        x = _COSET_SHIFT * pow(w, i, _GOLDILOCKS_P) % _GOLDILOCKS_P
        for c in range(3):
            acc = 0
            for k in range(deg - 1, -1, -1):
                acc = (acc * x + int(coeffs[k, c])) % _GOLDILOCKS_P
            out[i, c] = acc
    base = out.astype(np.uint64).astype(F)  # (n_ext, 3) base limbs -> montgomery
    return jnp.array(base.view(F3).reshape(n_ext))


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
            # The fri_prove fixtures fold random (high-degree) codewords, so the
            # low-degree test is run vacuously: n_bits == nBitsExt means zero
            # blowup, an empty coefficient range. The genuine low-degree test is
            # exercised by test_low_degree_final_pol below and seam_test.
            n_bits=case["steps"][0],
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

    def test_low_degree_final_pol(self) -> None:
        """A real low-degree `f` verifies; over-claiming the degree bound rejects.

        Fold a genuine low-degree FRI polynomial through the prover and verify it
        with the matching base size. Then re-verify with a stricter base size: the
        fold chain, Merkle openings, and grind are unchanged (all still pass), so
        only `check_final` can reject — isolating the low-degree test."""
        n_bits, n_bits_ext = 4, 6
        steps = [6, 4, 2]  # folds 16x; final pol is degree < 1 (a constant)
        arity = 4
        seed = u64(["1", "2", "3", "4"])

        fri_pol = _low_degree_codeword(n_bits, n_bits_ext, seed=0xF1)
        transcript = Transcript()
        transcript.put(seed)
        proof = prove(fri_pol, steps, arity=arity, transcript=transcript)
        indices, nonce = sample_query_positions(
            transcript,
            proof.final_pol,
            pow_bits=_POW_BITS,
            n_queries=4,
            n_bits_ext=steps[0],
        )
        openings = prove_queries(proof, indices)

        def vfy(n_bits_arg: int) -> bool:
            t = Transcript()
            t.put(seed)
            return verify(
                proof.roots,
                proof.final_pol,
                openings,
                steps=steps,
                arity=arity,
                transcript=t,
                pow_bits=_POW_BITS,
                nonce=nonce,
                n_bits=n_bits_arg,
            )

        self.assertTrue(vfy(n_bits), msg="valid low-degree proof rejected")
        # A stricter base size tightens the degree bound below the final pol's
        # true degree; nothing else changes, so check_final must reject.
        self.assertFalse(
            vfy(n_bits - 1), msg="check_final accepted an over-degree final pol"
        )


if __name__ == "__main__":
    absltest.main()
