"""Wiring smoke test for the end-to-end inner-proof spine.

Not a byte-match test — there is no golden inner proof yet (the DEEP stage that
would pin one is unimplemented). It asserts the spine runs, the shapes line up
across commit -> quotient -> FRI -> queries, the query phase opens every tree,
and the whole thing is deterministic in the Fiat-Shamir transcript.
"""

from __future__ import annotations

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F

from zisk_zorch.prover import InnerProver, _fold_steps
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import InnerClaim, InnerWitness

_N_BITS = 6
_N_COLS = 8
_N_CONSTRAINTS = 8
_DEGREE = 2
_BLOWUP_BITS = 1
_ARITY = 4  # the only arity ZisK proves at
_FOLD_BITS = 3
_FINAL_BITS = 5
_POW_BITS = 8
_N_QUERIES = 4


def _trace(seed: int) -> fnp.ndarray:
    """A random `(2^_N_BITS, _N_COLS)` base-field trace (canonical u64 -> F, the
    `golden.u64` construction)."""
    ints = np.random.default_rng(seed).integers(0, 1 << 30, (1 << _N_BITS, _N_COLS))
    return fnp.array(ints.astype(np.uint64), dtype=F)


def _eval_fn(trace: fnp.ndarray) -> fnp.ndarray:
    """`_N_CONSTRAINTS` degree-`_DEGREE` column products in the trailing axis — a
    field-mul proxy for an AIR's constraint expression (`bench_inner_proof`'s)."""
    cols = []
    for k in range(_N_CONSTRAINTS):
        c = trace[:, k % _N_COLS]
        for d in range(1, _DEGREE):
            c = c * trace[:, (k * _DEGREE + d) % _N_COLS]
        cols.append(c)
    return fnp.stack(cols, axis=-1)


def _prove(seed: int = 0, echo_deep: bool = True, jit: bool = True):
    prover = InnerProver(
        _eval_fn,
        n_bits=_N_BITS,
        blowup_bits=_BLOWUP_BITS,
        arity=_ARITY,
        fold_bits=_FOLD_BITS,
        final_bits=_FINAL_BITS,
        pow_bits=_POW_BITS,
        n_queries=_N_QUERIES,
        echo_deep=echo_deep,
        jit=jit,
    )
    claim = InnerClaim(n_bits=_N_BITS, n_cols=_N_COLS, n_constraints=_N_CONSTRAINTS)
    result = prover.prove(claim, InnerWitness(_trace(seed)), Transcript())
    return result.reduction_proof


class FoldStepsTest(absltest.TestCase):
    """The FRI schedule's two degenerate shapes, which nothing downstream
    catches: `Pil2FriCode` accepts a one-step schedule, so both would run and
    produce a proof that folds nothing."""

    def test_rejects_a_final_at_or_above_the_extended_domain(self) -> None:
        for final_bits in (7, 8):
            with self.subTest(final_bits=final_bits):
                with self.assertRaisesRegex(ValueError, "final_bits"):
                    _fold_steps(7, 3, final_bits)

    def test_rejects_a_non_positive_fold(self) -> None:
        for fold_bits in (0, -1):
            with self.subTest(fold_bits=fold_bits):
                with self.assertRaisesRegex(ValueError, "fold_bits"):
                    _fold_steps(7, fold_bits, 5)

    def test_accepts_a_schedule_that_folds(self) -> None:
        self.assertEqual(_fold_steps(11, 3, 5), [11, 8, 5])


class InnerProverTest(absltest.TestCase):
    def test_spine_shapes(self):
        proof = _prove()
        opening = proof.opening
        n_bits_ext = _N_BITS + _BLOWUP_BITS
        # 4-element Poseidon2 roots for every committed tree.
        self.assertEqual(proof.trace_root.shape, (4,))
        self.assertEqual(proof.quotient_root.shape, (4,))
        for root in opening.fri.roots:
            self.assertEqual(root.shape, (4,))
        # Final polynomial is the last FRI layer, sent uncompressed.
        self.assertEqual(opening.fri.final_pol.shape, (1 << _FINAL_BITS,))
        # Every query opens every committed tree exactly once.
        self.assertEqual(len(opening.positions), _N_QUERIES)
        self.assertEqual(len(opening.trace_openings), _N_QUERIES)
        self.assertEqual(len(opening.quotient_openings), _N_QUERIES)
        self.assertEqual(len(opening.fri_openings), _N_QUERIES)
        # Positions land inside the extended domain.
        self.assertTrue(np.all(opening.positions < (1 << n_bits_ext)))

    def test_deterministic_transcript(self):
        a, b = _prove(0), _prove(0)
        np.testing.assert_array_equal(a.opening.positions, b.opening.positions)
        self.assertEqual(a.opening.nonce, b.opening.nonce)
        np.testing.assert_array_equal(
            np.asarray(a.trace_root), np.asarray(b.trace_root)
        )
        np.testing.assert_array_equal(
            np.asarray(a.quotient_root), np.asarray(b.quotient_root)
        )

    def test_distinct_traces_diverge(self):
        a, b = _prove(0), _prove(1)
        self.assertFalse(
            np.array_equal(np.asarray(a.trace_root), np.asarray(b.trace_root))
        )

    def test_default_opening_runs(self):
        # The default opening is the real OpeningProver (opens the committed
        # columns at the OOD point, absorbs, batches) — exercise the whole pil2
        # spine, not just the quotient-passthrough fallback.
        proof = _prove(echo_deep=False)
        self.assertEqual(proof.opening.fri.final_pol.shape, (1 << _FINAL_BITS,))
        self.assertEqual(len(proof.opening.positions), _N_QUERIES)

    def test_jit_zone_matches_eager_bytes(self):
        # The DEEP-leg jit zone threads the transcript through the boundary as
        # a pytree; the eager path hops between the inner zones. The whole
        # Fiat-Shamir stream must agree exactly — one diverged absorb would
        # move every later challenge.
        a = _prove(echo_deep=False, jit=True)
        b = _prove(echo_deep=False, jit=False)
        np.testing.assert_array_equal(
            np.asarray(a.trace_root), np.asarray(b.trace_root)
        )
        np.testing.assert_array_equal(
            np.asarray(a.quotient_root), np.asarray(b.quotient_root)
        )
        np.testing.assert_array_equal(
            np.asarray(a.opening.evals), np.asarray(b.opening.evals)
        )
        np.testing.assert_array_equal(
            np.asarray(a.opening.fri.final_pol), np.asarray(b.opening.fri.final_pol)
        )
        np.testing.assert_array_equal(a.opening.positions, b.opening.positions)
        self.assertEqual(a.opening.nonce, b.opening.nonce)


if __name__ == "__main__":
    absltest.main()
