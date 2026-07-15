"""Wiring smoke test for the end-to-end inner-proof spine.

Not a byte-match test — there is no golden inner proof yet (the DEEP stage that
would pin one is unimplemented). It asserts the spine runs, the shapes line up
across commit -> quotient -> FRI -> queries, the query phase opens every tree,
and the whole thing is deterministic in the Fiat-Shamir transcript.
"""

from __future__ import annotations

import frx.numpy as jnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F

from zisk_zorch.prover import prove_inner, quotient_as_fri_polynomial

_N_BITS = 6
_N_COLS = 8
_N_CONSTRAINTS = 8
_DEGREE = 2
_BLOWUP_BITS = 1
_ARITY = 2
_FOLD_BITS = 3
_FINAL_BITS = 5
_POW_BITS = 8
_N_QUERIES = 4


def _trace(seed: int) -> jnp.ndarray:
    """A random `(2^_N_BITS, _N_COLS)` base-field trace (canonical u64 -> F, the
    `golden.u64` construction)."""
    ints = np.random.default_rng(seed).integers(0, 1 << 30, (1 << _N_BITS, _N_COLS))
    return jnp.array(ints.astype(np.uint64), dtype=F)


def _eval_fn(trace: jnp.ndarray) -> jnp.ndarray:
    """`_N_CONSTRAINTS` degree-`_DEGREE` column products in the trailing axis — a
    field-mul proxy for an AIR's constraint expression (`bench_inner_proof`'s)."""
    cols = []
    for k in range(_N_CONSTRAINTS):
        c = trace[:, k % _N_COLS]
        for d in range(1, _DEGREE):
            c = c * trace[:, (k * _DEGREE + d) % _N_COLS]
        cols.append(c)
    return jnp.stack(cols, axis=-1)


def _prove(seed: int = 0, fri_polynomial_fn=quotient_as_fri_polynomial):
    return prove_inner(
        _trace(seed),
        _eval_fn,
        n_constraints=_N_CONSTRAINTS,
        blowup_bits=_BLOWUP_BITS,
        arity=_ARITY,
        fold_bits=_FOLD_BITS,
        final_bits=_FINAL_BITS,
        pow_bits=_POW_BITS,
        n_queries=_N_QUERIES,
        fri_polynomial_fn=fri_polynomial_fn,
    )


class ProveInnerTest(absltest.TestCase):
    def test_spine_shapes(self):
        proof = _prove()
        n_bits_ext = _N_BITS + _BLOWUP_BITS
        # 4-element Poseidon2 roots for every committed tree.
        self.assertEqual(proof.trace_root.shape, (4,))
        self.assertEqual(proof.quotient_root.shape, (4,))
        for root in proof.fri.roots:
            self.assertEqual(root.shape, (4,))
        # Final polynomial is the last FRI layer, sent uncompressed.
        self.assertEqual(proof.final_pol.shape, (1 << _FINAL_BITS,))
        # Every query opens every committed tree exactly once.
        self.assertEqual(len(proof.query_positions), _N_QUERIES)
        self.assertEqual(len(proof.trace_openings), _N_QUERIES)
        self.assertEqual(len(proof.quotient_openings), _N_QUERIES)
        self.assertEqual(len(proof.fri_openings), _N_QUERIES)
        # Positions land inside the extended domain.
        self.assertTrue(np.all(proof.query_positions < (1 << n_bits_ext)))

    def test_deterministic_transcript(self):
        a, b = _prove(0), _prove(0)
        np.testing.assert_array_equal(a.query_positions, b.query_positions)
        self.assertEqual(a.nonce, b.nonce)
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

    def test_default_deep_stage_runs(self):
        # The default fri_polynomial_fn is the real DEEP combiner (opens the
        # committed columns at the OOD point, absorbs, batches) — exercise the
        # whole pil2 spine, not just the quotient-passthrough fallback.
        proof = _prove(fri_polynomial_fn=None)
        self.assertEqual(proof.final_pol.shape, (1 << _FINAL_BITS,))
        self.assertEqual(len(proof.query_positions), _N_QUERIES)


if __name__ == "__main__":
    absltest.main()
