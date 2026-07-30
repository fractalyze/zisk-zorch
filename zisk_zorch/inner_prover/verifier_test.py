"""Inner-proof verifier round trip: the verifier accepts an honest proof and
rejects every way of lying about it.

The acceptance case is the weak half — a verifier that returns True is trivial
to write. Each rejection case below breaks exactly one of the four independent
checks (Merkle, DEEP, FRI, constraint), so a check silently dropped from the
verifier turns its case red rather than passing unnoticed.
"""

from __future__ import annotations

from dataclasses import replace

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocks as F

from zisk_zorch.inner_prover.prover import InnerProver
from zisk_zorch.inner_prover.verifier import InnerVerifier
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import InnerClaim, InnerWitness

_N_BITS = 6
_N_FREE = 4  # freely chosen columns; each constraint pins one derived column
_N_COLS = 2 * _N_FREE
_N_CONSTRAINTS = _N_FREE
_BLOWUP_BITS = 1
_ARITY = 2
_FOLD_BITS = 3
_FINAL_BITS = 5
_POW_BITS = 8
_N_QUERIES = 4


def _trace(seed: int) -> fnp.ndarray:
    """A SATISFYING witness: columns `[0, _N_FREE)` are random, column
    `_N_FREE + k` is the row-wise product of columns `k` and `(k+1) % _N_FREE`
    — exactly what `_eval_fn`'s constraints demand. The prover spine runs on
    any trace, but the verifier's constraint check at `z` holds only when the
    quotient's zerofier division is exact, i.e. when the AIR is satisfied on
    every row."""
    ints = np.random.default_rng(seed).integers(0, 1 << 30, (1 << _N_BITS, _N_FREE))
    free = fnp.array(ints.astype(np.uint64), dtype=F)
    derived = [free[:, k] * free[:, (k + 1) % _N_FREE] for k in range(_N_FREE)]
    return fnp.concatenate([free, fnp.stack(derived, axis=-1)], axis=1)


def _eval_fn(trace: fnp.ndarray) -> fnp.ndarray:
    """`_N_CONSTRAINTS` degree-2 product constraints `t[k]·t[(k+1)%F] −
    t[F+k]`, each vanishing on a `_trace` witness. Row-wise, so it evaluates
    at the OOD row too."""
    cols = []
    for k in range(_N_CONSTRAINTS):
        c = trace[:, k] * trace[:, (k + 1) % _N_FREE] - trace[:, _N_FREE + k]
        cols.append(c)
    return fnp.stack(cols, axis=-1)


def _random_trace(seed: int) -> fnp.ndarray:
    """A NON-satisfying trace of the same shape — the constraint check's
    rejection case."""
    ints = np.random.default_rng(seed).integers(0, 1 << 30, (1 << _N_BITS, _N_COLS))
    return fnp.array(ints.astype(np.uint64), dtype=F)


def _claim(n_constraints: int = _N_CONSTRAINTS) -> InnerClaim:
    return InnerClaim(n_bits=_N_BITS, n_cols=_N_COLS, n_constraints=n_constraints)


def _prove_trace(trace: fnp.ndarray, echo_deep: bool = False, jit: bool = True):
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
    result = prover.prove(_claim(), InnerWitness(trace), Transcript())
    return result.reduction_proof


def _prove(seed: int = 0, jit: bool = True):
    """An honest proof through the real opening — the echo placeholder opens
    no columns, so it produces nothing a verifier could check."""
    return _prove_trace(_trace(seed), jit=jit)


def _accepts(proof, n_constraints: int = _N_CONSTRAINTS) -> bool:
    verifier = InnerVerifier(
        _eval_fn,
        n_bits=_N_BITS,
        blowup_bits=_BLOWUP_BITS,
        arity=_ARITY,
        fold_bits=_FOLD_BITS,
        final_bits=_FINAL_BITS,
        pow_bits=_POW_BITS,
        opening_points=(0,),
    )
    result = verifier.verify(_claim(n_constraints), proof, Transcript())
    return bool(result.ok)


def _bump(value):
    """`value + 1` in whatever field it lives in — a tamper that stays a legal
    element rather than an out-of-range int."""
    return value + fnp.ones((), value.dtype)


def _tamper(proof, **fields):
    """`proof` with the named opening-section fields replaced — every tamper
    below rewrites the opening discharge, never the committed roots."""
    return replace(proof, opening=replace(proof.opening, **fields))


class InnerVerifierTest(absltest.TestCase):
    def test_accepts_an_honest_proof(self) -> None:
        self.assertTrue(_accepts(_prove()))

    def test_accepts_an_eager_proof(self) -> None:
        # The jit knob must not change the byte stream the verifier replays.
        self.assertTrue(_accepts(_prove(jit=False)))

    def test_rejects_a_tampered_nonce(self) -> None:
        """Breaks the grind, before any position is read."""
        proof = _prove()
        self.assertFalse(_accepts(_tamper(proof, nonce=proof.opening.nonce - 1)))

    def test_rejects_a_tampered_trace_opening(self) -> None:
        """Breaks that row's Merkle path against the stage-1 root."""
        proof = _prove()
        openings = [list(o) for o in proof.opening.trace_openings]
        first = openings[0][0]
        openings[0][0] = first.at[0].set(_bump(first[0]))
        self.assertFalse(_accepts(_tamper(proof, trace_openings=openings)))

    def test_rejects_a_tampered_quotient_opening(self) -> None:
        proof = _prove()
        openings = [list(o) for o in proof.opening.quotient_openings]
        first = openings[0][0]
        openings[0][0] = first.at[0].set(_bump(first[0]))
        self.assertFalse(_accepts(_tamper(proof, quotient_openings=openings)))

    def test_rejects_tampered_evals(self) -> None:
        """The openings are absorbed before `vf` is squeezed, so changing them
        re-derives a different batching challenge and every later check moves
        with it — the transcript binding, not one comparison."""
        proof = _prove()
        evals = proof.opening.evals.at[0].set(_bump(proof.opening.evals[0]))
        self.assertFalse(_accepts(_tamper(proof, evals=evals)))

    def test_rejects_a_tampered_final_polynomial(self) -> None:
        proof = _prove()
        fri = proof.opening.fri
        final = fri.final_pol.at[0].set(_bump(fri.final_pol[0]))
        self.assertFalse(_accepts(_tamper(proof, fri=replace(fri, final_pol=final))))

    def test_rejects_a_proof_of_a_different_trace(self) -> None:
        """Roots from one trace, openings from another: each half is internally
        well-formed, so only cross-checking them catches it."""
        honest, other = _prove(seed=0), _prove(seed=1)
        self.assertFalse(
            _accepts(_tamper(honest, trace_openings=other.opening.trace_openings))
        )

    def test_rejects_an_unsatisfied_air(self) -> None:
        """A random trace commits, folds, and opens perfectly well — but its
        constraints do not vanish on the base domain, so the zerofier division
        is not exact and the AIR identity at `z` fails. The other three checks
        cannot see this: it is the constraint check's whole reason to exist."""
        self.assertFalse(_accepts(_prove_trace(_random_trace(0))))

    def test_rejects_a_wrong_constraint_count(self) -> None:
        """`n_constraints` is the verifier's half of the statement: folding by
        a different alpha vector checks a different claim."""
        proof = _prove()
        with self.assertRaises(ValueError):
            _accepts(proof, n_constraints=_N_CONSTRAINTS - 1)

    def test_refuses_an_unverifiable_echo_proof(self) -> None:
        """`EchoOpening` opens no columns, so there is nothing to bind the FRI
        instance to the trace. Refusing beats returning True vacuously."""
        proof = _prove_trace(_trace(0), echo_deep=True)
        self.assertIsNone(proof.opening.evals)
        with self.assertRaises(ValueError):
            _accepts(proof)


if __name__ == "__main__":
    absltest.main()
