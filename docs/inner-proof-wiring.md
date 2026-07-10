# Inner-proof wiring

`zisk_zorch/prover.py` (`prove_inner`) runs the inner-proof stages in
pil2-proofman's order over a single Fiat-Shamir `Transcript`. Until now each
stage existed only as an isolated primitive, exercised on random inputs by
`bench_inner_proof.py`; this module is the connective tissue.

## The transcript spine

```
commit_trace(trace)          -> root₁      transcript.put(root₁)
                                            alpha  = transcript.get_field()   (powers → constraint fold)
quotient_from_constraints(…)  -> Q          commit Q → rootQ, transcript.put(rootQ)
fri_polynomial_fn(ctx)        -> fri_pol    ← DEEP seam (see below)
fri.prove(fri_pol, …)         -> layers     fold betas squeezed off the same transcript
sample_query_positions(…)     -> positions  finalPol absorb → grind → getPermutations
prove_queries + group_proof   -> openings   every committed tree opened per query
```

The single shared `Transcript` is what makes the challenges depend on the
committed roots — the property the isolated benchmarks cannot exercise.

## The one seam: DEEP / FRI-polynomial construction

pil2's `calculateFRIPolynomial` squeezes an out-of-domain point, evaluates the
committed polynomials there, and batches the openings into the codeword FRI
folds. **That stage has no primitive in this repo** (the bench feeds FRI a random
codeword). Fabricating its byte stream would violate the repo's byte-match
contract, so it is injected as `fri_polynomial_fn` rather than hand-rolled.

- `quotient_as_fri_polynomial` — a runnable placeholder: the quotient is itself a
  cubic codeword of length `2^n_bits_ext`, so it drives the whole spine. It does
  **not** byte-match pil2 (no trace openings), so it is for wiring/shape tests
  only, not conformance.
- A future golden-backed DEEP combiner drops in at the same seam; it reads the
  committed evaluations off `FriPolynomialContext` and squeezes its out-of-domain
  challenge from the live transcript.

## Status

`prover_test.py` is a shape/determinism smoke test (green on the zkx GPU
backend), not a golden byte-match — the golden that would pin an inner proof
needs the DEEP stage first. When that stage lands, add a golden inner-proof
vector and assert `prove_inner` reproduces it.
