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
deep_fri_polynomial(ctx)      -> fri_pol    ← DEEP stage (zisk_zorch/deep/, see below)
fri.prove(fri_pol, …)         -> layers     fold betas squeezed off the same transcript
sample_query_positions(…)     -> positions  finalPol absorb → grind → getPermutations
prove_queries + group_proof   -> openings   every committed tree opened per query
```

The single shared `Transcript` is what makes the challenges depend on the
committed roots — the property the isolated benchmarks cannot exercise.

## The DEEP / FRI-polynomial stage (`zisk_zorch/deep/`)

pil2's `calculateFRIPolynomial` squeezes the out-of-domain point `z`, evaluates
the committed polynomials there, absorbs the openings, squeezes a batching
challenge, and builds the FRI codeword. It is the default `fri_polynomial_fn`.

- `deep/opening.py` — pil2 `computeLEv` + `evmap`: the OOD-evaluation primitive.
  `LEv[k] = INTT((z·g^p·shift⁻¹)^k)` are the Lagrange weights with
  `Σ_k LEv[k]·p(shift·g^k) = p(z·g^p)`; `open_columns` subsamples each extended
  column to the base coset and dots. Pinned by the round-trip identity
  (`opening_test.py`), no pil2 dump needed.
- `deep/fri_polynomial.py` — pil2 `calculateFRIPolynomial`: the DEEP-ALI batched
  quotient `f(x) = Σ_m vf^m·(p_m(x) − p_m(ξ))/(x − ξ)`. Verified by the FRI
  low-degree property (`fri_polynomial_test.py`): a correct opening folds low,
  a wrong one does not.

**Byte-match boundary.** `deep_composition` implements the *generic* DEEP-ALI
formula. pil2's `friExp` in a real proving key bakes in which columns are
batched, their order, and each one's challenge power (`expressions.bin`) —
matching a specific AIR byte-for-byte needs that compiled op list (the machinery
`cexp_ref` already interprets for the quotient) plus a pil2 golden. That is the
next slice. `quotient_as_fri_polynomial` remains as a trivial FRI-over-quotient
fallback.

## Status

Tests are shape/determinism/property checks, green on the zkx GPU backend, not
golden byte-matches — a golden inner proof needs the AIR-specific `friExp` op
list. When that lands, add a golden vector and assert `prove_inner` reproduces
it end to end.
