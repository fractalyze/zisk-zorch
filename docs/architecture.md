# Architecture: the inner proof over one transcript

`prove_inner_chain` ([`../zisk_zorch/prover.py`](../zisk_zorch/prover.py)) is a
`ProveChain` of five **Stages** — trace commit, quotient, DEEP, FRI, queries —
running in pil2-proofman's `genProof` order over a single Fiat-Shamir
`Transcript` and one **Bridge** (`InnerBridge`). This page maps the proof onto
those Stages and names the pil2 vocabulary each one mirrors. Every primitive that
mirrors pil2 is pinned against pil2-proofman v1.0.0-alpha's `fields` crate via
[`../tools/fixture-gen/`](../tools/fixture-gen/).

## Stage / Bridge in this repo

- **Stage** — one step of the inner proof's heterogeneous sequence, a
  `zorch.round.Round` subclass named `*Stage`. Each runs its own inner rounds
  (FRI's layer chain).
- **Bridge** — the state a Stage hands the next (`InnerBridge`): the trace
  commitment, the quotient and its tree, the DEEP codeword, the FRI proof. It
  holds only what a later Stage reads from an earlier one — a Stage writes its
  own fields via `replace` and passes the rest through. Static config (arity, the
  fold schedule, `eval_fn`) lives on the Stage, not the Bridge.

For coding style see [conventions.md](conventions.md); to build, test, and
benchmark it see [development.md](development.md).

## The transcript spine

```
commit_trace(trace)           -> root₁     transcript.put(root₁)
                                            alpha = transcript.get_field()   (powers → constraint fold)
quotient_from_constraints(…)  -> Q         commit Q → rootQ, transcript.put(rootQ)
deep_fri_polynomial(ctx)      -> fri_pol   ← DEEP stage
fri.prove(fri_pol, …)         -> layers    fold betas squeezed off the same transcript
sample_query_positions(…)     -> positions finalPol absorb → grind → getPermutations
prove_queries + group_proof   -> openings  every committed tree opened per query
```

The one shared `Transcript` is what makes each challenge depend on the committed
roots — the property the per-stage benchmarks cannot exercise.

## Stages

| Stage | pil2 name | What it does | Module | Golden |
|---|---|---|---|---|
| Trace commit | `extendAndMerkelize` (`commitStage(1)`) | INTT each column, coset-7 RS encode to `N·blowup` rows in pil2 domain order, pil2 linear-hash each row to a 4-Goldilocks leaf, k-ary Poseidon2 fold to the root | `commit/` | `lde`, `linear_hash`, `merkle_root`, `merkle_proof`, `stage1_commit` |
| Constraint ingest | — (rw-exported) | Load each ZisK chip's constraints + bus interactions from the `rw_constraints` wheel (`constraints/zisk/v1`), the same export `sp1-zorch` consumes | `constraints/` | — (pinned by the quotient's byte-match) |
| Quotient | `calculateQuotientPolynomial` | Fold constraints by powers of `alpha` (zorch's agnostic `constraint_eval`), divide by the inverse zerofier, commit `Q` | `quotient/` | `cexp_eval`, `zerofier_inv`, `gsum` |
| DEEP | `calculateFRIPolynomial` | Squeeze the OOD point `z`, open the committed polynomials there (`computeLEv`+`evmap`), absorb, squeeze `vf`, build the DEEP-ALI codeword | `deep/` | **none** — see below |
| FRI | `FRI::fold` / `proveQueries` | Fold the codeword down the layer chain committing each layer, grind, open every tree per query | `fri/` | `fri_fold`, `fri_prove`, `fri_final`, `grinding`, `query_sample` |

Each stage's pil2 conventions — the Poseidon2 M4 choice, the NTT domain order,
the linear-hash chaining, the transcript's buffer discipline, the opening layout,
the α-power order — live in the module docstring of the code that implements
them, per [conventions.md](conventions.md).

## The DEEP byte-match boundary

`deep_composition` implements the *generic* DEEP-ALI quotient
`f(x) = Σ_m vf^m·(p_m(x) − p_m(ξ))/(x − ξ)`. A real proving key's `friExp` also
fixes **which** columns are batched, their order, and each one's challenge power
(`expressions.bin`). Matching a specific AIR byte-for-byte needs that compiled op
list — the machinery `cexp_ref` already interprets for the quotient — plus a pil2
golden.

Until then DEEP is the one stage with no golden. It is pinned by properties
instead: the OOD opening by the round-trip identity
`Σ_k LEv[k]·p(shift·g^k) = p(z·g^p)`, the composition by the FRI low-degree test
(a correct opening folds low, a wrong one does not). Both hold for *any* correct
DEEP-ALI implementation, so neither pins pil2's choices.
`quotient_as_fri_polynomial` remains as a trivial FRI-over-quotient fallback.
