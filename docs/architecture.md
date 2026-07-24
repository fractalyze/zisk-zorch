# Architecture: the inner proof over one transcript

`InnerProver` ([`../zisk_zorch/prover.py`](../zisk_zorch/prover.py)) is a
composite `zorch.stage.ProverStage` reducing the inner statement to the
trivial claim in pil2-proofman's `genProof` order over a single Fiat-Shamir
`Transcript`: the trace commit binds the witness, then the quotient and
opening Stages each discharge the claim the one before produced. This page
maps the proof onto that structure and names the pil2 vocabulary each part
mirrors. Every primitive that mirrors pil2 is pinned against pil2-proofman
v1.0.0-alpha's `fields` crate via [`../tools/fixture-gen/`](../tools/fixture-gen/).

## Stages and claims in this repo

- **Stage** — one claim reduction with paired roles, each stage's pair in its
  own package: `quotient/prover.py`'s `QuotientProver` reduces `InnerClaim`
  (some trace satisfies the AIR) to `QuotientBoundClaim` (the committed pair
  is consistent), and `opening/prover.py`'s `OpeningProver` discharges that to
  the trivial claim. pil2's DEEP, FRI, and query phases are one terminal
  stage: each sub-step consumes a challenge the previous one squeezed and
  produces the next one's prover input, so their seams are prover-data seams,
  not claim boundaries.
- **Claim** — what crosses a stage seam: only data both roles derive (the
  verifier from the wire), never prover-only state. The claims live in
  [`../zisk_zorch/types.py`](../zisk_zorch/types.py), below every stage,
  alongside the commit-data handoffs and witness types.
  The committed trees' prover data — the extended trace, the quotient
  codeword and layers — rides witness wrappers (`OpeningWitness`) the
  composite assembles. Static config (arity, the fold schedule, `eval_fn`)
  lives on the role instances, the statement on the claim, the trace on the
  witness.
- **Verifier duals** — every Stage's `VerifierStage` role sits beside its
  prover twin (`quotient/verifier.py`, `opening/verifier.py`), consuming the
  same claims over the same transcript schedule; the composite
  `InnerVerifier` ([`../zisk_zorch/verifier.py`](../zisk_zorch/verifier.py))
  mirrors `InnerProver` step for step (Merkle, DEEP, FRI, and the AIR
  constraint check at the OOD point).

For coding style see [conventions.md](conventions.md); to build, test, and
benchmark it see [development.md](development.md).

## The transcript spine

```
commit_trace(trace)           -> root₁     transcript.put(root₁)
                                            alpha = transcript.get_field()   (powers → constraint fold)
quotient_from_constraints(…)  -> Q         commit Q → rootQ, transcript.put(rootQ)
deep_fri_polynomial(ctx)      -> fri_pol   ← DEEP
fri.prove(fri_pol, …)         -> layers    fold betas squeezed off the same transcript
sample_query_positions(…)     -> positions finalPol absorb → grind → getPermutations
prove_queries + group_proof   -> openings  every committed tree opened per query
```

The one shared `Transcript` is what makes each challenge depend on the committed
roots — the property the per-stage benchmarks cannot exercise.

## pil2 phases

The pipeline phase → module map. Phases are pil2's `genProof` vocabulary (the
grid [development.md](development.md)'s per-stage baseline measures), not
`zorch.stage` Stages: the quotient phase is the one standalone Stage, DEEP/
FRI/queries share the terminal `OpeningProver`, and the trace commit is
`OpeningProver.commit` — the opening scheme's other half, bound by the
composite through `bind_trace_commitment`.

| Phase | pil2 name | What it does | Module | Golden |
|---|---|---|---|---|
| Trace commit | `extendAndMerkelize` (`commitStage(1)`) | INTT each column, coset-7 RS encode to `N·blowup` rows in pil2 domain order, pil2 linear-hash each row to a 4-Goldilocks leaf, k-ary fold to the root in the selected hash family (Poseidon2 or the key's Poseidon1 default) | `commit/` | `lde`, `linear_hash`, `merkle_root`, `merkle_proof`, `stage1_commit`, `poseidon1_{linear_hash,stage1_commit}`, + real-program trace roots per family (`testdata/fullprogram/`, see [development.md](development.md#fixtures)) |
| Constraint ingest | — (rw-exported) | Load each ZisK chip's constraints + bus interactions from the `rw_constraints` wheel (`constraints/zisk/v1`), the same export `sp1-zorch` consumes | `constraints/` | — (pinned by the quotient's byte-match) |
| LogUp bus | `calculateWitnessSTD` (`std_sum`) | Combine each bus tuple into one cubic denominator, prefix-sum the row-local terms into the committed `gsum` column, export its last row as the airgroup `gsum_result` | `logup/` | `gsum` |
| Quotient | `calculateQuotientPolynomial` | Fold constraints by powers of `alpha` (zorch's agnostic `constraint_eval`), divide by the inverse zerofier, commit `Q` | `quotient/` | `cexp_eval`, `zerofier_inv` |
| DEEP | `calculateFRIPolynomial` | Squeeze the OOD point `z`, open the committed polynomials there (`computeLEv`+`evmap`), absorb, squeeze `vf`, build the DEEP-ALI codeword | `deep/` | — (pinned by the LEv round-trip identity and the FRI low-degree test; a pil2 golden additionally needs the proving key's compiled `friExp` op list) |
| FRI | `FRI::fold` / `proveQueries` | Fold the codeword down the layer chain committing each layer, grind, open every tree per query | `fri/` | `fri_fold`, `fri_prove`, `fri_final`, `grinding`, `query_sample` |

Each phase's pil2 conventions — the Poseidon2 M4 choice, the NTT domain order,
the linear-hash chaining, the transcript's buffer discipline, the opening layout,
the α-power order — live in the module docstring of the code that implements
them, per [conventions.md](conventions.md).
