# Architecture: the inner proof over one transcript

`prove_inner` ([`../zisk_zorch/prover.py`](../zisk_zorch/prover.py)) runs the
inner-proof stages in pil2-proofman's `genProof` order over a single Fiat-Shamir
`Transcript`. This page maps the proof onto those stages, then names the
conventions that make each one byte-identical to pil2-stark. Every primitive that
mirrors pil2 is pinned against pil2-proofman v1.0.0-alpha's `fields` crate via
[`../tools/fixture-gen/`](../tools/fixture-gen/).

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

## The conventions that make it byte-identical

- **Poseidon2 parameters** ([`goldilocks.py`](../zisk_zorch/poseidon2/goldilocks.py)):
  pil2's external M4 is the HorizenLabs reference matrix, NOT the Plonky3 one
  zorch defaults to, so every width passes its matrix explicitly. zorch#264
  carries it as an `external_m4` marker, so widths 8/12/16 lower to the dedicated
  `zorch.poseidon2` emitter; width 4's plain single-block M4 is not
  marker-carried and stays on the generic fused region.
- **NTT domain order** ([`trace_commit.py`](../zisk_zorch/commit/trace_commit.py)):
  zk_dtypes' two-adic generator is Plonky3's; pil2's `W[32]` differs
  (`pil2 = zk^4168946053`). `_PIL2_GENERATOR` puts every transform on pil2's root
  — `extend` hands it to `ReedSolomon`, `fri.fold.intt` to `frx.lax.ntt`.
- **Leaf hashing** ([`linear_hash.py`](../zisk_zorch/commit/linear_hash.py)):
  pil2's chained linear hash (zero-padded blocks, capacity chaining) is NOT
  zorch's padding-free sponge, so it lives here and duck-types the leaf-hasher
  seam. One width-`4*arity` permutation hashes both leaves and nodes; incomplete
  levels complete with zero digests.
- **Transcript** ([`transcript.py`](../zisk_zorch/transcript/transcript.py)):
  pil2's pending/out buffer discipline; 3-limb cubic challenges; 63-bit query
  index packing.
- **Openings** ([`openings.py`](../zisk_zorch/commit/openings.py)):
  `MerkleTreeGL::getGroupProof`'s flat `[row..., mp levels...]` array. zorch's
  k-ary `open` already packs siblings in pil2's mp order, so serialization is
  flatten-and-concatenate.
- **Quotient leaf layout**: each cubic row commits as its 3 contiguous Goldilocks
  limbs — pil2's `FIELD_EXTENSION` memory order, matching the FRI seam.
- **α-power order** must follow pil2's eSTARK convention (proving-key
  `expressionsinfo`, incl. `imPols`), NOT rw's `constraint_order`, which is SP1
  `eval_block` / zerocheck indexing.
- **Ingest dtypes** ([`chip_loader.py`](../zisk_zorch/constraints/chip_loader.py)):
  constraints *and* interactions are `goldilocks` — the registry's `jnp.uint32`
  default is SP1-specific (its interaction code is bitwise), while ZisK's is pure
  field arithmetic. Evaluating chip code needs `jax_enable_x64`, and Bazel
  runfiles need a symlink-following tree copy
  ([riscv-witness#1580](https://github.com/fractalyze/riscv-witness/issues/1580)).

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
