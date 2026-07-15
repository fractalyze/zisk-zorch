# Architecture: the inner proof over one transcript

`zisk_zorch/prover.py` (`prove_inner`) runs the inner-proof stages in
pil2-proofman's `genProof` order over a single Fiat-Shamir `Transcript`. Each
stage below is a slice of that spine; every primitive that mirrors pil2-stark is
byte-matched against pil2-proofman v1.0.0-alpha's `fields` crate via
[`../tools/fixture-gen/`](../tools/fixture-gen/).

For coding style see [conventions.md](conventions.md); for how to build, test,
and benchmark it see [development.md](development.md).

## The transcript spine

```
commit_trace(trace)           -> root₁     transcript.put(root₁)
                                            alpha = transcript.get_field()   (powers → constraint fold)
quotient_from_constraints(…)  -> Q         commit Q → rootQ, transcript.put(rootQ)
deep_fri_polynomial(ctx)      -> fri_pol   ← DEEP stage (zisk_zorch/deep/)
fri.prove(fri_pol, …)         -> layers    fold betas squeezed off the same transcript
sample_query_positions(…)     -> positions finalPol absorb → grind → getPermutations
prove_queries + group_proof   -> openings  every committed tree opened per query
```

The single shared `Transcript` is what makes each challenge depend on the
committed roots — the property the isolated benchmarks cannot exercise.

## Stage 1 — trace commit

pil2-stark's `extendAndMerkelize` (`commitStage(1)`): commit the execution-trace
columns before any challenge is drawn.

```
trace (N x n_cols, Goldilocks evals on the order-N subgroup)
  │  INTT per column                      zorch native NTT
  ▼
coefficients
  │  coset-7 RS encode, blowup 2^(nBitsExt-nBits)   zorch ReedSolomon
  ▼
extended trace (N·blowup x n_cols, rows in pil2 domain order)
  │  pil2 linear hash per row             zisk_zorch.commit.linear_hash
  ▼
leaf digests (4 Goldilocks each)
  │  k-ary Poseidon2 fold (arity 2/3/4 → width 8/12/16)   zorch MerkleTree
  ▼
root (4 Goldilocks) ──▶ transcript.put(root)   zisk_zorch.transcript
```

### The conventions that make it byte-identical

- **Poseidon2 parameters** ([`../zisk_zorch/poseidon2/goldilocks.py`](../zisk_zorch/poseidon2/goldilocks.py)):
  pil2's external M4 is the HorizenLabs reference matrix, NOT the Plonky3 one
  zorch defaults to, so every width passes its matrix explicitly. zorch#264
  carries it as an `external_m4` marker, which zkx#676 taught the compiler to
  apply via multiply-free add-chains, so the block-structured widths (8/12/16)
  lower to the dedicated `zorch.poseidon2` emitter. Width 4's plain single-block
  M4 is not marker-carried and stays on the generic fused region.
- **NTT domain order** ([`../zisk_zorch/commit/trace_commit.py`](../zisk_zorch/commit/trace_commit.py)):
  zk_dtypes' Goldilocks two-adic generator is Plonky3's; pil2's `W[32]` differs
  (`pil2 = zk^4168946053`). `extend` re-indexes rows into and out of the native
  NTT so the extended matrix lands in pil2's row order. A root-parameterized
  native NTT is the gather-free follow-up.
- **Leaf hashing** ([`../zisk_zorch/commit/linear_hash.py`](../zisk_zorch/commit/linear_hash.py)):
  pil2's chained linear hash (zero-padded blocks, capacity chaining) is NOT
  zorch's padding-free sponge, so it lives here and duck-types the leaf-hasher
  seam.
- **Tree**: one width-`4*arity` permutation hashes both leaves and nodes;
  incomplete levels complete with zero digests (zorch's k-ary MerkleTree stores
  the padded layers).
- **Transcript** ([`../zisk_zorch/transcript/transcript.py`](../zisk_zorch/transcript/transcript.py)):
  pil2's pending/out buffer discipline; 3-limb cubic challenges; 63-bit query
  index packing.
- **Openings** ([`../zisk_zorch/commit/openings.py`](../zisk_zorch/commit/openings.py)):
  `MerkleTreeGL::getGroupProof`'s flat `[row..., mp levels...]` array. zorch's
  k-ary `open` already packs siblings in pil2's mp order (group order, own slot
  skipped), so serialization is flatten-and-concatenate; the `merkle_proof`
  golden pins both directions.

## Stage 2 — constraints and interactions

Stage-2 needs each ZisK chip's AIR **constraints** and **lookup/permutation
(bus) interactions**. zisk-zorch does not parse ZisK's native pilout for these —
it **ingests riscv-witness's exported `constraints/zisk/v1`** through the shared
`rw_constraints` package, the same way `sp1-zorch` consumes `constraints/sp1`.
The byte-match against pil2-proofman is what verifies rw's authored
constraints+interactions (rw's per-chip CPU test cannot — interactions are
CPU-erased there).

```
rw-constraints wheel  (bundles data/constraints/zisk/v1: manifest.json + 15 chip *.py)
  │  ConstraintRegistry(_constraints_root()).load("zisk", "v1", …)
  ▼
dict[name, rw_constraints.Chip]   # arith, binary, mem, keccak, sha256, main, …
  │  Chip.eval_constraints(trace[, pv]) -> (N, K) violations
  │  Chip.get_sends()/get_receives() -> typed Interaction (VirtualPairCol tuples)
  ▼
constraint-eval + alpha-fold + Z_H division -> quotient commit
```

[`../zisk_zorch/constraints/chip_loader.py`](../zisk_zorch/constraints/chip_loader.py)
exposes `load_zisk_chips(version, chip_names)`. Unlike sp1-zorch's loader there is
no name-mapping seam — the rw manifest's ZisK chip names are already the names
stage-2 uses.

### The conventions that make ingestion correct

- **Field dtype — both constraints and interactions are `goldilocks`.** The chip
  modules reference an unbound `FIELD_DTYPE` the registry injects. The registry
  default `jnp.uint32` is SP1-specific (SP1 interaction code is bitwise), whereas
  ZisK's `*_interaction` functions are pure field arithmetic, so the bus tuples
  are Goldilocks-valued too.
- **JAX x64 must be enabled to evaluate chip code.** rw's exported code
  materializes field constants via `jnp.full(..., dtype=jnp.uint64).view(
  FIELD_DTYPE)`. With x64 off, `jnp.uint64` truncates to 4 bytes and the view
  fails.
- **Bazel runfiles need a tree copy.** Runfiles expose the wheel as a per-file
  symlink farm, which trips the registry's containment check (each chip file must
  `resolve()` inside its version dir). `_constraints_root()` materializes one
  symlink-following copy per process; plain pip installs skip it. Drop once
  [riscv-witness#1580](https://github.com/fractalyze/riscv-witness/issues/1580)
  makes the check runfiles-safe.
- **The wheel must bundle the export.** The `rw-constraints` wheel historically
  packaged only `sp1/*`; bundling `constraints/zisk/v1` was added in
  riscv-witness#1889 (multi-zkvm) / #1891 (main). Dev wheels are cut only from
  `main`, so the consumable pin comes from the `main`-side change.

### The interaction model

Each interaction is a typed `Interaction` built from the manifest:
`values: tuple[VirtualPairCol]`, a `multiplicity: VirtualPairCol`, `kind` (= the
native ZisK bus id), and `is_send`. A `VirtualPairCol` is the affine (degree ≤ 2)
form `constant + Σ wᵢ·colᵢ + Σ wⱼ·col_a·col_b`; `apply_batch(preprocessed_trace,
main_trace)` materializes it at the trace dtype. Native `assumes`→send,
`proves`→receive; bus ids per
`riscv-witness/docs/zisk/conventions/interaction-bus-mapping.md` (ARITH_TABLE 331,
ARITH_RANGE_TABLE 330, BINARY_TABLE 125, OPERATION_BUS 5000).

**Byte-match trap:** the α-power order must follow pil2's eSTARK convention
(proving-key pilout / `expressionsinfo`, incl. `imPols`), NOT rw's
`constraint_order` (which is SP1 `eval_block` / zerocheck indexing). The α-fold
itself is not ZisK-specific — it reuses zorch's agnostic `constraint_eval`; this
repo adds only the coset extension, the `Z_H` division, and the commit wrapper.

## DEEP — the FRI polynomial

pil2's `calculateFRIPolynomial` squeezes the out-of-domain point `z`, evaluates
the committed polynomials there, absorbs the openings, squeezes a batching
challenge, and builds the FRI codeword. It is the default `fri_polynomial_fn`.

- [`../zisk_zorch/deep/opening.py`](../zisk_zorch/deep/opening.py) — pil2
  `computeLEv` + `evmap`: `LEv[k] = INTT((z·g^p·shift⁻¹)^k)` are the Lagrange
  weights with `Σ_k LEv[k]·p(shift·g^k) = p(z·g^p)`; `open_columns` subsamples
  each extended column to the base coset and dots. Pinned by that round-trip
  identity (`opening_test.py`), no pil2 dump needed.
- [`../zisk_zorch/deep/fri_polynomial.py`](../zisk_zorch/deep/fri_polynomial.py) —
  pil2 `calculateFRIPolynomial`: the DEEP-ALI batched quotient
  `f(x) = Σ_m vf^m·(p_m(x) − p_m(ξ))/(x − ξ)`. Verified by the FRI low-degree
  property (`fri_polynomial_test.py`): a correct opening folds low, a wrong one
  does not. Its coset is built host-side and passed in as an input — building it
  inside the jit fed a cubic reciprocal that crashed the NVPTX compiler (#67).

**Byte-match boundary.** `deep_composition` implements the *generic* DEEP-ALI
formula. pil2's `friExp` in a real proving key bakes in which columns are
batched, their order, and each one's challenge power (`expressions.bin`) —
matching a specific AIR byte-for-byte needs that compiled op list (the machinery
`cexp_ref` already interprets for the quotient) plus a pil2 golden. That is the
next slice, and it is why DEEP is the one stage with no golden.
`quotient_as_fri_polynomial` remains as a trivial FRI-over-quotient fallback.

## Status

`prove_inner`'s tests are shape / determinism / property checks, not golden
byte-matches: a golden inner proof needs the AIR-specific `friExp` op list. When
that lands, add a golden vector and assert `prove_inner` reproduces it end to
end.
