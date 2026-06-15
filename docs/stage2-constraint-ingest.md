# Stage-2 constraint/interaction ingestion

Stage-2 (constraint evaluation on the extended domain, the grand-sum / bus
argument, and the DEEP quotient) needs each ZisK chip's AIR **constraints** and
**lookup/permutation (bus) interactions**. zisk-zorch does not parse ZisK's
native pilout for these — it **ingests riscv-witness's exported
`constraints/zisk/v1`** through the shared `rw_constraints` package, the same
way `sp1-zorch` consumes `constraints/sp1`. The byte-match against
pil2-proofman is what verifies rw's authored constraints+interactions (rw's
per-chip CPU test cannot — interactions are CPU-erased there).

This page covers the ingestion seam (`zisk_zorch/constraints/`). The
constraint-eval / quotient byte-match that consumes it is a later slice
(see "What stage-2 consumes next").

## The loader

```
rw-constraints wheel  (bundles data/constraints/zisk/v1: manifest.json + 15 chip *.py)
  │  ConstraintRegistry(_constraints_root()).load("zisk", "v1", …)
  ▼
dict[name, rw_constraints.Chip]   # arith, binary, mem, keccak, sha256, main, …
  │  Chip.eval_constraints(trace[, pv]) -> (N, K) violations
  │  Chip.get_sends()/get_receives() -> typed Interaction (VirtualPairCol tuples)
  ▼
stage-2 constraint-eval + grand-sum (next slice)
```

[`../zisk_zorch/constraints/chip_loader.py`](../zisk_zorch/constraints/chip_loader.py)
exposes `load_zisk_chips(version, chip_names)`. Unlike sp1-zorch's loader there
is no name-mapping seam — the rw manifest's ZisK chip names are already the names
stage-2 uses, so a chip is addressed by its manifest name directly.

## The conventions that make ingestion correct

- **Field dtype — both constraints and interactions are Goldilocks.** The chip
  modules reference an unbound `FIELD_DTYPE` that the registry injects per dtype.
  Constraints get `goldilocks_mont`; interactions **also** get `goldilocks_mont`
  — the registry default `jnp.uint32` is SP1-specific (SP1 interaction code is
  bitwise), whereas ZisK's `*_interaction` functions are pure field arithmetic,
  so the bus tuples are Goldilocks-valued.
- **JAX x64 must be enabled to evaluate chip code.** rw's exported code
  materializes field constants via `jnp.full(..., dtype=jnp.uint64).view(
  FIELD_DTYPE)`. With x64 off, `jnp.uint64` truncates to 4 bytes and the view
  fails — the same u64 trap the trace-commit path sidesteps by constructing in
  numpy first (see [`conventions.md`](conventions.md)). Constraint/interaction
  *evaluation* therefore runs under `jax_enable_x64`.
- **Bazel runfiles need a tree copy.** Runfiles expose the wheel as a per-file
  symlink farm, which trips the registry's containment check (each chip file
  must `resolve()` inside its version dir). `_constraints_root()` materializes
  one symlink-following copy per process; plain pip installs skip it. Drop once
  [riscv-witness#1580](https://github.com/fractalyze/riscv-witness/issues/1580)
  makes the check runfiles-safe.
- **The wheel must bundle the export.** The `rw-constraints` wheel historically
  packaged only `sp1/*`; bundling `constraints/zisk/v1` was added in
  riscv-witness#1889 (multi-zkvm) / #1891 (main). Dev wheels are cut only from
  `main`, so the consumable pin comes from the `main`-side change.

## The interaction model (what the grand-sum will consume)

Each interaction is a typed `Interaction` built from the manifest:
`values: tuple[VirtualPairCol]`, a `multiplicity: VirtualPairCol`, `kind`
(= the native ZisK bus id), and `is_send`. A `VirtualPairCol` is the affine
(degree ≤ 2) form `constant + Σ wᵢ·colᵢ + Σ wⱼ·col_a·col_b`; `apply_batch(
preprocessed_trace, main_trace)` materializes it at the trace dtype. Native
`assumes`→send, `proves`→receive; bus ids per
`riscv-witness/docs/zisk/conventions/interaction-bus-mapping.md`
(ARITH_TABLE 331, ARITH_RANGE_TABLE 330, BINARY_TABLE 125, OPERATION_BUS 5000).

## What stage-2 consumes next

- **Constraint evaluation + DEEP quotient**: fold each chip's
  `eval_constraints` output by powers of the stage-`nStages+1` challenge into the
  composite constraint column on the extended (coset, blowup ≥ max constraint
  degree) domain, divide by the vanishing polynomial `Z_H`, commit the quotient.
  Mirrors pil2's `calculateQuotientPolynomial`
  ([gen_proof.hpp](https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/gen_proof.hpp#L157-L178)).
  The α-fold itself is **not** ZisK-specific — reuse zorch's agnostic
  `constraint_eval(eval_fn, trace, alpha)` (the fused `zorch.constraint_eval`
  composite sp1-zorch's zerocheck is built on); zisk-zorch only adds the
  coset-extension + `Z_H` division + commit wrapper. **Byte-match trap:** the
  α-power order must follow pil2's eSTARK convention (proving-key
  pilout / `expressionsinfo`, incl. `imPols`), NOT rw's `constraint_order`
  (which is SP1 `eval_block` / zerocheck indexing).
- **Grand-sum / bus (LogUp) witness**: build the running grand-sum column from
  the typed interactions under the stage-2 challenge, commit it. Mirrors pil2's
  `calculateWitnessSTD` / `gsum_col`.
- These unblock the remaining FRI-into-full-STARK wiring (`calculateFRIPolynomial`
  = the random linear combination `f`, and `proveQueries` over the *stage* trees,
  which today only opens the FRI-layer trees).
