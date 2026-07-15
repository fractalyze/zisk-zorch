# Coding conventions

`zisk-zorch` inherits `zorch`'s conventions (`@jit` usage, type annotations,
`_`-private naming, snake_case); this page only adds the rules specific to a
byte-match consumer repo.

## Comments carry WHY, never WHAT

Names, types, and tests already say what the code does. A comment exists to
state what the code cannot: which pil2-stark convention a line mirrors, why a
shape or ordering is load-bearing for the byte-match, which reference decision
forced a non-obvious choice.

## Pin external references

Link pil2-proofman and ZisK sources as GitHub permalinks at tag `v1.0.0-alpha`
(or a short commit SHA) with line ranges — never a branch. A branch link rots
silently; a permalink stays true to the constant it pins.

## Golden tests are the spec

Every primitive that mirrors pil2-stark (Poseidon2 permutation, linear hash,
Merkle tree, transcript, LDE) is pinned by a golden vector generated from the
reference's own `fields` crate by
[`../tools/fixture-gen/`](../tools/fixture-gen/). Conventions:

- Goldens live in `testdata/golden/*.json` next to the test that consumes
  them, are small (KBs), and are committed.
- A golden test compares with exact equality (`jnp.array_equal`), never a
  tolerance — field elements either match or they don't.
- Regenerate with `cd tools/fixture-gen && cargo run --release`; the harness is
  deterministic (fixed seeds), so a regeneration must be a no-op unless the
  reference pin changed.
- Never let an unordered container's iteration order feed the RNG stream (or
  anything serialized) in the generator: Rust's `HashMap` seeds per-process, so
  drawing randoms while iterating one yields non-reproducible goldens. Use
  `BTreeMap` / sorted keys. The drift hides from tests (they reload the file
  they just wrote) — verify with a regen-of-regen diff, not "the test passes".

## What lives here vs in zorch

ZisK / pil2-stark glue lives here: Poseidon2-Goldilocks round constants, the
pil2 transcript's pending/out buffer discipline, the chained linear hash, the
trace-commit pipeline and its conventions (coset shift 7, row-major leaves,
4-element roots). Anything reusable by another scheme — the Poseidon2
permutation core, k-ary Merkle folding, Reed-Solomon/NTT, the duplex
transcript — belongs in `zorch`.
