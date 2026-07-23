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
reference's own `fields` crate by [`../golden/`](../golden/). Conventions:

- Goldens live in `testdata/golden/*.json` next to the test that consumes
  them, are small (KBs), and are committed.
- A golden test compares with exact equality (`fnp.array_equal`), never a
  tolerance — field elements either match or they don't.
- Regenerate with `cd golden && cargo run --release`; the harness is
  deterministic (fixed seeds), so a regeneration must be a no-op unless the
  reference pin changed.
- Never let an unordered container's iteration order feed the RNG stream (or
  anything serialized) in the generator: Rust's `HashMap` seeds per-process, so
  drawing randoms while iterating one yields non-reproducible goldens. Use
  `BTreeMap` / sorted keys. The drift hides from tests (they reload the file
  they just wrote) — verify with a regen-of-regen diff, not "the test passes".

## Cubic ⇄ base limbs cross host, not device

JAX has no device-side view from the `goldilocksx3_mont` cubic dtype to its
three `goldilocks_mont` base limbs (a `(N,)` cubic array cannot `reshape` to
`(N, 3)` base — the dtype is opaque, and `x64` is off). So any cubic→base or
base→cubic step (absorbing a cubic codeword into the transcript, decoding a
squeezed challenge, regrouping a FRI layer for Merkle leaves) round-trips
through host NumPy `.view`, never an on-device `bitcast`/`reshape`. This is
fine because the code that needs it — transcript, FRI prover orchestration — is
host-driven by design; the jitted hot path (LDE/NTT/hash) stays base-field.
See `transcript._canonical` and `fri/prover._cubic_to_base`.

## What lives here vs in zorch

ZisK / pil2-stark glue lives here: Poseidon2-Goldilocks round constants, the
pil2 transcript's pending/out buffer discipline, the chained linear hash, the
trace-commit pipeline and its conventions (coset shift 7, row-major leaves,
4-element roots). Anything reusable by another scheme — the Poseidon2
permutation core, k-ary Merkle folding, Reed-Solomon/NTT, the duplex
transcript — belongs in `zorch`.
