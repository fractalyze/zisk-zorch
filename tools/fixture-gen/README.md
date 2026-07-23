# `tools/fixture-gen/` — pil2-reference byte-match fixture generator

Reproducibly (re)generates the golden vectors under
`zisk_zorch/**/testdata/golden/` from the **pinned pil2-proofman reference**, so
every committed golden has a single authoritative provenance. A single Rust crate
(`zisk-zorch-fixture-gen`) links pil2-proofman's own `fields` crate — the same
code the ZisK prover delegates to — runs the reference computations in-process,
and writes the JSON fixtures directly. No Python convert step, no scattered debug
dumps.

## Pin

pil2-proofman [`v1.0.0-alpha`](https://github.com/0xPolygonHermez/pil2-proofman/tree/v1.0.0-alpha)'s
`fields` crate, `git`-pinned by tag in [`Cargo.toml`](Cargo.toml) with
`features = ["verify"]` (the reference Poseidon2, linear hash, INTT, Merkle,
transcript, and cubic-extension arithmetic). The x86_64 path deliberately does
**not** build the pil2-stark C++ FFI — this is pure reference field code, so the
generator needs no CUDA and no GPU.

Determinism: every random input is drawn from a fixed `splitmix64` seed, so the
fixtures are byte-identical run to run. Regeneration is a no-op unless the pin
changes — **a clean `git status` after regenerating in place is the byte-match.**

For local iteration against a sibling pil2-proofman checkout, swap the `fields`
dependency in `Cargo.toml` from the `git` tag to a `path` dep temporarily.

## Recipe (cargo, outside Bazel; CPU only)

```sh
cd tools/fixture-gen
cargo run --release
```

Argless: one run regenerates **all** fixtures in place. Each write prints
`wrote <path>`; the paths are anchored on `CARGO_MANIFEST_DIR`, so the run writes
the same files from any working directory (never CWD-relative).

## What it emits

16 golden fixtures, one per byte-matched primitive:

| module | fixtures |
|---|---|
| `poseidon2/` | `permutation` |
| `commit/` | `linear_hash`, `merkle_root`, `merkle_proof`, `lde`, `stage1_commit` |
| `transcript/` | `transcript` |
| `quotient/` | `zerofier_inv`, `gsum`, `cexp_eval` |
| `fri/` | `fri_fold`, `fri_prove`, `fri_final`, `query_sample`, `grinding` |
| `evals/` | `compute_lev` |

## Provenance notes (load-bearing)

- **Encoding.** Field elements are emitted as **canonical-u64 decimal strings**,
  not JSON numbers — a JSON number cannot carry a 64-bit Goldilocks value exactly.
  The Python golden loader (`zisk_zorch/golden.py`, `u64` / `u64x3`) reads them
  back losslessly.
- **`cexp_eval` is anchored to real pil2 dumps, not synthetic input.** The three
  constraint captures it evaluates —
  `zisk_zorch/quotient/testdata/{memalign_readbyte,binary,arith}_cexp.json` — are
  `include_str!`'d from committed pil2 dumps and run through the reference
  evaluator, so the quotient golden reflects a real AIR's constraint expression,
  not a proxy.
- **Anchoring.** `write()` joins `CARGO_MANIFEST_DIR/../..` before the fixture
  path, so a `cargo run` from any directory lands the files in the tree (this is
  why the generator is safe to invoke from Bazel wrappers or scripts).
