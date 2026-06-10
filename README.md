# zisk-zorch

A lean **ZisK prover** built on [`zorch`](https://github.com/fractalyze/zorch)'s
scheme-agnostic SNARK building blocks. `zorch` provides the reusable pieces
(hashing, Merkle commitment, Reed-Solomon LDE, transcript, …); `zisk-zorch`
adds only the ZisK-specific glue on top — the pil2-stark Poseidon2-Goldilocks
parameters, the pil2 transcript and linear-hash conventions, and the
byte-match against the [pil2-proofman](https://github.com/0xPolygonHermez/pil2-proofman)
reference prover that ZisK uses.

```
JAX  ──▶  zorch (scheme-/zkVM-agnostic blocks)  ──▶  zisk-zorch (ZisK / pil2-stark glue)
```

Why a separate repo: ZisK proves with Polygon's eSTARK (pil2-stark) — a
FRI-based STARK over the Goldilocks field. None of that scheme-specific
knowledge belongs in `zorch` (its hard rule); building directly on `zorch`'s
blocks keeps this prover small and gives a focused target to grow ZisK glue
and benchmark against the pil2-stark CUDA reference.

## Status

Early bootstrap. First slice: the **stage-1 trace commit** (≈ pil2-stark's
`extendAndMerkelize`) — coset-7 NTT LDE onto the extended domain, pil2
linear-hash row leaves, k-ary Poseidon2 Merkle tree, and the pil2 transcript —
byte-matched against golden vectors generated from pil2-proofman v0.18.0's
`fields` crate (see [`golden/`](golden/)).

## The scheme (what ZisK actually runs)

ZisK delegates proving to pil2-proofman (eSTARK / pil2-stark). The constants
that pin this repo's glue, all from
[pil2-proofman v0.18.0](https://github.com/0xPolygonHermez/pil2-proofman/tree/v0.18.0):

- **Field**: Goldilocks (2^64 − 2^32 + 1); FRI challenges in the cubic
  extension x³ − x − 1.
- **Hash**: Poseidon2 over Goldilocks, widths 4/8/12/16, capacity always 4,
  x⁷ S-box, 4+4 full rounds, 21–22 partial rounds.
- **Merkle**: configurable arity (2/3/4 → node hash Poseidon8/12/16), rows
  leaf-hashed with pil2's chained `linear_hash`, 4-element roots.
- **LDE**: NTT with coset shift 7, blowup 2^(nBitsExt − nBits).
- **Transcript**: Poseidon2 sponge with a pending/out buffer discipline
  (not a duplex sponge — see `zisk_zorch/transcript/`).

## Development

`zisk-zorch` is pure Python on JAX + the ZKX PJRT plugin, built with Bazel
(bzlmod). It consumes `zorch` as a Bazel module, pinned in `MODULE.bazel` via
`git_override` for reproducible builds.

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

**Dev against a local `zorch` checkout** instead of the pinned commit — create
`.bazelrc.user` (gitignored):

```
common --override_module=zorch=/abs/path/to/your/zorch/checkout
```

Run the tests (CPU is the default for determinism):

```sh
bazel test //...
```

### Regenerating the golden vectors

The byte-match fixtures under `zisk_zorch/**/testdata/golden/` are produced by
the Rust harness in [`golden/`](golden/), which links the same `fields` crate
pil2-proofman v0.18.0 ships:

```sh
cd golden && cargo run --release
```

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
