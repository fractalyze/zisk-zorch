# zisk-zorch

A lean **ZisK prover** built on [`zorch`](https://github.com/fractalyze/zorch)'s
scheme-agnostic SNARK building blocks. `zorch` provides the reusable pieces
(hashing, Merkle commitment, Reed-Solomon LDE, transcript, …); `zisk-zorch`
adds only the ZisK-specific glue on top — the pil2-stark Poseidon2-Goldilocks
parameters, the pil2 transcript and linear-hash conventions, and the
byte-match against the [pil2-proofman](https://github.com/0xPolygonHermez/pil2-proofman)
reference prover that ZisK uses.

```text
frx  ──▶  zorch (scheme-/zkVM-agnostic blocks)  ──▶  zisk-zorch (ZisK / pil2-stark glue)
```

ZisK proves with Polygon's eSTARK (pil2-stark) — a FRI-based STARK over
Goldilocks. None of that scheme-specific knowledge belongs in `zorch` (its hard
rule), so it lives here.

## Status

`prove_inner` runs the inner proof end to end over one Fiat-Shamir transcript —
trace commit → quotient → DEEP → FRI. The primitives it is built from are
byte-matched against golden vectors generated from pil2-proofman v1.0.0-alpha's
`fields` crate ([`tools/fixture-gen/`](tools/fixture-gen/)); DEEP is the one
stage with no golden. No stage is yet byte-matched against a real pil2 dump, so
the per-stage timings in [`docs/development.md`](docs/development.md) are
engineering signal, not a sealed baseline. See
[`docs/architecture.md`](docs/architecture.md).

## Development

`zisk-zorch` is pure Python on frx (Field, Ring Accelerated), run against the
Fractalyze [xla](https://github.com/fractalyze/xla) fork's PJRT plugin (the
`frx-cuda12` wheels), built with Bazel (bzlmod). It consumes `zorch` as a
dev-release wheel from the Fractalyze index, pinned in
[`requirements.in`](requirements.in), so `frx` and `zk_dtypes` resolve once here.

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

**Dev against a local `zorch` checkout** instead of the pinned wheel — create
`.bazelrc.user` (gitignored):

```
common --override_module=zorch=/abs/path/to/your/zorch/checkout
```

Run the tests (CPU is the default for determinism):

```sh
bazel test //...
```

## Documentation

See [`docs/`](docs/README.md) — the [architecture](docs/architecture.md) (the
inner proof as stages over one transcript, plus the pil2 vocabulary they
mirror), the [development guide](docs/development.md) (environment, testing,
fixtures, CI, and the per-stage pil2 baseline), and the
[conventions](docs/conventions.md).

## License

Licensed under the Apache License, Version 2.0 (see [LICENSE](LICENSE)).
