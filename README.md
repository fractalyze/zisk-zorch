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

`InnerProver` runs the inner proof end to end over one Fiat-Shamir transcript —
trace commit → quotient → DEEP → FRI. The primitives it is built from are
byte-matched against golden vectors generated from pil2-proofman v1.0.0-alpha's
`fields` crate ([`tools/fixture-gen/`](https://github.com/fractalyze/zisk-zorch/tree/main/tools/fixture-gen));
DEEP is the one phase with no golden. No phase is yet byte-matched against a real
pil2 dump, so the per-stage timings in
[`docs/development.md`](https://github.com/fractalyze/zisk-zorch/blob/main/docs/development.md)
are engineering signal, not a sealed baseline. See
[`docs/architecture.md`](https://github.com/fractalyze/zisk-zorch/blob/main/docs/architecture.md).

## Installation

**Python 3.11 on Linux x86_64, or macOS on Apple Silicon.** (`frxlib` ships a
cp311 wheel for those two platforms only — not 3.12/3.13, not Intel Macs.)

### CPU

```sh
pip install zisk-zorch
```

### GPU (CUDA 12)

```sh
pip install zisk-zorch 'frx[cuda12]' \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
```

The extra index carries the CUDA plugin wheels: `frx-cuda12-pjrt` is over PyPI's
per-file limit, and `frx-cuda12-plugin` is not published there. It is not needed
for the CPU tier.

### Verify

```sh
python -c \
    "import frx, zisk_zorch.prover; print(frx.devices()); print(zisk_zorch.__version__)"
```

`[CpuDevice(id=0)]` is the CPU tier. If you followed the GPU command and still
see it, the CUDA plugins did not take effect and everything will run on the CPU
without saying so.
Importing `zisk_zorch.prover` rather than the package is deliberate: the package
`__init__` is a docstring and a version string, so a bare `import zisk_zorch`
touches neither `frx` nor `zorch` and stays green on an install that resolved
neither.

## Development

From a git checkout, not a pip install — nothing below ships in the
distribution.

`zisk-zorch` is pure Python on frx (Field, Ring Accelerated), run against the
Fractalyze [xla](https://github.com/fractalyze/xla) fork's PJRT plugin (the
`frx-cuda12` wheels), built with Bazel (bzlmod). It consumes `zorch` as a
dev-release wheel from the Fractalyze index, pinned in
[`requirements.in`](https://github.com/fractalyze/zisk-zorch/blob/main/requirements.in),
so `frx` and `zk_dtypes` resolve once here. Those pins are the development set,
not the packaged dependency set — a release resolves from PyPI via
[`pyproject.toml`](https://github.com/fractalyze/zisk-zorch/blob/main/pyproject.toml).

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

See [`docs/`](https://github.com/fractalyze/zisk-zorch/blob/main/docs/README.md) —
the [architecture](https://github.com/fractalyze/zisk-zorch/blob/main/docs/architecture.md)
(the inner proof as composite Stage roles over one transcript, plus the pil2
vocabulary they
mirror), the [development guide](https://github.com/fractalyze/zisk-zorch/blob/main/docs/development.md)
(environment, testing, fixtures, CI, and the per-stage pil2 baseline), and the
[conventions](https://github.com/fractalyze/zisk-zorch/blob/main/docs/conventions.md).

Install the git hooks with both stages named. Plain `pre-commit install` wires
only the `pre-commit` stage, which leaves the commit-message linter inactive —
a malformed commit message then sails through to CI:

```sh
pre-commit install --install-hooks --hook-type pre-commit --hook-type commit-msg
```

Commit messages follow [Conventional Commits](https://www.conventionalcommits.org):
a valid type, a lowercase summary with no trailing period, a header of at most
80 characters, and a body on everything but `docs`. The scope is the package the
change lives in — `commit`, `constraints`, `deep`, `evals`, `fri`, `logup`,
`poseidon2`, `quotient`, `transcript` — or `prover`, `golden`, `bench`,
`release` for the modules directly under `zisk_zorch/`. A change spanning
several takes no scope.
The same linter runs in CI over every commit in a pull request and over the PR
title.

## License

Licensed under the Apache License, Version 2.0 (see
[LICENSE](https://github.com/fractalyze/zisk-zorch/blob/main/LICENSE)).
