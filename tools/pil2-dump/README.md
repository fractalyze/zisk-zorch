# `tools/pil2-dump/` — pil2-reference per-stage dump generator

Reproducibly (re)generates the per-stage reference dumps the `verify_*`
runnables byte-match against, by running **one real pil2-proofman `genProof`
in-process** with the pinned fork's dump hooks armed. A single Rust crate
(`zisk-zorch-pil2-dump`) drives the prove and the prover writes each stage
buffer as it computes it — no CUDA harness, no re-implementation of pil2's
kernels, nothing for a reviewer to diff against upstream by eye.

Sibling of [`../fixture-gen/`](../fixture-gen/), which pins the same project
for the *CPU* `fields`-crate golden vectors: those pin each primitive in
isolation on synthetic inputs, these pin whole stages of a real proof.
Together they are the ZisK analog of SP1's `SP1_DUMP_PHASES` phase dump, and
this crate mirrors sp1-zorch's [`tools/fixture-gen`][sp1-fg], which git-pins
`fractalyze/sp1` for the same reason.

[sp1-fg]: https://github.com/fractalyze/sp1-zorch/tree/main/tools/fixture-gen

## Pin

pil2-proofman fork [`fractalyze/pil2-proofman`](https://github.com/fractalyze/pil2-proofman)
@ `4424b7ebae571905c3518b2ceaa3deec84411010` — the `v1.0.0-alpha` base plus one
fork change: `PIL2_DUMP_DIR`-gated per-stage buffer dumps in `gen_proof.hpp`
(`pil2_dump.hpp`). `git`-pinned by `rev` in [`Cargo.toml`](Cargo.toml).

The hooks are **writes only**: with `PIL2_DUMP_DIR` unset every hook is a no-op,
and with it set the fibonacci-square proof still self-verifies — so the dumped
buffers are what an unmodified prove computes.

**Why a fork rather than upstream.** The stages this repo gates — DEEP, the FRI
fold chain, evals, the LogUp grand sum — have no entry point outside
`genProof`. pil2's Rust FFI exposes `commit_witness_c`, `gen_proof_c`, and
`stark_verify_c`, and its C++ host wrappers (`fold_inplace`,
`calculateFRIExpression`, `evmap_inplace`, …) each take a fully-built
`SetupCtx`/`StepsParams`/`MerkleTreeGL` prover context. There is no supported
way to run one stage on chosen inputs, so observing an intermediate means
proving for real and having the prover hand it out.

For local iteration against a sibling checkout, swap the `git`/`rev` deps in
`Cargo.toml` for `path` deps temporarily.

Determinism: same pin, same proving key, same inputs → byte-identical dumps.
**A clean `git status` after regenerating in place is the byte-match.**

## Recipe (cargo, outside Bazel)

The proving key and witness library are pil2 toolchain build artifacts; this
crate drives the prove, it does not build them. Build them once with pil2's own
flow (`compile-pil` → `setup` → `pil-helpers` → `cargo build -p <example>`),
then:

```sh
cd tools/pil2-dump
cargo build --release --features cpu-only

./target/release/pil2-dump \
    --proving-key   <pil2-proofman>/examples/fibonacci-square/build/provingKey \
    --witness-lib   <pil2-proofman>/target/release/libfibonacci_square.so \
    --public-inputs <pil2-proofman>/examples/fibonacci-square/src/inputs.json \
    --out           /path/to/dump
```

`--gpu` proves on CUDA instead; drop `--features cpu-only` to build that path
(its `build.rs` probes `nvidia-smi` and builds the CUDA starks lib — masking
`nvcc` out of `PATH` does not stop it).

Point the gate at the result:

```sh
bazel run //zisk_zorch:verify_inner_proof -- \
    --dump=/path/to/dump --instance=ag0_air0_inst0 \
    --starkinfo=<provingKey>/build/<air>/air/<Air>.starkinfo.json
```

The committed CI capture is one instance of this dump — fibonacci-square's
SpecifiedRanges AIR (`ag0_air2_inst5`, N=2^8), 260 KB including the proving-key
files the gate reads. To refresh it, copy that instance's `*.npy` plus the
AIR's `.starkinfo.json`, `.expressionsinfo.json`, and `.const` into
`zisk_zorch/testdata/fibsq_specifiedranges/`; a clean `git status` afterwards
is the byte-match.

## Provenance notes (load-bearing)

1. **Setups default to Poseidon1.** pil2's `--hash Poseidon2` flag is
   case-sensitive and `globalInfo.json` records the choice; a Poseidon1 key
   cannot match this repo, which models Poseidon2 only (zisk-zorch#90).
2. **Stage-1 columns fill asynchronously.** Hint-computed columns are still
   settling during `STEP_1`, so the trace is dumped *after* `commitStage(1)`;
   an earlier capture disagrees with the committed root.
3. **The quotient section is only valid post-commit.** `computeQ` writes the
   committed form during the commit, so a pre-commit capture of `q` is stale
   garbage — and `q_ext` must be read before the in-place INTT.
4. **Committed buffers are tiled.** pil2 stores committed polynomials 256×4
   column-major-within-tile (`getBufferOffset`); the loaders untile before
   comparing. LEv is tiled too.
5. **`vf2` powers are assigned by Horner** — column 0 carries the *highest*
   power — the reverse of a natural ascending `vf^m`; multi-opening DEEP folds
   `vf1` across groups in `openingPoints` order.
