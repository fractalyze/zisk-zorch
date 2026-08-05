# `tools/rw-fixture-gen/` — the native-ZisK byte-parity oracle

Drives **native ZisK state machines** to emit the trace-parity fixtures and
goldens the riscv-witness zisk gates consume (`zisk_trace_dump_test`, the
full-program parity suites, the perf-bench ZROM/ZPIN). The chip-side sibling of
[`../fixture-gen/`](../fixture-gen/README.md): where fixture-gen pins
pil2-proofman's `fields` crate for the prover primitives, this crate pins the
ZisK state machines themselves — a standalone cargo crate, no fork workspace to
check out.

The fork carries only what an external crate cannot: the targeted SM
re-exports (`BinaryBasicSM`, `BinaryAddSM`, …) and the direct guest programs,
all reachable from the pinned rev.

## Pin

All ZisK crates are git-pinned in [`Cargo.toml`](Cargo.toml) to one
`fractalyze/zisk` rev on the `rw/zisk-v1.0.0-alpha` line (upstream
`v1.0.0-alpha` + the re-exports and guests above); the proofman crates come
from `pil2-proofman` at tag `v1.0.0-alpha`, the same pin `../fixture-gen` uses.
Bump the rev and `cargo update` in lockstep — the riscv-witness genrules
cache-bust on the oracle rev (`--action_env=FIXTURE_BUST=<rev>`).

## Recipe (cargo, outside Bazel)

```sh
cd tools/rw-fixture-gen
cargo build --release
export ZISK_FIXTURE_GEN_BIN=$PWD/target/release/rw-fixture-gen
```

`$ZISK_FIXTURE_GEN_BIN` is the contract the riscv-witness
`requires-zisk-toolchain` genrules consume (absolute path — Bazel genrules run
in the execution root). Building the proofman deps needs the usual native
toolchain (gmp, nlohmann-json); the fixture-emitting modes additionally read
the ziskup proving key from `~/.zisk` (or `$ZISK_PROVING_KEY`) and want
`libsodium` at runtime.

## Modes

One binary, mode per flag — see `--help` for the full list and per-flag docs:

| flags | what it emits |
|---|---|
| `--chip --case` | single-opcode fixtures (binary/arith/mem families, precompiles) |
| `--elf [--inputs]` | full-program Gate-0 fixtures: one run, every single-instance op-bus chip |
| `--mem` / `--mem-align[-byte/-read-byte/-write-byte]` / `--rom-data` / `--input-data` | segmented mem-family fixtures (count→plan→expand) |
| `--binary-multi` / `--keccak-multi` | plan-driven multi-instance fixtures (#2347) |
| `--main-multi` | Main-SM per-segment native goldens (metadata-only) |
| `--emu-trace-dump` | ZROM/ZPIN for the perf-bench (proofman-free) |
| `--checkpoint-reference` | per-chunk execute-seed JSON (proofman-free) |
| `--stage1-root [--hash-family]` | native stage-1 commit root over a `--debug-trace` dump |

`--debug-trace` additionally writes the full native trace (gzipped `.npy`, the
exact `golden_sha256` preimage) next to each fixture for first-diff hunting.

## Consumers

- riscv-witness `tools/zisk/v1/zisk_fullprogram_fixture_gen`,
  `zisk_fullprogram_block_fixture_gen`, `zisk_block_rom_input_fixture_gen` —
  genrules that invoke `$ZISK_FIXTURE_GEN_BIN` and gate the outputs.
- zisk-zorch real-trace stage-1 fixtures (`--debug-trace` + `--stage1-root`).
