# zisk-zorch-bridge

pil2-proofman's `gen_proof` over zisk-zorch's exported StableHLO artifacts:
the host half of genProof in Rust (transcript, challenges, query draw, the
`proof2pointer` layout), the device half compiled from each AIR's
manifest through PJRT. Design, byte-gates and the run recipe live in
[`docs/bridge.md`](../docs/bridge.md).

## Build

```bash
# needs XLA_PJRT_PLUGIN (see docs/bridge.md) and clang/libclang for
# xla-pjrt's bindgen
cargo build --release --features standalone      # zz_prove, the byte-gate tool
```

`standalone` builds proofman's `fields` crate with its pure-Rust field
arithmetic; inside a proofman build (the crate's real consumer) leave the
feature off so `fields` stays the copy proofman uses.

Until the xla-pjrt session-options branch is merged, build against a
working copy through `.cargo/config.toml` (gitignored):

```toml
[patch."https://github.com/fractalyze/xla-pjrt"]
xla-pjrt = { path = "/path/to/xla-pjrt" }
```

## Byte-gate

```bash
FRX_PLATFORMS=cuda python -m zisk_zorch.export.cases \
    --proving_key=$PK --artifacts=$ARTIFACTS --air=RomData --out=/tmp/case
./target/release/zz_prove $ARTIFACTS /tmp/case --repeat 3
```

prints `byte-identical: N words` when the crate reproduces the Python
replay's proof, and the warm prove times.

```bash
./target/release/zz_prove --warm $ARTIFACTS            # compile + cache every AIR
./target/release/zz_prove --warm $ARTIFACTS Main_n22   # or just these
```

fills the executable cache (`$ARTIFACTS/.pjrt-cache`, or `ZZ_COMPILE_CACHE`)
so a prove finds its programs compiled: minutes per AIR the first time,
seconds after.
