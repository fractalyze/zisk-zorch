# The gen_proof bridge

How ZisK proves through zisk-zorch: pil2-proofman keeps everything it
does around a basic proof (witness generation, contributions, the global
challenge, aggregation) and only the inner proof of each instance runs on
our stack. The seam is proofman's `gen_proof` wrapper: with the bridge on
it hands the instance to `zisk-zorch-bridge` instead of `gen_proof_c`.

```
proofman (Rust)  ──gen_proof──▶  zisk-zorch-bridge (Rust)  ──PJRT──▶  StableHLO artifacts (one dir per AIR)
                                   host half: transcript, challenges,          device half: every stage,
                                   query draw, proof2pointer layout             exported once per proving key
```

## Pieces

| Piece | Where | Role |
|---|---|---|
| Exporter | `zisk_zorch/export/export_air.py`, `stages.py` | Lowers each schedule stage of `Pil2InnerProver` to StableHLO bytecode; writes `<out>/<Air>_n<nBits>/*.mlirbc` + `manifest.json` (schedule facts, and every program's inputs/outputs by name, dtype, shape). |
| Python replay | `zisk_zorch/export/replay.py`, `runtime.py` | genProof's host half in Python over the artifacts. The reference the bridge mirrors step for step. |
| Bridge crate | `bridge/` | The same host half in Rust: `driver.rs` (schedule), `transcript.rs` (proofman's `fields::Transcript`), `artifact.rs` (compile + run by manifest name over `xla-pjrt`), `lib.rs` (client slots, the `gen_proof` entry), `ab.rs` (in-process A/B). `zz_prove` is the standalone byte-gate tool. |
| proofman fork | `fractalyze/pil2-proofman` branch `feat/zisk-zorch-bridge` | `proofman.rs::gen_proof` calls the bridge when `ZZ_ARTIFACTS` is set; the bridge is created before proofman sizes its GPU buffers; the A/B comparison runs after the basic phase collects pil2's proofs. |

## The byte-gates

Three links, each pinned bit for bit:

1. **Python prover ↔ artifact replay** — `zisk_zorch/export/stages_test.py`
   proves a random instance with `Pil2InnerProver` and with `replay.prove`
   over the exported artifacts and compares the flat proofs.
   Needs `ZISK_PROVING_KEY` and a GPU; `ZISK_EXPORT_AIR` picks the AIR
   (RomData, Rom for a custom commit, Mem for a witness_calc AIR are the
   ones run), `ZISK_ARTIFACTS` reuses an export.
2. **Artifact replay ↔ Rust bridge** — `python -m zisk_zorch.export.cases`
   dumps the instance, the key sections and the replay's proof;
   `zz_prove <artifacts> <case>` proves it through the crate and
   compares.
3. **pil2 ↔ bridge, in process** — `ZZ_AB=1` makes proofman prove every
   basic instance both ways and compare after collection (the run panics
   on a mismatch). This is the gate every wall-time number must pass
   first.

The Python prover itself is pinned against pil2 by the harness's capture
byte-gates (`docs/development.md`).

## Running

```bash
# once per proving key: export every basic AIR (~4 min on an RTX 5090)
FRX_PLATFORMS=cuda python -m zisk_zorch.export.export_air \
    --proving_key=$PK --air=all --out=$ARTIFACTS

# the bridge's inputs
export ZZ_ARTIFACTS=$ARTIFACTS
export XLA_PJRT_PLUGIN=<venv>/site-packages/frx_plugins/xla_cuda12/xla_cuda_plugin.so
export ZZ_LOG=1               # per-instance timing on stderr
export ZZ_CLIENTS=3           # PJRT clients (default 3)
export ZZ_MEMORY_FRACTION=0.5 # share of the card the clients claim before pil2 sizes its buffers
# export ZZ_AB=1              # prove through pil2 too and compare
# export ZZ_COMPILE_CACHE=dir # serialized executables (default $ARTIFACTS/.pjrt-cache)

cargo-zisk prove -e guest.elf -i input.bin -k $PK -g -y -o proof
```

`ZZ_ARTIFACTS` unset means pil2's own `gen_proof` runs: the fork is a
drop-in cargo-zisk with the bridge dormant.

## Status (2026-09-02, RTX 5090, go hello-world guest)

`cargo-zisk prove -g -y` through the bridge completes and its final proof
verifies. All 11 basic instances (Rom, Main, Mem, InputData, RomData,
MemAlign, BinaryExtension, Binary, Arith, both virtual tables) are
byte-identical to native pil2's, compared as per-instance proof dumps
(`ZZ_DUMP_PROOFS`) from a native run and a bridge run of the same guest.
The in-process `ZZ_AB=1` variant reproduces the same verdict per instance
but still crashes once the card fills; the dump comparison is the gate to
quote.

| | native (1 stream) | native (3 streams) | bridge, first run | bridge, warm |
|---|---|---|---|---|
| `cargo-zisk prove` wall | 10.3 s | 16.8 s | 28.0 s | 21.6–23.0 s |
| proofman init (both stacks) | ~8 s | ~9 s | ~9 s | ~5–9 s |
| per-instance prove, warm | pil2 `STARK_GPU_PROOF` 0.07–0.27 s | | | RomData 0.15 s, Rom 0.37 s, Main 0.89 s |
| fixed sections per AIR (const LDE + tree) | precomputed on disk | | | 0–1.6 s (88-column tables at 1.5 s) |
| executable load per AIR | — | | 2.3–3 s, overlapped | hidden under init |

The bridge started at 80.7 s. Where the 59 s went, from a per-program
profile (`ZZ_LOG=2`) and `perf` of a cached load:

- **Executable loads** were 5.6 s per AIR and every instance paid one,
  serially. Half of it was XLA re-formatting the Python stack frames jit
  had left as MLIR locations on every op (`SourceLocationVisitor` in the
  profile): the exporter now strips debug info. A third was 30–70 MB of
  literals per commit program: XLA constant-folds the coset power series
  from its scalar seed, so every LDE-bearing program carried a
  2^nBitsExt table (and the fold its own); the seeds now cross an
  optimization barrier. The rest is XLA rebuilding the optimized HLO,
  which stays. Loads now run six in parallel and start at bridge
  creation for the AIRs the previous run used (`.last-used` beside the
  artifacts), so they finish under proofman's own initialization.
- **The proofman worker was blocked.** pil2's GPU `gen_proof` returns
  after enqueueing; ours ran the whole prove on proofman's single worker,
  so recursion witnesses queued behind it (one waited 33 s). The bridge
  now copies the instance out and proves on its own thread, firing the
  completion callback itself, exactly pil2's contract.

What remains above native is the bridge's own GPU time: about 10 s for
this guest's 11 instances (fixed sections ~5 s, proves ~5 s) against
pil2's ~2 s. That is zisk-zorch's per-stage kernel performance, the
subject of the baseline in `docs/development.md`, not the bridge. The
block-sized workload (the zec-reth example needs the ASM emulator with
hints; it starts under it but the guest exits early) is the open item
for the wall-clock comparison the issue asks for.

Facts the gate surfaced, all now handled by the bridge:

- **Packed traces.** For AIRs carrying `witness_bits` hints the witness
  library bit-packs rows (`num_packed_words` per row, bits per column);
  pil2 unpacks on the device. The bridge unpacks on the host from the
  packing proofman registers (`set_packed_info`). The virtual tables are
  not packed, which is why they matched before this was found.
- **Custom-commit fixed file.** The `_gpu.bin` proofman hands over is a
  32-byte root, then the base section, extended section and tree in the
  prover's tiled device layout (256x4 column-major tiles); the bridge
  untiles the base section and recomputes the rest.
- **Out-of-range words.** A trace holds raw machine words, some above the
  modulus; they are reduced on the way in (pil2 reads them as residues).

## Design notes

- **Synchronous by contract.** proofman frees the instance's host trace
  right after `gen_proof` returns, so the bridge uploads and proves before
  returning and fires the completion callback itself. pil2's GPU path is
  asynchronous (proofs land through stream collectors); ours never
  touches those streams, and a streamed instance's stream is simply left
  free after its commit was collected.
- **One PJRT client per pil2 stream.** A client serializes its
  executions, so the streamed instances proofman proves concurrently get
  a client each. Slot choice: the stream's own slot for a streamed
  instance, the first idle one otherwise.
- **Memory is claimed first.** pil2 sizes its stream buffers from the GPU
  memory free at init and would take the whole card, so the bridge is
  created before that and its clients preallocate `ZZ_MEMORY_FRACTION`
  of the card between them (unset: allocate on demand, which only works
  when pil2 leaves room).
- **The trace is re-uploaded.** pil2 keeps a tiled device copy of the
  trace from `commit_witness`; adopting it would tie the bridge to
  pil2's layout. Re-uploading costs a host-to-device copy per instance.
- **A `uint64` boundary.** Every field-typed program input and output is
  exported as plain `uint64` words (a cubic element as three limbs on a
  trailing axis) with a free bitcast inside the trace: the PJRT C API's
  host-buffer entry converts only XLA's standard element types, so a
  field-typed parameter cannot be fed from Rust. The manifest describes
  field data that way; the bridge moves words and never sees a field type.
- **Scalars ride packed** exactly as pil2 dumps them (one word for a
  stage-1 value, three for a later one); the stage-2 hints rewrite the
  air values inside the `logup` program and every later stage reads
  those, as pil2 does.
- **Memory budget (RTX 5090, 32 GB).** pil2 keeps at least one basic
  stream (7.85 GB) plus its constant areas (5 GB) even with the bridge on,
  since `commit_witness` still runs there. A bridge client holds an AIR's
  fixed sections resident (the extended constants, their tree, the base
  constants the stage-2 hints read: 4.6 GB for a table AIR with 88
  constant columns) plus one prove's working set (3-6 GB), so a client
  needs 8-10 GB and this card fits ONE (`ZZ_CLIENTS=1
  ZZ_MEMORY_FRACTION=0.55`). Two clients at 25% each ran out of memory on
  `VirtualTableZisk0`. Trimming the resident set (re-upload the base
  constants per prove, drop digest layers after the openings) is the way
  to a second client here; a larger card needs nothing.
- **Compile cost.** Compiling an AIR's programs takes minutes
  (RomData: 245 s for 34 programs on an RTX 5090), so the bridge keeps
  the serialized executables on disk and a later client loads them in
  2–3 s. The cache is keyed by bytecode hash and is plugin-version
  specific: drop it with the plugin. `zz_prove --warm` fills it.
- **Loads never overlap a prove.** A deserialization on a client while
  one of its executions is in flight wedges both; deserializations
  alongside each other are fine. The client gate admits loads together
  and a prove alone, a pending prove holds new loads back, and the
  background preload of the whole key (`ZZ_PRELOAD=all`) stops once
  proving starts; requested AIRs still load on demand.
