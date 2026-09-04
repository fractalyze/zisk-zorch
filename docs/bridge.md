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

# the bridge's inputs (the full list is the table below)
export ZZ_ARTIFACTS=$ARTIFACTS
export XLA_PJRT_PLUGIN=<venv>/site-packages/frx_plugins/xla_cuda12/xla_cuda_plugin.so
export ZZ_LOG=1

cargo-zisk prove -e guest.elf -i input.bin -k $PK -g -y -o proof
```

`ZZ_ARTIFACTS` unset means pil2's own `gen_proof` runs: the fork is a
drop-in cargo-zisk with the bridge dormant.

| variable | meaning | default |
|---|---|---|
| `ZZ_ARTIFACTS` | directory of `<Air>_n<nBits>/` exports | unset: bridge off |
| `XLA_PJRT_PLUGIN` | the frx CUDA PJRT plugin `.so` (read by xla-pjrt) | required |
| `ZZ_CLIENTS` | PJRT clients; proofman spawns this many basic-proof workers | 3 |
| `ZZ_MEMORY_FRACTION` | share of the card the clients claim up front, split evenly, before pil2 sizes its buffers | unset: allocate on demand |
| `ZZ_GPU_HEADROOM_GB` | (fork) GPU memory pil2 leaves out of its stream sizing | 0 |
| `ZZ_PRELOAD` | executables loaded at bridge creation: the previous run's AIRs (`.last-used`), `all`, or `0` | last used |
| `ZZ_PRELOAD_THREADS` | AIRs loading at once | 6 |
| `ZZ_PENDING` | proves admitted per client on the device (one running, the rest uploaded ahead) | 2 |
| `ZZ_RESIDENT_AIRS` | AIRs whose fixed sections stay on a client at once, least recently used evicted | 1 |
| `ZZ_HOST_THREADS` | threads for the host-side copies and key reads | half the cores, at most 8 |
| `ZZ_COMPILE_CACHE` | directory of serialized executables | `$ZZ_ARTIFACTS/.pjrt-cache` |
| `ZZ_LOG` | `1` per-instance timing on stderr, `2` per program; lines carry the seconds since bridge-up | off |
| `ZZ_AB` | prove through pil2 too and compare per instance | off |
| `ZZ_DUMP_PROOFS` | (fork) write every basic proof as raw words into this directory | off |
| `ZZ_DUMP_INPUTS` | write each instance as a `zz_prove` case directory under this one | off |
| `ZZ_DUMP_TRACES` | (fork) write each host trace as `gen_proof` receives it | off |

## Status (2026-09-03, RTX 5090, go hello-world guest)

`cargo-zisk prove -g -y` through the bridge completes and its final proof
verifies. All 11 basic instances (Rom, Main, Mem, InputData, RomData,
MemAlign, BinaryExtension, Binary, Arith, both virtual tables) are
byte-identical to native pil2's, compared as per-instance proof dumps
(`ZZ_DUMP_PROOFS`) from a native run and a bridge run of the same guest.
The in-process `ZZ_AB=1` variant reproduces the same verdict per instance
but still crashes once the card fills; the dump comparison is the gate to
quote.

Warm, on an otherwise idle host (a run right after another process has
churned the page cache adds 4–5 s of file reading to either stack's init;
take the second of two consecutive runs):

| | native (1 stream) | bridge, warm |
|---|---|---|
| `cargo-zisk prove` wall | 10.7 s | 21.2–21.5 s |
| proofman init | 2.9 s | 6.2 s (bridge up 0.5 s, then init beside the executable loads) |
| inner-proof leg | 3.5 s (28 proofs on 3 basic + 1 recursive streams) | 9.9–10.1 s |
| ├ proves, one client, back to back | | 9.1 s (RomData 0.4, MemAlign 0.5, InputData 0.6, VirtualTableZisk1 0.7, Arith 0.8, Mem 0.8, VirtualTableZisk0 0.9, Rom 0.9, BinaryExtension 1.1, Main 1.2, Binary 1.3) |
| ├ gaps between proves | | 0.2 s |
| └ recursion after the last basic proof | | 0.7 s |
| fixed sections, inside the proves above | precomputed on disk | 0.45 s for the two 88-column tables, ≤0.2 s otherwise (the key read ahead of the slot) |
| host copy of an instance (proofman's worker) | | 0.01–0.1 s |

Native's leg holds the same 11 basic proofs plus 17 recursion proofs
whose per-proof timers sum to 10.5 s, overlapped four-wide; ours runs the
basic proves one at a time on one client (two do not fit beside pil2's
14 GB: a table AIR's constant setup and logup are single 4.5–5.5 GiB
allocations) with the tower interleaved on pil2's stream. Inside a prove
the host no longer idles the device: Main's 1.2 s is kernel time end to
end, against pil2's 0.15 s commit plus 0.27 s proof for the same instance.

What remains above native is on the device, and a per-program profile
(frx profiler over the Python replay, kernels attributed by execution
order; pil2's own `TIMERS FOR INSTANCE` categories at `-vv`) says where.
Main, single stream on both sides:

| stage | bridge | pil2 |
|---|---|---|
| commit1 (LDE + Merkle of cm1) | 0.385 s | 0.165 s (Merkle 0.099, NTT 0.032, H2D 0.032) |
| commit2, quotient_commit, fri_commit | 0.343 s | inside its 0.444 s proof: Merkle 0.165, NTT 0.068 |
| const_setup (the constant tree) | 0.100 s | read from disk |
| logup + quotient (expressions) | 0.096 s | 0.191 s |
| evals, DEEP, folds, openings, grind | 0.024 s | small |

Expressions and the NTT passes (36 ms in the whole prove) are ahead of
pil2; the entire lag is the Merkle hashing in the commit programs, 0.83 s
against pil2's 0.36 s of Merkle plus NTT. In commit1's optimized HLO the
Poseidon linear layer is lowered as a sum of rank-1 outer products over
the whole leaf set (`state + outer(column, row)` on a
goldilocks[8388608,15] array, one full pass per term and per round): its
484 fusions write 44.5 GiB, 24 GiB of it at the leaf level, for about
4 GiB of unavoidable traffic, and XLA already runs them as CUDA graphs.
That is zorch's Poseidon and tree lowering, not the bridge nor
zisk-zorch's stages; the same HLO size is what makes each large
executable 0.5–0.7 s to load. The block-sized workload (the zec-reth
example needs the ASM emulator with hints; it starts under it but the
guest exits early) is the open item for the wall-clock comparison the
issue asks for.

Facts the gate surfaced, all now handled by the bridge:

- **Packed traces.** For AIRs carrying `witness_bits` hints the witness
  library bit-packs rows (`num_packed_words` per row, bits per column);
  pil2 unpacks on the device. The bridge unpacks on the host from the
  packing proofman registers (`set_packed_info`). The virtual tables are
  not packed, which is why they matched before this was found.
- **Custom-commit fixed file.** The `_gpu.bin` proofman hands over is a
  32-byte root, then the base section, extended section and tree in the
  prover's tiled device layout (256x4 column-major tiles); the bridge
  untiles the base section and recomputes the rest. A CPU run (`prove`
  without `-g`) writes the same file row-major and names its constants
  `.const` rather than `.const_gpu`, which is what the bridge keys the
  layout on.
- **Out-of-range words.** A trace holds raw machine words, some above the
  modulus; they are reduced on the way in (pil2 reads them as residues).

## Design notes

- **Asynchronous, as pil2's GPU path is.** The bridge copies the
  instance out first (`Bridge::take`, on proofman's worker, which never
  waits on the bridge), then proves it on a thread of its own and fires
  the completion callback itself; `gen_proof` returns at once, like
  pil2's, which enqueues and lets its stream collectors deliver. The
  instance's trace buffer stays in proofman's pool until that callback
  (the fork's `gen_proof` defers the release), so witness generation
  throttles on the pool exactly as it does natively; the bridge's own
  cap, two proves per client on the device (one running, one with its
  uploads done ahead), is taken on the prove thread. Ours never touches
  pil2's streams, and a streamed instance's stream is simply left free
  after its commit was collected. The completion side checks the proof's
  length against pil2's own size for the AIR before storing it, since
  the bridge sizes the proof from the manifest and a stale export would
  otherwise hand the aggregation a wrong-sized buffer.
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
- **Exports carry no debug info and no folded power tables.** XLA
  re-formats every op's source location on load (half of a 5.6 s load
  once), so the exporter strips them; and it constant-folds the coset
  power series from its scalar seed into a 2^nBitsExt literal per
  LDE-bearing program (30–70 MB each) unless the seed crosses an
  optimization barrier, which it now does. Loads are CPU-bound (XLA
  rebuilds the executable from its HLO, ~4.5 CPU-s per AIR); more preload
  threads slow proofman's init by as much as they gain.
- **Compile cost.** Compiling an AIR's programs takes minutes
  (RomData: 245 s for 34 programs on an RTX 5090), so the bridge keeps
  the serialized executables on disk and a later client loads them in
  2–3 s. The cache is keyed by the bytecode's hash and the plugin's
  identity (`XLA_PJRT_PLUGIN` path, size, mtime), an entry the plugin
  rejects is recompiled in place, one client compiles an entry at a
  time, and a directory that cannot be created (a read-only export)
  means compiling without a cache, not failing. `zz_prove --warm` fills
  it; a key scheme change (this one included) leaves the old entries
  unused on disk, so delete the directory when reclaiming the space.
- **Loads never overlap a prove.** A deserialization on a client while
  one of its executions is in flight wedges both; deserializations
  alongside each other are fine. The client gate admits loads together
  and a prove alone, taken per program rather than per AIR so a prove
  that arrives mid-load waits for one program (well under a second), not
  an AIR's thirty; a pending prove holds new loads back, and the
  background preload of the whole key (`ZZ_PRELOAD=all`) stops once
  proving starts; requested AIRs still load on demand.
- **Host copies run across cores.** The instance leaves pil2's buffers on
  proofman's proof worker (`Bridge::take`), before `gen_proof` returns;
  unpacking and reducing a gigabyte there single-threaded held the worker
  for seconds per instance, so the row work is split over the host's
  cores (half of them: proofman's own pools run beside it), and the fixed
  sections are read from the key straight into word buffers in parallel
  slices, the custom commit untiled the same way. The next instance's
  uploads and key read happen while the previous prove has the client,
  and everything through commit2 is enqueued before the first transcript
  wait (the stage-2 challenges do not depend on root1 in this schedule).
