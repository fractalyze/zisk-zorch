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
# once per proving key: export every basic AIR (~5 min on an RTX 5090)
FRX_PLATFORMS=cuda python -m zisk_zorch.export.export_air \
    --proving_key=$PK --air=all --out=$ARTIFACTS

# once per export and plugin build: compile the AIRs a guest needs into the
# cache. The fused Poseidon1 kernels compile slowly (about a minute for a
# leaf sponge, 15 s per tree level; ~40 min for the 11 hello-world AIRs on
# 11 threads), and a compile inside a prove trips proofman's 10-minute
# watchdog, so fill the cache ahead of the first run.
ZZ_WARM_THREADS=11 zz_prove --warm $ARTIFACTS Main_n22 Rom_n22 ...   # or no list: all

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

## Status (2026-09-06, RTX 5090, block-shaped sha-hasher workload)

The wall-clock comparison the issue asks for, on the closest stand-in for
block 21740136 this host can run: the `sha-hasher` example guest at
14,000 iterations, hint-free, under the ASM emulator. Its 51.1 M steps plan
into 38 instances across 16 families — 13 Main, 6 Binary, 5
BinaryExtension, 2 BinaryAdd, and one each of Arith, Dma, Dma64AlignedMem,
DmaPrePost, DmaUnaligned, InputData, Mem, MemAlign, Rom, RomData and the
two virtual tables — where the block was 38 instances with 12 Main. (The
block's captures and the zec-reth guest's hints are not on this host; the
guest uses the `sha2` crate's software path, so no precompile family
appears.) Same binary for both stacks, alternating runs, three per stack,
proof dumps compared after every bridge run; every bridge run's 38 basic
proofs were byte-identical to native's and its final proof verified.

| | native (3 basic streams + 1 recursive) | bridge (1 client at 45 % of the card; pil2 on 1 basic stream, recursion on it too) |
|---|---|---|
| `cargo-zisk prove` wall | 31.3–31.9 s | 34.5–38.0 s |
| proofman init | 5.2–7.2 s | 4.3–7.4 s |
| contributions | 3.5–3.6 s | 4.1–4.3 s |
| inner-proof leg (38 basic + their recursion) | 15.0–15.3 s | 20.3–20.7 s |
| ├ the 38 proves' own time on the client | | 18.9–19.5 s (Main 0.53 s ×13, Binary 0.53 ×6, BinaryExtension 0.48 ×5, BinaryAdd 0.36 ×2, the rest 0.35–0.82 once each) |
| ├ of which fixed sections rebuilt on family switches | | 4.5–5.0 s over 30–34 switches |
| └ waiting for the client, summed over instances | | 147–157 s (the serialization) |

So on a block-shaped mix the bridge's leg is 1.35× native's and its wall
1.10–1.19×, against 1.75× / 1.4× on the hello-world guest: the fixed
per-run costs amortize, and per instance the proves are where pil2's are
(Main 0.53 s here against pil2's ~0.6 s single-stream). Two things
separate the legs, both already named in #170:

- **One client.** The 38 proves run back to back; pil2 overlaps three.
  The instances' summed wait says the client is never idle from the first
  prove to the last (19 s span for 19 s of proves).
- **Family switches.** With `ZZ_RESIDENT_AIRS=1` (the default) every
  switch re-uploads and re-hashes the incoming family's constants, ~4.7 s
  per run — Main alone comes and goes 13 times. Raising the resident set
  does not fit on a 32 GB card at this share: `ZZ_RESIDENT_AIRS=2`, 3, 4
  and 8 all abort once the second or third family is resident (PJRT
  `Out of memory` from the client's BFC pool, which xla-pjrt's `check`
  turns into a panic rather than an error the bridge could evict on), and
  a larger share (`ZZ_MEMORY_FRACTION=0.55`) leaves pil2 13.3 GB, below
  the minimum it will start with. The lever is the resident-set trim
  (drop digest layers after the openings, re-upload base constants per
  prove) so that two or three families fit beside a prove's working set.

Reproduce with the scripts in [`../bridge/bench/`](../bridge/bench/):
`mk_input.py 14000 in.bin` for the guest's input (a ZiskStdin frame of a
bincode-varint `u32`), `run.sh <tag> native|bridge` for a prove with its
dumps, `compare_dumps.py` for the byte-gate, `summarize.py` for the table's
rows. The guest builds with `cargo-zisk build --release` in
`examples/sha-hasher/guest` of the ZisK checkout after
`cargo-zisk toolchain install`; the bridge's cache needs BinaryAdd and the
four Dma AIRs warmed beyond the hello-world set (58 min on 5 threads
here). Start a run only once `nvidia-smi` shows the card empty: a
process still releasing its memory makes pil2 size 20 streams from the
1.6 GB it sees and exit.

## Status (2026-09-04, RTX 5090, go hello-world guest)

`cargo-zisk prove -g -y` through the bridge completes and its final proof
verifies. All 11 basic instances (Rom, Main, Mem, InputData, RomData,
MemAlign, BinaryExtension, Binary, Arith, both virtual tables) are
byte-identical to native pil2's, compared as per-instance proof dumps
(`ZZ_DUMP_PROOFS`) from a native run and a bridge run of the same guest.
The in-process `ZZ_AB=1` variant reproduces the same verdict per instance
but still crashes once the card fills; the dump comparison is the gate to
quote.

Quiet host, three consecutive runs per stack, the second and third quoted
(a run right after another process has churned the page cache — a Bazel
build, the other stack, a cache warm — adds 4–5 s of file reading to
either stack's init):

| | native (3 basic streams + 1 recursive) | bridge (1 client) |
|---|---|---|
| `cargo-zisk prove` wall | 11.2–11.6 s | 15.3–16.9 s |
| proofman init | 3.0 s | 5.4 s (bridge up 0.2 s, then init beside the executable loads) |
| inner-proof leg | 3.7 s (28 proofs) | 6.5 s |
| ├ proves, one client, back to back, own time | | 5.45 s (InputData 0.17, RomData 0.30, MemAlign 0.32, Arith 0.42, VirtualTableZisk1 0.45, Rom 0.49, BinaryExtension 0.57, VirtualTableZisk0 0.57, Mem 0.62, Binary 0.75, Main 0.64–0.79) |
| ├ waiting for the client, summed over the 11 instances | | 28 s (the serialization) |
| └ executable loads, per AIR from the cache | | 0.52 s |
| Main, single stream on both sides | 0.61 s (commit 0.165 + proof 0.444) | 0.64–0.79 s |

Before #168 (2026-09-03) the same table read 21.2–21.5 s wall, a 9.9–10.1 s
leg with 9.1 s of proves, Main at 1.2 s and ~4.5 CPU-s of executable loads
per AIR. All of that difference was one cause: the hash-frx wheel pinned
then emitted marker spellings the frx plugin had retired (fractalyze/xla#557),
so every Poseidon permutation and sponge inlined into hundreds of loop
fusions over the whole leaf set — Main's `commit1` held 2034 fusions, 44.5
GiB of writes, and took 0.385 s where the fused kernel takes 0.08 s. Bytes
never changed, so no golden noticed; `//zisk_zorch/commit:fusion_test` now
compiles a commit on the GPU leg and asserts one custom fusion per level.
The same `commit1` now holds 77 fusions (17 NTT passes, 13 hash kernels,
the rest reshapes and slices) and loads in 75 ms.

What remains above native is structural, tracked in #170: the bridge
proves the 11 instances back to back on one client while pil2 overlaps
three basic streams and its recursion (two clients did not fit beside
pil2's 14 GB when a table AIR's `const_setup`/`logup` allocated 4.5–5.5 GiB
in one piece; to be re-measured on the smaller executables); the bridge
comes up beside proofman's init on the same cores; and `const_setup`
recomputes each AIR's constant tree per run where pil2 reads it from disk.
Per instance, Main is within 5–30 % of single-stream pil2. The block-shaped
comparison is the section above.

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
  rebuilds the executable from its HLO) and scale with the instruction
  count: an AIR's programs load in ~0.5 s with the hash kernels fused,
  ~4.5 CPU-s when the markers inlined (#168); more preload threads slow
  proofman's init by as much as they gain.
- **Compile cost.** Compiling an AIR's programs takes many minutes:
  the fused Poseidon1 sponge and permute kernels are each a fully
  unrolled straight-line body (3358 multiplies, 1.9 MB of LLVM IR for one
  width-16 permute against Poseidon2's 296 and 0.27 MB), which XLA and
  LLVM optimize for ~1 min per leaf sponge and ~15 s per tree level while
  ptxas itself takes 2 s — about 40 min for the 11 hello-world AIRs on
  11 warm threads (`ZZ_WARM_THREADS`), 245 s for RomData's 34 programs
  when the markers still inlined. So the bridge keeps the serialized
  executables on disk and a later client loads them in ~0.5 s per AIR.
  The cache is keyed by the bytecode's hash and the plugin's identity
  (`XLA_PJRT_PLUGIN` path, size, mtime), an entry the plugin rejects is
  recompiled in place, one compile per entry across clients, and a
  directory that cannot be created (a read-only export) means compiling
  without a cache, not failing. Fill it with `zz_prove --warm` before
  the first run: a compile inside a prove trips proofman's 10-minute
  watchdog. A key scheme change (this one included) leaves the old
  entries unused on disk, so delete the directory when reclaiming the
  space.
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
