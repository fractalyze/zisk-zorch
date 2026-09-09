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
# watchdog, so fill the cache ahead of the first run. Threads spread across the
# AIRs named first, and the rest go inside them -- one AIR at 11 threads
# compiles 11 programs at once, and 6 AIRs at 11 threads run 5 of them two-wide.
# Warming a single AIR is what bisecting a plugin build does.
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
| `ZZ_CLIENTS` | PJRT clients; proofman spawns this many basic-proof workers | 3 — more than this card fits, see "Memory budget" |
| `ZZ_MEMORY_FRACTION` | share of the card the clients claim up front, split evenly, before pil2 sizes its buffers | unset: allocate on demand |
| `ZZ_GPU_HEADROOM_GB` | (fork) GPU memory pil2 leaves out of its stream sizing | 0 |
| `ZZ_PRELOAD` | executables loaded at bridge creation: the previous run's AIRs (`.last-used`), `all`, or `0` | last used |
| `ZZ_PRELOAD_THREADS` | AIRs loading at once | 6 |
| `ZZ_EAGER_MODULES` | executables load their modules into the CUDA context as they are deserialized, not on first execute: `0` off, anything else on, empty or unset follows `ZZ_PRELOAD` | on unless `ZZ_PRELOAD=0` |
| `ZZ_PENDING` | proves admitted per client on the device (one running, the rest uploaded ahead) | 2 |
| `ZZ_RESIDENT_AIRS` | AIRs whose fixed sections stay on a client at once, least recently used evicted | 1 |
| `ZZ_HOST_THREADS` | threads for the host-side copies and key reads | half the cores, at most 8 |
| `ZZ_COMPILE_CACHE` | directory of serialized executables | `$ZZ_ARTIFACTS/.pjrt-cache` |
| `ZZ_LOG` | `1` per-instance timing on stderr, `2` per program; lines carry the seconds since bridge-up | off |
| `ZZ_AB` | prove through pil2 too and compare per instance | off |
| `ZZ_DUMP_PROOFS` | (fork) write every basic proof as raw words into this directory | off |
| `ZZ_DUMP_INPUTS` | write each instance as a `zz_prove` case directory under this one | off |
| `ZZ_DUMP_TRACES` | (fork) write each host trace as `gen_proof` receives it | off |

## Profiling

Where a prove's device time goes program by program, and the same question
asked of pil2. The two totals are not like for like: `nvtx_kern_sum` counts
kernels only and the bridge's uploads happen outside the ranges (they are not
in `Artifact::run`), so its total excludes H2D and D2H, while pil2's totals
count its `H2D_COPY` category. Compare the kernel work, and subtract pil2's
`H2D_COPY` before comparing totals. Both halves want a quiet host and a warm
executable cache — a compile inside the capture buries the numbers.

```bash
# The bridge: one NVTX range per program, then nsys over a prove of an
# instance dumped with ZZ_DUMP_INPUTS. --cuda-graph-trace=node is not
# optional; XLA runs the fusions as CUDA graphs and nsys sees nothing
# through them. --repeat 1 proves twice, so the capture holds two proves.
cargo build --release --features standalone,nvtx
nsys profile --cuda-graph-trace=node -t cuda,nvtx -o main \
    zz_prove $ARTIFACTS $CASE --repeat 1
nsys stats --report nvtx_kern_sum --format csv -o main main.nsys-rep
bench/nvtx_programs.py main_nvtx_kern_sum.csv

# pil2: its own per-instance timers at -vv, on ONE basic stream. With more
# the blocks of the streams interleave and nothing can be attributed; the
# fork's headroom knob is what forces one (15 GB on a 32 GB card).
ZZ_GPU_HEADROOM_GB=15 cargo-zisk-dev prove -e guest.elf -k $PK -g -y -vv \
    -o proof > native.log
bench/pil2_timers.py native.log --global-info $PK/pilout.globalInfo.json

# Where the uploads sit: nsys over a WHOLE run (not one instance), then the
# host-to-device time each prover's own kernels did not hide. --sample=none
# --cpuctxsw=none is not optional either -- with CPU sampling on, nsys
# 2026.1.3 collects the run fine and then deadlocks in report generation,
# leaving an unusable .qdstrm and no .nsys-rep.
ZZ_ARTIFACTS=$ARTIFACTS ZZ_CLIENTS=1 ZZ_MEMORY_FRACTION=0.45 ZZ_GPU_HEADROOM_GB=3 \
XLA_PJRT_PLUGIN=<venv>/site-packages/frx_plugins/xla_cuda12/xla_cuda_plugin.so \
nsys profile --cuda-graph-trace=node -t cuda --sample=none --cpuctxsw=none \
    -o run --force-overwrite true \
    cargo-zisk prove -e guest.elf -k $PK -g -y -o proof -vv
nsys stats --report cuda_gpu_trace --format csv -o run run.nsys-rep
bench/h2d_overlap.py run_cuda_gpu_trace.csv
```

`h2d_overlap.py` reports the two provers apart, because one bridged run has
both on the card: XLA writes a fusion's name with no argument list, pil2's
kernels are C++ signatures, and a transfer stream belongs to whoever owns
the kernels on it or, for a dedicated one, the kernels that follow its
copies. Per side it prints the transfer time that side's own kernels did
not hide, split at the leg so the share and the exposure share a window;
the same time measured against the *other* prover's kernels, which is the
control for a zero; and how long the side's device had already been idle
when each copy started, which separates a copy the runtime would not
overlap from one that had nothing to overlap with. Every variable the run
needs is inlined above, including the `XLA_PJRT_PLUGIN` path that "Running"
exports, because this recipe wraps the prover directly rather than
`bench/run.sh` — nsys tracking the shell and `/usr/bin/time` between it and
the prover is the other way to reach the same hang, and run.sh is also what
would otherwise export `ZZ_CLIENTS=1` for you. Leaving that one out is the expensive mistake: the default is 3,
three clients splitting one `ZZ_MEMORY_FRACTION` are each below a client's
floor, and the run aborts mid-prove with no GPU data in the capture at
all. The other three values are the ones the numbers below were taken at.

The `nvtx` feature is off by default and stays off in proofman builds: it
links the CUDA toolkit's `libnvtx3interop` and the ranges say nothing outside
a profiler. `nvtx_programs.py` counts only the bridge's ranges — XLA opens
its own around the same kernels — and reports per prove, except for a program
that ran fewer times than the capture has proves, which it reports whole
(`const_setup` builds the constant tree once per family). A program that runs
several times per prove, like the quotient over its chunks, is one row
carrying all of them. `pil2_timers.py` is per instance, and its rows are per
air: a row carries the `x<n>` instances it sums and their average, because a
workload runs several instances of the same air.

### Where the leg's idle goes

The per-program table above says what the device *did*; on the hello-world
guest it is busy for under half the bridge's leg, so the larger question is
what the host was doing for the rest. The bridge opens a second family of
NVTX ranges, prefixed `host/`, one per step of `Bridge::take`, `prove_owned`
and the schedule in `AirDriver::prove`; `host_idle.py` charges every idle
nanosecond of the leg to the phase that was running.

Only the prove *holding the client* can explain the idle. The bridge proves
one instance at a time per client but gives every instance a thread, so a
dozen threads are alive and all but one are queued: a queued thread's
`host/admit` and `host/slot_wait` cover almost the whole leg by construction
and are waits, not costs. So the report splits the idle at a prove's turn —
from where its thread leaves `host/slot_wait` holding the slot mutex to the
end of its `host/prove` — and reports what the other threads were doing
separately, as overlapping rather than additive.

```bash
# A whole run, both provers on the card. The bridge must be built with the
# feature ON inside the proofman build, which the fork does not expose: add
# `default = ["nvtx"]` to the [features] of the bridge that zisk's Cargo.toml
# [patch] points at, build cargo-zisk, and take it out again afterwards.
nsys profile --cuda-graph-trace=node -t cuda,nvtx --sample=none --cpuctxsw=none \
    -o run cargo-zisk prove -e guest.elf -k $PK -g -y -o proof -vv
nsys stats --report cuda_gpu_trace --report nvtx_pushpop_trace \
    --report cuda_api_trace --format csv -o s run.nsys-rep
bench/host_idle.py s_cuda_gpu_trace.csv s_nvtx_pushpop_trace.csv \
    s_cuda_api_trace.csv
```

The third CSV is optional and cuts the same idle a second way: which CUDA
driver call the holding thread was inside. The phase cut says which step of a
prove starved the device; this one says what the driver was doing there, and
the two are answers about the same nanoseconds rather than separate budgets.
`-t cuda` already collects it, so an existing capture can be re-exported
without re-running anything.

`--sample=none --cpuctxsw=none` is not optional here either: with CPU
sampling on, nsys 2026.1.3 collects a run this size and then deadlocks in
report generation. Set `ZZ_CLIENTS=1` by hand when wrapping the binary
directly — the bridge's own default is 3, and three clients splitting one
`ZZ_MEMORY_FRACTION` land under a client's floor and abort mid-prove.

Two things about reading the result, both learned by getting them wrong.

**Quote the share of the *leg*, not of the idle.** The two denominators
differ by about 2x, and the milestone's criterion is wall time. The same
`cuModuleLoadFatBinary`, on the two workloads measured 2026-09-09: on the
hello-world guest 1.53-1.60 s, which is 53-56 % of that leg's idle but
28-29 % of the leg itself (1.9x); on the block-shaped `sha-hasher` mix
2.94 s, 34 % of the idle and 15 % of the leg (2.25x). One number, two
denominators — and the leg is the one that decides anything.

**Module loads are once per (AIR, program) pair, not per execution and not
per instance.** A program that runs four times in a prove loads once, and
every later instance of an AIR already seen loads nothing — measured on the
block-shaped workload, where 16 of 38 proves load a full program set and the
other 22 load zero. Eviction does not undo it: a module lives in the CUDA
context, and `ZZ_RESIDENT_AIRS` only drops device buffers. So this cost
scales with how many *families* a workload touches, and a guest whose
instances are all distinct AIRs — hello-world — is its worst case and a bad
place to size it from. Per-load cost is not constant either (~4.2 ms on
hello-world against ~5.5 ms on the block-shaped mix, over the calls that
contributed idle — not over every load made), so scaling by program
count alone under-predicts.

Cross-check any figure this produces against `ZZ_LOG=2`, which prints each
`Artifact::run`'s enqueue time from the bridge's own clock with no profiler
attached; on the 2026-09-09 runs the two agreed to within 8 % (3.448 s of
enqueue summed, against 3.457-3.731 s of NVTX range time under nsys), which
is what says the dispatch cost is real and not an artifact of tracing.

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
  turns into a panic rather than an error the bridge could evict on — the
  read-ahead's upload catches that unwind and falls back to uploading
  under the slot, so a full card costs it the head start rather than the
  run), and
  a larger share (`ZZ_MEMORY_FRACTION=0.55`) leaves pil2 13.3 GB, below
  the minimum it will start with. The resident-set trim was the candidate
  lever and has since been measured: it does not move the floor, because
  what binds a client is a single program's own working set rather than
  anything kept between proves ("Memory budget" below).

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
| proofman init | 3.0 s | 5.4 s (bridge up 0.2 s, then init beside the executable loads) — closed since, see "Bridge start-up" |
| inner-proof leg | 3.7 s (28 proofs) | 6.5 s |
| ├ proves, one client, back to back, own time | | 5.45 s (InputData 0.17, RomData 0.30, MemAlign 0.32, Arith 0.42, VirtualTableZisk1 0.45, Rom 0.49, BinaryExtension 0.57, VirtualTableZisk0 0.57, Mem 0.62, Binary 0.75, Main 0.64–0.79) |
| ├ waiting for the client, summed over the 11 instances | | 28 s (the serialization) |
| └ executable loads, per AIR from the cache | | 0.52 s |
| Main, single stream on both sides | 0.61 s (commit 0.165 + proof 0.444) | 0.64–0.79 s |

Reproduce with the same `bridge/bench/` scripts as the block-shaped
section, minus the input: the guest takes none, and it needs
`ZISK_PROVE_FLAGS=` (empty) on a host with no ASM emulator built, since
run.sh's default is the ASM emulator's `-a -u`. So
`ZISK_PROVE_FLAGS= run.sh <tag> native|bridge`, then `compare_dumps.py`
for the byte-gate and `summarize.py` for the rows. "Memory budget" below
was measured this way, adding `ZZ_MEMORY_FRACTION` and
`ZZ_GPU_HEADROOM_GB` per run.

### Bridge start-up (2026-09-09, post-#176)

The bridge's start is hidden inside proofman's init with time to spare.
`ZZ_LOG` timestamps a run against that start: the client is up at
+0.19 s, `INITIALIZING_PROOFMAN` runs from there to +3.44 s, and the whole
preload — 11 AIRs, 380 programs out of the cache — is done at +1.03 s,
leaving about 2.4 s of init it does not use. What is left beside native is
0.07–0.26 s on proofman's own timer, or 0.26–0.45 s counting the client
creation that precedes it.

So neither lever #178 proposed has anything to buy. Hooking the bridge in
earlier moves work that already finishes with slack; deferring the client
to the first prove would give up what `ZZ_MEMORY_FRACTION` is for, since
the clients claim their share before pil2 sizes its stream buffers from
the memory it sees free. Nor is the residual the preload's own cost:
`ZZ_PRELOAD_THREADS` at six, three and two lands within 0.1 s, and
`ZZ_PRELOAD=0` reaches native's init only by moving the loads into the
contributions phase rather than removing them (that spelling turns eager
module loads off as well, so it moves two things at once). Of what is
left, 0.3 s is the #176 plugin bump's own share — the same run on the
previous plugin costs that much more init.

The ordering this rests on — that an AIR the run will prove is loaded
before a prove wants it — is pinned by two tests rather than by the
timing: the preload queue drains the run's own AIRs ahead of the rest of
the key, in order (`lib.rs`), and a prove waits for the loads already in
flight instead of passing them (`artifact.rs`).

### The uploads, measured (2026-09-09, post-#192)

Three `nsys` captures of the whole run on a quiet card, read by
`bench/h2d_overlap.py` (recipe under "Profiling"). Every figure in the table
below is a line that tool prints over those captures. Three numbers in the
prose are not its, and each says so where it appears: proofman's own
`GENERATING_INNER_PROOFS` timer, the device-idle share `host_idle.py`
produces, and the pinning projection at the end, which is arithmetic on the
table rather than a measurement. The bridge's leg here is its first kernel to
its last, which is 0.8–1.2 s inside `GENERATING_INNER_PROOFS` (6.29, 6.63,
6.62 s here against 5.93–6.01 s uninstrumented — nsys costs the leg 5–11 %).

| per run | run 1 | run 2 | run 3 |
|---|---|---|---|
| bridge leg | 5.45 s | 5.47 s | 5.59 s |
| its kernels, busy | 2.60 s | 2.60 s | 2.61 s |
| its uploads, 9.75 GB | 0.58 s | 0.44 s | 0.49 s |
| ├ overlapped by its own kernels | 0.00 s | 0.00 s | 0.00 s |
| ├ on the critical path | 0.58 s | 0.44 s | 0.49 s |
| ├ … of that, inside the leg | 0.31 s | 0.31 s | 0.31 s |
| └ … of that, before the leg's first kernel | 0.27 s | 0.14 s | 0.19 s |
| copies starting >1 ms after a kernel ended | 71 % | 74 % | 75 % |
| ├ median idle before a copy | 1.62 ms | 1.90 ms | 1.98 ms |
| └ p90, longest | 27, 217 ms | 29, 125 ms | 27, 156 ms |
| pil2's copies overlapped by *the bridge's* kernels | 0.09 s | 0.08 s | 0.07 s |
| pageable, 5.29 GB | 11.2 GB/s | 15.7 GB/s | 13.4 GB/s |
| pinned, 4.46 GB | 42.0 GB/s | 41.9 GB/s | 44.9 GB/s |

The two parts of the critical path are each rounded to a hundredth, so they
do not always re-add to it: run 2 is 0.308 s inside the leg and 0.136 s
before it, against 0.444 s in total.

Two things this settles. Uploads are **0.44–0.58 s**, not the ~1.9 s the
pre-#175 profile put on them, and pageable transfers on this card run at
11–16 GB/s rather than the 3–6 GB/s that number assumed — most of the
difference is #182's read-ahead and #183's parallel key reads, which took
the host-side staging out of the transfer. And **none of it overlaps the
prove it belongs to**: 0.00 s against its own client's kernels in all three
runs, on both provers. The separate host-to-device stream
(`local_device_state.h`) exists and is never busy at the same time as that
client's compute stream.

That zero is enforced, and not by the hardware: in the same captures
pil2's copies overlap *the bridge's* kernels for 0.07–0.09 s, so the card
runs copy and compute together happily. What neither prover overlaps is
its own kernels, and for the bridge PJRT is why. A GPU client is
`kComputeSynchronized` (`xla/pjrt/local_device_state.h`): a buffer the
allocator returns at time t may only be written once the compute stream
has drained everything enqueued before t. So `AllocatedRawSEDeviceMemory`
records a compute-stream sync point when it allocates
(`tracked_device_buffer.cc`), and both `BufferFromHostBuffer` paths call
`WaitForAllocation`, which makes the host-to-device stream wait on that
sync point's event (`pjrt_stream_executor_client.cc`). An upload into a
*freshly allocated* buffer therefore cannot start until the client's own
compute stream is empty — no host-buffer-semantics flag changes that,
which is why the 2026-09-03 `kImmutableOnlyDuringCall` attempt only moved
the wait into the next `Execute`. It also explains the shape of the
capture: the 25–29 % of copies that start within a millisecond of the last
kernel ending had waited on exactly that event.

The host is separately late: the other 71–75 % start into a device that
has been idle longer than that, a median of 1.6–2.0 ms and a p90 of 27–29
ms. **Removing the ordering was tried and does not help.** Allocating an
instance's four input buffers together, up front, through PJRT's async
host-to-device transfer manager — so the sync point is taken at admission
rather than once per buffer behind the previous copy — leaves the overlap
at 0.00/0.01/0.00 s and proofman's `GENERATING_INNER_PROOFS` unmoved:
6110/6188/6254 ms before against 6218/6140/6223 ms after. Those are the
uninstrumented timer on a separate same-session A/B — old bridge and new
bridge built one after the other on the same card, three runs each — so
they are comparable to each other and to nothing else on this page, neither
the nsys legs in the table above nor the 5.93–6.01 s beside them.

The second constraint is what binds, and the idle distribution is what
makes it visible. The device is not idle for the whole leg — its own
kernels are busy 2.60 s of 5.45 s — but it is idle when the copies run:
three in four start after it has already been doing nothing for over a
millisecond, so an upload freed to run beside kernels finds none to run
beside. `host_idle.py` says where that idle goes: 76–83 % of it is
host-side dispatch inside `Artifact::run` — #197's measurement, on its own
`-t cuda,nvtx` captures of the same guest and card on the same day, not on
the three here. So at the moment the next instance uploads, the prove
holding the client is on the host rather than on the device.

Which also means the 0.31 s inside the leg is an upper bound that
overstates its own cost here: an upload landing in idle the leg would have
had anyway is not paid for twice. Uploads are not this leg's problem, and
no change to the upload path makes them one. Bandwidth is smaller still: on
the table's own rates, pinning the pageable 5.29 GB at the 42 GB/s the
already-pinned copies reach would take 0.34–0.47 s to about 0.13 s.

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
three basic streams and its recursion (re-measured on the #171 artifacts,
a second client is still 8.1 GiB more than this card has — "Memory budget"
below); and `const_setup` recomputes each AIR's constant tree per run
where pil2 reads it from disk. The bridge's own start is no longer one of
them: it finishes well inside proofman's init ("Bridge start-up" below).
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
- **Memory budget (RTX 5090, 31.8 GiB).** Three pools share the card,
  and what each gets is `ZZ_MEMORY_FRACTION` (the clients' share, claimed
  up front), `ZZ_GPU_HEADROOM_GB` (held back from pil2's sizing) and
  whatever is left (pil2's). The first two floors were measured on the
  hello-world key by walking the fraction down until a run failed, one
  client, repeats at each fraction — a single run at the boundary is a
  race and lands either way. The client's floor is from #191's re-walk
  (2026-09-09, both arms on one binary); pil2's and the module loads'
  are from 2026-09-08 and the blocked LDE does not touch them:
  - **A client needs 11.8 GiB**, at headroom 3 — the fraction every run
    survives, below which the outcome is a coin flip rather than a
    cliff. Both columns are `main` at c8da072, so only the artifacts
    differ:

    | `ZZ_MEMORY_FRACTION` | the client's share | before #191 | after |
    |---|---|---|---|
    | 0.45 | 14.3 GiB | — | 2/2 |
    | 0.39 | 12.4 GiB | 3/3 | 3/3 |
    | 0.37 | 11.8 GiB | 1/3 | 3/3 |
    | 0.35 | 11.1 GiB | 0/3 | 4/5 |
    | 0.34 | 10.8 GiB | 0/3 | 1/3 |
    | 0.32 | 10.2 GiB | — | 0/3 |
    | 0.30 | 9.5 GiB | — | 0/3 |
    | 0.28 | 8.9 GiB | — | 0/2 |

    #177 first put the floor at 12.1 GiB off one run per fraction and
    #188 revised it to 12.4 off seven; a repeated figure supersedes a
    single run. #188's own table below reads 0/4 at 0.37 where this one
    reads 1/3 — that arm predates #190's stage-tree release, and is not
    this binary.
    That 11.8 GiB is what a client holds at once — one AIR's fixed
    sections (the extended constants, their tree, the base constants the
    stage-2 hints read: 4.6 GB for a table AIR with 88 constant columns),
    the next AIR's sections read ahead of its slot, and a prove's working
    set — though the sweep measures the total, not the split. Which AIR
    aborts is not fixed, and at one fraction it varies run to run:
    whichever wide one first finds the arena dry, so read the floor off
    the fraction rather than off the AIR named in the log.
  - **pil2 needs 14.3 GiB left to it and refuses to start below that**,
    since `commit_witness` stays on the card. Left to it means the card
    minus the clients' share minus the headroom, so one fraction can go
    either way: at headroom 0, fraction 0.55 leaves 14.3 GiB and pil2
    comes up with one basic stream and 5.05 GB of fixed pols, while 0.58
    leaves 13.4 GiB and it exits with `Not enough GPU memory to run the
    proof`; at headroom 3 that same 0.55 leaves 11.3 GiB and it refuses.
    (The block-shaped section above reports 0.55 leaving pil2 13.3 GB,
    which this model reproduces at neither headroom; that run's headroom
    is not recorded, so the two are not the same measurement. #170
    carries the discrepancy.)
  - **Module loads come out of neither**, which is what
    `ZZ_GPU_HEADROOM_GB` buys: at headroom 0 a run both pools fit in
    still dies on `Failed to get module function:
    CUDA_ERROR_OUT_OF_MEMORY`, with the card at 31.4 GiB. The bench's 3
    is enough and 0 is not; the totals below budget ~2.
  - **What pil2 actually allocates — 12.9 GiB — is already sized for
    recursion**, so taking its basic proofs away frees none of it
    (#194). It is 1.72 GiB of basic fixed pols, 3.33 GiB of aggregation
    fixed pols and one 7.85 GiB auxiliary trace, and three lines every
    run prints just above the stream count place that last term
    ([`proof_ctx.rs:961-988`](https://github.com/fractalyze/pil2-proofman/blob/daf3a598/common/src/proof_ctx.rs#L961-L988)):
    `Max prover buffer size: 7.85 GB` is `max(basic, recursion)`, `Max
    prover recursive buffer size: 7.85 GB` is the recursion term alone,
    and `Max prover recursive1/recursive2 buffer size: 1.53 GB` is the
    per-recursive-stream buffer, `max(recursive1, recursive2)`. The
    first two being equal says only that recursion is at least basic —
    enough to know basic proving does not size the buffer, not enough to
    say what does. The third takes recursive1 and recursive2 out of the
    five-way max
    ([`setup_ctx.rs:149-154`](https://github.com/fractalyze/pil2-proofman/blob/daf3a598/common/src/setup_ctx.rs#L149-L154)),
    leaving the compressor and the two vadcop finals, and the proving
    key separates those. Holding every committed section at extended
    size plus the trace — a lower bound on the `mapTotalN` the buffer is
    cut from — the largest compressor (Keccakf's) comes to 6.41 GiB
    against 2.36 for `vadcop_final` and 0.60 for
    `vadcop_final_compressed`, with `recursive2` at 1.22 against its
    logged 1.53. Only the compressor is in range. It is also sized over
    the whole proving key rather than the workload: none of
    hello-world's 11 airs has a compressor at all, and the buffer is
    still Keccakf's.
  - **Nor can the basic stream itself go.** Contributions stay on pil2
    under the bridge, and `commit_witness_gpu` takes a *non-recursive*
    stream, reads the basic fixed pols and writes that same auxiliary
    trace
    ([`starks_api.cu:1231-1292`](https://github.com/fractalyze/pil2-proofman/blob/daf3a598/pil2-stark/src/api/starks_api.cu#L1231-L1292));
    so does the compressor, since `gen_recursive_proof_gpu` sets
    `aggregation` for `recursive1` and `recursive2` only
    ([`:894-901`](https://github.com/fractalyze/pil2-proofman/blob/daf3a598/pil2-stark/src/api/starks_api.cu#L894-L901)).
    At zero basic streams `selectStream` has no candidate for either and
    spins in its wait loop
    ([`:1746-1812`](https://github.com/fractalyze/pil2-proofman/blob/daf3a598/pil2-stark/src/api/starks_api.cu#L1746-L1812)).
    The mirror image is in every bridge run already: it sizes **0**
    recursive streams and still finishes, recursion falling back to the
    basic stream.

  So one client's floors total 11.8 + 14.3 + ~2 = **28.1 GiB** of the
  31.8 available. (A run at the bench's `ZZ_MEMORY_FRACTION=0.45` peaks
  at 28.7 GiB, which is not this sum: there the client claims 14.3 GiB,
  well above its floor, and pil2 sizes itself down to what is left.)
  **Two clients need 2 × 11.8 + 14.3 + ~2 = 39.9 GiB and are 8.1 GiB
  short.** 6.1 GiB of that is the measured floors alone (2 × 11.8 +
  14.3 = 37.9 against 31.8, before any headroom at all); the rest is the
  headroom, which is estimated but cannot be zero. #194's counterfactual
  does not close it either: had pil2 been sizable for recursion alone —
  1.72 + 3.33 + one 1.53 GiB recursive stream, 6.3 GiB below what it
  holds — two clients would still be 1.8 GiB short. The bullet above is
  why that 6.3 GiB is not on offer.

  The runs bear it out: `ZZ_CLIENTS=2` aborts in three runs out of three
  at fraction 0.45 (headroom 3), and again in three out of three with at
  most one prove admitted per client (`ZZ_PENDING=1`, which does not stop
  the fixed sections going up ahead of the slot — all 11 still do). No
  fraction rescues it: pil2's floor caps the clients' total share near
  0.55, so two clients can have at most ~8.8 GiB each, 3.0 GiB below the
  floor, and that ceiling leaves the module loads nothing. **The unset
  default is 3**, which 3 × 11.8 = 35.4 GiB puts past the whole card
  before pil2 gets any; the bench pins `ZZ_CLIENTS=1`, and every number
  here is from one client.

- **The resident-set trim does not reach a second client** (#188). Scoped
  as "re-upload the base constants per prove, drop the digest layers once
  the openings are done", built in full, and walked down the same fraction
  (one client, headroom 3, hello-world; a pass is all 11 proofs and a
  verified final proof):

  | `ZZ_MEMORY_FRACTION` | the client's share | before | the trim in full | shipped |
  |---|---|---|---|---|
  | 0.45 | 14.3 GiB | 4/4 | 5/5 | 6/6 |
  | 0.39 | 12.4 GiB | 7/7 | 7/10 | 5/7 |
  | 0.38 | 12.1 GiB | 0/4 | 4/7 | 2/3 |
  | 0.37 | 11.8 GiB | 0/4 | 2/4 | — |
  | 0.36 | 11.5 GiB | 0/1 | 0/1 | — |

  Before the change the floor is a cliff: every run at 0.39 and above, none
  below. With the trim there is no cliff, only a band from 0.39 down to 0.37
  where the outcome is a coin flip, and no fraction a run can be counted on
  at is lower than before. Read the columns as "not told apart at these
  counts" rather than as a gain — a fraction at the boundary is a race
  between the read-ahead's upload and the running prove's peak, which is
  what the before column revises the floor above for. Run repeats and quote
  the counts. The bench's 0.45 is unaffected in every arm.

  With the trim in full, `ZZ_CLIENTS=2` still fails 0/3 at fraction 0.45
  (headroom 3) and 0/3 at 0.54 (headroom 0). That arm frees strictly more
  than what shipped, so the verdict is the conservative one.

  What binds is the same allocation before and after, and the resident set
  never held it: one block twice the size of an extended section — an
  extend's input and output alive at once inside a single program. 5.50 GiB
  for `const_setup` on `VirtualTableZisk0_n21` (2 × its 2.75 GiB
  `const_ext`), 4.56 GiB for `VirtualTableZisk1_n21`, 4.88 GiB for `commit1`
  on `Binary_n22` (2 × its 2.44 GiB `cm1_ext`). `const_base` is the *input*
  to the largest of them, so releasing it after `logup` cannot reach it.
  That block is now gone (#191 below) and two clients still do not fit: it
  was the largest single allocation a client made, which is not the same
  thing as the floor.

  Half the trim shipped: a stage tree is released as its openings reach the
  wire, which costs nothing since nothing re-reads or recomputes it. The
  base sections stay resident. Making them per prove leaves residency with
  nothing to reuse, so two consecutive instances of one AIR would carry a
  second `const_base` beside the running prove's — invisible to the
  hello-world guest every number here comes from, whose 11 AIRs are
  distinct, and paid by the block-shaped mix above, which is 13 Main and
  6 Binary.
- **The extend transforms a column block at a time** (#191). `extend` turned
  a section into its LDE in one transform, which is what put two extended
  copies of it on the device beside the result. It now splits the columns
  into blocks of at most `LDE_BLOCK_BYTES` (256 MiB) of the extended domain
  and writes each into the result, the blocks ordered against each other so
  XLA does not schedule several of their transforms at once. Each column's
  LDE is independent of every other's, so the codeword does not move: only
  the four LDE-bearing programs re-export (21 of the 380 behind the
  hello-world set), every golden is unchanged, and all 11 basic proofs stay
  byte-identical to native's. From the compiled executables' own accounting
  (`memory_analysis`), on the LDE at the two shapes above:

  | RTX 5090 | VirtualTableZisk0_n21's constants | Binary_n22's `cm1` |
  |---|---|---|
  | the section, extended | 88 columns, 2.75 GiB | 39 columns, 2.44 GiB |
  | columns a block | 8 | 4 |
  | temporaries | 5.50 → **0.75 GiB** | 4.88 → **0.75 GiB** |
  | argument + result + temporaries | 9.63 → **4.88 GiB** | 8.53 → **4.41 GiB** |
  | the LDE's own device time | 31.9 → 39.5 ms | 34.0 → 54.5 ms |

  Whole programs move less than their LDEs do, because the Merkle half
  has temporaries of its own: `const_setup` on VirtualTableZisk0_n21 goes
  from 9.79 to **6.42 GiB** (1.38 argument + 2.92 results, scratch 5.50 →
  2.13), and what is left of the scratch is the tree's. Compiling it is
  unaffected, 262.8 s against 266.4 s — the blocks multiply the NTT
  passes and the Poseidon kernels are what the minutes go to. A program
  cannot go below its own argument and results, 4.13 GiB there, and both
  are fixed sections the client holds either way.

  The time lands where the per-LDE figures predict. A whole hello-world
  leg is 5.64–5.86 s before and 5.74–6.14 s after, three runs each at
  fraction 0.39 on the same binary; `zz_prove` on the dumped Main case
  proves it in 0.501 s against 0.519 s, which over 11 instances is the
  0.2 s the leg moves.

  **Freeing memory inside a program is not the same as lowering the
  floor.** 3.4 GiB out of the biggest one buys 0.6 GiB of the client
  floor above, because that block was the arena's largest single
  allocation rather than most of its high-water; the largest any failing
  run now reports is 1.56 GiB, inside `commit2` on
  `VirtualTableZisk0_n21` at fraction 0.28. So the extend is not what
  stands between this card and a second client, and the next lever is
  what a client keeps rather than what one program computes.
- **Exports carry no debug info and no folded power tables.** XLA
  re-formats every op's source location on load (half of a 5.6 s load
  once), so the exporter strips them; and it constant-folds the coset
  power series from its scalar seed into a 2^nBitsExt literal per
  LDE-bearing program (30–70 MB each) unless the seed crosses an
  optimization barrier, which it now does. Loads are CPU-bound (XLA
  rebuilds the executable from its HLO) and scale with the instruction
  count: an AIR's programs load in ~0.5 s with the hash kernels fused,
  ~4.5 CPU-s when the markers inlined (#168). The thread count is not a
  lever in either direction: six, three and two preload threads put
  proofman's init within 0.1 s of each other, because the preload is done
  long before init is ("Bridge start-up").
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
  and so does the upload of the key's fixed sections, which leaves the
  slot the setup programs alone instead of a table AIR's 1.2-1.4 GB
  `const_base` out of pageable memory. One AIR's sections go up beyond
  the running prove's at a time (a per-client permit, taken at the read
  and handed on when they are installed), so the read-ahead costs the
  card one key's worth of memory however many proves queue up; a prove
  that finds the permit taken uploads under the slot. Running the setup
  programs ahead as well does not fit: `logup` reads `const_base` through
  the prove, so the next AIR's whole fixed set would have to live beside
  the running prove's, and every hello-world run aborted with the client
  out of memory. What the slot still pays is the setup programs
  themselves, and that is device time rather than the transfer this
  read-ahead removes: `const_setup` runs 103.5 ms on Main and, with a
  custom commit, 100.0 ms beside `custom_setup_0`'s 110.6 ms on Rom,
  against warm `set_fixed` walls of 103 ms and 213 ms (RTX 5090, kernel
  time per program from the `nvtx` build under `nsys`). Only 21-23 % of a
  first `set_fixed` is the module load that #176 moves to init, so no
  scheduling change reaches a target set at the transfer's device floor;
  #183 tracks caching the constant tree on disk the way pil2 ships
  `.const_tree`. Everything through commit2 is then enqueued before the
  first transcript wait (the stage-2 challenges do not depend on root1 in
  this schedule).
