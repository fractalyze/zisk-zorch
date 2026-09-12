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

# before any run whose init or wall you intend to quote: the proving-key files
# proofman reads before it sizes its buffers. Cold, they cost seconds of init
# on either stack, and any other tenant of the host can empty them out of the
# page cache between two runs -- see "What sets a run's init is the page
# cache". bench/run.sh does this for you unless ZZ_WARM_KEY=0.
bench/pagecache.py --warm --proofman-init $PK

# the bridge's inputs (the full list is the table below)
export ZZ_ARTIFACTS=$ARTIFACTS
export XLA_PJRT_PLUGIN=<venv>/site-packages/frx_plugins/xla_cuda12/xla_cuda_plugin.so
export ZZ_LOG=1

# CUDA_MODULE_LOADING=EAGER is deliberately NOT exported here. It used to be
# worth 0.48 s of this leg, but that was before fractalyze/xla#698 moved the
# same work into the plugin's own module load: the leg now reaches its old
# EAGER ceiling with the variable unset, and what it adds on top is unmeasured.
# See "The plugin materializes the kernels now". If you do set it, set it in
# the environment -- the driver reads it at initialization and pil2 has
# initialized CUDA before any bridge client exists, so the bridge cannot.

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
| `ZZ_STAGING_THRESHOLD` | bytes at or above which the plugin DMAs a host-to-device transfer out of pageable memory instead of copying it through its pinned staging pool. Raising it above the bridge's 1.2–1.4 GiB sections grows the pinned pool inside the prove and costs more than the faster copies return on a guest that uploads each section once — measured, see "Staging the big uploads is a faster copy and a slower leg" | off: no option sent, so the plugin's own 1 GiB stands (also what a plugin older than fractalyze/xla#718 needs) |
| `ZZ_PENDING` | proves admitted per client on the device (one running, the rest uploaded ahead) | 2 |
| `ZZ_FIXED_AHEAD` | AIRs whose fixed sections may be uploaded ahead of the running prove's, per client; `0` sends every upload under the slot, and the value is capped at `ZZ_PENDING` — the permit is taken and given back inside that admission, so no more proves than it admits can hold one | 1 |
| `ZZ_RESIDENT_AIRS` | AIRs whose fixed sections stay on a client at once, least recently used evicted | 1 |
| `ZZ_HOST_THREADS` | threads for the host-side copies and key reads | half the cores, at most 8 |
| `ZZ_COMPILE_CACHE` | directory of serialized executables | `$ZZ_ARTIFACTS/.pjrt-cache` |
| `ZZ_LOG` | `1` per-instance timing on stderr, `2` per program; lines carry the seconds since bridge-up | off |
| `ZZ_AB` | prove through pil2 too and compare per instance | off |
| `ZZ_DUMP_PROOFS` | (fork) write every basic proof as raw words into this directory | off |
| `ZZ_DUMP_INPUTS` | write each instance as a `zz_prove` case directory under this one | off |
| `ZZ_DUMP_TRACES` | (fork) write each host trace as `gen_proof` receives it | off |
| `CUDA_MODULE_LOADING` | the CUDA driver's, not the bridge's: `EAGER` puts a module's kernel code on the device as it loads, process-wide. It was what made `ZZ_EAGER_MODULES` pay until fractalyze/xla#698 gave the plugin its own way to do the same thing for the bridge's executables alone — `ZZ_EAGER_MODULES` is now worth 6.167 → 5.517 s with this unset | driver default `LAZY` |

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

`--minus-call <name>` crosses the two, reporting what each phase keeps once
that driver call leaves the prove path. It is how a bridge-side lever is
sized against a bump that is already coming: with
`--minus-call cuGraphInstantiateWithFlags` the leg's largest phase rows on
hello-world (`lev`, the quotient chunks) fall to milliseconds, because they
were graph instantiation wearing a phase's name. It needs the API CSV and
refuses a name no call in the capture carries — subtracting nothing prints the
phase column back unchanged, which reads as "that call is free" rather than
"that is not its name". A call the capture *does* carry but which never ran
while the device starved is a real answer, and the report says so on the
header rather than leaving two identical tables to tell apart.

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

### The module-loading mode decides where the first-execution cost lands

`cuModuleLoadFatBinary` and `cuGraphInstantiateWithFlags` are not two costs.
They are one piece of first-execution work, and which of them pays depends on
the CUDA driver's module-loading mode.

CUDA 12 defaults `CUDA_MODULE_LOADING` to `LAZY`, under which loading a module
only registers its fatbin: each kernel's code reaches the device when something
first references it, and for the bridge that reference is the
`cuGraphInstantiateWithFlags` building a program's graph inside the prove slot.
So `eager_load_executable_modules` (#176) moves the registration to preload and
leaves the code load on the prove path. Under the profiler (2026-09-09, two
capture pairs, the bump-jax binary) enabling it took `cuModuleLoadFatBinary`
from 1.45-1.51 s to zero inside the leg and took `cuGraphInstantiateWithFlags`
from 0.06-0.12 s to 1.16-1.33 s, which moved the leg 5.10-5.11 s to
4.94-5.04 s — 2.3 %.

Unprofiled and interleaved (the arms below, five passes on `main`) the same
flag moves the leg 5.77 s to 5.72 s, 0.9 %, with the ranges overlapping. Take
that as the better-controlled figure and the 2.3 % as its profiled upper
bound: the flag's own effect on the leg is at most ~2 % and this design cannot
tell it from zero. What #176 bought is real but it is the driver-call
bookkeeping, not the leg.

`CUDA_MODULE_LOADING=EAGER` makes the driver do both at load time, which
`ZZ_PRELOAD` has already put off the prove path. Measured 2026-09-09 on one
binary, arms interleaved, five passes each on hello-world and six on the
block-shaped mix:

| arm | hello-world leg | block-shaped leg |
|---|---|---|
| `ZZ_EAGER_MODULES=0` | 5.77 s | 19.71 s |
| `=1` | 5.72 s | 19.61 s |
| `=1` + `CUDA_MODULE_LOADING=EAGER` | 5.24 s | 18.53 s |
| `=0` + `CUDA_MODULE_LOADING=EAGER` | 5.71 s | — |

Byte-gate green on every run gated — a sample of the sweep, not all of it: on
hello-world, passes 1/3/5 of each arm, 12 of the 20 runs, at 11 of 11 native
dumps each; on the block-shaped mix, passes 1/2/3 of the three arms that ran it
(`ZZ_EAGER_MODULES=0`, `=1`, and `=1` + `CUDA_MODULE_LOADING=EAGER`), nine of
the 18, at 38 of 38 — except one `ZZ_EAGER_MODULES=0` pass that aborted
mid-proof and matched on the seven dumps it had written. That abort is also
why the block-shaped `ZZ_EAGER_MODULES=0` leg above is a mean over five passes;
the other two arms have all six.

**The two knobs are only worth anything together.** The last row is the whole
argument: the driver variable with lazy executable loading buys nothing,
because there is no earlier place for the code load to go. Paired with the
preload it returns 0.48 s of the hello-world leg (8.4 %) and 1.08 s of the
block-shaped one (5.5 %), with graph instantiation dropping 1.32 s to 0.56 s
over a whole run.

That last row does a second job. `CUDA_MODULE_LOADING` is process-wide, so the
0.48 s arm also changed how pil2 loads its own modules, and a plugin-side
option scoped to the bridge's executables would not. The `=0` + `EAGER` arm is
what bounds that share: with the bridge's executables loading lazily, the
variable moves the leg 5.77 s to 5.71 s with the ranges overlapping. So at most
~0.06 s of the 0.48 s belongs to everything that is not a bridge executable,
and a scoped change should expect ~0.42-0.48 s rather than the whole of it.

Read the two shares the way the difference implies rather than picking one:
first-execution cost is paid once per (AIR, program) pair however many
instances follow, so the *share* falls as instances per AIR rise and the
*absolute* figure travels. 5.5 % is an upper bound for block-shaped work, not
a constant; 1.08 s is the portable number.

**The bridge cannot set this itself.** The driver reads the variable when it
initializes, and pil2 has initialized CUDA before any bridge client exists:
setting it in `artifact::new_session` was measured as a no-op (leg 5.65 s
against 5.17 s for the same binary with the variable set in the environment,
four interleaved passes each). It has to be set before the process starts, or
the plugin has to materialize the kernels itself after loading a module —
which is where the durable fix belongs: beside `eager_load_executable_modules`,
scoped to the executables loaded through it rather than to every module the
process loads.

### The plugin materializes the kernels now, and it is worth what the bound said

[fractalyze/xla#698](https://github.com/fractalyze/xla/pull/698) made
`eager_load_executable_modules` enumerate an executable's kernels and load each
one (`cuModuleEnumerateFunctions` + `cuFuncLoad`, CUDA ≥ 12.3), so the code
load happens where the module load already does rather than at first
reference. Measured on the wheel that carries it, five arms interleaved pass by
pass, four passes each, one binary and one artifacts directory with both plugin
builds warm. Four of the arms are the module-loading question; the fifth is the
staging threshold, and it has its own section below:

| arm | | leg, median | sd |
|---|---|---|---|
| `old` | the previous wheel, eager module loads on | 5.970 s | 0.087 |
| `oldctl` | + `CUDA_MODULE_LOADING=EAGER` | 5.675 s | 0.307 |
| `newoff` | this wheel, eager module loads **off** | 6.167 s | 0.018 |
| `new` | this wheel, eager module loads on | **5.517 s** | 0.240 |
| `newstg` | this wheel, eager on + staging at 2 GiB | 5.986 s | 0.262 |

The short names are this page's handle for these five arms; sections below
cite them.

**−0.453 s**, against the ~0.42–0.48 s the `=0` + `EAGER` arm above bounds it
at. The scoped change reaches the process-wide variable's ceiling — `oldctl`
and `new` overlap — without changing how pil2 loads its own modules.

The driver calls say the same thing directly. Over one capture per arm,
`cuGraphInstantiateWithFlags` falls **1.457 s → 0.359 s** across the same 245
calls, `cuFuncLoad` appears where it did not exist (0 → 6140 calls, 0.162 s),
and `cuModuleLoadFatBinary` is unchanged at 0.21–0.23 s over 369 calls —
that one was already moved by
[#664](https://github.com/fractalyze/xla/pull/664). So the first-execution work
is not removed, it is relocated a second time: out of the prove's
`cuGraphInstantiateWithFlags` and into the load, which `ZZ_PRELOAD` has already
put off the prove path.

Note what this does to the knob's history. The same
`ZZ_EAGER_MODULES=0 → 1` that was worth 5.77 → 5.72 s (null) before #698 is
worth 6.167 → 5.517 s after it. The flag was never the lever; it was the
place to put one.

### Staging the big uploads is a faster copy and a slower leg

The same wheel carries
[fractalyze/xla#718](https://github.com/fractalyze/xla/pull/718), which turns
the plugin's 1 GiB host-to-device staging cutoff into the
`staging_threshold_bytes` create option. The bridge's four largest uploads sit
above that cutoff, so they were being DMA'd out of pageable memory; setting the
option above them (`ZZ_STAGING_THRESHOLD`) moves them onto the pinned path, and
it does exactly that:

| one capture each | option off | option at 2 GiB |
|---|---|---|
| bridge uploads, pageable | 1205 copies, 5.29 GB at 11.1 GB/s | 1201 copies, 0.01 GB at 11.3 GB/s |
| bridge uploads, pinned | 356 copies, 4.46 GB at 43.0 GB/s | 360 copies, 9.75 GB at 46.4 GB/s |
| the four over 1 GiB | 105, 73, 66, 234 ms — all pageable | 26, 23, 30, 28 ms — all pinned |
| upload time inside the leg | 0.31 s | 0.12 s |

**And the leg gets worse by 0.469 s** (5.517 s → 5.986 s, four interleaved
passes each — the `new` and `newstg` arms of the same sweep). The pinned pool
has to grow to hold a 1.4 GB transfer and pays for it inside the prove:
`cuMemHostAlloc` goes from 0.258 s over 16 calls to 1.071 s over 17. One
allocation costs more than every faster copy returns, because this guest
uploads each large section once.

So the option ships **off**. A workload that uploads the same large section
repeatedly would amortize the pool growth this one cannot — the block-shaped
mix is where that would show, and it is unmeasured, which is why the knob
exists and why its default is the case that was measured.

Two traps for anyone re-running this. The pageable copy *count* barely moves
(1205 → 1201): those are sub-megabyte XLA runtime internals on the compute
stream, not the bridge's uploads, so read the bytes and the rate, not the
count. And `bench/h2d_overlap.py` is the instrument — it attributes by stream
and reports `SrcMemKd`; a hand-rolled filter on `nsys_trace.owner()` gives
`bridge` for both provers' copies, because `owner()` keys on `(` in the name
and no memcpy row has one.

A caveat for anyone sizing a lever off a per-program table. Across three
captures of one arm (`ZZ_EAGER_MODULES=1`), a program's instantiate cost moves
by more than most levers are worth: `fri_fold_0` 0.004 / 0.181 / 0.002 s,
`deep` 0.106 / 0.108 / 0.002 s, `quotient_1048576` 0.490 / 0.387 / 0.275 s.
The third capture is a different binary, which accounts for some of `deep`'s
spread but not `fri_fold_0`'s — that one moves 0.18 s between two captures of
one arm on one binary. So per-program attribution from a single capture
supports claims above roughly **0.2 s** and nothing below, and two captures of
one arm is the cheapest way to confirm that floor before trusting a table.

### A phase's share of the idle is where the device waits, not what for (2026-09-10)

The report above charges every idle nanosecond to the host phase that was
running. That is an exact split, and it is still not a list of levers: twice
now, a change that removed a large share outright has left the leg where it
was, because the cost re-appeared in the phase next door.

Measured on the hello-world guest with `ZZ_EAGER_MODULES=1`, one binary per
arm, arms interleaved pass by pass so run order cannot favour one, leg from
proofman's `GENERATING_INNER_PROOFS`:

| arm | what it removes | leg, mean [min-max] |
|---|---|---|
| baseline | — | 5963 ms [5818-6121] |
| `FIXED_AHEAD = 0` | the fixed-section read-ahead, so every upload is under the slot | 6047 ms [5886-6210] |
| `constants` shared per program | 9 of 11 runs of `constants` | 6010 ms [5820-6198] |

> Read these against each other, not against the 5.72 s the `=1` arm shows
> above: that table is another session's, and the absolute leg and init on
> this host are not reproducible across sessions. 39 runs over two of them
> failed to explain the level — run order moved init 0.57 s in one session and
> nothing in another, and two same-binary populations ten minutes apart
> differed by 1.18 s. Every arm here is interleaved against the baseline beside
> it, minutes apart, which is what makes the comparison sound while the level
> is not. Take a baseline in your own session and never quote a cross-session
> delta.

Both arms are nulls, and the phase table says why. Dropping the read-ahead
grows `host/fixed_install` (0.37-0.50 s to 0.62-0.89 s) and shrinks `constants`
(0.46-0.77 s to 0.32-0.64 s); sharing `constants` takes its row to zero and
grows `host/fixed_install` to 0.74-0.90 s with `const_setup` and `commit1`
taking the rest. The sum over the fixed-section install — `constants`,
`host/fixed_install`, `const_setup`, `custom_setup_*` — is what stays put. It
is one quantity, and which phase is holding the bag when the device starves is
not a property of the bridge's scheduling.

This is the same shape as the module-load result above, where moving the loads
off the prove path re-priced them into `cuGraphInstantiateWithFlags` instead of
recovering them, and it is why the report's own header calls a phase's share an
upper bound on what removing it returns.

**So run a positive control before believing a null on this leg.**
`CUDA_MODULE_LOADING=EAGER` is the one to use: same binary, an environment
variable, no build, and an effect of the size most bridge-side levers are
sized at. Five interleaved passes each on the arms above's baseline binary:

| arm | leg, mean [min-max] |
|---|---|
| unset (the driver's `LAZY` default) | 6099 ms [5987-6219] |
| `CUDA_MODULE_LOADING=EAGER` | 5510 ms [5264-5690] |

0.59 s apart with the ranges disjoint, which is what says a 0.4-0.6 s effect
would have shown in the table above had one been there. A null quoted without
a control like this says only that the harness did not see anything.

**Why neither removal recovered anything.** The same three captures answer it,
because `nsys` sees more than kernels. Splitting each phase's idle by whether
the device was moving bytes or doing nothing at all — counting only the
*bridge's* copies, which under `cargo-zisk` means the ones PJRT issues through
the CUDA driver API, since pil2 shares the process and owns more of the
traffic than we do — the fixed-section install (`constants` +
`host/fixed_install` + `const_setup` + `custom_setup_*`) is 0.93-1.51 s of
idle, of which only 0.27-0.35 s is host-to-device transfer: **71-77 % is dead
device time, no kernel and no copy.** The bridge's whole H2D is 9.75 GB a run
("The uploads, measured" below), and its upload calls cost exactly their DMA
(`const_base` 1408 MB in 68.07 ms against 67.88 ms of DMA), so the uploads are
neither a bandwidth floor nor a staging cost. Nor is the dead time the
allocator reclaiming the AIR just evicted: against the size of what was freed,
r = -0.15 over 30 installs, and the proves that freed the most were faster.

What it is, is a cost with no per-AIR structure. `constants` is the cleanest
probe in the leg — one program, no inputs, identical outputs on every prove of
a given size — and its dispatch cost for **the same AIR** across three captures
of one arm runs 0.82 / 62.97 / 95.94 ms (`Rom_n22`), 233.03 / 0.63 / 26.99 ms
(`Binary_n22`), 8.13 / 3.51 / 386.91 ms (`VirtualTableZisk0_n21`). Correlating
the eleven AIRs between captures gives r = -0.28, -0.27, -0.30 — no structure,
if anything anti-correlated. The per-run total carries (473 / 614 / 779 ms);
which prove pays it is redrawn every run.

So the cost is not in the phase, which is why removing a phase cannot remove
it, and not in the AIR, which is why residency and ordering cannot reach it. It
lands wherever the holder happens to be. Almost none of it is inside a CUDA
driver call, so what is left is XLA/PJRT host code between `Artifact::run` and
the device having work — the same place graph instantiation lives, but making
no driver call at all. **Read this as a bound on bridge-side scheduling work in
the leg, not as a lever waiting to be pulled**: three captures cannot separate
"no per-AIR structure" from "structure far below the share being sized", and
either way a change to what the bridge schedules is not what reaches it.

**The check to run before building any lever that moves or removes a phase.**
It costs one extra capture of the arm you already have, and it predicts the
result:

```bash
# Two or three captures of ONE arm, then the same report on each.
for c in c1 c2 c3; do
    bench/host_idle.py ${c}_cuda_gpu_trace.csv ${c}_nvtx_pushpop_trace.csv \
        ${c}_cuda_api_trace.csv --minus-call cuGraphInstantiateWithFlags
done
```

Compare the phase you mean to attack across the captures. A phase whose share
moves by more than the win you are sizing is not a lever, however large its
mean: the cost is landing there rather than living there, and moving the phase
will move the cost somewhere else in the same run. `constants` above swings
0.46-0.77 s across three captures of one arm while the win being sized was
0.4-0.6 s — the check fails, and both removals that were built on it measured
null. Sharper still if the capture lets you name the per-prove unit: correlate
the same AIR's cost between captures, and an r near zero says the phase is not
where the cost lives.

This is the same discipline as the positive control, from the other side. The
control asks whether the harness could see the effect; this asks whether the
effect is attached to the thing you are about to change.

## Status (2026-09-06, RTX 5090, block-shaped sha-hasher workload)

> Measured 2026-09-06, on that date's binary and plugin. Do not adjust these
> figures for `CUDA_MODULE_LOADING=EAGER`: the 2026-09-09 arms above read
> 19.61 s of leg without it and 18.53 s with it, both under the 20.3-20.7 s
> here, so this table is stale by more than that one knob. Take the shape of
> the gap from here and the leg from those arms.

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

## Status (2026-09-11, RTX 5090, go hello-world guest)

Where #170 leaves this guest. On `main` at f5bd8fb, frx quad pinned to
`0.10.2.dev20260910150749` (fractalyze/jax@1b7c92fe, xla `fff9509ab012`, which
carries fractalyze/xla#698 and #718), proving key v1.0.0-alpha, artifacts
`zz-artifacts-191`, warmth probed for the pinned plugin before the sweep
(`zz_prove --warm`, three AIRs, every program from the cache).

Every row names the session that measured it, and none of them is re-derived
from another. That is not bookkeeping: on this rig both init and the leg drift
between sessions by more than most of the levers #170 chased are worth, so a
figure without a native baseline taken in the same session and interleaved with
it says nothing about either stack — which is what "Bridge start-up" below had
to establish the expensive way.

| | the figure | where it comes from |
|---|---|---|
| inner-proof leg | bridge **5.327 s**, native **3.455 s** (1.54x) | #214, both arms in one session, one binary, interleaved pass by pass and rotated within a pass, eight passes each; medians over passes 2–8, bridge sd 0.136 [5.195–5.608], native sd 0.124 [3.319–3.668] |
| proofman init, bridge minus native | **0.248 s** behind, on a page cache warmed to the set init reads | #217, on the bumped wheel, both arms interleaved and rotated in one session, three passes each, that set warmed immediately before every run: bridge 3.107 s [3.046–3.123] against native 2.859 s [2.820–2.881], ranges disjoint. Counting each bridged run's own client creation (0.181–0.192 s, `ZZ_LOG`'s `bridge up`) it is 0.43 s. **The cache state is part of the figure**, which is why this row names it: with that same set evicted and nothing else changed the gap is 3.478 s (bridge 8.434 s against native 4.956 s), and on the uncontrolled cache of #217's own predecessor sweep it is 0.128 s. All three are inside the criterion; none of them is the difference without a state. A difference, not two levels: neither stack's init has a level this page will quote (see "Bridge start-up"). #214's 0.17–0.22 s is the same quantity on an uncontrolled cache and is not contradicted, only unquotable on its own — see "What sets a run's init is the page cache" for which to read and why |
| basic proofs byte-identical to native's | 11 of 11 | #214, two independent pass pairs out of the interleaved sweep, each a native and a bridged run in the shipped configuration, compared by `bench/compare_dumps.py` |
| `MemAlign_n21` first prove, bridge | 0 of 60 wrong | #204, 60 fresh processes doing one first prove each under `CUDA_LAUNCH_BLOCKING=1` |

The leg is proofman's own `GENERATING_INNER_PROOFS` on both sides.

**Both leg figures are from one session and one binary**, which is what the
rule above asks for and what the table carried until #214 spent the run. The
pair it replaces was #204's bridge arm against #209's native arm, taken a day
apart; the two designs agree on the gap to within 0.055 s (1.817 s then,
1.872 s now), so nothing downstream of the older pairing moves. Where the gap
sits is "The gap is the basic phase's wall" below.

Three figures the older table carries are absent here rather than stale: the
`cargo-zisk prove` wall, the eleven proves' own time on the client, and the
summed client waiting. Nothing since that table has quoted them, and the runs
that could have yielded them are off the rig, so the rows would need a fresh
sweep rather than a re-read. Take their shape from "The per-stage shape" below
without carrying the values forward. The fourth, the fixed sections uploaded
under the prove's own slot, is current and lives in "Raising the read-ahead
permit": 1.42 s summed at the default permit, on the pre-bump wheel.

Reproduce with the same `bridge/bench/` scripts as the block-shaped
section, minus the input: the guest takes none, and it needs
`ZISK_PROVE_FLAGS=` (empty) on a host with no ASM emulator built, since
run.sh's default is the ASM emulator's `-a -u`. So
`ZISK_PROVE_FLAGS= run.sh <tag> native|bridge`, then `compare_dumps.py`
for the byte-gate and `summarize.py` for the rows. "Memory budget" below
was measured this way, adding `ZZ_MEMORY_FRACTION` and
`ZZ_GPU_HEADROOM_GB` per run. An arm is an env swap rather than a rebuild,
so interleave the arms pass by pass and rotate them within a pass.

### The acceptance #170 set, and where it lands

| criterion | verdict |
|---|---|
| inner-proof leg within 1.2x of native's (≤ 4.15 s against 3.455 s) | **not met** — 5.327 s is 1.54x, over by 1.181 s |
| proofman init within 0.5 s of native's | **met on the baseline the recipe now guarantees** — the set init reads warmed: 0.248 s behind by proofman's own timer, 0.43 s counting the bridge's client creation, #217. Not met on a cold page cache, where the same pair is 3.478 s apart. The criterion is only readable with the cache state named, which is why `bench/run.sh` sets that state and records it per run |
| all 11 basic proofs byte-identical to native's dumps | met — 11 of 11 |
| this section traces every number to a run recipe and a commit | met by the table above |

The leg is the criterion that did not close, and nothing on the list below
closes it: of everything #170 tried, only eager kernels moved the leg, and the
1.181 s the bridge sits above the 1.2x bar is more than twice the 0.453 s that
one was worth. The gap to native is a wider figure measured to a different
reference — 1.872 s, of which the bar forgives the first 0.691 s — so the two
are not quantities to subtract from each other. What the rest of it is, the
section below now says: it is the basic phase's wall, and one client is why.

### The gap is the basic phase's wall

#214 spent the run the Status table had been waiting for: both arms in one
session, one binary, one plugin, interleaved pass by pass and rotated within a
pass, eight passes each. Medians over passes 2–8, the first pass of each arm
dropped as the sweep's own first run:

| | native, pil2's three streams | bridge, one client | difference |
|---|---|---|---|
| inner-proof leg | 3.455 s | 5.327 s | **+1.872 s** |
| the basic phase's wall | 2.343 s | 4.279 s | **+1.936 s** |
| `leg − basic phase` | 1.112 s | 1.048 s | −0.064 s |

The basic phase is the wall in which that arm's eleven basic proofs were
running — the union of their intervals, not their sum. On native they are
proofman's `GEN_PROOF_n` spans. Under the bridge they are not: `gen_proof`
returns as soon as the work is handed to a worker, so those spans run 1–438 ms
against proves that take seconds, and the phase is the union of the `ZZ_LOG`
per-instance intervals from where an instance takes the client to where it
gives it back.

Every difference in this section is between the two columns' medians, which is
what makes the rows sum to the gap exactly. Where the median of the per-pass
differences disagrees it is given too: here it is −0.219 s (native 1.230 s,
bridge 1.011 s, sd ~0.2 on each) against the −0.064 s above, and both sit
inside the ~0.2 s floor, so read `leg − basic phase` as "small, sign not
established".

**The gap is the basic-phase row — and `leg − basic phase` is a residual, not
the recursion's cost.** Both arms run the same seventeen recursive proofs on
pil2, and the residual holds only the part of them the basic phase did not
already cover. How much that is moves with how long the basic phase is:

| | native, n=7 | bridge, n=7 | native, one basic stream, n=3 |
|---|---|---|---|
| the recursion's wall | 2.640 s [2.330–2.764] | 3.006 s [2.231–3.513] | 2.695 s [2.509–2.938] |
| of which inside the basic phase, at least | 1.528 s | 1.958 s | 1.977 s |
| `leg − basic phase` | 1.112 s | 1.048 s | 0.718 s |

The overlap row is `basic + recursion − leg`: both phases sit inside the leg,
so whatever they cover past its length they cover at once. It is a lower bound
and it needs no common clock, which matters because the bridge's basic phase is
read off the bridge's clock and its recursion off proofman's. Like every other
row here it is computed from the medians above it, so it reconciles with them;
`leg_phases.py` also prints the median of the per-pass bounds, which is a
different statistic of a different thing and reads 1.419 / 1.914 / 1.965 s.

**The bridge's recursion wall sits 0.366 s above native's, and that is not a
share of the gap.** Its range contains native's whole range — it is the widest
quantity in this section, sd 0.422 against the leg's 0.136 — so the difference
between the two medians is not resolvable on seven passes. What is structural
is the shape rather than the size: native's seventeen recursive proofs overlap
one another, 5.413 s of spans inside a 2.640 s wall (2.05x), because they
contend with its three basic streams and each one's span inflates while it
waits; under the bridge, where pil2 has no basic proofs of its own to run, they
go through clean and serial at 1.00x. Either way it does not reach the leg,
because the residual row is where it would show and that row is −0.064 s.

And the one-stream column shows why a residual must not be read as the
recursion's cost even inside one stack: its recursion is unchanged while its
residual falls to 0.718 s, purely because a longer basic phase hides more of
it. What the leg table establishes is the identity — leg is the basic phase's
wall plus whatever is left — and that the whole of the gap sits in the wall.

Within the basic phase the bridge achieves 1.00x concurrency — 4.283 s of
proving in 4.279 s of wall, which is one client doing eleven proves back to
back and nothing else. Native's is 2.22x by the same arithmetic (5.199 s of
spans in 2.343 s of wall; 2.39x if the ratio is taken per pass and those
medianed), but either overstates what its streams buy, because proofs on three
contending streams each take longer than they would alone. The
honest pivot is to force the same stack serial, which the fork's headroom knob
does (`ZZ_GPU_HEADROOM_GB=15`, one basic stream, three runs):

| | wall of the basic phase | concurrency | leg |
|---|---|---|---|
| native, three basic streams (the baseline arm, n=7) | 2.343 s | 2.22x | 3.455 s |
| native, one basic stream (diagnostic, n=3) | 3.164 s [3.123–3.172] | 1.00x | 3.882 s |
| bridge, one client (n=7) | 4.279 s | 1.00x | 5.327 s |

So the phase's +1.936 s splits into **+0.821 s** that pil2's three streams buy
it (3.164 → 2.343 s, a 1.35x speedup rather than 3x, because the streams
contend for one card) and **+1.115 s** by which the bridge's serial prove is
dearer than pil2's serial prove (4.279 against 3.164 s). With the rest row those
three sum to the 1.872 s gap. Two caveats on the split and neither on the sum.
The headroom knob shrinks pil2's own buffers as well as its stream count, so
the 3.164 s pivot carries that and the shares either side of it do too. And the
pivot run's leg is not the baseline's with one term swapped: its remainder is
0.718 s against the baseline arm's 1.112 s, which is the residual above moving
with the basic phase rather than the recursion changing. The +1.936 s phase
share depends on neither — it is two walls, each measured on its own arm.

**Of the bridge's serial phase, its own kernels cover 2.58 s.** Three captures
of the bridge arm put it at 2.58 / 2.59 / 2.58 s — the steadiest figure in this
section — against phases of 4.41 / 4.53 / 5.08 s, so 1.83 / 1.94 / 2.51 s of it
is the client's device idle. Carried onto the unprofiled 4.279 s phase that is
about 1.7 s, a number that crosses a profiled quantity into an unprofiled
window and should be read to one digit. The part that costs the leg is the part
with the client held, 1.611 / 1.939 / 2.506 s across the three.

**None of that idle is a lever, and the captures say so themselves.** Its
phases move more between captures of one arm than any of them is worth:
`constants` 0.364 / 0.735 / 0.937 s, `host/fixed_install` 0.519 / 0.429 /
0.566 s, on one binary in one session. That is #205's result, re-confirmed on
the shipped wheel and now with the thing it is a share *of* measured beside it.
`--minus-call cuGraphInstantiateWithFlags` takes the held idle from 1.611 s to
1.320 s in the first capture, and that 0.291 s is the plugin's rather than the
bridge's.

The bridge's kernels are not where its leg goes. Its eleven proves carry
2.58 s of kernel time where pil2's own per-instance timers put its eleven at
3.310–3.678 s (`bench/pil2_timers.py` on the one-stream runs). Those are
different instruments — nsys kernel spans against pil2's CUDA-event timers,
which for the same eleven proofs exceed proofman's span for them by 0.19, 0.41
and 0.51 s over the three one-stream runs — so it is a direction, not a
subtraction.

What this sizes for a second client (#215), as arithmetic on this section's
shares rather than a measurement: at pil2's own 1.35x the bridge's basic phase would
be 3.17 s and its leg about 4.2 s, still over the 4.15 s that 1.2x of native
allows; at perfect packing the phase cannot go below the 2.58 s of kernel time
it carries, which puts the leg at about 3.6 s. So a second client is the only
share anyone has left to take, it is worth about 1.1 s of the 1.9, and it does
not on its own close the criterion.

It also does not fit on this card. #215 re-walked the memory floors on this
wheel: a client needs an 11.60 GiB arena and two of them can have at most 8.47
GiB each before pil2 refuses to start, so the second client is 3.1-3.5 GiB per
client out of reach and the 1.1 s above stays arithmetic. "Memory budget" has
the walk, what is in the arena, and the sizes a memory unit would have to move.

Reproduce: `ZISK_PROVE_FLAGS= bench/run.sh <tag> native|bridge`, the bridge arm
with `ZZ_EAGER_MODULES=1 ZZ_STAGING_THRESHOLD=0` and run.sh's own `ZZ_CLIENTS=1
ZZ_MEMORY_FRACTION=0.45`, alternating the arm order pass by pass. Probe warmth
before the first timed run: `zz_prove --warm <artifacts> <one AIR>`, per plugin
the sweep will use. run.sh warms the proving-key set init reads before each run
and records the census beside the log; if you drive the prover directly, warm
it yourself (`bench/pagecache.py --warm --proofman-init $PK`) or the init
column is measuring the host's page cache.

Every wall in this section — 2.343, 4.279, 3.164 s and the recursion and
overlap figures beside them — comes out of **`bench/leg_phases.py <run.log>...`**,
which is where the union-of-spans arithmetic and the rule for finding a
bridged run's basic proofs live; hand it a whole arm's logs and it prints the
median and range per quantity. The one-stream pivot is the same script on a run
made at `ZZ_GPU_HEADROOM_GB=15`, and `bench/pil2_timers.py` on that run is what
gives its per-instance timers. For the device idle inside the phase, the
capture recipe in "Profiling" and `bench/host_idle.py`;
`bench/compare_dumps.py` between a native and a bridged run's dumps is the
byte-gate.

**What the rotation caught about init.** A run's `INITIALIZING_PROOFMAN`
tracks the *previous run's arm*, in both stacks. Of the sixteen runs, fifteen
have a predecessor, and they split 3.07–3.85 s (n=7) after a native run against
4.04–7.74 s (n=8) after a bridged one; dropping the two that immediately follow
the sweep's start — the runs still filling a cold page cache, at 3.851 and
7.737 s — leaves 3.07–3.33 s and 4.04–4.27 s, which are disjoint. It is not the
arm being timed: native and bridge each appear in both groups, and the
alternation rules out drift. That is a candidate for the regime #178 found and
could not select ("Bridge start-up"), and it is worth about 0.95 s — four to
five times the arm difference underneath it. The predecessor's arm turned out
to be a proxy rather than a mechanism: #217 isolated it and found what it
stands for, which is the section after next.

Which is why the arm difference has to be read *within* a predecessor group,
and can be: the rotation puts both arms in both groups.

| init | after a native run | after a bridged run |
|---|---|---|
| native | 3.101 s [3.074–3.109], n=3 | 4.081 s [4.042–4.133], n=4 |
| bridge | 3.320 s [3.307–3.326], n=3 | 4.254 s [4.246–4.269], n=3 |
| bridge − native | +0.219 s | +0.173 s |

Ranges are disjoint in both groups, and the two groups agree on the difference
to 0.046 s while disagreeing on the level by 0.95 s. Add each bridged run's own
client creation, which proofman's timer starts after — median 0.186 s,
[0.183–0.309], from `ZZ_LOG`'s `bridge up` line — and the bridge is 0.36–0.41 s
behind. Both readings are what the Status table's init row carries, and both
agree with #178's pre-bump pair (0.10–0.23 s by the timer, 0.28–0.41 s with
client creation). The two runs dropped above are dropped for the level, not the
difference: they are one arm each and sit either side of it.

### What sets a run's init is the page cache (2026-09-11, #217)

The rotation above read init against the arm that ran *before* it. Isolate that
predecessor and it stops predicting anything. Four cells — native or bridged
first, native or bridged measured — three repeats each, the pairs run back to
back and the cell order rotated each repeat so a cell's grand-predecessor is
not the same arm every time. Only each pair's second run is quoted:

| init | after a native run | after a bridged run |
|---|---|---|
| native | 3.286 s [3.255–3.712], n=3 | 3.279 s [3.204–4.850], n=3 |
| bridge | 3.409 s [3.349–3.412], n=3 | 3.437 s [3.376–3.535], n=3 |

No step. Read down instead of across and the arm difference is still there at
+0.12 to +0.16 s (pooled over the predecessor, which is a null: +0.128 s);
read across and the 0.95 s #214 measured is gone. **That is not the figure the
Status table quotes**, and the difference between them is this section's
subject rather than a discrepancy in it — see "Which of the two to quote"
below. Both outlying runs — 4.850 s, and 5.094 s on
the predecessor side of a pair — are in the sweep's first four, and both
started with part of one file family missing from the page cache. Every other
run in the sweep started with that family whole.

**Which family, and why init cares.** `INITIALIZING_PROOFMAN` splits at two
landmarks proofman prints: the buffer sizes it announces, then
`LOADING_FIXED_POLS`. Across all four cells above and both states below, the
second and third parts do not move — the GPU allocation stays inside
0.27–0.34 s and `LOADING_FIXED_POLS` inside 0.67–0.87 s. Everything that moves
is in the first part, before proofman has sized anything, and that part is a
file read. A run polled from a fully evicted key fetched 8.57 GiB from storage
inside it (`/proc/<pid>/io` `read_bytes`, sampled against the log's own
landmarks) and gained 8.44 GiB of resident proving key over the same window:
the const pols in GPU layout, the `.exec` and `.dat` files of the recursion
setups, and the small binaries and JSON beside them. The constant trees are not
in it — the allocation and `LOADING_FIXED_POLS` pull one air's worth each, and
the rest of that 34 GiB belongs to the leg.

**So set it directly.** The same runs with only that set's cache state flipped
immediately before each one — read back in, or dropped with `posix_fadvise` —
interleaved and rotated, three repeats a cell:

| init | the set warm | the set evicted |
|---|---|---|
| native | 2.859 s [2.820–2.881] | 4.956 s [4.918–4.979] |
| bridge | 3.107 s [3.046–3.123] | 8.434 s [8.361–8.472] |

Ranges are disjoint by seconds, and it lands where the poll said it would: the
pre-allocation part goes 1.877 → 3.973 s on native and 2.132 → 7.481 s on the
bridge, while the allocation and `LOADING_FIXED_POLS` sit still in every cell.
The leg is the control and does not move (native 3.445 s warm against 3.380 s
cold, bridge 5.182 against 5.382): the arms differ in the files init reads and
in nothing else, which is why `.const` and the constant trees are left alone by
both. **That makes it a control for this experiment, not a general one** — the
leg reads the constant trees, and no arm here evicts them, so nothing above
says what the leg does when *they* are cold.

**Which of the two to quote.** These are two measurements of one difference on
two different baselines, and the page's own rule is that a figure names its
baseline. The four cells ran on whatever cache state the previous run left:
`.const_gpu` was resident in 11 of the 12, but nothing censused `.exec` and
`.dat`, and their absolute init sits about 0.4 s above the warmed arms' —
which is what a partly cold set looks like. The warmed arms are the only ones
whose state was set. So **criterion 3 reads the warmed figure, +0.248 s**, and
the sweep's +0.128 s is a consistency check rather than a second quote: it was
designed to test the predecessor, not to resolve a tenth of a second between
the arms, and its native arm carries a 4.850 s outlier that its median only
survives by being a median. Both are far inside the 0.5 s the criterion
allows. What neither licenses is a bare number: the same difference is
+0.128 s on an uncontrolled cache, +0.248 s warmed and +3.478 s cold, and it is
not a monotone function of warmth, because the two arms are not hurt equally by
a partial one.

**The bridge is not the actor here, and neither is either stack.** The cold
penalty is 2.1 s on native and 5.3 s on the bridge — a bridged run's init also
holds its own client creation and preload, which queue behind proofman on the
same disk — but the state that decides it belongs to the host, not to the run
before. In this session the set survived every bridged run; what emptied it was
a sibling session's build server starting between two of this unit's own
sweeps, after which the key held none of the 18 GiB it had held minutes
earlier — while free memory went *up*. #214's session was one
where a bridged run did the evicting, and the arm inherited the credit. Any
process on this machine that reads a few GiB is a large enough actor.
fractalyze/zisk-zorch#222 would shrink the bridge's own contribution to it; on
this evidence that makes the bridge a smaller tenant, not the one that matters.

**What this costs a reader of init.** The bridge is 0.248 s behind native with
the set warm and 3.478 s behind with it cold, same binary, same session. A
criterion of the form "init within 0.5 s of native's" is therefore a statement
about the page cache as much as about either stack, and an init figure with no
cache state beside it says nothing. `bench/run.sh` warms the set before every
run and leaves the census in `pagecache.txt` beside the log, so the state is
recorded rather than assumed — on every run, warmed or not, since a cold run is
exactly the one whose figure depends on the state. The cold arm above is that
step inverted: `--evict` the set, then `ZZ_WARM_KEY=0` so run.sh censuses it
and leaves it evicted.

```bash
# the reset, the check that it took, and the cold arm on purpose
bench/pagecache.py --warm --proofman-init $PK
bench/pagecache.py --proofman-init $PK          # census only, faults nothing in
bench/pagecache.py --evict --proofman-init $PK
```

Three native/bridge pairs out of these sweeps were compared with
`bench/compare_dumps.py`: 11 of 11 basic proofs byte-identical in each.

### The levers, each closed with a measurement

| lever | what it was worth |
|---|---|
| eager kernels at module load (xla#698, landed by #204) | **−0.453 s** — #204's wheel bump with eager module loads already on (`old` 5.970 → `new` 5.517 s). The only lever that moved this leg, and it lands at the process-wide `CUDA_MODULE_LOADING=EAGER` ceiling. "The plugin materializes the kernels now" |
| eager module loads on their own (#176 / xla#661) | null — 5.77 → 5.72 s toggling the flag on the pre-#698 wheel, ranges overlapping. It moves the registration to preload and leaves the kernels' code on the prove path, so there is nothing to collect until #698 puts that at module load too. "The module-loading mode" |
| upload overlap (#193) | null — an upload into a freshly allocated buffer waits on the client's own compute stream, so it never overlaps that client's kernels. "The uploads, measured" |
| host-idle remainder (#205) | null — two built changes measured null against an `EAGER` control on the same binary; the cost re-prices into the phase next door. "A phase's share of the idle" |
| read-ahead depth (#209) | null — −26 ms paired, smaller than the −120 ms that two labels of a single configuration differed by in the same sweep. "Raising the read-ahead permit" |
| constant tree over the extended domain (#183, #206) | worse — the same 8.4 GB over the same read-ahead path took the leg 6.1–6.6 s to 7.3–7.5 s |
| H2D staging threshold (#204 / xla#718) | **+0.469 s**, so it ships off — the copies do get faster, and the pinned pool's growth inside the prove costs more than they return. "Staging the big uploads" |
| XLA fusion cap (#149) | retracted — the flag has no occurrence in this wheel, its replacement measured a net tree regression, and under the bridge pil2 proves the recursion tree on its own CUDA, where an XLA fusion cap has no surface |

**The two eager rows do not add, and must not be subtracted from each other.**
Each is a different baseline: −0.453 s toggles the wheel with the flag on,
−0.05 s toggles the flag on the wheel that predates #698, and toggling the
same flag on the post-#698 wheel is worth −0.650 s (`newoff` 6.167 → `new`
5.517 s, #204). The parts sum to −0.503 s against that −0.650 s, and the
0.147 s is not a lever anyone has left to claim: the two mechanisms gate each
other, since #698 has nothing to do without an eager module load and the flag
had nothing to collect before #698. Read the pair from one session's own two
arms, never by adding a row here to a row there.

### The per-stage shape (2026-09-04, superseded)

Kept for the per-instance breakdown, which nothing since re-measures. Every
figure here is from 2026-09-04's binary and plugin and none of them is current:
the leg alone has moved 6.5 s to the 5.327 s the Status table now carries,
across the units #170 landed and a wheel bump, so do not adjust these rows for
one knob and do not quote them. Take the
shape of the gap from here and every number from the table above.

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

### Bridge start-up (2026-09-09/10, post-#176)

The bridge's start is hidden inside proofman's init with time to spare.
`ZZ_LOG` timestamps a run against that start, so the three figures after
this colon are on the bridge's zero rather than proofman's: over 23 runs at
the default six preload threads, the client is up at 0.16–0.19 s, the
whole preload — 11 AIRs, 380 programs out of the cache — is done at
1.03–1.19 s, and `INITIALIZING_PROOFMAN` does not end until 3.40–5.19 s.
proofman's own timer starts once the client is up, so the same runs read
0.16–0.19 s shorter on it — 3.23–5.00 s — and that is the figure the rest
of this section compares against native. The preload finishes
2.30–4.09 s before init does, in every one of them, a difference of two
timestamps on the same zero and so the same on either clock. Starving it
makes that point rather than breaking it: at three preload threads it
lands 1.90–2.03 s early, and at two, where it takes nearly twice as long
to run, still 1.49–1.56 s early.

Beside native, treat the bridge's init as bounded rather than known. It
was 0.10–0.23 s behind on one session, taken pass by pass across four
interleaved passes rather than as an envelope over the two arms' ranges,
or 0.28–0.41 s adding each run's own client creation; and ahead of
native on both paired sets of another. Do not read the levels as a
property of either stack: over 31 bridge runs and 15 native ones, on
identical source, wheel and artifacts, the bridge's init ranged
3.16–5.00 s and native's 3.00–5.75 s. The two covariates that look explanatory each fail
somewhere. Run order was worth 570 ms on 2026-09-10 — a bridge run
after a native one took 3.38 s against 3.95 s after another bridge,
reproduced in a second block either side of the control — and did
nothing at all across twelve runs the day before. Blocks read
(`/usr/bin/time -v`, which `bench/run.sh` already captures) tracks init
inside a bridge-after-bridge sequence at r ≈ +0.9, then inverts between
the arms, where the faster arm read *more*; and it counts a whole run,
not an init. Nor does run order exhaust it: a sibling session's
bridge-after-bridge runs on the same binary averaged 5.13 s against
3.95 s for the same arm here about ten minutes later — a residual
larger than the ordering effect and with no account of its own. Those
runs are not among the 31 above, which is why the range there stops at
5.00 s.

**#217 names it, and it is neither covariate.** Both of those are the same
thing seen through different windows. proofman's init reads 8.4 GiB of the
proving key before it sizes its buffers, and the page cache decides how much
of that comes off the disk. That is why blocks read tracks init *within* a
sequence and inverts *between* the arms — the reads that cost a run are not
its own — and why run order was worth 0.57 s on one day and nothing on the
next: a bridged run evicts the set on a host under memory pressure and not on
one that is idle. It is also what the sibling session's slower runs were: a
sibling session is exactly the kind of tenant that empties it. Set the state
instead of measuring around it ("What sets a run's init is the page cache")
and the same pair sits 0.248 s apart warm and 3.478 s apart cold, both with
ranges that do not overlap.

That instability is why the comparison above is stated as a bound, and
also why the conclusion survives it: the slower init gets, the more of
it the bridge's start hides inside. The tightest margin of the 23 came
from the *fastest* init, not the slowest. What the instability does bind
is anyone quoting init later — a figure means nothing without a native
baseline taken in the same session and interleaved with it, and the run
order stated beside it.

So neither lever #178 proposed has anything to buy. Hooking the bridge in
earlier moves work that already finishes with slack; deferring the client
to the first prove would give up what `ZZ_MEMORY_FRACTION` is for, since
the clients claim their share before pil2 sizes its stream buffers from
the memory it sees free. Nor is the preload's own cost a lever.
Swept in one interleaved session, `ZZ_PRELOAD_THREADS` at six,
three and two moves when the preload *finishes*, by the better part of
a second, and leaves proofman's init within 0.1 s of itself — the
preload runs beside init, not inside it. In the same session
`ZZ_PRELOAD=0` reaches native's init only by moving the loads into the
contributions phase rather than removing them, and that spelling turns
eager module loads off as well, so it moves two things at once. The
#176 bump is worth 0.24–0.71 s of init on the same measure: the
pre-#176 configuration ran 3.50–3.94 s against the default's
3.23–3.26 s. That arm changed the plugin and disabled eager module
loads together, so the 0.24–0.71 s belongs to the pair, not the plugin.

Two tests pin the scheduling this leans on, and it is worth being exact
about which: `lib.rs` pins that the preload queue keeps its callers'
priority and order — a requested batch ahead of queued background work,
in the order given — and `artifact.rs` pins the load/prove gate, where a
prove waits for the loads already in flight and a waiting prove holds new
loads back. Neither pins that a requested AIR is loaded *before* a prove
wants it: a key still sitting in the queue is loaded by the prove itself
(`Bridge::artifact`), which is also what keeps an AIR outside
`.last-used` from waiting on the rest of the preload. That the preload
finishes first is the margin measured above, not an invariant.

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
a second client is still more than this card has — "Memory budget"
below, where the shortfall is 6.3 GiB on the shipped wheel); and
`const_setup` recomputes each AIR's constant tree per run
where pil2 reads it from disk. The bridge's own start is no longer one of
them: it finishes well inside proofman's init ("Bridge start-up" above).
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

### Raising the read-ahead permit moves the upload, not the leg

`ZZ_FIXED_AHEAD` is how many AIRs' fixed sections may be uploaded ahead of the
running prove's. **It has only two settings.** The permit is taken in `plan` and
given back in `installed`, both inside the admission `prove_owned` holds for the
whole prove, so at most `ZZ_PENDING` proves — two by default — can hold one at a
time: 1 is the permit refusing, anything at or above the admission is the permit
never refusing, and the value is capped to it. There is no third arm to run.

At 1, four to six of hello-world's eleven proves find the permit taken and
upload under their own slot; with it off, none do, and the bridge's own `fixed
sections for X` line shows the work moving out from under the slot. **The leg
does not follow it.** One binary, arms interleaved and rotated within each pass,
leg from proofman's `GENERATING_INNER_PROOFS`:

| | permit at 1 | permit off |
|---|---|---|
| leg, mean [min-max] | 6113 ms [5852-6247], n=14 | 6075 ms [5815-6351], n=20 |
| proves uploading under the slot | 4-6 of 11 | 0 |
| `under the slot`, summed | 1.42 s | 1.29 s |
| `ahead of it`, summed | 1.67 s | 1.93 s |

Paired inside each pass, which cancels the session drift, the permit off is
**-26 ms** (sd 109 ms, 14 passes, 8 of them favouring it). Two things size that
against the harness rather than against zero: a `CUDA_MODULE_LOADING=EAGER`
control on the same binary and session is -450 ms with every pass the same sign,
and two *labels for the same configuration* — the sweep ran the capped value as
if it were an arm of its own — differ by -120 ms (sd 194, 6 passes), more than
the effect. The run-to-run scatter is the whole of what the depth arms show.

The reason is the ordering in "The uploads, measured" above: an upload into a
freshly allocated buffer waits on the client's compute stream, so it never
overlaps that client's own kernels. Two captures, one per setting, hold the same
1205 copies and 9.75 GB in 0.47 s, still 0.00-0.01 s overlapped. A copy moved
earlier lands in device idle either way, and there is no leg time to win by
choosing which idle it lands in. This is the third arm on this leg to move
host-side work without moving the leg — #205 measured the other two, switching
the read-ahead off from below and sharing the `constants` program across AIRs
(that one removed nine of eleven executions outright and was still null, so the
shape is not "moving is free, removing pays").

Read it as a prior with a control attached, rather than as a law that host-side
work cannot matter. In those two captures the *capture's* leg — the bridge's
first kernel to its last, which is 0.8-1.2 s inside proofman's timer and so not
the 6.1 s above — has the client's kernels busy 2.4 s of 5.1-5.3 s, so host work
is most of what the leg is; what these arms show is that taking a piece of it
away lets the neighbouring pieces expand into the device idle it was living in.
Nobody has a mechanism for that conservation, and it is a prediction that can
fail — so a fourth arm is worth running, and what it has to beat is the `EAGER`
control, not zero.

Turning the permit off costs memory, so 1 stays the default: the sections held
ahead are the AIR's `const_base`, 16-96 MiB for nine of hello-world's eleven
AIRs but 1168 and 1408 MiB for the two virtual tables, and those two prove back
to back. It does not move the client's floor, because the floor is not set by
them — every failure walking `ZZ_MEMORY_FRACTION` down is the same 5.50 GiB
allocation on `VirtualTableZisk0_n21` (on the `-168` artifacts this arm ran
on: post-#191 the largest is that AIR's 2.75 GiB `const_ext`, and the verdict
is unchanged — see "Memory budget"), at either setting (3 repeats per cell:
0.41 passes 3/3 with the permit at 1 and 5/6 with it off, 0.39 passes 1/3 and
3/6, 0.37 passes 0/3 and 1/6). That is the aggregate floor "Memory budget"
describes, and the read-ahead's extra `const_base` neither raises nor lowers it
within these repeats.

Hello-world is the workload that puts the most pressure on this permit, not the
least: its eleven instances are eleven distinct AIRs, so every prove needs a key
no prove before it uploaded. On a mix where an AIR repeats, most proves find
their sections resident and never plan a read-ahead at all. The block-shaped mix
is unmeasured here for that reason, not overlooked.

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
  whatever is left (pil2's). The floors below were measured on the
  hello-world key by walking the fraction down until a run failed, repeats
  at each fraction — a single run at the boundary is a race and lands
  either way — on the shipped wheel (`0.10.2.dev20260910150749`, eager
  kernels on, staging off) against the #191 artifacts (#215, 2026-09-11).
  `bench/mem_budget.py <run.log>...` reads these tables back out of the
  logs they came from, and carries the traps below as its own rules.

  **A share is the arena the run allocated, not the fraction times the
  card.** XLA divides `ZZ_MEMORY_FRACTION` by the client count and applies
  it to its own base — 33670758400 B, which is 31.36 GiB, 0.48 GiB under
  this card's 31.84 GiB — and every client of a run gets the same arena.
  The run prints the product it allocated (`XLA backend allocating N bytes
  on device 0 for BFCAllocator`), so read that rather than computing it.
  An earlier version of this note computed the column as `fraction × 31.84
  GiB`: the fractions below are unchanged, the GiB beside them are 1.5 %
  lower than they were.
  - **A client needs an 11.60 GiB arena**, at headroom 3 — the arena every
    run survives, below which the outcome is a coin flip rather than a
    cliff. The right column turns the fixed-section read-ahead's upload
    off (`ZZ_FIXED_AHEAD=0`):

    | `ZZ_MEMORY_FRACTION` | the client's arena | shipped | read-ahead off |
    |---|---|---|---|
    | 0.45 | 14.11 GiB | 3/3 | — |
    | 0.39 | 12.23 GiB | 3/3 | — |
    | 0.37 | 11.60 GiB | 6/6 | 3/3 |
    | 0.35 | 10.98 GiB | 5/6 | 3/3 |
    | 0.34 | 10.66 GiB | 3/6 | 1/3 |
    | 0.32 | 10.03 GiB | 0/3 | 0/3 |
    | 0.30 | 9.41 GiB | 0/3 | 0/3 |
    | 0.28 | 8.78 GiB | 0/3 | — |

    The surviving fraction is where #191 left it, so the wheel's eager
    kernel loads and its pinned staging pool do not reach the client's
    floor. #177 first put the floor at 0.38 off one run per fraction and
    #188 revised it to 0.39 off seven; a repeated figure supersedes a
    single run, and 0.37 here is six.

    **Nor does the read-ahead reach it.** `ZZ_FIXED_AHEAD=0` takes the next
    AIR's `const_base` off the device — up to 1.38 GiB, and the two virtual
    tables that are 88 % of the 2928 MiB the eleven carry between them prove
    back to back (#209) — and the two columns
    are not told apart at these counts, with the same cliff between 10.66
    and 10.03 GiB. `MaxInUse` at 10.03 GiB is 8.69–9.73 GiB with it off
    against 8.73–9.55 GiB with it on. Only the *upload* is off at depth 0;
    the key read still runs ahead of the slot, which is why the `ahead`
    column of the `fixed sections for X` lines does not go to zero.

    **That null is a knob that was never on the right buffer**, not a
    read-ahead that costs nothing. `ZZ_FIXED_AHEAD` governs `const_base` and
    `custom_base`; the next instance's `trace` — 32 MiB to 1,248 MiB across
    these eleven AIRs, and the largest thing another instance leaves on the
    client — goes up under the per-client admission `ZZ_PENDING` instead
    (`lib.rs:966-971`). `ZZ_PENDING=1` does move it, by 1.2 GiB of client
    high-water; see "Where a prove's device memory goes" below.

    **What is in the 11.60 GiB.** A run that dies leaves BFC's own
    accounting in the log, and 20 of these 21 report the same largest
    single allocation: **2.75 GiB, `VirtualTableZisk0_n21`'s `const_ext`**
    (88 constants × 2²² × 8 B from its manifest; the twenty-first died
    earlier and got no further than 2.44 GiB). The running prove's own extended
    trace is the next size down — `cm1_ext` 2.44 GiB on `Binary_n22`, 2.38
    GiB on `Main_n22`, `cm2_ext` 1.50 GiB on `Main_n22` — and the last two
    are both live in the chunk list of a `Main_n22` abort. #209's 5.50 GiB
    is the pre-#191 figure on the `-168` artifacts: the blocked extend took
    that scratch out, and what stands now is the section itself.

    **The room above the data is a range, and its tight end is a few
    hundred MiB.** A run that *finishes* can be asked what its allocator
    held — `PJRT_Device_MemoryStats`, through the readout parked on branch
    `issue220-parked`, not through anything on `main`, where
    `bench/mem_budget.py` reads `MaxInUse` off the logs of runs that died.
    At the 11.60 GiB arena the client's own peak in use came back
    11.27, 10.68 and 10.22 GiB over three runs — 0.33, 0.92 and 1.38 GiB of
    arena above the live high-water. What binds is the tight end: an arena
    barely larger than the data it had to hold. An earlier version of this
    note put it at "at most 1.2 GiB", from the highest `MaxInUse` seen
    (10.40 GiB) on runs that *died*, where the figure is truncated at the
    abort; measured on runs that finish it is a range, because the live peak
    itself swings about a GiB between runs of one workload (#220).

    **The aborts are placement, but `LargestFreeBlock` is not the
    evidence.** Neither allocator the bridge can build writes
    `tsl::AllocatorStats::largest_free_block_bytes`, and it is not the only
    such field: of that dump BFC maintains `InUse`, `MaxInUse`, `NumAllocs`,
    `MaxAllocSize` and `Limit`, and assigns none of `Reserved`,
    `PeakReserved` or `LargestFreeBlock` — all three print the zero
    `AllocatorStats` initialises them to. So the `LargestFreeBlock: 0B` an
    earlier version of this note cited is printed whatever the heap holds,
    on a full pool and an empty one alike, and it is not a reading. The same
    dump does carry the statement — a #220 run at the 8.78 GiB arena:
    `Total size in pool: 8.78GiB ... available size: 40B` beside `Sum Total
    of in-use chunks: 7.54GiB`, with a 1.12 GiB request refused. 1.24 GiB
    free inside the pool and no block large enough is the placement finding,
    and it stands.

    **`cuda_async` is reachable, and it is not an arena.** The GPU client's
    create options take an `allocator` kind as a string — `default`,
    `platform`, `bfc`, `cuda_async`, `vmm`, parsed in
    [`pjrt_c_api_gpu_internal.cc`](https://github.com/fractalyze/xla/blob/64ebf90f17/xla/pjrt/c/pjrt_c_api_gpu_internal.cc#L96)
    and carried by the shipped wheel, whose refusal message names all five —
    so the kind is a create option rather than the plugin change #215 took it
    for. Walked with it (#220: one binary, the kind an env var, the two arms
    back to back inside each repeat, three repeats a cell). **The floor
    record is the table further up, not this one**: that walk was taken to
    site the floor, with six repeats at the boundary cells, while this one
    exists to compare two columns taken in one session. Its BFC column lands
    a rung harsher than the floor table's at the same fractions (2/3 against
    5/6 at 0.35, 0/3 against 3/6 at 0.34) — different session, a different
    build, a co-tenant on the host throughout and swap full, and boundary
    cells that are races either way. The floor cells were not re-run under
    these conditions, so the two tables are not a controlled comparison of
    each other; what this one measures is the gap between its own columns.

    | `ZZ_MEMORY_FRACTION` | the client's arena | BFC | `cuda_async` |
    |---|---|---|---|
    | 0.37 | 11.60 GiB | 3/3 | 3/3 |
    | 0.35 | 10.98 GiB | 2/3 | 3/3 |
    | 0.34 | 10.66 GiB | 0/3 | 3/3 |
    | 0.32 | 10.03 GiB | 0/3 | 3/3 |
    | 0.30 | 9.41 GiB | 0/3 | 3/3 |
    | 0.28 | 8.78 GiB | 0/3 | 3/3 |
    | 0.26 | 8.15 GiB | 0/3 | 3/3 |
    | 0.24 | 7.53 GiB | 0/3 | 2/3 |
    | 0.22 | 6.90 GiB | 0/3 | 0/3 |

    **The right column is not a smaller client.** That kind builds no arena:
    it allocates from the device's default CUDA memory pool, so the share is
    the pool's release threshold — claimed up front, and not a ceiling. The
    client reports itself past it: at the 8.15 GiB claim its allocator gives
    `limit 8348 MiB` against a peak in use of 10613 and 11093 MiB, 2.2 and
    2.7 GiB above its own limit. Those are the two runs that *finished* in a
    later set of three at that cell — a set that passed 2 of 3 where the
    table's walk passed 3 of 3, the cell being near its boundary either way.
    The third aborted on a 2.50 GiB allocation, so its peak is truncated at
    the abort, in the way "The room above the data is a range" above gives
    as the reason not to quote such a figure; it is a lower bound, and it is
    over the limit too. What the lower cliff measures is the pool growing
    into room pil2 did not take. The working set is not what moved: at one claim of 11.60 GiB the
    peaks are 10.22–11.27 GiB under BFC against 9.90–10.83 under
    `cuda_async`, ranges a one-GiB run-to-run swing cannot tell apart. All
    three `cuda_async` runs are byte-identical to a native run from the same
    session, 11 of 11. The table is one build; the peaks beside it are a
    second build of the same tree, which adds the off-by-default readout
    they are taken from and nothing else. Both were read off the run logs by
    hand — the arena from each run's own `XLA backend allocating N bytes on
    device 0 for CudaAsyncAllocator`, the peaks from the readout's `client 0
    memory: ... peak_in_use N MiB`. `bench/mem_budget.py` does **not** produce
    this table: its arena pattern matches `for BFCAllocator` only, and it
    knows nothing of the readout's line — the reader that handles both is
    parked with the option.

    So the kind is not the lever the 2.75 GiB `const_ext` made it look like.
    It buys no arena, it gives up the ceiling `ZZ_MEMORY_FRACTION` exists
    for — a client that outgrows its claim takes memory pil2 has already
    sized itself against — and what it could recover is the few hundred MiB
    above the data. It is measured, and parked rather than shipped: the
    bridge option and the memory-stats readback it was measured with are on
    branch `issue220-parked` (with fractalyze/xla-pjrt#6 behind it), not on
    `main`.

    Which AIR aborts is not fixed: 13 of these 21 on `Main_n22`, 7 on
    `VirtualTableZisk0_n21`, one on `Binary_n22`. Read the floor off the
    arena, not off the AIR named in the log.
  - **pil2 needs 12.904 GB left to it and refuses to start below that**,
    since `commit_witness` stays on the card. It is pil2's own check and
    pil2 prints both sides of it: at one client and headroom 0, fraction
    0.55 leaves it 12.927 GB and it comes up with one basic stream and
    5.05 GB of fixed pols, while 0.56 leaves 12.613 GB and it exits with
    `Insufficient memory. Need 12.904107 GB but only 12.612976 GB
    available`. The requirement is the same figure at either client count.

    **That figure already contains the module loads**, because it is free
    memory as pil2 finds it — after the clients have claimed their arenas
    and their module loads have begun. An earlier version of this note put
    pil2's floor at 14.3 GiB, which is the *card space* left at the last
    fraction pil2 survived, and then added ~2 GiB of module loads on top of
    it: that is the same memory counted twice, and it is most of why the
    two-client shortfall below is smaller than the 8.1 GiB this note used
    to carry.

    The block-shaped section above says `ZZ_MEMORY_FRACTION=0.55` leaves pil2
    13.3 GB and calls that below the minimum it will start with. That does not
    reconcile with either number here — 0.55 leaves 12.93 GB on this key, and
    12.93 is above the 12.904 pil2 asks for, so it starts. That run's headroom
    is not recorded and its workload is the block-shaped one, so the two are
    not the same measurement; #170 carries the discrepancy.

    pil2 refuses in a second sentence as well. When what it can see is
    small enough that its own stream sizing asks for a card nobody has, it
    prints `Not enough GPU memory to run the proof` and no figures — at two
    clients holding their floor it sized 20 basic streams and asked for
    162.077 GB. That `Need` is the sizing, not a requirement.
  - **Module loads come out of neither pool**, which is what
    `ZZ_GPU_HEADROOM_GB` buys: at headroom 0 a run both pools fit in still
    dies, on `Failed to get module function: CUDA_ERROR_OUT_OF_MEMORY` or
    `too many resources requested for launch`. The bench's 3 is enough and
    0 is not. The headroom does not come out of what pil2 reports as
    available — across these runs the fraction alone accounts for pil2's
    `Using minimum memory` to within 0.2 GB at headroom 0 and 3 alike — so
    it acts on the stream sizing that follows and on what is left for the
    loads, not on the check above.
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

  One client fits with room: at the bench's `ZZ_MEMORY_FRACTION=0.45` a
  whole run peaks at 28.33 GiB of the 31.84 the card has (`nvidia-smi`
  every 50 ms, the same figure in three runs; 27.35 GiB at 0.37, and native
  alone peaks at 30.95 GiB). The card is not full because pil2 sizes itself
  down to what is left — which is why taking 2.51 GiB off the client's
  arena moved the run's peak by 0.98 GiB, not by 2.51.

  **Two clients are at least 6.3 GiB short** (#215), and both ends of that
  are measured rather than summed. Give each client the 11.60 GiB it needs
  (`ZZ_CLIENTS=2 ZZ_MEMORY_FRACTION=0.74`) and pil2 is left 6.55 GB against
  the 12.904 GB it asks for. Walk down instead until pil2 will start, and
  the clients get 8.15 GiB each — 8.47 GiB is the last arena at which pil2
  still refuses, and it refuses by 0.076 GB:

  | `ZZ_CLIENTS=2` | each client | outcome |
  |---|---|---|
  | 0.74 | 11.60 GiB | pil2 refused, sees 6.55 GB — 0/4 |
  | 0.60 | 9.41 GiB | pil2 refused — 0/2 |
  | 0.56 | 8.78 GiB | pil2 refused — 0/2 |
  | 0.55 | 8.62 GiB | pil2 refused by 0.393 GB — 0/3 |
  | 0.54 | 8.47 GiB | pil2 refused by 0.076 GB — 0/2 |
  | 0.52 | 8.15 GiB | pil2 up on one basic stream, client OOM — 0/2 |
  | 0.48 | 7.53 GiB | pil2 up, client OOM — 0/2 |
  | 0.45 | 7.06 GiB | pil2 up, client OOM — 0/3 (headroom 3) |

  So **a client has to fit in at most 8.47 GiB and needs 11.60: the target
  is 3.1 to 3.5 GiB per client, 6.3 to 6.9 GiB over the pair.** It is a
  lower bound on the shortfall — every row but the last runs at headroom 0,
  where the module loads have no reserve and the clients abort anyway.
  #194's counterfactual comes close and still does not close it, as
  arithmetic on the rows above rather than a run: had pil2 been sizable for
  recursion alone — 1.72 + 3.33 + one 1.53 GiB recursive stream, 6.58 GiB
  against the 12.02 GiB it asks for — the rows' 31.37 GB per unit fraction
  puts its break-even near 0.72, which is about 11.35 GiB a client, still
  under the 11.60 they need. The bullet above is why that saving is not on
  offer anyway. **The unset `ZZ_CLIENTS`
  default is 3**, which puts three arenas past the whole card before pil2
  gets any; the bench pins `ZZ_CLIENTS=1`, and every number here is from
  one client.

  Where a client's 3.2 GiB could come from, sized above and **in this
  order**: the 1.19–1.22 GiB base trace the prove was holding past its last
  reader (#219, done — worth 0.31–0.95 GiB of arena, a bracket rather than a
  figure for the reason the walk below gives), then the 2.75 GiB `const_ext`
  it holds resident, which only becomes the binding shape once the trace is
  gone (#219 measured that hand-over), then the 2.38–2.44 GiB `cm1_ext` the
  running prove computes, with the placement room above the live set
  throughout — a range whose tight end is a few hundred MiB rather than the
  flat 1.2 GiB this paragraph used to carry, re-measured under "The room
  above the data is a range" above (#220). Not from the fixed-section
  read-ahead, which the table above measures as not binding, and not from
  pil2, which the two bullets above close off. An earlier version of this
  paragraph led with `const_ext` on the strength of its being the largest
  single allocation; the bullets below are why that is an argument about
  ordering rather than about size.

  **The `const_ext` step is not filed and nobody is working it.** #219
  closed on the trace release alone: releasing `const_ext` from stage 1 is
  worth 0.6–1.2 GiB, needs a new export program and a re-export of the 11
  AIRs, and leaves `Main_n22`'s `cm1_ext` + `cm2_ext` at 8.80–8.87 GiB — so
  a client would still be above the 8.47 GiB two of them can each have. The
  order above is what a future attempt at a second client has to work
  through, not a queue with owners.

  Reproduce a cell, then read the table back out of the runs it made:

  ```bash
  # one cell: three runs at one arena size, a pass being 11 proofs and a
  # verified final proof. Distinct tags -- run.sh clears the tag it is given.
  export ZZ_RUNS=./zz-runs            # run.sh's own default; name it so the
                                      # read line below can find the logs
  for r in 1 2 3; do
    ZZ_CLIENTS=1 ZZ_GPU_HEADROOM_GB=3 ZZ_MEMORY_FRACTION=0.37 \
      ZISK_PROVE_FLAGS= bench/run.sh walk-f0.37-r$r bridge
  done
  bench/mem_budget.py "$ZZ_RUNS"/walk-f0.37-r*/run.log
  ```

- **A lever is sized against the prove that actually fails, and the shape
  that fails moves when you fix one** (#219). Two levers were in play here
  and the order between them was the whole result: `const_ext` is the
  largest block a client holds, and it was not in the binding live set
  until the smaller lever landed.

  `const_ext`'s readers are in every AIR's manifest — `quotient`, `evals`,
  `deep` and `open_const` take it as an input, and nothing in stage 1 does.
  A prove holds it from `set_fixed`, which builds it, to the first opening,
  so the window it is resident without a reader is the whole of stage 1:
  `commit1`, `logup`, `commit2`. On `VirtualTableZisk0_n21` that is 2.75 GiB
  (88 constants × 2²² × 8 B) held across three programs that cannot read it,
  and it is the largest single allocation BFC reports on a failing run.

  It was still the wrong lever to reach for first. Each failing run prints
  BFC's in-use chunk list, and the chunk sizes name the sections against the
  manifests. Grouping #215's 21 one-client aborts (its `w1`/`w2`
  shipped-walk repeats and the `fa0` arm, which is `ZZ_FIXED_AHEAD=0`; all
  at one client and headroom 3):

  | the prove that aborted | aborts | at fractions | `MaxInUse` | that AIR's `const_ext` |
  |---|---|---|---|---|
  | `Main_n22` | 13 | 0.30–0.35 | 9.11–10.40 GiB | 0.19 GiB |
  | `VirtualTableZisk0_n21` | 7 | 0.28–0.32 | 7.62–8.83 GiB | 2.75 GiB |
  | `Binary_n22` | 1 | 0.32 | 9.10 GiB | 0.06 GiB |

  Every abort in the 0.34–0.37 band where a run is a coin flip is a
  `Main_n22` prove, whose `const_ext` is 0.19 GiB; the shape carrying the
  2.75 GiB never failed above 0.32, about 1.6 GiB below the shape setting
  the floor. The bullet below is what the `Main_n22` shape was carrying
  instead, and what happened to the ordering once it was gone.

  So: read the live set of the prove that fails, not the largest allocation
  in the run, and read it again after each change — the binding shape is not
  a property of the workload, it is a property of the current binary.

  ```bash
  # the last in-use chunk list of a run that died, sections named by size
  grep -B40 'Sum Total of in-use chunks' "$ZZ_RUNS"/<tag>/run.log | tail -40
  ```

- **A prove kept the base trace to its last opening, and releasing it makes
  `const_ext` the binding shape** (#219). `prove` drops the trace from its
  environment after `logup`, which is its last reader, so that a wide AIR's
  1.19–1.22 GiB is gone before the quotient's peak. The drop freed nothing:
  `upload_inputs` runs ahead of the slot and the caller held its result on
  `InstanceInputs::uploaded` for the whole prove, so the environment's
  handle was a clone and the device buffer outlived every release.

  It was the one dead buffer in the shape that bound. Of the 14
  `Main_n22`/`Binary_n22` aborts above, 13 are past `logup` and 10 of those
  hold a chunk of exactly the proving AIR's base-trace size, at `quotient`,
  `lev` or `evals` (the other three hold one in a 1.24–1.28 GiB bin, which
  is a base trace BFC placed in a larger chunk and does not say whose). The
  clean one is `w1-c1-h3-f0.35-r3`: aborting on `Main_n22` at `lev`, it
  holds 1.188 and 1.219 GiB at once — Main's own trace, six programs past
  its last reader, beside the next instance's `Binary_n22` trace, which is
  uploaded ahead and legitimately live. On an AIR with `witness_calc` two
  traces were live at once through `commit1` and `logup`, the uploaded one
  and the one the program computed over it.

  `prove` now takes `InstanceInputs` by value and moves the uploads into its
  environment, so the environment owns them and a removal frees. What makes
  that safe is which programs read the section, which the export decides and
  not the driver, so `RELEASED_EARLY` in `driver.rs` states the release
  points and every prove checks the manifest against them
  (`check_release_points`): an export that added a later reader fails the
  prove instead of binding a buffer that is gone.

  Walked the same fractions as "Memory budget" above, the two binaries
  interleaved run by run inside one session, three repeats a cell, on the
  #191 artifacts and wheel `0.10.2.dev20260910150749` (one client, headroom
  3, hello-world; a pass is all 11 proofs and a verified final proof). Every
  run here allocates a BFC arena, which is what makes a fraction walk a
  reading of the working set — each run names it itself, in the `XLA backend
  allocating N bytes on device 0 for BFCAllocator` line the share is read
  from:

  | `ZZ_MEMORY_FRACTION` | the client's arena | before | after |
  |---|---|---|---|
  | 0.37 | 11.60 GiB | 3/3 | 3/3 |
  | 0.35 | 10.98 GiB | 3/3 | 3/3 |
  | 0.34 | 10.66 GiB | 1/3 | 3/3 |
  | 0.33 | 10.35 GiB | 0/3 | 3/3 |
  | 0.32 | 10.03 GiB | 0/3 | 1/3 |
  | 0.30 | 9.41 GiB | 0/3 | 0/3 |

  **The lowest arena every run survives goes from 10.98 to 10.35 GiB**, and
  that is a bracket rather than a figure: the ladder's rungs are 0.31–0.63
  GiB apart, so the before arm's floor is somewhere in (10.66, 10.98] and
  the after arm's in (10.03, 10.35], which puts the shift between 0.31 and
  0.95 GiB. The before column reproduces #215's walk at three repeats rather
  than six — that walk put the floor at 0.37 off 6/6 with 0.35 at 5/6, and
  three repeats here cannot tell 0.35 from 0.37 — so read the arms against
  each other in this table, not against #215's.

  Wherever it falls in that bracket, the shift is well under the 1.19–1.22
  GiB of data the release takes out of the shape that was binding, which is
  the direction #191 found for the same reason: what is freed is data, and
  what a run needs is placement on top of it. **Do not subtract the two and
  call the remainder placement.** Two things changed between these arms, not
  one — the data is gone, *and* the shape that sets the floor is no longer
  the same shape (below). The release frees 1.19 GiB on `Main_n22` but only
  0.36 GiB on the `VirtualTableZisk0_n21` shape that now co-binds, so part
  of what did not convert is the hand-over rather than placement. Placement
  is a real term and #220 is measuring it; it is not this subtraction.
  `MaxAllocSize` is unchanged at 2.75 GiB, since `const_setup` still
  allocates `const_ext` whether or not the prove keeps it.

  **What binds now is `const_ext`.** The arms' aborts, same runs:

  | the prove that aborted | before | after |
  |---|---|---|
  | `Main_n22` | 7, `MaxInUse` 9.16–10.35 GiB | 2, 8.80–8.87 GiB |
  | `VirtualTableZisk0_n21` | 3, 7.91–8.77 GiB | 3, 7.20–8.79 GiB |
  | `Binary_n22` | 1, 9.10 GiB | — |

  Before, the `Main_n22` shape stood 1.6 GiB above the `const_ext` one and
  set the floor alone. After, the two are level — 8.80–8.87 against
  7.20–8.79 — and the `const_ext` shape is the majority of what is left.
  `after-f0.32-r1` is the mechanism in one dump: aborting on `Main_n22` at
  `quotient`, it holds the next instance's `Binary_n22` trace at 1.219 GiB
  and no trace of its own, where the same shape before held both.

  The leg pays nothing for it: 5.235 s [5.234–5.668] before against 5.130 s
  [5.074–5.337] after, three passes an arm interleaved at the bench's own
  `ZZ_MEMORY_FRACTION=0.45`, read with `bench/leg_phases.py` — inside the
  ~0.2 s floor, so the arms are not told apart. Both byte-gates are green on
  the changed binary: 11 of 11 basic proofs identical to native's dumps
  (`bench/compare_dumps.py`) and a clean `ZZ_AB=1` run.

  So #219's own lever is now worth what the issue claimed for it, and was
  not before: taking `const_ext` out of stage 1 would drop the
  `VirtualTableZisk0_n21` shape by 2.75 GiB of data and leave `Main_n22`'s
  `cm1_ext` + `cm2_ext` as the next wall. A client still needs 10.35 GiB
  against the 8.47 GiB two of them can have, so this is one step of three,
  not the step.

- **The resident-set trim does not reach a second client** (#188). Scoped
  as "re-upload the base constants per prove, drop the digest layers once
  the openings are done", built in full, and walked down the same fraction
  (one client, headroom 3, hello-world; a pass is all 11 proofs and a
  verified final proof):

  | `ZZ_MEMORY_FRACTION` | the client's arena | before | the trim in full | shipped |
  |---|---|---|---|---|
  | 0.45 | 14.11 GiB | 4/4 | 5/5 | 6/6 |
  | 0.39 | 12.23 GiB | 7/7 | 7/10 | 5/7 |
  | 0.38 | 11.92 GiB | 0/4 | 4/7 | 2/3 |
  | 0.37 | 11.60 GiB | 0/4 | 2/4 | — |
  | 0.36 | 11.29 GiB | 0/1 | 0/1 | — |

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
  allocation rather than most of its high-water; the largest a failing
  run at fraction 0.28 now *asks* for is 1.56 GiB, inside `commit2` on
  `VirtualTableZisk0_n21` — which is the request that found the arena dry,
  not the largest allocation, and BFC still reports 2.75 GiB for that
  ("Memory budget"). So the extend is not what
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
  lever in either direction: swept in one interleaved session, six, three
  and two preload threads put proofman's init within 0.1 s of each other,
  because the preload is done long before init is either way ("Bridge
  start-up", where the levels are only comparable within a session).
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

### Where a prove's device memory goes, against pil2's own buffer (2026-09-12, #226)

Every memory unit before this one sized allocations. None asked which
buffers are alive at each stage of one prove and why, so the ~3 GiB a
second client is short by (#215) had no owner. This is that inventory, and
the answer is that **it is not in the prove's sections**: section for
section the bridge holds what pil2 holds, and on one of the two binding
shapes it holds less.

That is the same conclusion #220 reached from the other side. Its ladder found
the allocator kind was not a lever — the room above the data is small — and
this comparison says the data was never the outlier either. Three units of
this family sized an excess that, per prove, is not there.

Read with `bridge/bench/mem_stages.py` over a `ZZ_MEM_STAGES=2` run and
`bridge/bench/pil2_layout.py` over the proving key — not with
`bench/mem_budget.py`, which answers a different question (what a whole run
asked the card for, and which allocation a dead run died on) and cannot
produce these tables. Twelve runs, four arms of three, go hello-world,
`ZZ_CLIENTS=1`, fraction 0.45, headroom 3, shipped wheel
`0.10.2.dev20260910150749`, artifacts `zz-artifacts-191`, page cache warmed
by `run.sh`. All 11 basic proofs byte-identical to a same-session native arm
on every one of the twelve. Host carried a full swap (7/7) throughout, as
#220's walk did.

**pil2's per-AIR need is printed by pil2 and is far below the 7.85 GiB
ceiling.** The `-vv` line `TOTAL PROVER MEMORY USAGE` gives it per AIR;
it is `prover_buffer_size * 8`, and `prover_buffer_size` is
`get_map_totaln_c`, i.e. `mapTotalN`
([`utils.rs:127`](https://github.com/fractalyze/pil2-proofman/blob/daf3a598/proofman/src/utils.rs#L127),
[`setup.rs:212`](https://github.com/fractalyze/pil2-proofman/blob/daf3a598/common/src/setup.rs#L212)).
For hello-world: Main 6.03, VirtualTableZisk0 7.35, Rom 4.52, Arith 4.40,
MemAlign 2.74. The 7.85 GiB is `max(basic, recursion)` over the whole
proving key and is the compressor's, so it bounds nothing about a basic AIR.

**Every "GB" pil2 prints is a GiB**, by two independent routes:
`format_bytes` divides by 1024 and labels the units KB/MB/GB
([`utils.rs:184-197`](https://github.com/fractalyze/pil2-proofman/blob/daf3a598/common/src/utils.rs#L184-L197)),
and the `Insufficient memory. Need X GB` line does not come from it at all —
it divides by `1024.0 * 1024.0 * 1024.0` inline
([`starks_api.cu:344`](https://github.com/fractalyze/pil2-proofman/blob/daf3a598/pil2-stark/src/api/starks_api.cu#L344)).
Both figures are already comparable to ours; "converting" either shrinks the
bridge's excess by 7 %.

**Where pil2's buffer differs from ours in kind.** pil2 takes one buffer per
stream and places sections at offsets inside it, so sections dead by the time
a later one is written share their bytes: `cm1` base sits where `cm2_ext` is
written, `cm2` base where the quotient section and its tree go. It does not
release them — it never allocated them apart. The bridge reaches the same
place by releasing (`driver::prove` drops `trace` after `logup` and `cm2`
after `commit2`; `release_tree` drops each stage tree as its openings reach
the wire), and the two come out level. Whether an AIR's constant tree is in
that per-stream buffer or preloaded once per GPU is decided **per AIR**:
Main's is shared, VirtualTableZisk0's is not.

#### The live sets, section for section

Bridge rows are the largest boundary of one prove under `ZZ_PENDING=1`
(nothing of another instance on the client); pil2's are its buffer, which is
allocated whole for the stream's life. MiB.

| section | Main: bridge | pil2 | diff | VirtualTableZisk0: bridge | pil2 | diff |
|---|---|---|---|---|---|---|
| `cm1_ext` | 2,432 | 2,432 | 0 | 736 | 736 | 0 |
| `cm2_ext` | 1,536 | 1,536 | 0 | 1,152 | 1,152 | 0 |
| `cm3_ext` (qsec) | 384 | 384 | 0 | 192 | 192 | 0 |
| `mt1` / `mt2` / `mt3` | 341 each | 341 each | 0 | 171 each | 171 each | 0 |
| const (base) | 96 | 96 | 0 | 1,408 | 1,408 | 0 |
| constant tree | 533 | 0 | **+533** | 2,987 | 2,987 | 0 |
| FRI layers + trees | 268 | 268 | 0 | 134 | 134 | 0 |
| `q/f` + codeword | 0 | 384 | −384 | 0 | 192 | −192 |
| `zi` / domain | 128 | 0 | +128 | 64 | 0 | +64 |
| quotient row windows | 32 | 0 | +32 | 16 | 0 | +16 |
| itemised above | 6,434 | 6,127 | +307 | 7,201 | 7,316 | −115 |
| **buffer pil2 allocates** | **6,434** | **6,170** | **+264** | **7,201** | **7,530** | **−329** |

Two totals because they answer different questions. The itemised row is
section against section. `mapTotalN` is larger than the sections placed in it:
it is the maximum of the placed layout and the scratch terms pil2 sizes the
buffer against but does not place in the table (`lev`, `mem_exps`, the
`tmp1`/`tmp3` expression memory) — 43 MiB of it on Main and 214 MiB on
VirtualTableZisk0. The second row is the comparison that matters, since pil2
holds the whole buffer for the stream's life whether a section is in it or
not.

**Before believing that parity, check the instrument could have seen a
difference** — this page's own rule, from "So run a positive control before
believing a null on this leg". Two things say it can. The table itself
resolves a per-section difference where one is known to exist and reports zero
where it is not: the constant-tree row is +533 MiB on Main and exactly 0 on
VirtualTableZisk0, which is pil2's own per-AIR branch (shared per GPU against
carried per stream) recovered independently from the live set. And turning a
knob moves it — `ZZ_PENDING=1` takes co-residency from 1,504–2,704 MiB to 0 and
the client high-water from 10,263 to 8,933 MiB. An inventory blind to a
gigabyte would have done neither.

The bridge's own live set is within 0.26 GiB of pil2's on Main and 0.32 GiB
*below* it on VirtualTableZisk0. The three rows that differ:

- **The constant tree, +533 MiB on Main and 0 on VirtualTableZisk0.** pil2
  preloads Main's once per GPU and shares it across streams; the bridge holds
  one per client. This is worth `533 MiB x (clients - 1)` and nothing at one
  client. On VirtualTableZisk0 pil2 did not preload — its 2,987 MiB tree is
  inside *every* stream's buffer — so the claim "we hold 2.75 GiB pil2 does
  not" was never true for that shape. That does not make our residency free;
  it means the comparison cannot size it.
- **`zi` / domain, +128 MiB.** pil2 places `zi`/`x` at the offset its
  expression scratch starts from, so they fall inside its maximum instead of
  adding to it; the bridge materialises them as buffers from `constants`.
- **`q/f` + codeword, −384 MiB.** pil2 reserves `q/f` and `buff_helper` for
  the buffer's life; the bridge has released the codeword by `openings`.

#### What the excess actually is

Two terms, neither of them a section.

**(b) Co-residency — the next instance's `trace`, 32–1,248 MiB, and 0 under
`ZZ_PENDING=1`.** The per-client admission (`ZZ_PENDING`, default 2) puts the
next instance's `trace` on the device during the running prove. Which AIR that
is moves run to run, and the eleven traces span 32 MiB (Rom) to 1,248 MiB
(Binary), so this is a bimodal jump rather than scatter. At Main's `openings`
boundary, per run:

| arm | r1 | r2 | r3 |
|---|---|---|---|
| default | Rom, 32 MiB | Binary, 1,248 MiB | BinaryExtension, 928 MiB |
| `ZZ_FIXED_AHEAD=0` | Mem, 416 MiB | Mem, 416 MiB | none |
| `ZZ_PENDING=1` | none | none | none |

`ZZ_FIXED_AHEAD` does not control it — that depth governs `const_base` /
`custom_base` only, while `trace` / `publics` / `airvalues` ride the
admission (`lib.rs:966-971`). This explains the null recorded above under
"Nor does the read-ahead reach it": the knob was never on the largest
co-resident buffer. `ZZ_PENDING=1` removes it in all three runs and takes the
client high-water from 9,007–10,263 MiB to **8,933–8,993 MiB**.

The trace is the largest of the next instance's uploads but not the only one;
counting its `const_base` and scalars too, everything on the client that is
not the running prove's comes to 32–1,376 MiB on Main and 1,504–2,704 MiB on
VirtualTableZisk0 at the boundaries above, and to zero under `ZZ_PENDING=1`.
One admission slot holds one next instance either way.

**(c) Transients inside one program, +838 to +1,957 MiB.** Under
`ZZ_PENDING=1` the client high-water is 8,933–8,993 MiB against a largest
boundary live set, over all eleven proves, of 7,201 MiB — 1.7 GiB that no
boundary ever sees, because it is reached *inside* an execution. The binding
one is `commit2` on VirtualTableZisk0, which raises the allocator's peak by
1,957 MiB while it runs; Main's own largest is `evals`, +838 MiB. (Main's
9,007-MiB-era gap is not Main's transient: the peak is the client's, and by
the time Main proves, VirtualTableZisk0 has already set it.) This is XLA's
own allocation
while a program runs — an extend's output beside its input, fusion scratch —
and it is on top of our live set, where pil2's equivalent (`mem_exps`,
`tmp1`/`tmp3`, `buff_helper`) is already inside `mapTotalN`.

**This term is the one an arena figure cannot be decomposed into.** It is
inside the client-lifetime peak that "Memory budget" above quotes, and it
belongs to no section — so anyone who reads that arena and tries to account
for it section by section is left with a gigabyte and a half that has no row,
whatever inventory they take. It is visible only between two programs, which
is what `ZZ_MEM_STAGES=2` exists for. Size a memory lever against the live set
plus this term, never against the live set alone.

**(a) Held past their last reader: 256 MiB on Main, 1,488 MiB on
VirtualTableZisk0**, dominated by `const_base` (96 / 1,408 MiB), whose last
reader is `logup` in stage 1 and which stays for the life of the prove. Both
figures are from the `ZZ_PENDING=1` arm and are identical across its three
runs. That arm is the one to read them from: the registry is per client rather
than per prove, so on a default-admission log the next instance's `trace` is
alive with no reader yet run, and counting it here would charge the largest
buffer in the workload to this category. `mem_stages.py` excludes it by size —
the eleven AIRs declare eleven different trace widths — but a figure quoted
from an arm where nothing is co-resident needs no such rule to be believed.
`ZZ_RESIDENT_AIRS=1` keeps it for the next prove of the same AIR — **which on
hello-world never comes**, because its 11 AIRs are all distinct. On the
block-shaped `sha-hasher` workload (38 instances over 16 AIRs) AIRs do
repeat, and that is the case the policy exists for. Read as a defect it
argues for dropping residency, which would regress the workload nobody in
this family is measuring.

**(d) Per-client copies of what pil2 shares:** the constant tree above, 533
MiB per extra client on Main-shaped AIRs, 0 on VirtualTableZisk0. **(e)** the
`zi`/domain and row-window rows, 160 MiB and 80 MiB.

The categories do not sum to a single "~3 GiB excess" because that figure was
a client-lifetime high-water compared against a per-stream ceiling. Per
prove, against pil2's own per-AIR need, the bridge is at parity; the client
peak sits above it by (b) and (c).

#### Fix candidates, sized from the table

Not filed here — one change each, for the supervisor.

1. **Bound the instance read-ahead by bytes, not by count** — (b), 0–1.2 GiB
   of client peak, and it makes the peak reproducible. `ZZ_PENDING=1` costs
   ~0.32 s of the 5.6 s leg (medians 5,953 vs 5,635 ms), but these arms were
   run in blocks rather than interleaved, so that figure is provisional and a
   fix unit must re-measure it interleaved (Decision 84 on #170). A byte cap
   would admit Rom's 32 MiB trace and hold back Binary's 1,248 MiB.
2. **Share the constant tree across clients** — (d), 533 MiB per extra client
   on const-light AIRs. Worth nothing at `ZZ_CLIENTS=1`, which is why it has
   to be sized against the two-client configuration it exists for.
3. **Make the fixed-section residency conditional on the plan** — (a), up to
   1,408 MiB on VirtualTableZisk0. proofman knows the instance list before
   proving, so an AIR that appears once need not keep `const_base` past
   `logup`. Must be measured on `sha-hasher`, not hello-world.
4. **The `commit2` / `evals` transient** — (c), the largest single term at
   1.0–1.6 GiB above the live set. Not reachable from the bridge: it is
   XLA's allocation inside one executable, so the lever is export-side
   (chunk the extend the way #191 chunked its predecessor) or plugin-side
   (donate the input buffer).
