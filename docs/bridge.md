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
# once per proving key AND per client count: export every basic AIR (~5 min
# on an RTX 5090). ZISK_CLIENTS is how many clients will share the card; it
# raises the quotient's window ceiling, because N clients each get 1/N of the
# card and so can afford 1/N of a transient. One client (the default) keeps
# the eight windows the cache curve picked, so a one-client run pays none of
# the dispatch cost of the higher count. Two client counts are two exports.
FRX_PLATFORMS=cuda ZISK_CLIENTS=1 python -m zisk_zorch.export.export_air \
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
# See "The module-loading knobs". If you do set it, set it in
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
| `ZZ_CLIENTS` | PJRT clients; proofman spawns this many basic-proof workers. The EXPORT is told the same number as `ZISK_CLIENTS`, because the quotient's window count is compiled in rather than read at run time | 3 — more than this card fits, see "Memory budget" |
| `ZZ_MEMORY_FRACTION` | share of the card the clients claim up front, split evenly, before pil2 sizes its buffers | unset: allocate on demand |
| `ZZ_GPU_HEADROOM_GB` | (fork) GPU memory pil2 leaves out of its stream sizing | 0 |
| `ZZ_PRELOAD` | executables loaded at bridge creation: the previous run's AIRs (`.last-used`), `all`, or `0` | last used |
| `ZZ_PRELOAD_THREADS` | AIRs loading at once | 6 |
| `ZZ_EAGER_MODULES` | executables load their modules into the CUDA context as they are deserialized, not on first execute: `0` off, anything else on, empty or unset follows `ZZ_PRELOAD` | on unless `ZZ_PRELOAD=0` |
| `ZZ_STAGING_THRESHOLD` | bytes at or above which the plugin DMAs a host-to-device transfer out of pageable memory instead of copying it through its pinned staging pool. Raising it above the bridge's 1.2–1.4 GiB sections grows the pinned pool inside the prove and costs more than the faster copies return on a guest that uploads each section once — measured, see "Staging the big uploads is a faster copy and a slower leg" | off: no option sent, so the plugin's own 1 GiB stands (also what a plugin older than fractalyze/xla#718 needs) |
| `ZZ_PENDING` | proves admitted per client on the device (one running, the rest uploaded ahead) | 2 |
| `ZZ_PENDING_BYTES` | bytes of `trace` + `publics` + `airvalues` + `proofvalues` an instance may upload while another prove still holds the client. The count above is the ceiling; this is what sizes the term the admission adds to the client's peak, because those uploads stay live for the whole of the running prove. It bounds the uploads it weighs and no more — the next AIR's `const_base` rides `ZZ_FIXED_AHEAD`'s permit, taken inside this admission, so a refusal holds that back too but an admission puts no cap on it. A client with nothing on it admits any size, so the largest AIR is never refused outright. What the budget is compared against is the admitted set's uploads **less its smallest member**, which at the default `ZZ_PENDING=2` is simply the larger of the two — so there, and only there, a budget above the workload's largest upload never refuses and is the count-only admission this replaced. Above that cap it can refuse instances that are each individually under it | 192 MiB |
| `ZZ_FIXED_AHEAD` | AIRs whose fixed sections may be uploaded ahead of the running prove's, per client; `0` sends every upload under the slot, and the value is capped at `ZZ_PENDING` — the permit is taken and given back inside that admission, so no more proves than it admits can hold one | 1 |
| `ZZ_RESIDENT_AIRS` | AIRs whose fixed sections stay on a client at once, least recently used evicted | 1 |
| `ZZ_FIXED_RESIDENT` | whether an AIR's fixed sections stay on the client after the prove that uploaded them: `1` every AIR's, `0` none, unset follows proofman's plan — an AIR the plan proves once lets each section go at its last reader instead (`const_base` after `logup`, the constant tree after its opening). The two forced values are measurement arms; an unreadable one says so and follows the plan | unset: the plan decides |
| `ZZ_HOST_THREADS` | threads for the host-side copies and key reads | half the cores, at most 8 |
| `ZZ_COMPILE_CACHE` | directory of serialized executables | `$ZZ_ARTIFACTS/.pjrt-cache` |
| `ZZ_LOG` | `1` per-instance timing on stderr, `2` per program; lines carry the seconds since bridge-up | off |
| `ZZ_AB` | prove through pil2 too and compare per instance | off |
| `ZZ_DUMP_PROOFS` | (fork) write every basic proof as raw words into this directory | off |
| `ZZ_DUMP_INPUTS` | write each instance as a `zz_prove` case directory under this one | off |
| `ZZ_DUMP_TRACES` | (fork) write each host trace as `gen_proof` receives it | off |
| `CUDA_MODULE_LOADING` | the CUDA driver's, not the bridge's: `EAGER` puts a module's kernel code on the device as it loads, process-wide. It was what made `ZZ_EAGER_MODULES` pay until fractalyze/xla#698 gave the plugin its own way to do the same thing for the bridge's executables alone — see "The module-loading knobs" for what the pair is worth and why a scoped option is the durable form | driver default `LAZY` |

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

### Reading a capture

- **Quote the share of the *leg*, not of the idle.** The two denominators
  differ by about 2x and the criterion is wall time, so a cost that is half
  the idle can be a quarter of the leg. One number, two denominators; the leg
  is the one that decides anything.
- **Module loads are once per (AIR, program) pair**, not per execution and
  not per instance. A program that runs four times in a prove loads once, and
  every later instance of an AIR already seen loads nothing. Eviction does not
  undo it — a module lives in the CUDA context and `ZZ_RESIDENT_AIRS` only
  drops device buffers. So the cost scales with how many *families* a workload
  touches, which makes hello-world, whose instances are all distinct AIRs, its
  worst case and a bad place to size it from. Per-load cost is not constant
  either, so scaling by program count alone under-predicts.
- **Cross-check any profiled figure against `ZZ_LOG=2`**, which prints each
  `Artifact::run`'s enqueue time from the bridge's own clock with no profiler
  attached. Agreement is what says a dispatch cost is real rather than an
  artifact of tracing.
- **A phase's share of the idle is where the device waits, not what for**, so
  it is not the size of a lever that removes the phase: the cost re-prices
  into the phase next door. Check that a candidate moves the leg by less than
  the win across repeated captures, and run a positive control — on this leg
  `CUDA_MODULE_LOADING=EAGER` is one — before believing a null.

### The module-loading knobs

The eager preload and `CUDA_MODULE_LOADING=EAGER` are only worth anything
together: the driver variable with lazy executable loading buys nothing,
because there is no earlier place for the code load to go. Paired, they move
both workloads' legs and drop graph instantiation substantially.

- **`CUDA_MODULE_LOADING` is process-wide**, so an arm that sets it also
  changes how pil2 loads its own modules. Bounding that share separately —
  the variable with the bridge's executables loading lazily — says most of
  the win belongs to bridge executables, so a plugin-side option scoped to
  them should expect nearly all of it but not the whole.
- **The bridge cannot set it itself.** The driver reads the variable when it
  initializes and pil2 has initialized CUDA before any bridge client exists,
  so setting it in `artifact::new_session` is a no-op. It has to be in the
  environment before the process starts, or the plugin has to materialize the
  kernels after loading a module — which is where the durable fix belongs,
  scoped to the executables loaded through it rather than to every module the
  process loads. The shipped plugin now does that, which is why `Running`
  does not export the variable.
- **Staging the big uploads is a faster copy and a slower leg.** Raising
  `ZZ_STAGING_THRESHOLD` above the bridge's section sizes does make the
  copies faster, and grows the pinned pool inside the prove by more than they
  return. It ships off.

## Status

The per-run tables are on the issue each reading names; what follows is what
stays true of the tree.

- **The byte-gate holds on the current export.** Re-exported into
  `zz-artifacts-243-c1`, all 11 basic proofs are byte-identical to native's
  dumps on every interleaved pass (#243; #241 and #239 are the same gate one
  and two exports earlier). Two of each AIR's programs re-lower, `evals_<size>`
  and `evals_sum`, and every other program's manifest entry is byte-identical
  to the previous export's, so the re-lowering moved no interface the bridge
  binds to. The schedule gains `evals_chunks`, which is the one manifest key
  that is new.
- **What sets a client's high-water is `Main_n22`'s quotient.** The client
  reaches 7,440 MiB (#243), against 8,278 MiB when `evals` was one dispatch
  (#241), 8,821 MiB at `deep` before that program was windowed (#239) and
  8,960-9,005 MiB at #228's shipped admission. The 838 MiB between 8,278 and
  7,440 is `evals`' own rise leaving and nothing else moving: in the same session the
  pre-change arm's high-water rose across `evals` by 838 MiB and the post-change
  arm's rose across `quotient_1048576` by 24 MiB.
  Each is one `art.run`, so one program is what to aim a trim at. Read the
  stage from `mem_stages.py`'s `peak stage` and never off a boundary label:
  `Stage::set` reports a boundary under the *incoming* stage's name, so the
  row carrying a peak is headed with the stage after the one that made it.
- **Two clients still fit at no fraction**, and the two ends are 0.076 GB and
  one allocation apart (#243, headroom 0, three repeats each). Above
  `ZZ_MEMORY_FRACTION=0.53` pil2 will not start — it needs 12.904 GB and 0.54
  leaves it 12.828. At 0.53 each client gets an 8.31 GiB arena and goes dry on
  a 1,566 MiB request, which is `quotient_524288`'s own temp arena at the
  sixteen windows two clients export, so what closes the window is still the
  largest in-program transient left rather than anything resident. #215's,
  #239's and #241's grids ended the same way.
  A single run at 0.54 that gets past pil2's check has read the card before the
  clients claimed their share: quote that fraction only from repeats, and take
  a free-memory figure that breaks the monotone walk as the race it is.

## What the measurements settled

Rules established by units under #170 and #213. Each names the issue that
holds the runs, the arms and the scatter; this page carries only what a
later change has to respect.

**The leg is the basic phase's wall plus a residual, and the whole gap to
native sits in the wall** (#214). The basic phase is the union of the
intervals in which that arm's basic proofs ran — under the bridge, the
`ZZ_LOG` per-instance intervals, since `gen_proof` returns as soon as the
work is handed to a worker and proofman's `GEN_PROOF_n` spans are then
meaningless. `bench/leg_phases.py` does that arithmetic.

- **`leg − basic phase` is a residual, never the recursion's cost.** It
  shrinks as the basic phase lengthens, because both phases sit inside the
  leg and a longer phase hides more of the recursion. Forcing native serial
  leaves its recursion unchanged and its residual falls anyway.
- **One client proves serially, at 1.00x.** pil2's three basic streams buy
  it 1.35x, not 3x — they contend for one card. The honest pivot is to force
  pil2 serial (`ZZ_GPU_HEADROOM_GB=15`), not to compare against its
  three-stream wall.
- **A second client is the only share left, and it is not enough.** At
  pil2's own 1.35x the bridge's leg lands above the 1.2x bar, and at perfect
  packing it cannot go below the kernel time the phase carries. It also does
  not fit on this card — "Memory budget", and the Status above.
- **The client's device idle is not a lever.** Its phases move more between
  captures of one binary in one session than any of them is worth, and
  removing one re-prices the cost into the phase next door (#205, #209).
  What a candidate has to beat is the `CUDA_MODULE_LOADING=EAGER` control on
  the same binary, not zero.

**What sets a run's init is the page cache, not either stack** (#217).
proofman reads 8.4 GiB of the proving key before it sizes any buffer, so
what init measures is how much of that set came off disk. The same pair is
0.128 s apart on an uncontrolled cache, 0.248 s apart with the set warmed
and 3.478 s apart with it evicted — not a monotone function of warmth,
since a partial cache does not hurt the two arms equally.

- **An init figure with no cache state beside it says nothing.**
  `bench/run.sh` warms the set and censuses it into `pagecache.txt` on every
  run, warmed or not, because a cold run is exactly the one whose figure
  depends on the state. `bench/pagecache.py --warm|--evict|--census`.
- The apparent effect of *the previous run's arm* is this same state seen
  through a proxy: any tenant that reads a few GiB empties the set, and a
  sibling session will do it between two of your own runs.

**The bridge's start-up is not a lever** (#178, #217). Its client is up in
0.16–0.19 s and its whole preload finishes one to four seconds before
proofman's init ends, at every thread count tried — the preload runs beside
init rather than inside it. Hooking in earlier moves work that already
finishes with slack; deferring the client gives up what `ZZ_MEMORY_FRACTION`
is for, since the clients claim their share before pil2 sizes its buffers
from what it sees free. That the preload finishes first is a measured
margin, not an invariant: a key still in the queue is loaded by the prove
that wants it.

**An upload never overlaps the prove it belongs to, and that is PJRT's
ordering rather than the hardware's** (#193, #204). A GPU client is
`kComputeSynchronized`: a buffer the allocator returns at time t may only be
written once the compute stream has drained everything enqueued before t, so
an upload into a freshly allocated buffer waits on that client's own
kernels. The card is willing — in the same captures pil2's copies overlap
the bridge's kernels. Allocating an instance's buffers up front through the
async transfer manager was built and measured, and moved neither the overlap
nor the leg.

- Uploads are a fraction of a second over the leg, and most copies start
  into a device that has already been idle for milliseconds, so an upload
  freed to run beside kernels would find none to run beside.
- Pageable transfers reach 11–16 GB/s on this card and pinned ones 42–45,
  but pinning the difference is worth less than the pinned pool's growth
  inside the prove costs — which is why `ZZ_STAGING_THRESHOLD` ships off.

**`ZZ_FIXED_AHEAD` has only two settings, and neither moves the leg**
(#209). The permit is taken in `plan` and returned in `installed`, both
inside the admission `prove_owned` holds for the whole prove, so at most
`ZZ_PENDING` proves can hold one: 1 is the permit refusing, anything at or
above the admission is it never refusing, and the value is capped there.
Paired within passes the difference is smaller than the difference between
two *labels for the same configuration* in the same sweep. It stays at 1
because turning it off costs memory — the sections held ahead include the
two virtual tables' `const_base`, and those two prove back to back.

Hello-world puts the most pressure on that permit, not the least: its eleven
instances are eleven distinct AIRs, so every prove needs a key no prove
before it uploaded. On a mix where an AIR repeats, most proves find their
sections resident and plan no read-ahead at all.

**Three things the byte-gate surfaced**, all handled by the bridge now:
traces are bit-packed for AIRs carrying `witness_bits` hints and are
unpacked on the host from the packing proofman registers
(`set_packed_info`); the custom-commit `_gpu.bin` is a 32-byte root then the
base, extended and tree sections in the prover's 256x4 tiled device layout,
which the bridge untiles and recomputes from (a CPU run writes the same file
row-major and names it `.const`, which is what the layout is keyed on); and
a trace holds raw machine words, some above the modulus, reduced on the way
in.

### The levers, each closed with a measurement

| lever | what it was worth |
|---|---|
| eager kernels at module load (xla#698, landed by #204) | **−0.453 s** — the only lever that moved this leg, and it lands at the process-wide `CUDA_MODULE_LOADING=EAGER` ceiling |
| eager module loads on their own (#176 / xla#661) | null — it moves the registration to preload and leaves the kernels' code on the prove path |
| upload overlap (#193) | null — an upload into a freshly allocated buffer waits on the client's own compute stream |
| host-idle remainder (#205) | null — two built changes, both null against an `EAGER` control; the cost re-prices into the phase next door |
| read-ahead depth (#209) | null — smaller than two labels of one configuration differed by in the same sweep |
| constant tree over the extended domain (#183, #206) | worse — the same 8.4 GB over the same read-ahead path |
| H2D staging threshold (#204 / xla#718) | worse, so it ships off — the copies get faster and the pinned pool's growth inside the prove costs more than they return |
| XLA fusion cap (#149) | retracted — no occurrence in this wheel, and under the bridge pil2 proves the recursion on its own CUDA, where the flag has no surface |

**The two eager rows do not add and must not be subtracted from each
other.** Each is a different baseline: one toggles the wheel with the flag
already on, the other toggles the flag on the wheel that predates #698, and
toggling that flag on the post-#698 wheel is a third figure again. The parts
do not sum to it, and the difference is not a lever anyone has left to
claim — the two mechanisms gate each other, since #698 has nothing to do
without an eager module load and the flag had nothing to collect before
#698. Read the pair from one session's own two arms.

### The block-shaped mix

The closest stand-in for a real block this host runs: the `sha-hasher` example
guest at 14,000 iterations, hint-free, under the ASM emulator. Its 51.1 M steps
plan into 38 instances across 16 families — 13 Main, 6 Binary, 5
BinaryExtension, 2 BinaryAdd, and one each of Arith, Dma, Dma64AlignedMem,
DmaPrePost, DmaUnaligned, InputData, Mem, MemAlign, Rom, RomData and the two
virtual tables — where block 21740136 was 38 instances with 12 Main. The guest
uses the `sha2` crate's software path, so no precompile family appears.

Two things separate the legs on such a mix, both named in #170 and neither
specific to it:

- **One client.** The proves run back to back while pil2 overlaps three, so the
  client is never idle from the first prove to the last.
- **Family switches.** With `ZZ_RESIDENT_AIRS=1` every switch re-uploads and
  re-hashes the incoming family's constants, and Main alone comes and goes 13
  times. Raising the resident set does not fit on a 32 GB card at this share:
  `ZZ_RESIDENT_AIRS` of 2, 3, 4 and 8 all abort once the second or third family
  is resident, on a PJRT `Out of memory` that xla-pjrt's `check` turns into a
  panic rather than an error the bridge could evict on. A larger share leaves
  pil2 below the minimum it will start with. The trim was the candidate lever
  and does not move the floor, because what binds a client is a single
  program's own working set rather than anything kept between proves.

Start a run only once `nvidia-smi` shows the card empty: a process still
releasing its memory makes pil2 size its streams from what it sees and exit.

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
- **Memory budget (RTX 5090).** Three pools share the card: the clients'
  share, claimed up front (`ZZ_MEMORY_FRACTION`), what is held back from
  pil2's sizing (`ZZ_GPU_HEADROOM_GB`), and pil2's own, which is whatever is
  left. Floors are found by walking the fraction down until a run fails, with
  repeats at each step — a single run at the boundary is a race and lands
  either way. `bench/mem_budget.py` reads those tables back out of the logs
  they came from and carries these traps as its own rules. The per-prove
  inventory is a different question and a different tool:
  `bench/mem_stages.py` over a `ZZ_MEM_STAGES` run, with
  `bench/pil2_layout.py` over the proving key.

  Five rules that a memory unit here has got wrong before:

  - **A share is the arena the run allocated, not the fraction times the
    card.** XLA divides the fraction by the client count and applies it to
    its own base, which is under this card's total; the run prints the arena
    it allocated, and that is the figure to quote.
  - **Every "GB" pil2 prints is a GiB**, by two independent routes in its
    source, so its figures are already comparable to ours and "converting"
    one silently shrinks the bridge's excess.
  - **pil2 prints its own per-AIR need** on the `-vv` `TOTAL PROVER MEMORY
    USAGE` line, which is `mapTotalN * 8`. The ceiling it prints elsewhere is
    `max(basic, recursion)` over the whole key and is the compressor's, so it
    bounds nothing about a basic AIR.
  - **A lever is sized against the prove that actually fails, and the shape
    moves when you fix one.** Size against the failing run's live set rather
    than its largest allocation, and re-read which shape fails after each
    fix: freeing N bytes buys well under N of arena, because what a run needs
    is placement on top of its data. Do not subtract the two.
  - **A client-lifetime high-water is not a per-prove peak.** Only the prove
    that raised it can be attributed from it, and the order the proves ran in
    decides which that is.

  Per prove the bridge is at parity: section for section it holds what pil2
  holds, and on one of the two binding shapes it holds less. pil2 takes one
  buffer per stream and places sections at offsets inside it, so sections
  dead by the time a later one is written share their bytes; the bridge
  reaches the same place by releasing at each section's last reader. What
  separates a client from pil2 is therefore co-residency plus an in-program
  transient, not the prove's own sections — which is why three units that
  sized an excess per prove found none.

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

### The fixed sections stay only while the plan proves the AIR again

Sections held past their last reader are not a leak; they are what residency
costs. The key's sections stay on the client so the *next* prove of the same
AIR skips re-reading and re-uploading them, and the bet pays only when that
next prove comes. On the block-shaped mix it does — "Family switches" puts
what it saves at seconds per run — and on hello-world it never does, since
its eleven instances are eleven distinct AIRs.

proofman knows which it is before the first prove, so it sends its instance
list with duplicates intact (`Bridge::set_plan`). **That half lives in the
proofman fork**, whose `gen_proof` call site pushes every instance's key.
An AIR the list names once hands its sections to the prove that uploads them
rather than lending them — `driver::prove` *takes* the fixed env off the
driver instead of cloning it, which is what makes the removes inside the
prove free — and each section goes at its own last reader. An AIR named more
than once keeps everything, exactly as before.

Three properties, each a way it could have been got wrong:

- **An AIR the plan does not name keeps its sections.** A caller that sends
  no plan — a build pinned behind the fork rev, and `zz_prove` in every
  build — leaves every AIR behaving as it did before there was a plan. That
  is what let this land ahead of the fork; it is also why a figure taken on
  such a build is not a figure for this policy.
- **The count is a run's, not a client's.** Two instances of one AIR can land
  on two clients and each prove it once, but a count taken before the slots
  are assigned cannot know that, so it keeps on both — the conservative way.
- **The releases are read off the manifest, not off the two names.**
  `base_dead_after_logup` asks which programs after `logup` list `const_base`
  or a `custom_base_<id>` as an input, and releases only what none of them
  does, so an export that gives one a later reader keeps it rather than
  proving against a buffer that is gone.

`ZZ_FIXED_RESIDENT` pins the policy for measurement: `1` keeps every AIR's
sections, `0` keeps none whatever the plan says, unset lets the plan decide.

### The instance read-ahead, bounded by bytes

The admission is bounded by bytes rather than by a count: `ZZ_PENDING_BYTES`
caps the `trace`, `publics`, `airvalues` and `proofvalues` an instance may
upload while another prove still holds the client, because those uploads stay
live for the whole of the running prove. `ZZ_PENDING` remains the ceiling on
how many. It bounds the uploads it weighs and no more — the next AIR's
`const_base` rides `ZZ_FIXED_AHEAD`'s permit, taken inside this admission, so
a refusal holds that back too but an admission puts no cap on it. A client
with nothing on it admits any size, so the largest AIR is never refused
outright. Any figure quoted against "the default admission" from before this
is a different configuration.

### The instance's host words go at the upload, not at the prove

`gen_proof` returns before the prove runs, and the `StepsParams` pointers it
was handed are valid only for that call, so the bridge copies each instance
into host words of its own (`OwnedRequest`). `driver::upload_inputs` is that
copy's only reader: a prove binds the device buffers the upload returned, and
the one section it reads on the host is `global_challenge`, through the
transcript. So the words leave the request where the upload is
(`OwnedRequest::take_host_words`) and go when it returns. A base trace is a
wide AIR's whole `2^nBits x cm1` section — over a gigabyte at `n22` — and
admission allows more than one prove per client, so that is what holding them
to the end of the prove costs in host RAM.

The types carry the rule rather than a comment: the words the upload consumes
are their own shape (`driver::HostInputs`), and what a prove takes
(`driver::InstanceInputs`) carries an `Uploaded` that is not optional, so
nothing on the prove's side can read host words at all. `ZZ_DUMP_INPUTS` is
their one other reader, and it runs where they are still there, ahead of the
slot.

### What the in-program transient is made of

What a client holds beyond its registered buffers is what XLA allocated
inside an execution — the difference between the bridge's own registry and the
allocator's `in_use`, which `bench/buffer_assignment.py` itemises per
executable out of XLA's buffer assignment. On the LDE-bearing programs it is
dominated by the extend's transposed copy of its input rather than by anything
the driver keeps. `extend` splits the columns into blocks of at most `LDE_BLOCK_BYTES`
of the extended domain and writes each into the result, ordered so XLA does
not schedule several transforms at once; each column's LDE is independent, so
the codeword does not move and only the LDE-bearing programs re-export.

The blocking does not by itself cover the input's re-layout: the transform
reads a column and the section is stored by row, so each block is transposed
on the way in, and taking the field view over the whole section before
slicing lets those per-block transposes merge into one transpose of the
entire section, live from the first block to the last. `extend_words` takes
the view a block at a time, which leaves them where the exporter emitted
them. The section's declared layout does not move and neither does the
manifest — `raw_boundary` reports field inputs as `uint64` either way — so
the bridge uploads the same bytes to the same specs and the goldens pin the
codeword at three block sizes.

The loop's `optimization_barrier` does its job without surviving: no
`opt-barrier` is left in the optimized module, so nothing in the final
program enforces the order. It constrains the passes that run before it is
dropped, and taking it out of the source grows the arena by a block set.

The openings programs (`deep`, `evals`) hold a different shape: N of the
evMap's cubic columns over the extended domain, alive at once, one whole
column each. They are N independent results rather than a copy, so the only
thing that shrinks them is evaluating fewer rows at a time — and there the
division has to be **one dispatch per row window**, the shape the quotient's
chunks already use. Windowing inside one program leaves the windows
independent, and XLA is then free to compute them together, which holds them
together; a barrier chaining one window to the next does not recover it
either (#241 has both dumps). `pil2_prover._OPENING_ROW_CHUNKS` sets the count
for both, the schedule declares the windows as `deep_chunks` and
`evals_chunks`, and `deep_concat` and `evals_sum` compose them.

The two compose differently because the two reduce differently: `deep` is
elementwise over the extended domain, so its windows concatenate, while an
`evals` opening is a sum over the base domain, so its windows add. Addition in
the field is exactly associative, so adding the window partials is the same
value as the whole-domain sum rather than an approximation of it — which is
what lets the byte-gate check the composition. `evals`' window is therefore
counted in BASE rows, the domain `lev` and the sum are indexed by, and carries
the `stride`-times-longer extended window of the section under it.

`evals_sum` is its own program rather than folded into a consumer the way the
quotient's concatenation is folded into `quotient_commit`. It has to be: the
host downloads the openings and absorbs them into the transcript before the
DEEP challenges are squeezed, so the composition falls between the windows and
their only device-side reader and must materialise a result the host can read.

**A row window does not divide the quotient the way it divides the openings.**
Eight of `quotient_<size>`'s temporaries are whole extended cubic columns
(192 MiB each at Main's width, 1,536 MiB together) no matter how many windows
the cExp is evaluated in, because the quotient takes its window as an index
vector gathered *after* the columns are joined — which is what pil2's row-offset
operands need, a plain slice would read past the window's edge. Those 1,536 MiB
are the floor the count cannot reach under; the arena measured around it is
1,904 MiB at eight windows and 1,566 MiB at sixteen (#243). Read the floor and
the two totals, not a split: the 338 MiB between them is not the per-window
part halving, and what else moved is not established here. Raising the count
alone cannot halve the arena; what would is the shape `committed_column`'s
`rows` already fixes for the openings.
