# ZisK-vs-zisk-zorch per-stage inner-proof benchmark

How to compare zisk-zorch's GPU prover against ZisK's native pil2-proofman /
pil2-stark CUDA reference, **per stage**, on the premise that **both prove the
same instance, at the same scope, and produce the same output** (byte-match).
Only under that premise is a wall-clock comparison meaningful.

> **Read this first — two independent reasons no number here is a baseline yet.**
>
> 1. **Provenance.** Every row marked ✔ was measured on 2026-07-15, both sides,
>    on the same box (RTX 5090, driver 595.71.05) against main's shipped pins.
>    **extend, commit, quotient, LogUp and FRI** run on `origin/main`. **evals and
>    DEEP were measured on the wiring this branch brings in** (#61's `deep/` plus
>    #67's compile fix) — they do not exist on `origin/main` yet, so they are what
>    ships *if this branch lands*, not what ships today.
>    Where a re-measurement disagreed with the recorded figure, the measurement
>    wins and the delta is called out.
> 2. **No same-output gate.** Not one stage has a reproducible byte-match against
>    a real pil2-stark dump. The ratios are therefore *engineering signal, not
>    baselines*. Do not quote one as "zisk-zorch is Nx pil2" outside this doc
>    until its row's golden column says so.

## Two numbers that must not be re-quoted

- **The "45 ms quotient".** An earlier `quotient` proxy ran 64 constraints of
  degree 3 — **1/55th** of the real Main AIR's op density (#66). It was
  drastically undersized, not a win. The calibrated proxy (900 x degree-9)
  measures **77.0 ms** at 2^23.
- **The "~270 ms quotient" that replaced it.** Also not a real number: it was
  extrapolated from sizes where XLA had not yet fused the proxy. Measured, 2^23
  is 77.0 ms. See the quotient note below.
- **Any per-leg ms against the 24.6 s `GENERATING_INNER_PROOFS`.** That phase is
  the **whole** inner proof across **111 AIR instances** of block 24654300 (#30);
  `bench_inner_proof.py` times **one** AIR at one height. Dividing one into the
  other is a ~111× scope error — exactly the confound sp1-zorch had to retract.
  A real total awaits an all-AIR run.

## The benchmark that would be valid (and what's missing)

Both sides would prove the **same inner proof** (e.g. the block-24654300 one the
bench already targets) and their per-stage intermediates would be byte-identical,
so per-stage wall-clocks compare the same computation.

**The gate does not exist yet.** `golden/` pins each primitive against
pil2-proofman v1.0.0-alpha's `fields` crate, but only on *tiny synthetic* inputs —
it cannot show that the assembled prover reproduces pil2 on a real block (#59).
The blocking sub-task is a **pil2-proofman per-stage dump** of a real inner proof,
plus the `verify_*` runnables that consume it (#59).

> Stage-1's commit root **was** byte-matched once against a real pil2-stark CUDA
> dump (2026-07-10, #59's first slice). **No artifact survives** — the harness was
> never committed on any branch (only orphaned `.pyc` remain), and its dump
> fixture, loader, and capture recipe are gone. It is not currently reproducible,
> so stage-1 cannot carry a `byte-match` mark.

## zisk-zorch side — `bench_inner_proof.py`

**A venv from `requirements.in` alone is CPU-only and will silently benchmark the
CPU.** The pins name `jax-cuda12-plugin` but none of the `nvidia-*-cu12`
libraries it loads, so `jax.devices()` returns `[CpuDevice(id=0)]` and
`JAX_PLATFORMS=cuda` fails with *"Backend 'cuda' is not in the list of known
backends"*. Install the extra at the **same pinned version**:

```sh
pip install -r requirements.in \
    "jax-cuda12-plugin[with-cuda]==$(sed -n 's/^jax-cuda12-plugin==//p' requirements.in)" \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
python -c 'import jax; print(jax.devices())'    # must show CudaDevice, not CpuDevice
```

Then, the run that produced the ✔ extend rows above:

```sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
PYTHONPATH=<zisk-zorch> \
  python -m zisk_zorch.bench_inner_proof \
    --stages=extend --n_bits=22 --n_cols=38 --blowup_bits=1 --phase runtime \
    -o report.json      # --n_cols=24 for cm2
```

Stages: `extend`, `commit`, `full`, `quotient`, `divide`, `fri` (`--stages` to
select). zkbench owns warmup (3) + timed iterations (20) and reports warm
`latency`, `compile_time`, and a device-memory high-water mark.

- **`CUDA_VISIBLE_DEVICES=0` is load-bearing** on this box — device enumeration
  wedges otherwise (#65).
- **Use `--arity=4` for FRI** — production arity, and what the pil2 native runs.
  (It only started working with the warm-list fix in this change; before that it
  raised `TracerArrayConversionError`.)
- **A `RESOURCE_EXHAUSTED` at 2^23 is usually not the card.** JAX caps the BFC
  allocator at ~75% of the GPU (23.5 GiB of this 5090's 31.8), so a working set
  in the 24–30 GiB band OOMs at the default and runs with
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95`. This is what forced #68/#69 to
  extrapolate; at 0.95 both measure directly at the native's size. Reach for it
  before assuming a stage does not fit.
- **Do NOT `jax.profiler.trace` at 2^23.** It leaks host RAM, spikes load, and
  wedges CUDA (#65). Decompose by direct sub-op timing instead.
- **Run `--phase compile` and `--phase runtime` as separate invocations.** The
  memory peak is process-cumulative — PJRT exposes no portable per-op reset.
- **The default flags are not production-shaped.** `--n_cols` defaults to 64,
  which matches no real AIR (Main is 38 cm1 / 24 cm2), and `--arity` defaults to
  2 while production is 4. `--fold_bits=3` is the uniform factor-8 drop every
  ZisK starkStruct uses; a drop-2 schedule no ZisK config uses would add tail
  rounds and overstate the fold.
- The `output_hash` in the report is a self-consistency hash **across zisk-zorch
  runs** — it is *not* a pil2 byte-match and cannot back the golden column.

## Native pil2 side

**Full inner proof** — the only invocation recorded anywhere (#30):

```
cargo-zisk 0.18.0 [gpu]  --emulator --no-aggregation     # mainnet block 24654300
```

→ ~47 s wall, 7.41 GB peak, 111 AIR instances, of which `GENERATING_INNER_PROOFS`
= **24.6 s**. Block choice is constrained: block 21740136's procured input is
version-incompatible with the bundled `zec-reth.elf` (`Deserialization
UnexpectedEof`) — use 24654300 or re-procure.

> **Version skew.** `cargo-zisk 0.18.0` is pil2-proofman **v0.18.0**, but the
> goldens pin **v1.0.0-alpha**. The 24.6 s was measured against a different
> reference than this repo byte-matches.

**Per-stage** — ⚠️ **no native per-stage bench is reproducible *from this repo*.**
Every native figure came from Google-Benchmark binaries and hand-lifted CUDA
kernels built ad-hoc under `/tmp/claude-1006/` on one box (`main_bench`,
`fri_bench`, `evmap_bench`, `friexp_bench`, `ntt_bench`, `merkle_bench`; pil2
source at `/tmp/claude-1006/pil2-proofman/pil2-stark`, build scripts alongside).
Rebuilding them behind a committed script is open work.

While that scratch dir survives, the extend and quotient natives do run:

```sh
cd /tmp/claude-1006
LD_LIBRARY_PATH=/tmp/claude-1006/gmp-prefix/lib CUDA_VISIBLE_DEVICES=0 \
  ./main_bench --benchmark_filter='MAIN_EXPR_PATTERN|MAIN_LDE_CM1|MAIN_LDE_CM2|MAIN_MERKLE' \
    --benchmark_min_time=5x --benchmark_repetitions=3 \
    --benchmark_report_aggregates_only=true
```

`bench_main_proof_gpu.cu` pins the Main AIR's real production dims and is the
authority for what each row must match: N=2^22, N_ext=2^23, cm1=38 / cm2=24
columns, Merkle **W=16 arity=4**.

Re-run 2026-07-15 (RTX 5090, driver 595.71.05): `MAIN_LDE_CM1` **32.5 ms**
(cv 0.10%), `MAIN_LDE_CM2` **20.4 ms** (cv 0.03%), `MAIN_EXPR_PATTERN` **133 ms**
(cv 0.15%), `MAIN_MERKLE_STAGE1` **36.6 ms** (cv 1.07%), `MAIN_MERKLE_STAGE2`
**19.9 ms** (cv 0.22%) — reproducing the recorded 32.5 / 20.4 / 134; the merkle
absolutes were previously unrecorded.

FRI likewise (`./fri_bench`, same env; it takes no flags — arity 4 and the
`[23,20,17,14,11,8,5]` schedule are compiled in): fold **3.27 ms**, merkle
**4.56 ms**, total **7.84 ms** (best of 7), reproducing the recorded
3.29 / 4.56 / 7.86.

evals / DEEP: `./evmap_bench` and `./friexp_bench` (optional arg = reps,
default 7). Each prints **two** cases — 1 opening point then 2; the rows above
use **1 opening**, so do not read the last block. Re-run 2026-07-15: evmap
**3.72 ms**, friexp **8.88 ms** (recorded 3.70 / 8.90).

LogUp: `./gsum_bench` (no flags; sweeps log2N, I=8 compiled in). At log2N=22 it
pairs pil2's own Blelloch scan with the cubic inv+fold: invfold **2.213 ms** +
scan **0.238 ms** = **2.451 ms**, reproducing the recorded 2.45 ms.

`/tmp` is not durable; treat all of this as a stopgap, not the fix.

## Per-stage comparison

RTX 5090, one AIR, N=2^22 → N_ext=2^23 (`blowup_bits=1`) unless noted. **Each row
brackets a different span** — read the notes. ✔ = re-measured 2026-07-15;
everything else is transcribed from the cited issue.

| stage | native pil2 | zisk-zorch | ratio | golden | on main? |
|---|---|---|---|---|---|
| **extend cm1 (38 col)** | **32.5 ms** ✔ | **23.7 ms** ✔ | **0.73×** ✔ | unit golden only | ✅ **shipped** |
| **extend cm2 (24 col)** | **20.4 ms** ✔ | **14.9 ms** ✔ | **0.73×** ✔ | unit golden only | ✅ **shipped** |
| extend cm1, with #63 | 32.5 ms ✔ | 18.7 ms | 0.58× | unit golden only | ❌ needs #63 |
| extend cm2, with #63 | 20.4 ms ✔ | 11.8 ms | 0.58× | unit golden only | ❌ needs #63 |
| **commit stage1 (38 col)** | **36.6 ms** ✔ | **38.8 ms** ✔ | **1.06×** ✔ | unit golden only | ✅ **shipped** |
| **commit stage2 (24 col)** | **19.9 ms** ✔ | **20.9 ms** ✔ | **1.05×** ✔ | unit golden only | ✅ **shipped** |
| quotient (constraint eval) | 133 ms ✔ | **77.0 ms** ✔ | 0.58× ⚠️ | unit golden only | ✅ shipped — but proxy/proxy |
| — zerofier divide | *(bracketed)* | 0.32 ms | parity | unit golden only | ✅ |
| **LogUp grand-sum (I=8)** | **2.45 ms** ✔ | **9.61 ms** ✔ | **3.92×** ✔ | unit golden only | ✅ **shipped** |
| LogUp grand-sum, with #64 | 2.45 ms ✔ | 3.20 ms | 1.30× | unit golden only | ❌ #64 open |
| **evals (`evmap`)** | **3.72 ms** ✔ | **15.8 ms** ✔ | **4.25×** ✔ | none | ✅ this branch (#61) |
| **DEEP (`computeFRIExpression`)** | **8.88 ms** ✔ | **15.6 ms** ✔ | **1.76×** ✔ | **none** | ✅ this branch (#61 + #67) |
| **FRI total (queries excl.)** | **7.84 ms** ✔ | **19.5 ms** ✔ | **2.49×** ✔ | unit golden only | ✅ **shipped** |
| — FRI fold | 3.27 ms ✔ | 14.75 ms | 4.48× | unit golden only | ✅ shipped |
| — FRI merkle | 4.56 ms ✔ | ~5.09 ms | ~1.12× | unit golden only | ✅ shipped |
| FRI total (INTT fold) | 7.84 ms ✔ | ~6.5 ms | ~0.83× | unit golden only | ❌ #70 + zorch#456 |

⚠️ **= proxy or extrapolation. Specifically:**

- **`quotient` is still proxy-vs-proxy, but #66's "~270 ms / ~2.0× slower" does
  not survive measurement.** #66 measured at ≤2^21 and extrapolated to ~270 ms,
  believing 2^22+ OOMs at ~44 GB. On main's shipped pins (2026-07-15) the real
  2^23 run is **77.0 ms at 5.13 GiB peak** — no OOM. The extrapolation failed
  because the proxy changes regime: at ≤2^21 XLA materializes the intermediates
  (9.91 GiB, ~33 ns/row); at ≥2^22 it fuses them (5.13 GiB, **~9.2 ns/row**), so
  projecting from the unfused sizes over-predicts by ~3.5×. The work is real —
  time scales with `--n_constraints` (225 / 450 / 900 → 33.2 / 47.6 / 77.0 ms).
  **The "we are ~2× slower on quotient" reading is unsupported; measured, the
  proxy is faster than the native proxy.** What has *not* changed is that both
  sides are proxies — ours is 900 independent degree-9 products, the native is a
  density mimic tuned to the real ~135 ms kernel, not the interpreter — so 0.58×
  is not a baseline. Settling it still needs the real Main constraints through
  `constraint_eval` (#66).
- **`evals` and `DEEP` zisk-zorch numbers are extrapolated** from 2^19–2^21 (both
  OOM at 2^22+); the native figures are real runs. Their code lives on the
  unmerged wire branch (#61), and DEEP additionally depends on #67.
- **evals and DEEP are branch numbers, and both extrapolations held.** #68
  predicted ~4.3× and #69 ~1.8×; measured directly at the native's config
  (n_ext=2^23, M=68) they are **4.25×** (15.80 vs 3.72 ms) and **1.76×** (15.64 vs
  8.88 ms). Unlike quotient, these projections were sound. Both are measured on
  the wiring this branch adds (#61 + #67), not on `origin/main`. The DEEP gap is the one #69 names: our
  committed buffer is 68 *cubic* columns (12.75 GiB) where the native holds 62
  base + 6 cubic = 80 gl64 (5.37 GiB), so base columns embedded to cubic cost
  ~2.4x the reads.
- **Bracket boundaries differ**: `fri` excludes the query phase; `commit` excludes
  the extend it depends on; native `MAIN_EXPR` excludes the INTT-back and Merkle.
- **`extend` 0.58× is cumulative over three levers** (xla#254 + #63 + xla#257).
  The plugin ones shipped (#71); **#63 has not**. Main's shipped extend was
  measured on 2026-07-15 at **0.73× on both column counts** (23.7 ms / 14.9 ms) —
  already past parity. #63 is what takes it to 0.58×.
- **`LogUp` on main is 3.92×, not the 1.30× #64 reports.** The recorded arc
  (11.39 → 7.09 → 3.51 → 3.20 ms) was captured on locally-built plugins, so its
  intermediate points are not comparable to a shipped-pin run; main measured
  **9.61 ms** on 2026-07-15. #64 (fused fold + [I,N] layout) is what closes it to
  1.30×. There is no `gsum` leg in `bench_inner_proof` — this was measured by
  calling `grand_sum` directly at the native's config (N=2^22, I=8, `[N,I]`
  cubic).
- **`commit` runs at arity 4 at production dims**, despite `--arity`'s help text
  warning that "arity>=3 hits the `merkle_commit` power-of-two leaf-layer limit at
  scale". It did not bite at 2^23 x 38 — but the default is still 2, so pass
  `--arity=4` explicitly or you will measure a tree the native never builds.
- **The `fri` leg runs at production arity as of this change.** It used to warm a
  hardcoded widths 8 and 12, so `--arity=4` — the arity #49/#65 and the pil2
  native all use — traced a width-16 perm and died with
  `TracerArrayConversionError`. It now warms `merkle_tree(arity)`'s width,
  derived from the flag. FRI itself was never broken: `prove` runs fine at arity 4
  (verified), and the earlier recorded numbers came from a separate harness, so
  the short warm list never blocked them.

## Measure shipped code

A number is only a baseline if it runs what the team **ships**. zisk-zorch has
three ways to accidentally measure something else:

```sh
# 1. Are the pins the shipped ones? (must be empty)
git fetch origin && git diff origin/main -- requirements.in requirements_lock_3_11.txt

# 2. Does the venv match the pins?
grep -E '^(zorch|jax|jaxlib|jax-cuda12-(plugin|pjrt)|zk-dtypes)==' requirements.in
pip show zorch jax jax-cuda12-pjrt | grep -E 'Name|Version'

# 3. Any local source override in play? (must be absent/empty)
test ! -s .bazelrc.user || echo "LOCAL OVERRIDE ACTIVE — move it aside"
```

- **zorch is a pip wheel pinned in `requirements.in`**, not a Bazel
  `git_override` — that knob was removed (#55). A `--override_module=zorch=` line
  is a no-op today.
- **`.bazelrc.user` can point `zkx` / `prime_ir` at local checkouts.** Move it
  aside before a baseline run.
- **The GPU plugin is whichever `jax-cuda12-pjrt` wheel the venv has** (the
  fractalyze/xla fork's backend). Perf work often hot-swaps a locally built
  `pjrt_c_api_gpu_plugin.so` over the wheel's
  `jax_plugins/xla_cuda12/xla_cuda_plugin.so` — if you did that, **restore the
  `.orig`** or you are not measuring the shipped plugin.

> This is not hypothetical. #60 exists partly because sp1-zorch captured a
> baseline against a `zorch` override weeks behind `origin/main` and misread it as
> the shipped number. The same trap is live here.

## Size caveat

Never compare across differently-sized inputs.

- **One block is many inner proofs.** Block 24654300 has 111 AIR instances; a
  per-AIR ms and the 24.6 s phase total are not the same scope.
- **AIRs differ in width.** Main is 38 (cm1) / 24 (cm2) columns; DEEP/evals see 68.
  Always state the column count with the number.
- **Height is the axis everything scales on.** Every figure here is anchored to
  N=2^22 / N_ext=2^23. An extrapolated number and a measured one are also a size
  mismatch — mark them.
- **The FRI schedule must be production**: uniform drop-3 to nBits 5
  (`[22,19,16,13,10,7,5]` dominant). ZisK uses no non-uniform schedule.

## References

- Template: sp1-zorch [`docs/sp1-baseline.md`](https://github.com/fractalyze/sp1-zorch/blob/499fe71852de/docs/sp1-baseline.md).
- This doc: #60. Byte-match runnables (the missing gate): #59.
- Per-stage work: extend #58 / #63, commit #54, quotient #66, LogUp #64,
  evals #68, DEEP #67 / #69, FRI #65 / #70, roadmap #3.
