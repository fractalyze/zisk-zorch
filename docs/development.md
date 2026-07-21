# Development guide

Everything needed to build, test, and benchmark zisk-zorch: the environment, the
test conventions, and the per-stage baseline against native pil2. For the
prover's structure see [architecture.md](architecture.md); for coding style see
[conventions.md](conventions.md).

## Development environment

Pure Python on frx + the Fractalyze [xla](https://github.com/fractalyze/xla)
fork's PJRT plugin, built with Bazel 9 (bzlmod). `zorch` arrives as a dev-release
wheel from the Fractalyze index, pinned in
[`../requirements.in`](../requirements.in), so `frx` and `zk_dtypes` resolve once
there.

```sh
python3.11 -m venv .venv && . .venv/bin/activate
pip install -r requirements.in \
    --extra-index-url https://fractalyze.github.io/pypi/simple/
bazel test //...                 # hermetic, sandboxed; FRX_PLATFORMS=cpu default
```

For iterative dev outside Bazel: `export PYTHONPATH="$PWD"`.

**A venv from `requirements.in` has no GPU.** The pins name `frx-cuda12-plugin`,
but `frx_plugins/xla_cuda12` resolves its CUDA extensions by importing
`jax_cuda13_plugin` / `jax_cuda12_plugin` / `jaxlib.cuda` — never
`frx_cuda12_plugin`. `initialize()` then asserts on `cuda_versions is None`, the
`cuda` backend never registers, and work silently runs on the CPU. Assert the
device before trusting any GPU number:

```sh
python -c 'import frx; print(frx.devices())'   # must show CudaDevice, not CpuDevice
```

## Testing

```sh
bazel test //...     # hermetic, sandboxed; FRX_PLATFORMS=cpu by default
```

[`.bazelrc`](../.bazelrc) pins `FRX_PLATFORMS=cpu` so a plain `bazel test` is
deterministic on any machine — CPU is the default, not a requirement. CI
overrides it per matrix leg. `//...` is the whole suite on either backend; the
`-gpu` tag filter currently matches nothing.

### Test sizing & timeouts

`size` and `timeout` are independent knobs: **`size`** (`small`/`medium`/`large`)
is a resource hint governing parallelism; **`timeout`**
(`short`/`moderate`/`long`/`eternal` = 60/300/900/3600 s) is the wall-clock cap,
derived from `size` when unset. Every test here declares a `size` and none
declares a `timeout`. Measured locally (warm, CPU, under parallel load):

| test | size | cap | actual |
|---|---|---|---|
| `fri:verifier_test` | medium | 300 s | **135 s** |
| `commit:openings_test` | medium | 300 s | **130 s** |
| `commit:trace_commit_test` | large | 900 s | 90 s |
| everything else | small/medium | 60–300 s | 2–49 s |

The two to watch are `verifier_test` and `openings_test`, **not** the `large`
one: they sit at ~45% of a 300 s cap while `trace_commit_test` uses 10% of its
900 s. Declare a **`timeout` explicitly** if you push either past ~150 s. A
dependency bump invalidates the Bazel cache, so the suite re-runs **cold** on the
shared CI runner — slower than a local box under parallel load, and a test that
finishes in 150 s here can blow the 300 s `medium` cap there and fail as
`TIMEOUT`.

> A green CI on a branch with no dep bump is usually an all-cache-hit run (the
> remote cache is shared with dev boxes), not evidence the tests fit their caps.

Sizes are loose in the other direction too — `--test_verbose_timeout_warnings`
flags `trace_commit_test` as oversized and nine `medium` tests as `small`-able.

### Fixtures

Two kinds, both vendored, small, and compared with exact equality — field
elements either match or they don't.

- **Goldens** (`zisk_zorch/*/testdata/golden/*.json`) pin every primitive that
  mirrors pil2-stark against pil2-proofman v1.0.0-alpha's own `fields` crate.
  Regenerate with `cd tools/fixture-gen && cargo run --release`; the rules that
  keep them reproducible live in
  [conventions.md](conventions.md#golden-tests-are-the-spec). A clean
  `git status` afterwards is the byte-match.
- **Proving-key artifacts** (`quotient/testdata/<air>_{cexp,constraints}.json`)
  carry a ZisK AIR's stage-2 composite-cExp fragment and per-constraint SSAs,
  extracted from the ziskup proving key by
  [`../scripts/extract_cexp.py`](../scripts/extract_cexp.py).

## Per-stage baseline against native pil2

How to compare this prover against ZisK's native pil2-stark CUDA reference **per
stage**, on the premise that both prove the same instance, at the same scope, and
produce the same output. Only under that premise is a wall-clock comparison
meaningful.

> **No row here is a baseline yet.** Not one stage has a reproducible byte-match
> against a real pil2 dump — `tools/fixture-gen` pins primitives on tiny
> synthetic inputs, and #59's real-dump harness was never committed. The ratios
> are engineering signal. Do not quote one as "zisk-zorch is Nx pil2" outside
> this page.

**Numbers that must not be re-quoted:**

- **"45 ms quotient"** — a proxy of 64 degree-3 constraints, 1/55th of Main's op
  density (#66).
- **"~270 ms quotient"** — an extrapolation to a size the *proxy* cannot reach,
  not a measurement; likewise the ~370 ms the corrected proxy would extrapolate
  to. Both are proxy artifacts. The real Main air, folded through
  `constraint_eval`, does run at 2^23 and measures 8.1 ms (#66, below) — quote
  that, never the proxy extrapolations.
- **"77 ms quotient" / "0.58x"** — measured before `_make_eval_fn` drew distinct
  columns, so CSE folded 900 constraints down to 38: ~1/24th of the claimed
  density.
- **Any per-leg ms against the 24.6 s `GENERATING_INNER_PROOFS`** — that phase is
  the whole inner proof across **111 AIR instances** of block 24654300 (#30);
  this bench times **one** AIR. A ~111x scope error.

### zisk-zorch side — `bench_inner_proof.py`

```sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  bazel run //zisk_zorch:bench_inner_proof -- \
    --stages=extend --n_bits=22 --n_cols=38 --blowup_bits=1 --phase runtime \
    -o report.json
```

zkbench owns warmup (3) + timed iterations (20) and reports warm `latency`,
`compile_time`, and a device-memory high-water mark.

- **`CUDA_VISIBLE_DEVICES=0` is load-bearing** here — device enumeration wedges
  otherwise (#65).
- **A `RESOURCE_EXHAUSTED` at 2^23 is usually not the card.** frx caps the
  allocator at ~75% of the GPU (23.5 of this 5090's 31.8 GiB);
  `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` runs sizes that otherwise appear not to
  fit. This is what forced #68/#69 to extrapolate.
- **Do NOT `frx.profiler.trace` at 2^23** — it leaks host RAM and wedges CUDA
  (#65). Decompose by direct sub-op timing.
- **Run `--phase compile` and `--phase runtime` separately** — the memory peak is
  process-cumulative.
- **The defaults are not production-shaped**: `--n_cols` defaults to 64 (Main is
  38 cm1 / 24 cm2) and `--arity` to 2 (production is 4).
- **`--n_constraints` is only as real as the products are distinct.**
  `_make_eval_fn` draws each constraint's columns from a seeded RNG so the 900
  products over 38 columns are distinct; picking them in index order repeats a
  tuple every `n_cols` and CSE folds the repeats away, silently measuring a
  fraction of the requested density. `bench_inner_proof_test` pins this.
- The report's `output_hash` is a self-consistency hash across zisk-zorch runs —
  *not* a pil2 byte-match.

### Native pil2 side

**Full inner proof** — the only invocation recorded (#30): `cargo-zisk 0.18.0
[gpu] --emulator --no-aggregation` on mainnet block 24654300 → ~47 s wall, 111
AIR instances, of which `GENERATING_INNER_PROOFS` = **24.6 s**. Note the version
skew: that is pil2-proofman **v0.18.0**, while the goldens pin **v1.0.0-alpha**.

**Per-stage** — ⚠️ not reproducible *from this repo*. The figures come from
Google-Benchmark binaries and hand-lifted CUDA kernels built ad-hoc under
`/tmp/claude-1006/` (`main_bench`, `fri_bench`, `evmap_bench`, `friexp_bench`,
`gsum_bench`; pil2 source alongside). Rebuilding them behind a committed script is
open work. `bench_main_proof_gpu.cu` is the authority on what each row must match:
N=2^22, N_ext=2^23, cm1=38 / cm2=24, Merkle **W=16 arity=4**.

```sh
cd /tmp/claude-1006 && LD_LIBRARY_PATH=$PWD/gmp-prefix/lib CUDA_VISIBLE_DEVICES=0 \
  ./main_bench --benchmark_filter='MAIN_EXPR_PATTERN|MAIN_LDE_CM1|MAIN_LDE_CM2|MAIN_MERKLE' \
    --benchmark_min_time=5x --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
./fri_bench     # no flags; arity 4 and [23,20,17,14,11,8,5] compiled in
./gsum_bench    # no flags; sweeps log2N, I=8
./evmap_bench; ./friexp_bench   # each prints 1-opening THEN 2-opening — the rows below use 1
```

### Per-stage comparison

RTX 5090, one AIR, N=2^22 → N_ext=2^23, both sides re-measured 2026-07-16. Each
row brackets a different span (FRI excludes the query phase; commit excludes its
extend; `MAIN_EXPR` excludes the INTT-back and Merkle), so rows do not sum.

| stage | native pil2 | zisk-zorch | ratio | on main? |
|---|---|---|---|---|
| extend cm1 (38 col) | 32.5 ms | 23.6 ms | **0.73×** | ✅ |
| extend cm2 (24 col) | 20.4 ms | 15.0 ms | **0.73×** | ✅ |
| commit stage1 (38 col) | 36.6 ms | 38.9 ms | **1.06×** | ✅ |
| commit stage2 (24 col) | 19.9 ms | 21.0 ms | **1.05×** | ✅ |
| quotient ⚠️ #66 | 133 ms @2^23 (synthetic mimic) | 8.1 ms @2^23 (real Main air) | — | ✅ |
| LogUp grand-sum (I=8) | 2.45 ms | 7.91 ms | **3.23×** | ✅ (not in the spine) |
| evals (`evmap`) | 3.72 ms | 15.8 ms | **4.25×** | ❌ #61 |
| DEEP (`friExp`) | 8.88 ms | 15.6 ms | **1.76×** | ❌ #61 |
| FRI total (queries excl.) | 7.84 ms | 19.7 ms | **2.49×** | ✅ |

The LogUp row reflects #64 (the `@frx.jit` + `fnp.cumsum` fold fusion, now
merged). Its win is the fusion: the pre-#64 path — a Hillis-Steele scan
materializing the `[N, I]` ratio to HBM — measures ~22 ms (9.0×) at this pin;
#64 fuses div + fold + scan into one pass and lands at 7.91 ms (3.23×), matching
the PR's own pre-Fermat figure (25.1 → 7.85 ms). The residual is the cubic
inverse — `num/den` over 33.5 M cubic elements is ~7 of the 7.91 ms, so the fold
and scan are no longer the bottleneck. #64's headline 1.43× is on the
Fermat-inverse plugin; closing this pin's 3.23× to it needs that extension base
reciprocal (prime-ir #398), not more fold work.

Open PRs move two more: #63 takes extend to 0.58×, and #70 + zorch#456 take FRI
to 0.83× (its fold is 14.75 of the 19.7 ms). #69 is the evals/DEEP gap — our
committed buffer embeds every base column to cubic, so it is 2.55× the native's
reads.

**⚠️ quotient — the real number, and why it still has no clean ratio (#66).** The
re-authored Main air (`rw_constraints` `zisk/v1` `main`, 38 columns, 19
constraints) folded through the production `quotient_from_constraints`
(`constraint_eval` + zerofier) measures **8.1 ms at 2^23** on this card
(0.99 ns/row, 4.3 GiB peak) — a direct measurement *at* Main's size, no
extrapolation. Sibling airs on the same path, all at 2^23: `binary` (39 col, 14
con) 4.0 ms, `arith` (44 col, 33 con) 14.4 ms. `keccak` (2137 col, 1602 con) is
the one heavy air, ~55 ns/row and bandwidth-bound by its 17 KB/row trace, but
ZisK runs it at small heights. Reproduce with `--stages=quotient --chip=main`
(the flag forces x64 and folds the chip's actual `eval_constraints`).

**The real quotient fuses.** Peak memory is linear in size and single-digit GiB —
the compiled DAG does *not* materialize per-constraint intermediates. That is
what overturns the retracted "~270 ms": the **proxy** went HBM-bound because it
forms 900 *independent* degree-9 products (the `900 × rows × 8 B` table below);
a real air reuses columns, and `constraint_eval` fuses the shared-subexpression
graph. The direction #66 left open — competitive or 2× behind — resolves to
**single-digit ms, register/bandwidth-resident**, not the proxy's 2×.

The proxy's own ceiling (why the retracted numbers came from extrapolation): at
900 degree-9 constraints it measures ~44 ns/row flat and cannot reach 2^22.

| rows | predicted (`900 × rows × 8 B`) | measured |
|---|---|---|
| 2^21 | 14.06 GiB | 13.65 GiB peak, 92.5 ms |
| 2^22 | 28.1 GiB | **OOM** — a single 21.97 GiB allocation fails |
| 2^23 | 56.2 GiB | unreachable |

**But 133 ms is not the counterpart, so there is still no ratio to quote.** pil2's
`MAIN_EXPR_PATTERN` is itself a synthetic mimic: `bench_main_proof_gpu.cu`
hardcodes `EXPR_BASE_FMA_OPS = 7000` and `EXPR_FP3_OPS = 900` per row, "tuned so
the bench runtime lands near" a target. The real Main air is ~19 field muls/row
(mostly booleanity `x·(x−1)`) — the mimic over-states its density by ~370×. Both
sides of the original comparison were proxies, wrong by ~two orders of magnitude
in opposite directions. A true head-to-head needs pil2's per-air
`STARK_CALCULATE_QUOTIENT_POLYNOMIAL` timer under a real witness (block
24654300) — the same "no real block setup" wall as FRI #65. What is now settled:
our real quotient is register-resident and single-digit ms, not the
materialization-bound 2× the proxy implied.

### End-to-end (`prove_inner`)

Proves at 2^18 × 38 in 101.1 s through the real DEEP combiner. Whole-proof peaks,
real DEEP stage, this RTX 5090 (`XLA_PYTHON_CLIENT_MEM_FRACTION` raised to fit):

| base | N_ext | full prove | peak |
|---|---|---|---|
| 2^20 | 2^21 | 69.7 s | 10.46 GiB |
| 2^21 | 2^22 | 71.6 s | 20.98 GiB (pre-#69) → fits |
| 2^22 | 2^23 | 74.2 s | **17.43 GiB** (post-#69; **OOM'd pre-#69**) |
| 2^23 | 2^24 | — | OOM on the query grind (`queries.py`), not DEEP |

**#69 lifted the ceiling one doubling.** DEEP used to embed every base column to
cubic, so its committed buffer was `2^23 × (M+1) × 24B` = 7.31 GiB at 38 columns
(pil2 holds the same columns as 80 gl64 = 5.00 GiB) — which OOM'd the whole proof
at 2^22. Keeping base columns base (#69) drops the DEEP fold from 8.19 → 0.92 GiB
and 26.6 → 11.3 ms at 2^21 (byte-identical), and 2^22 now proves at 17.43 GiB.
The next ceiling at 2^23 is a *different* stage — a 9.5 GiB alloc in the
proof-of-work grind — not DEEP.

No whole-proof ratio appears above because none can: the only native total covers
111 AIRs. And nothing verifies these proofs — there is no `verify_inner`, so "it
proved" means the spine ran, not that the output is correct.

### Measure shipped code

A number is only a baseline if it runs what the team **ships**:

```sh
git fetch origin && git diff origin/main -- requirements.in requirements_lock_3_11.txt  # must be empty
pip show zorch frx frx-cuda12-pjrt | grep -E 'Name|Version'                             # venv == pins?
test ! -s .bazelrc.user || echo "LOCAL OVERRIDE ACTIVE — move it aside"
```

`zorch` is a pip wheel pinned in `requirements.in`, not a Bazel `git_override`
(#55) — an `--override_module=zorch=` line is a no-op. `.bazelrc.user` can point
`zkx` / `prime_ir` at local checkouts. And perf work often hot-swaps a locally
built `pjrt_c_api_gpu_plugin.so` over the wheel's — restore the `.orig` or you are
not measuring the shipped plugin.

> Not hypothetical: #60 exists partly because sp1-zorch captured a baseline
> against a `zorch` override weeks behind `origin/main` and misread it as shipped.

### Size caveat

Never compare across differently-sized inputs. One block is 111 AIR instances;
AIRs differ in width (Main is 38 cm1 / 24 cm2, DEEP/evals see 68); height is the
axis everything scales on, and every figure here is anchored to N=2^22 /
N_ext=2^23. An extrapolated number and a measured one are also a size mismatch —
mark them. The FRI schedule must be production: uniform drop-3 to nBits 5
(`[22,19,16,13,10,7,5]`); ZisK uses no non-uniform schedule.

### References

Template: sp1-zorch [`docs/development.md`](https://github.com/fractalyze/sp1-zorch/blob/main/docs/development.md).
This page: #60. The missing byte-match gate: #59. Per-stage work: extend #58/#63,
commit #54, quotient #66, LogUp #64, evals #68, DEEP #67/#69, FRI #65/#70.
