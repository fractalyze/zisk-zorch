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
bazel test //...                 # hermetic, sandboxed; JAX_PLATFORMS=cpu default
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
bazel test //...     # hermetic, sandboxed; JAX_PLATFORMS=cpu by default
```

[`.bazelrc`](../.bazelrc) pins `JAX_PLATFORMS=cpu` so a plain `bazel test` is
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
- **"~270 ms quotient"** — an extrapolation to a size the proxy cannot reach, not
  a measurement. Do not re-quote it — and do not quote the ~370 ms that
  extrapolating the *corrected* proxy would give either: 2^23 is two doublings
  past the OOM boundary below. Quotient has no number at Main's size, in either
  direction.
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
| quotient ⚠️ | 133 ms @2^23 | 92.5 ms @2^21 | — | sizes differ; see below |
| LogUp grand-sum (I=8) | 2.45 ms | 7.91 ms | **3.23×** | ✅ (not in the spine) |
| evals (`evmap`) | 3.72 ms | 15.8 ms | **4.25×** | ❌ #61 |
| DEEP (`friExp`) | 8.88 ms | 15.6 ms | **1.76×** | ❌ #61 |
| FRI total (queries excl.) | 7.84 ms | 19.7 ms | **2.49×** | ✅ |

The LogUp row reflects #64 (the `@frx.jit` + `jnp.cumsum` fold fusion, now
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

**⚠️ quotient has no ratio, and cannot yet.** At Main's density (900 degree-9
constraints over 38 columns) the proxy measures 47.4 ms at 2^20 rows and 92.5 ms
at 2^21 — a flat ~44 ns/row. **2^22 is its ceiling, and it is a hard one**: the
proxy materializes ~900 full-height base-field intermediates, so it needs
`900 × rows × 8 B`, and the model predicts the measurements closely.

| rows | predicted | measured |
|---|---|---|
| 2^21 | 14.06 GiB | 13.65 GiB peak, 92.5 ms |
| 2^22 | 28.1 GiB | **OOM** — a single 21.97 GiB allocation fails |
| 2^23 | 56.2 GiB | unreachable |

Against ~23.5 GiB usable (frx caps at ~75% of the card's 31.8), 2^22 is already
out. The native's 133 ms exists only at 2^23 (`MAIN_EXPR_PATTERN` has Main's dims
compiled in, no sweep), so there is no same-size pair to divide, and closing the
gap by extrapolation would cross two doublings and the OOM boundary — which is
how both "~270 ms" and "77 ms" came about.

That memory gap is the finding: pil2's bytecode interpreter keeps operands
register-resident where we materialize — the same shape as #69. The real number
needs the re-authored Main constraints (#66), which reuse columns instead of
forming 900 independent products.

### End-to-end (`prove_inner`)

Proves at 2^18 × 38 in 101.1 s through the real DEEP combiner. **At 2^22 it does
not fit**: DEEP's committed buffer is `2^23 × (M+1) × 24B` = **7.31 GiB** at 38
columns, where native pil2 holds the same columns as 80 gl64 = **5.00 GiB**.
Predicted and observed agree exactly at both 24 and 38 columns. So **#69 is a
blocker on running at all**, not a perf item; #64 cannot help (`grand_sum` is not
in the spine) and #70 targets the fold's speed, not DEEP's footprint.

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
