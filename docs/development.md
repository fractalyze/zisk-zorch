# Development guide

Everything needed to build, test, and benchmark zisk-zorch: the environment, the
test conventions, and the per-stage baseline against native pil2. For the
prover's structure see [architecture.md](architecture.md); for coding style see
[conventions.md](conventions.md).

## Development environment

Pure Python on frx + the Fractalyze [xla](https://github.com/fractalyze/xla)
fork's PJRT plugin, built with Bazel 9 (bzlmod). `zorch` arrives as the
`pyzorch` wheel from the Fractalyze index, pinned in
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
declares a `timeout`. The two to watch are `fri:verifier_test` and
`commit:openings_test`: ~135 s warm on an idle box against a 300 s `medium`
cap. Declare a **`timeout` explicitly** if you push either past ~150 s — a
dependency bump invalidates the Bazel cache, the suite re-runs **cold** on the
shared CI runner under parallel load, and a test that fits locally blows the
cap there as a `TIMEOUT`.

> A green CI on a branch with no dep bump is usually an all-cache-hit run (the
> remote cache is shared with dev boxes), not evidence the tests fit their caps.

### Fixtures

Three kinds, all vendored, small, and compared with exact equality — field
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
- **Real-program stage-1 fixtures**
  (`zisk_zorch/commit/testdata/fullprogram/<air>/`) pin the assembled
  trace-commit pipeline against a real ZisK witness trace at proving-key shape
  (`fullprogram_commit_test` re-commits the trace and matches the native
  root). Regeneration needs the private `fractalyze/zisk` fork's
  `rw-fixture-gen` (branch `rw/zisk-v1.0.0-alpha`) plus the ziskup
  v1.0.0-alpha proving key; both steps are deterministic:

  ```sh
  # 1. Dump the native trace for the AIR (fixture + golden-preimage blob):
  cargo run --release -p rw-fixture-gen -- \
      --elf "$PWD/go-program-hello-world.elf" --input-data \
      --out /tmp/fullprogram --debug-trace

  # 2. Capture the stage-1 root the native prover commits for that dump.
  #    --hash-family Poseidon2: the family this repo models; the installed
  #    key's own default is Poseidon1, which zisk-zorch does not implement.
  cargo run --release -p rw-fixture-gen -- \
      --stage1-root /tmp/fullprogram/input_data/fullprogram/seg00 \
      --hash-family Poseidon2
  ```

  Larger AIRs follow the same recipe but stay uncommitted — drop the
  regenerated dir under `testdata/fullprogram/` locally and the test runs
  every fixture dir present.

## Per-stage baseline against native pil2

A wall-clock comparison against ZisK's native pil2-stark CUDA reference means
something only when both sides prove the **same instance**, at the **same
scope**, and produce the **same output**. Only the trace commit meets the
output half today (the `fullprogram/` fixtures match the native root for a
real trace); every other stage is pinned by `tools/fixture-gen`'s synthetic
inputs. The ratios below are engineering signal — do not quote one as
"zisk-zorch is Nx pil2" outside this page.

> **Retracted numbers — never re-quote:** "45 ms quotient" (a 1/55th-density
> proxy), "~270 ms quotient" (an extrapolation the proxy cannot reach),
> "77 ms / 0.58×" (CSE folded 900 constraints to 38), and any per-stage ms
> against the 24.6 s `GENERATING_INNER_PROOFS` (that phase covers **111 AIR
> instances**; this bench times one — a ~111× scope error).

### zisk-zorch side — `bench_inner_proof`

```sh
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  bazel run //zisk_zorch:bench_inner_proof -- \
    --stages=extend --n_bits=22 --n_cols=38 --blowup_bits=1 --phase runtime \
    -o report.json
```

zkbench owns warmup (3) + timed iterations (20) and reports warm `latency`,
`compile_time`, and a device-memory high-water mark.

- **`CUDA_VISIBLE_DEVICES=0` is load-bearing** — device enumeration wedges
  otherwise (#65).
- **A `RESOURCE_EXHAUSTED` at 2^23 is usually not the card**: frx caps the
  allocator at ~75% of the GPU; `XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` runs
  sizes that otherwise appear not to fit.
- **Do NOT `frx.profiler.trace` at 2^23** — it leaks host RAM and wedges CUDA
  (#65). Decompose by direct sub-op timing.
- **Run `--phase compile` and `--phase runtime` separately** — the memory peak
  is process-cumulative.
- **The defaults are not production-shaped**: `--n_cols` defaults to 64 (Main
  is 38 cm1 / 24 cm2) and `--arity` to 2 (production is 4). For the quotient,
  `--stages=quotient --chip=main` folds the chip's actual `eval_constraints`;
  the synthetic `--n_constraints` path is only as real as its products are
  distinct (`bench_inner_proof_test` pins the density).
- The report's `output_hash` is a self-consistency hash across zisk-zorch
  runs — *not* a pil2 byte-match.

### Native pil2 side

**Full inner proof** — the only recorded invocation (#30): `cargo-zisk 0.18.0
[gpu] --emulator --no-aggregation` on mainnet block 24654300 → ~47 s wall, 111
AIR instances, `GENERATING_INNER_PROOFS` = **24.6 s**. Version skew: that is
pil2-proofman v0.18.0; the goldens pin v1.0.0-alpha.

**Per-stage** — ⚠️ not reproducible *from this repo*: Google-Benchmark binaries
and hand-lifted CUDA kernels built ad-hoc under `/tmp/claude-1006/`
(`main_bench`, `fri_bench`, `evmap_bench`, `friexp_bench`, `gsum_bench`;
rebuilding them behind a committed script is open work, #60).
`bench_main_proof_gpu.cu` is the authority on what each row must match:
N=2^22, N_ext=2^23, cm1=38 / cm2=24, Merkle W=16 arity=4.

```sh
cd /tmp/claude-1006 && LD_LIBRARY_PATH=$PWD/gmp-prefix/lib CUDA_VISIBLE_DEVICES=0 \
  ./main_bench --benchmark_filter='MAIN_EXPR_PATTERN|MAIN_LDE_CM1|MAIN_LDE_CM2|MAIN_MERKLE' \
    --benchmark_min_time=5x --benchmark_repetitions=3 --benchmark_report_aggregates_only=true
./fri_bench     # no flags; arity 4 and [23,20,17,14,11,8,5] compiled in
./gsum_bench    # no flags; sweeps log2N, I=8
./evmap_bench; ./friexp_bench   # each prints 1-opening THEN 2-opening — the rows below use 1
```

### Per-stage comparison

RTX 5090, one AIR, N=2^22 → N_ext=2^23, both sides re-measured 2026-07-23
(zorch `dev20260722235316`, frx `dev20260723085209`; not yet re-measured on
frx 0.10.1). Each row brackets a different span (FRI excludes the query phase;
commit excludes its extend; `MAIN_EXPR` excludes the INTT-back and Merkle), so
rows do not sum.

| stage | native pil2 | zisk-zorch | ratio | pinned by |
|---|---|---|---|---|
| extend cm1 (38 col) | 32.5 ms | 23.6 ms | **0.72×** | golden (`lde`) |
| extend cm2 (24 col) | 20.4 ms | 14.8 ms | **0.73×** | golden (`lde`) |
| commit stage1 (38 col) | 37.5 ms | 39.0 ms | **1.04×** | real-trace root |
| commit stage2 (24 col) | 20.3 ms | 21.1 ms | **1.04×** | real-trace root |
| quotient ⚠️ | 134 ms (synthetic mimic) | 12.0 ms (real Main air) | — (#66) | goldens (`cexp_eval`) |
| LogUp grand-sum (I=8) | 2.45 ms | 3.56 ms | **1.45×** | golden (`gsum`) |
| evals (`evmap`) | 3.73 ms (M=68) | 7.8 ms | **2.1×** | LEv round-trip |
| DEEP composition | 8.91 ms (`friExp`, 62+6 col) | 15.3 ms | **1.72×** | low-degree test |
| FRI total (queries excl.) | 7.88 ms | 6.5 ms | **0.83×** | goldens (`fri_*`) |

How to read the table:

- **The quotient row has no ratio and cannot get one from these tools** (#66):
  pil2's `MAIN_EXPR_PATTERN` hardcodes a density ~370× the real Main air, and
  our 12.0 ms is the real air folded through the production
  `quotient_from_constraints` — a direct measurement at 2^23, register/
  bandwidth-resident, peak memory linear. Reproduce with
  `--stages=quotient --chip=main`. A true head-to-head needs pil2's per-air
  quotient timer under a real witness.
- **The evals ratio pairs like-for-like shapes**: native's 3.73 ms opens M=68
  columns, so its counterpart is the 62+6 run (wired M=39 measures 4.1 ms).
  The residual on evals/DEEP is the extension-typed reduce and AoS cubic
  arithmetic (upstream: zorch#512, xla#306, xla_fork#258).
- **The LogUp row is not in the prover's spine** — it pins the grand-sum
  primitive. Its native side was re-measured 2026-07-28 (`gsum_bench` at
  2^22: invfold 2.211 + scan 0.240 = 2.451 ms), confirming the figure the
  table had carried forward.
- The FRI fold byte-match requires the compile-time-constant field-divide fix
  first carried in frx `dev20260723085209` (present in 0.10.1); `fold_test` is
  green on GPU and CPU under it.

### End-to-end (`prove_inner`)

Whole-proof peaks through the real DEEP combiner, this RTX 5090
(`XLA_PYTHON_CLIENT_MEM_FRACTION` raised to fit), one fresh process per row;
"warm re-prove" is the second prove of the same size in that process — the
marginal per-proof cost in a compile-once, prove-many server, which is
production's shape (111 AIR instances per block):

| base | N_ext | full prove (cold) | warm re-prove | peak |
|---|---|---|---|---|
| 2^20 | 2^21 | 68.8 s | 28.4 s | 2.69 GiB |
| 2^21 | 2^22 | 70.7 s | 28.6 s | 5.38 GiB |
| 2^22 | 2^23 | 73.2 s | **30.6 s** | **10.75 GiB** |
| 2^23 | 2^24 | — | — | OOM (see below) |

- **The warm wall is host-bound, and that is the e2e story**: ~30 s and nearly
  flat from 2^20 to 2^22, while the per-stage GPU table sums to ~0.2 s at
  production size. The warm prove is ~99% eager per-op dispatch and Python
  sequencing; per-stage kernel ratios are second-order until the spine itself
  is batched/jitted or moved off Python.
- **The 2^23 ceiling is the stage-1 LDE forward NTT**, whose lowering holds
  two full ping-pong copies of the `(38, 2^24)` matrix as one 9.50 GiB
  scratch allocation. The traceback blames the query grind only because the
  transcript is device-resident, so the grind's `_canonical` is the first
  host sync in the chain. Splitting the LDE into column blocks (exact — the
  transform is per-column) is the known fix, currently parked.
- **No measured whole-proof ratio exists**: the only native total covers 111
  AIRs. The bracket from the native per-stage sum (~0.27 s) and the
  block-level average (~0.22 s) puts native at ~0.2–0.3 s per proof against
  our 30.6 s warm — a ~100× gap that is entirely the host-bound spine.
  Derived, not measured — do not quote as a benchmark row.

Proofs verify: `verifier.verify_inner` replays the transcript and checks
Merkle, DEEP, FRI, and the AIR constraint at the out-of-domain point
(`verifier_test` round-trips prove → verify, honest and tampered).

### Measure shipped code

A number is only a baseline if it runs what the team **ships**:

```sh
git fetch origin && git diff origin/main -- requirements.in requirements_lock_3_11.txt  # must be empty
pip show pyzorch frx frx-cuda12-pjrt | grep -E 'Name|Version'                           # venv == pins?
test ! -s .bazelrc.user || echo "LOCAL OVERRIDE ACTIVE — move it aside"
```

`zorch` is a pip wheel (`pyzorch`) pinned in `requirements.in`, not a Bazel
`git_override` (#55) — an `--override_module=zorch=` line is a no-op.
`.bazelrc.user` can point `zkx` / `prime_ir` at local checkouts. And perf work
often hot-swaps a locally built `pjrt_c_api_gpu_plugin.so` over the wheel's —
restore the `.orig` or you are not measuring the shipped plugin.

> Not hypothetical: #60 exists partly because sp1-zorch captured a baseline
> against a `zorch` override weeks behind `origin/main` and misread it as
> shipped.

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
