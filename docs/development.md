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

**A venv from `requirements.in` has no GPU.** The `cuda` backend registers
only when the `nvidia-*` runtime wheels the plugin dlopens are present, and
the lock does not carry them — a bare `frx-cuda12-plugin` install leaves the
backend unregistered and work silently runs on the CPU. Install the extra:

```sh
pip install 'frx-cuda12-plugin[with-cuda]==<pinned version>'
```

(pip skips extras when the base requirement is already satisfied, so install
the extra form first — or force it after the fact.) On the pinned frx the
sm_120 card runs the full path: backend init, the LDE `lax.ntt`, the DEEP jit
zone, and a prove → verify round trip. Assert the device before trusting any
GPU number:

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
  (`fullprogram_commit_test` re-commits each fixture dir's trace with the
  `hash_family` recorded beside it and matches the native root). The same trace
  appears under two dirs, one per family — `input_data_seg00` (Poseidon2) and
  `input_data_seg00_poseidon1` — because the dump is family-independent and only
  the root differs. Poseidon1 is what native ZisK commits with by default (the
  installed key's globalInfo carries no `hash`, so pil2's `DEFAULT_HASH_ID`
  wins); Poseidon2 needs an explicit override. Regeneration needs the private
  `fractalyze/zisk` fork's `rw-fixture-gen` (branch `rw/zisk-v1.0.0-alpha`) plus
  the ziskup v1.0.0-alpha proving key; both steps are deterministic:

  ```sh
  # 1. Dump the native trace for the AIR (fixture + golden-preimage blob):
  cargo run --release -p rw-fixture-gen -- \
      --elf "$PWD/go-program-hello-world.elf" --input-data \
      --out /tmp/fullprogram --debug-trace

  # 2a. Root at the installed key's default family (Poseidon1 — no override):
  cargo run --release -p rw-fixture-gen -- \
      --stage1-root /tmp/fullprogram/input_data/fullprogram/seg00

  # 2b. Root at Poseidon2 (explicit override via a shadow proving key):
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
output half today (the `fullprogram/` fixtures match the native root for a real
trace, in both hash families); every other stage is pinned by
`tools/fixture-gen`'s synthetic inputs. The ratios below are engineering signal
— do not quote one as "zisk-zorch is Nx pil2" outside this page.

> **Only numbers on this page are quotable.** Figures from issues and PR
> threads have been retracted for density-proxy, extrapolation, and scope
> errors — most often per-stage ms held against the 24.6 s
> `GENERATING_INNER_PROOFS`, which covers **111 AIR instances** to this bench's
> one.

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

RTX 5090, one AIR, N=2^22 → N_ext=2^23, both sides re-measured on the pins
current when each row was taken (`git log docs/development.md` for a row's
provenance). Each row brackets a different span — FRI excludes the query phase,
`MAIN_EXPR` excludes the INTT-back and Merkle — so rows do not sum.

| stage | native pil2 | zisk-zorch | ratio | pinned by |
|---|---|---|---|---|
| trace commit (extend+merkle, 38+24 col) | 110.7 ms | 98.5 ms | **0.89×** | golden (`lde`) + real-trace root |
| quotient ⚠️ | 133 ms (synthetic mimic) | 12.1 ms (real Main air) | — (#66) | goldens (`cexp_eval`) |
| LogUp grand-sum (I=8) | 2.45 ms | 2.53 ms | **1.03×** | golden (`gsum`) |
| evals (`evmap`) | 3.74 ms (M=68) | 3.14 ms | **0.84×** | LEv round-trip |
| DEEP composition | 8.88 ms (`friExp`, 62+6 col) | 15.5 ms | **1.75×** | low-degree test |
| FRI total (queries excl.) | 7.88 ms | 6.5 ms | **0.83×** | goldens (`fri_*`) |

How to read the table:

- **The quotient row has no ratio and cannot get one from these tools** (#66):
  pil2's `MAIN_EXPR_PATTERN` hardcodes a density ~370× the real Main air, while
  ours is the real air through the production `quotient_from_constraints`
  (`--stages=quotient --chip=main`). A head-to-head needs pil2's per-air
  quotient timer under a real witness.
- **The evals ratio pairs like-for-like shapes**: native's 3.74 ms opens M=68
  columns, so its counterpart is the 62+6 run; the wired 38+1 shape measures
  1.58 ms.
- **The LogUp row is not in the prover's spine** — it pins the grand-sum
  primitive. `grand_sum` takes its inputs interaction-major, and that is
  load-bearing at 1.4× (see its docstring).
- **DEEP is the one row above native**, and its residual is neither bandwidth
  nor arithmetic: the same columns read whole stream at 1442 GB/s against the
  kernel's 366, and removing the divide changes nothing. What is left is
  register pressure in one 68-way unrolled kernel — fractalyze/zorch#541 splits
  it, and the column-major input it wants is a cross-stage layout call (#69).
- The FRI fold byte-match requires the compile-time-constant field-divide fix
  first carried in frx `dev20260723085209` (present in 0.10.1).

### End-to-end (`InnerProver`)

Whole-proof peaks through the real DEEP combiner, this RTX 5090
(`XLA_PYTHON_CLIENT_MEM_FRACTION=0.95` to fit), one fresh process per row,
re-measured 2026-07-28 on the merged pins (pyzorch `dev20260727041631`,
frx 0.10.1); "warm re-prove" is the second prove of the same size in that
process — the marginal per-proof cost in a compile-once, prove-many server,
which is production's shape (111 AIR instances per block):

| base | N_ext | full prove (cold) | warm re-prove | peak |
|---|---|---|---|---|
| 2^20 | 2^21 | 89.0 s | 0.6 s | 3.03 GiB |
| 2^21 | 2^22 | 89.0 s | 0.5 s | 5.38 GiB |
| 2^22 | 2^23 | 87.8 s | **0.7 s** | **10.75 GiB** |
| 2^23 | 2^24 | — | — | OOM (see below) |

- **Warm re-prove is dispatch-bound, not kernel-bound**: it is flat across
  sizes while the per-stage table sums to ~0.2 s. Running the spine eager
  (`jit=False`) lands at the same 0.7 s, so the DEEP jit zone is not what
  holds it.
- **Cold is compile time**: ~88–89 s and flat from 2^20 to 2^22 (up from
  ~70 s as more of the spine jits), so the cold column prices the
  compile-once half of the server shape, not the proving.
- **The 2^23 ceiling is the stage-1 LDE forward NTT**, whose lowering holds
  two full ping-pong copies of the `(38, 2^24)` matrix as one 9.50 GiB
  scratch allocation. The traceback blames the query grind only because the
  transcript is device-resident, so the grind's `_canonical` is the first
  host sync in the chain. Splitting the LDE into column blocks (exact — the
  transform is per-column) is the known fix, currently parked.
- **No measured whole-proof ratio exists**: the only native total covers 111
  AIRs. The bracket from the native per-stage sum (~0.27 s) and the
  block-level average (~0.22 s) puts native at ~0.2–0.3 s per proof against
  our 0.7 s warm — a ~2–3× gap. Derived, not measured — do not quote as a
  benchmark row.

Proofs verify: `verifier.InnerVerifier` replays the transcript and checks
Merkle, DEEP, FRI, and the AIR constraint at the out-of-domain point
(`verifier_test` round-trips prove → verify, honest and tampered). An accepting
verify at 2^22 with 64 queries takes ~765 s — the verifier's per-query host
loops are the minutes-scale wall, and the next per-stage issue candidate.

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
