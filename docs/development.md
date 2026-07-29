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
- **The inner-proof byte-gate** (fractalyze/zisk-zorch#59),
  [`../zisk_zorch/verify_inner_proof.py`](../zisk_zorch/verify_inner_proof.py),
  replays every assembled stage — stage-1 commit, stage-2 witness and commit,
  quotient, evals, DEEP, the FRI fold chain — against a dump of a **real
  pil2-proofman `genProof`**, the ZisK analog of SP1's `SP1_DUMP_PHASES`. Field
  arithmetic is exact, so each gate is equal-or-wrong.

  No capture is committed and the gate is a runnable, never a test —
  sp1-zorch's pattern for anything needing a host-provided native artifact
  (its `verify_prove_shard` + `SP1_JAX_FFI_LIB`): CI covers only hermetic
  tests, and operators run the gate on provisioned hosts. Argless runs read
  the bundle directory named by `ZISK_PIL2_CAPTURE` — a real
  fibonacci-square prove regenerated once per machine by
  [`../tools/pil2-dump/`](../tools/pil2-dump/) — and skip loudly when it is
  unset; `--dump` points at any other capture.

  ```sh
  ZISK_PIL2_CAPTURE=<bundle dir> bazel run //zisk_zorch:verify_inner_proof
  bazel run //zisk_zorch:verify_inner_proof -- \
      --dump=<dir> --instance=ag0_air0_inst0 --starkinfo=<starkinfo.json>
  ```

## Per-stage baseline against native pil2

A wall-clock comparison against ZisK's native pil2-stark CUDA reference means
something only when both sides prove the **same instance**, at the **same
scope**, and produce the **same output**. The fibonacci-square instance below
is the one basis meeting all three for every stage: both sides prove the same
real witness and every timed stage byte-matches the native dump. Quote
ratios only from that table; the Main-shape microbenches further down are
engineering signal for per-stage issues, not baselines.

> **Retracted numbers — never re-quote:** "45 ms quotient" (a 1/55th-density
> proxy), "~270 ms quotient" (an extrapolation the proxy cannot reach),
> "77 ms / 0.58×" (CSE folded 900 constraints to 38), and any per-stage ms
> against the 24.6 s `GENERATING_INNER_PROOFS` (that phase covers **111 AIR
> instances**; this bench times one — a ~111× scope error).

### Main-shape microbenches — zisk-zorch side (`bench_inner_proof`)

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

### Main-shape microbenches — native pil2 side

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
./evmap_bench; ./friexp_bench   # each prints 1-opening THEN 2-opening; use 1
```

For per-stage timings of a real native prove, no ad-hoc binary is needed:
pil2's CUDA-event stage timers (`LOG_TIME_GPU` is compiled in) print per
instance at `-vv` — the same-instance table below is built from them.

### Per-stage comparison — same instance, proof order

fibonacci-square FibonacciSquare @2^22 → 2^23, RTX 5090, measured
2026-07-29. Rows are the proof's stages in transcript order and each side's
rows sum to its total. Native numbers are pil2's own CUDA-event stage
timers (`TimerGPU`, printed at `-vv`, medians of 3 runs, cv < 1%; the
0.4 ms `STARK_STEP_0` transcript setup is the only part not in a row); ours
are `bench_prove_e2e --backend=device` warm medians, every row byte-gated
against the native dump in the same run.

| stage (proof order) | native pil2 | zisk-zorch | native timer |
|---|---|---|---|
| stage-1 commit (extend + merkle) | 15.0 ms | 14.9 ms | `STARK_COMMIT_STAGE_1` |
| stage-2 witness (STD hints → im, gsum, ImPol) | 1.6 ms | 0.8 ms | `WITNESS_STD` + `IM_POLS` |
| stage-2 commit | 18.0 ms | 17.6 ms | `STARK_COMMIT_STAGE_2` |
| quotient (cExp eval + commit Q) | 22.3 ms | 16.2 ms | `STARK_STEP_Q` |
| evals (LEv + openings) | 11.4 ms | 5.8 ms | `STARK_STEP_EVALS` |
| DEEP + FRI + queries | 16.2 ms | 14.4 ms | `STARK_STEP_FRI` |
| **total** | **85.0 ms** (device events) | **69.6 ms** (stage sum) | `STARK_GPU_PROOF` |

How to read the table:

- **The 49 ms `GEN_PROOF` wall and the 85 ms device total are both real.**
  The Rust-side `GEN_PROOF` bracket (49 ms) returns while ~36 ms of the
  instance's kernels are still draining on the stream — the tail overlaps
  the next instance's host work, which is how the six-instance program
  proves in 166 ms. The event-timed 85 ms is what the stages actually cost
  on the device; it is the apples-to-apples side for our synchronized
  stage sum. Against the pipelined wall we are ~1.4×; against device time
  **~0.8× — ahead of native**. The rows were measured with the interim
  xla#340 rotation workaround in place; with xla#341 letting rotations
  fuse back, the harness as committed measures **67.4 ms** (stage-2
  witness 0.71) — refresh the rows when the first post-#341 wheel lands.
- **Native's total carries ~13 ms of `H2D_COPY`** (witness upload lands in
  its commit rows); our bracket starts from a device-resident witness.
  Netting it out puts native at ~72 ms — parity with our 69.6.
- Where the stages differ: the two commits and stage-1 are hash-bound and
  identical (both sides sit at the same ~1.3 G perms/s Poseidon2 floor);
  our wins are the quotient (their `STARK_QUOTIENT_POLYNOMIAL` expression
  pass alone is 6.3 ms vs our whole interpreter at 4.6 ms), evals (11.4 →
  5.8), and the stage-2 hint chain (1.6 → 0.8). Their DEEP polynomial
  (`STARK_FRI_POLYNOMIAL`, 6.3 ms) ≈ ours (5.9 ms).
- The old Main-shape rows this table replaced (trace commit 0.89×, LogUp
  grand-sum, evmap, friExp, fri_bench) were cross-harness comparisons at
  mixed shapes — the LogUp row in particular benched the grand-sum
  *primitive* at I=8, not a proof stage. Their history lives in #58–#70;
  re-measure with the Main-shape microbenches below when a per-stage issue
  needs them, at production shapes only.

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

- **The 30 s host-bound warm wall is gone**: 30.6 s → **0.7 s** at 2^22. The
  collapse is the accumulated spine work — vmapped query openings (#88),
  shared jit cache keys (#92), frx 0.10.1 — not the DEEP jit zone alone: an
  eager (`jit=False`) warm prove also lands at 0.7 s, so the zone's warm
  delta is noise. The remainder is still flat across sizes (dispatch, not
  kernels — the per-stage table sums to ~0.2 s), just 40× smaller.
- **Cold is compile time**: ~88–89 s and flat from 2^20 to 2^22 (up from
  ~70 s as more of the spine jits), so the cold column prices the
  compile-once half of the server shape, not the proving.
- **The 2^23 ceiling is the stage-1 LDE forward NTT**, whose lowering holds
  two full ping-pong copies of the `(38, 2^24)` matrix as one 9.50 GiB
  scratch allocation. The traceback blames the query grind only because the
  transcript is device-resident, so the grind's `_canonical` is the first
  host sync in the chain. Splitting the LDE into column blocks (exact — the
  transform is per-column) is the known fix, currently parked.
- **No measured whole-proof ratio exists at ZisK shape**: the only native
  total covers 111 AIRs. The bracket from the native per-stage sum (~0.27 s)
  and the block-level average (~0.22 s) puts native at ~0.2–0.3 s per proof
  against our 0.7 s warm — a ~2–3× gap. Derived, not measured — do not quote
  as a benchmark row. The measured whole-proof ratio that does exist is on
  the fibonacci-square instance below.

Proofs verify: `verifier.InnerVerifier` replays the transcript and checks
Merkle, DEEP, FRI, and the AIR constraint at the out-of-domain point
(`verifier_test` round-trips prove → verify, honest and tampered). An
accepting verify at 2^22 with 64 queries measures 825.7 s first / 765.1 s
repeated — the verifier's per-query host loops are now the minutes-scale
wall, and the next per-stage issue candidate.

### End-to-end vs native pil2 — same instance (fibonacci-square)

The one whole-proof pairing that meets this page's bar — same instance, same
scope, same output — is pil2-proofman's own `fibonacci-square` example:
FibonacciSquare at N=2^22 → N_ext=2^23 (5 cm1 + 9 cm2 columns, arity 4, FRI
schedule `[23,20,17,14,11,8,5]`, 229 queries, 16 grinding bits). It is the
instance `verify_inner_proof` byte-gates 16/16 against a real `genProof`
dump, so every timed stage below produced byte-identical output first.

**Bracket** — runtime only, on both sides. Native: the per-instance CUDA
event timers (settled witness in, proof out; setup load, const-tree
regeneration, and witness computation excluded), with the host-side
`GEN_PROOF` wall shown alongside. Ours:
`zisk_zorch/bench_prove_e2e.py`, the timing twin of `verify_inner_proof` —
production stage functions in proof order on the same dump, each jitted
once, one compiling warmup, median of the warm repetitions. Neither side
includes compile time; a stage timing is reported only when its byte-gate
passed in the same run.

RTX 5090 / 16-core host, measured 2026-07-29 (native at v1.0.0-alpha plus
the non-perturbing `dump/per-stage-genproof` hooks, disarmed):

| side | GPU | CPU |
|---|---|---|
| native pil2, device time (`STARK_GPU_PROOF`) | **85.0 ms** | — |
| native pil2, pipelined wall (`GEN_PROOF`) | 49 ms | 2685 ms |
| zisk-zorch, all-device (stage sum, warm) | **67.4 ms** | 70.5 s |

The per-stage split of both GPU columns is the same-instance table above:
against native's device time we are **~0.8×**; the 49 ms wall is native's
host bracket whose async kernel tail overlaps the next instance, so
matching it is a pipelining property (prove-many overlap), not a per-proof
compute gap — our composed wall (no per-stage syncs) is ~73 ms and would
overlap the same way across instances.

The device row needs a frx newer than `0.10.1.dev20260729002119`: xla#335
(field-mul outlining), xla#327 (stride-aware unroll gating; run with
`XLA_FLAGS=--xla_gpu_experimental_max_unroll_factor=1`), and xla#341 (the
fused-rotation wrap-row miscompile, #340 — on wheels without it the
stage-2 and quotient byte-gates fail). On the shipped 0.10.1 pins the same
pipeline runs GPU-hybrid — quotient and stage-2 CPU-pinned per xla#334 —
at 1373 ms, which is the shipped-pins number until the next frx bump.

What the campaign from 1373 ms pulled, in order of yield:

- **xla#335 unpin** (−812 ms): both interpreter stages onto the device —
  stage-2 witness 242→0.8, quotient 570→4.6.
- **Device-paced query phase** (−347 ms): the production phase is
  host-paced (`_GRIND_CHUNK = 256` → ~256 round-trips for pow 16, Python
  bit-unpack in `get_permutations`, per-call re-`vmap` of every tree
  opening). Same math jitted once, 2^17-nonce grind batches, vectorized
  unpack — 373→2.0 ms, byte-checked against `sample_query_positions`.
- **`--xla_gpu_experimental_max_unroll_factor=1`** (−13 ms): DEEP
  17.7→6.0. The #327 structural gate catches slice-strided fusions but not
  the goldilocksx3 AoS limb interleave this fold reads (the known gl
  residual); the flag closes it.
- **Montgomery batch inversion in DEEP** (−6 ms): one Fermat chain for all
  five opening-group denominators instead of five, 24.1→17.7 pre-flag.
- **Fused evals** (−5 ms): `compute_lev` traced into the sums jit via its
  `LevConstants` prologue, 11.1→6.0.

**The floor, and the one lever left.** The four Merkle trees (three commits
+ FRI's layers) hash ~49M Poseidon2-width-12 permutations ≈ 39 ms of the
69.6 — and that is *pil2's own* floor: native's commit rows match ours to
within 0.5 ms, and its merkle benches at identical shape (11.4 ms at 5
cols / 36.4 ms at 38 cols, 2^23) put its CUDA at the same ~1.2–1.3 G
perms/s as frx's Poseidon2Fusion emitter. Pushing the total further down
therefore needs the permutation itself faster — an emitter/occupancy
project (utilization is single-digit % of the card's int throughput, so
headroom exists in principle for both sides), not a model-level change.
Every other stage is at or below native's counterpart and within ~2 ms of
its measured device floor.

Reading the CPU column: pil2's CPU prover is AVX2 + OpenMP across 16 cores;
70.5 s of ours is 59.6 s of Poseidon2 Merkle commits on the frx CPU backend.
The CPU path is a correctness lane, not a performance target.

Reproduce (native — the GPU build is `make starks_lib_gpu` at sm_120 plus a
cargo build *without* `proofman-starks-lib-c/cpu-only`):

```sh
# per-instance GEN_PROOF timers are debug-level (-vv); -t 1 serializes the
# streams so per-instance walls don't overlap; -g selects the GPU
./target/release/proofman-cli prove -g -t 1 -vv \
  --witness-lib ./target/release/libfibonacci_square.so \
  --proving-key examples/fibonacci-square/build2/provingKey/ \
  --public-inputs examples/fibonacci-square/src/inputs.json \
  --custom-commits rom=examples/fibonacci-square/build2/rom.bin \
  --output-dir examples/fibonacci-square/build2/proofs -y
```

```sh
# ours: --backend=device on a post-#335 frx; --backend=hybrid on the
# shipped 0.10.1 pins; FRX_PLATFORMS=cpu + --backend=cpu for the CPU row
CUDA_VISIBLE_DEVICES=0 XLA_PYTHON_CLIENT_PREALLOCATE=false \
  XLA_FLAGS=--xla_gpu_experimental_max_unroll_factor=1 \
  python -m zisk_zorch.bench_prove_e2e --dump=<PIL2_DUMP_DIR capture> \
    --instance=ag0_air0_inst0 --starkinfo=<...FibonacciSquare.starkinfo.json> \
    --backend=device --reps=5
```

Caveats that keep this row honest:

- **FibonacciSquare is not a ZisK AIR** — 5+9 columns vs Main's 38+24, so
  this measures the *pipeline* (per-proof overheads, stage seams, hashing),
  not ZisK-shape arithmetic throughput. The ZisK-shape counterpart stays
  open until a ziskup-key native per-instance timing exists.
- **Ours is a stage sum, not one integrated prove**: challenges are replayed
  from the dump (the wired `InnerProver` has no stage-2 slot yet), and
  transcript squeezes / proof serialization are excluded — both µs-scale
  next to the stages.
- **The rotation gates double as a compiler canary**: the fused-rotation
  wrap-row miscompile this benchmark surfaced (fractalyze/xla#340, fixed by
  xla#341) corrupted exactly one limb of one row — only the byte-gates
  caught it. A gate failure isolated to stage-2 witness + quotient on a new
  wheel is the signature of that bug class reappearing.
- **Native GPU quirk**: with `-g`, every instance verifies ✓ individually
  yet the run exits 1 with "Basic proofs were not verified" — a
  verified-flag bookkeeping issue in the GPU path, not a proof failure.
  Timings and per-proof verification are unaffected.
- All six fibsq instances, for scale: native GPU proves the whole program in
  166 ms (9 concurrent streams; 245 ms serialized), native CPU in 5.4 s.

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
