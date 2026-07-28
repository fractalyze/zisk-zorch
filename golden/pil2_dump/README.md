# pil2 reference-dump harnesses

The `verify_*` runnables (e.g. [`zisk_zorch/commit/verify_trace_commit.py`](../../zisk_zorch/commit/verify_trace_commit.py))
byte-match an assembled zisk-zorch stage against a **real pil2-proofman
reference dump** — the assembled-path counterpart to the `../` Rust golden
vectors, which pin each primitive in isolation on tiny synthetic inputs.

These harnesses capture that dump from pil2-stark's **real Goldilocks CUDA
kernels** — the code ZisK actually runs on GPU — rather than reconstructing it
from the `fields` crate. They are the ZisK analog of SP1's `SP1_DUMP_PHASES`
phase dump.

## `pil2_dump_stage1` — stage-1 trace commit (`extendAndMerkelize`)

Runs the production stage-1 commit path on a **deterministic** trace:

```
splitmix64 trace (row-major)  ->  fromRowMajorToTiled  ->  NTT LDE (coset)
  ->  Poseidon2 k-ary Merkle (linear-hash leaves)  ->  root (4 Goldilocks)
```

The trace is drawn with the exact splitmix64/`rand_fe` stream the Rust golden
generator ([`../src/main.rs`](../src/main.rs)) uses, so the harness needs only a
`(dims, arity, seed)` to reproduce a trace — the dump stores the trace bytes so
the consumer needs no RNG.

### Build

Needs a pil2-proofman **v1.0.0-alpha** checkout, CUDA `nvcc`, gmp, and a GPU
(the tiled NTT requires `n_bits >= 8`). Point `PIL2_STARK` at the checkout's
`pil2-stark/`:

```sh
PIL2_STARK=/path/to/pil2-proofman/pil2-stark \
  CUDA_ARCH=sm_120 GMP_PREFIX=/usr \
  ./build.sh
```

### Capture a dump

```sh
# small, production-shaped (arity 4) — this is the committed fixture:
./pil2_dump_stage1 --n_bits=8 --blowup_bits=1 --n_cols=8 --arity=4 --seed=0x515A31 \
    --dump=../../zisk_zorch/commit/testdata/dump/stage1

# real Main-air dimensions (N=2^22 -> N_ext=2^23, 38 cols, arity 4):
./pil2_dump_stage1 --n_bits=22 --blowup_bits=1 --n_cols=38 --arity=4 --seed=0x1 \
    --dump=/data/zisk_dumps/main_stage1
```

Each `--dump=<dir>` writes:

- `trace.bin`  — raw little-endian u64, row-major `N x n_cols` (the trace pil2 committed);
- `commit.json` — `{n_bits, blowup_bits, n_cols, arity, seed, root[4]}`.

Then gate zisk-zorch against it:

```sh
bazel run //zisk_zorch/commit:verify_trace_commit -- --dump=<dir>
```

`--arity` selects the Poseidon2 node width (2→8, 3→12, 4→16); production
basic-air Merkle trees use arity 4. Omitting `--dump` prints the root only.

### Why the root is the whole gate

The root is a Poseidon2 image of every extended row, so a single 4-element match
proves the coset LDE order, the linear-hash leaves, and the k-ary fold all
reproduced pil2 byte-for-byte — the same "one scalar seals the stage" discipline
the downstream stages will follow. (Verified: this harness's root byte-matches
`commit_trace` across arities 2/3/4 and several sizes, and — at a golden's
dims+seed — the committed `stage1_commit.json` root.)
