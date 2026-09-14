# Bench fixtures

All captures come from the go hello-world guest on an RTX 5090
(build-server-2, 2026-09-08, and the 2026-09-09 captures named below),
against the artifacts exported from proving key `v1.0.0-alpha`. They are
trimmed, not synthesised: every row and every timer block is what the tool
wrote. What was cut is named below, and nothing else was edited, so the
numbers the scripts print from these files are the numbers of that run.

- `nvtx_kern_sum.csv` — `nsys stats --report nvtx_kern_sum --format csv` over
  a `zz_prove <artifacts> <RomData case> --repeat 1` capture (two proves, so
  a per-prove figure is a total halved). Cut to seven of the artifact's
  programs — enough to carry one program that runs once per prove
  (`commit1`), one that runs once per family (`const_setup`) and one that
  runs once per quotient chunk (`quotient_524288`) — plus two of XLA's own
  `TSL` ranges, which `nvtx_programs.py` must leave out of the totals.

  Two rows in this file come from a **different capture** and are the one
  exception to the paragraph above: `host/prove` and `host/stage1`, PID
  576916, from a whole-run capture of 2026-09-09. Every other row is PID
  366825 from the 2026-09-08 run described above. They could not come from
  that run — it predates the `host/` ranges entirely — and they are here
  because `nvtx_programs.py` must exclude phases: a phase encloses the
  programs it runs and `nvtx_kern_sum` counts a kernel under every enclosing
  range, so counting them doubles every kernel. Read this file as one run's
  program rows plus two later phase rows, not as one capture.

- `pil2_prove.log` — `cargo-zisk-dev prove -vv` on the same guest through
  pil2's own GPU prover at one basic stream (`ZZ_GPU_HEADROOM_GB=15` on this
  card). Cut to the `TIMERS FOR INSTANCE` blocks of two airs, Main (`[0:0]`)
  and RomData (`[0:4]`), keeping every `>>> GEN_PROOF_<n>` marker because
  that is what tells a basic proof from the recursive proofs over it. Main's
  air carries four blocks — its commit, its basic proof, and the Recursive1
  and Recursive2 proofs above it — which is the case the discriminator is
  for.

- `pilout.globalInfo.json` — the `airs` section of the proving key's
  global info, the only place the air ids in a timer block are named.

- `host_cuda_gpu_trace.csv` / `host_nvtx_pushpop_trace.csv` — a few rows out
  of one `nsys` capture of a whole bridged `cargo-zisk prove` run
  (2026-09-09, artifacts `zz-artifacts-191`, `ZZ_CLIENTS=1`), which
  `host_idle.py` reads together. These pin the schema and nothing else:
  `host_idle_test.py` builds its attribution cases from a handful of ranges
  in the test, because pinning that logic against a real capture would mean
  carrying thousands of rows to make one assertion. So what is kept is one
  row per thing a reader has to tell apart — three of the bridge's kernels,
  two of pil2's, two copies and a memset for the kernel filter; a prove's
  phases on one thread, a `host/take/trace` from a proof worker on another,
  and one of XLA's `TSL` ranges for the domain filter. The leading `:` on
  every bridge range name is how `nsys` writes the default (unnamed) NVTX
  domain, and the `(ns)` in each time column is the unit the reader scales
  by.

- `host_cuda_api_trace.csv` — four rows of the same capture's
  `cuda_api_trace`, the optional third input. Three are the prove thread's
  own driver calls (a module load, a graph instantiation, a kernel launch)
  and the fourth is a module load on another thread, which must not be
  counted: only a holder's calls can explain the device being idle.

- `cuda_gpu_trace.csv` — `nsys stats --report cuda_gpu_trace --format csv`
  over a whole bridged `cargo-zisk prove` (one client), cut to the 0.22 s
  window around one prove's trace upload: the 1.31 GB pageable copy on the
  bridge's transfer stream, the bridge kernels either side of it, pil2's
  recursion kernels running in the same window on its own stream, and the
  smaller copies on both. Nothing else was cut, so the window carries the
  three cases `h2d_overlap.py` has to tell apart — a stream that carries a
  side's kernels (13 the bridge's, 66 pil2's) and a dedicated transfer
  stream that carries none (14) — and it carries the awkward part of the
  last one: some of stream 14's copies are followed by a pil2 kernel, so
  only the majority puts the stream on the right side.

- `xla_dump/` — the three after-optimizations reports XLA writes per
  executable under `--xla_dump_to`: `-buffer-assignment.txt`,
  `-live-range.txt` and `-memory-usage-report.txt`. Written by the shipped
  plugin (`0.10.2.dev20260910150749`, the pin in `requirements.in`) compiling
  a 512x512 `a @ a + (a * 2).sum()`, not a bridge program, and it is the one
  fixture here that does not come from the hello-world guest. That is
  deliberate: what `buffer_assignment.py` parses is the *plugin's* report
  format, identical for every program it compiles, and one small module
  carries every case the reader has to tell apart -- a parameter, a
  `maybe-live-out` output, nine `thread-local` allocations and a
  `preallocated-temp` arena that seven values share at overlapping offsets.
  A bridge AIR's own dump is `xla_dump_tree/` below. The only edit is the
  repo's own `trailing-whitespace` / `end-of-file-fixer` hook, which took two
  blank lines off the end of the usage report; every row is as written.

- `xla_dump_tree/commit2/` — the same three reports for a **real** bridge
  executable: `VirtualTableZisk0_n21`'s `commit2`, compiled from
  `zz-artifacts-191` by the shipped plugin on 2026-09-14 (`zz_prove --warm
  … --only commit2` into an empty `ZZ_COMPILE_CACHE`). Untrimmed, all
  45 allocations, so the allocations still sum to the `Total bytes` the usage
  report states and the reader's check against it stays live here.

  Laid out as `<program>/` under an AIR, which is how the dump is taken --
  one compile per directory, because XLA numbers modules per process -- so
  this fixture exercises the tree walk as well as the parse. It carries three
  cases the small one above cannot: a returned tuple and the 112-byte index
  table XLA allocates for its 14 elements (not a program output, and not
  declared in the manifest), 28 `constant` allocations placed at load rather
  than per execution, and a 1,600 MiB temp arena whose regions are an NTT's
  stage buffers plus one transpose of the input.
