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
