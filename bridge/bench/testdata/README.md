# Bench fixtures

Both captures come from the go hello-world guest on an RTX 5090
(build-server-2, 2026-09-08), against the artifacts exported from proving key
`v1.0.0-alpha`. They are trimmed, not synthesised: every row and every timer
block is what the tool wrote. What was cut is named below, and nothing else
was edited, so the numbers the scripts print from these files are the numbers
of that run.

- `nvtx_kern_sum.csv` — `nsys stats --report nvtx_kern_sum --format csv` over
  a `zz_prove <artifacts> <RomData case> --repeat 1` capture (two proves, so
  a per-prove figure is a total halved). Cut to seven of the artifact's
  programs — enough to carry one program that runs once per prove
  (`commit1`), one that runs once per family (`const_setup`) and one that
  runs once per quotient chunk (`quotient_524288`) — plus two of XLA's own
  `TSL` ranges, which `nvtx_programs.py` must leave out of the totals.

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
