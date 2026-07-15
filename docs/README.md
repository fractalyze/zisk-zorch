# Documentation

Index of the zisk-zorch docs. Jump here from [`../README.md`](../README.md)
(product intro) or [`../CLAUDE.md`](../CLAUDE.md) (the agent / contributor
pointer).

| Doc | What's in it |
| --- | --- |
| [conventions.md](conventions.md) | Comment scoping (why-not-what), how pil2-proofman / ZisK references are pinned, and the golden-test rules. |
| [stage1-commit.md](stage1-commit.md) | The stage-1 trace commit (pil2's `extendAndMerkelize`) as a pipeline, and the conventions that make it byte-identical. |
| [stage2-constraint-ingest.md](stage2-constraint-ingest.md) | The `rw_constraints` ingestion seam: how ZisK chip constraints and bus interactions reach stage-2. |
| [inner-proof-wiring.md](inner-proof-wiring.md) | `prove_inner`: the stages run in pil2's genProof order over one Fiat-Shamir transcript, plus the DEEP stage that fills the FRI-polynomial seam. |
| [testing.md](testing.md) | Running tests, `size` vs `timeout` conventions, and the golden / proving-key fixtures. |
| [zisk-baseline.md](zisk-baseline.md) | The per-stage zisk-zorch vs native pil2 benchmark protocol: same-data / same-scope / same-output, and what each quoted number is really worth. |
