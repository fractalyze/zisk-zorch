# Project context for Claude Code

Everything load-bearing lives in repo docs. Treat those as the source of truth;
this file is just the map plus the rules every change must respect.

- **Project overview & quick start:** [`README.md`](README.md)
- **Architecture:** [`docs/architecture.md`](docs/architecture.md) — the inner proof over one transcript: commit, constraint ingestion, DEEP.
- **Development:** [`docs/development.md`](docs/development.md) — env setup, testing, and the per-stage baseline any perf number must trace to.
- **Coding conventions:** [`docs/conventions.md`](docs/conventions.md)

## Non-negotiables

- **ZisK-specific only.** This repo holds the ZisK / pil2-stark glue:
  Poseidon2-Goldilocks parameters, the pil2 transcript and linear-hash
  conventions, the trace-commit pipeline, and the byte-match against
  pil2-proofman. Anything scheme- or zkVM-agnostic belongs upstream in
  `zorch`, never here.
- **Byte-match is the contract.** Every primitive that mirrors pil2-stark
  must be pinned by a golden vector generated from the pil2-proofman
  reference (`tools/fixture-gen/`). A change that breaks a golden test is wrong until
  the reference says otherwise.
- **Pin external references.** Link pil2-proofman / ZisK sources as GitHub
  permalinks at tag `v1.0.0-alpha` (or a short commit SHA), never a branch.
