# Stage-1 trace commit

The first slice of the ZisK prover: pil2-stark's `extendAndMerkelize`
(commitStage(1)) — commit the execution-trace columns before any challenge is
drawn. Everything here is byte-matched against pil2-proofman v0.15.0's
`fields` crate via [`../golden/`](../golden/).

## Pipeline

```
trace (N x n_cols, Goldilocks evals on the order-N subgroup)
  │  INTT per column                      zorch native NTT (lax.fft IFFT)
  ▼
coefficients
  │  coset-7 RS encode, blowup 2^(nBitsExt-nBits)   zorch ReedSolomon
  ▼
extended trace (N·blowup x n_cols, rows in pil2 domain order)
  │  pil2 linear hash per row             zisk_zorch.commit.linear_hash
  ▼
leaf digests (4 Goldilocks each)
  │  k-ary Poseidon2 fold (arity 2/3/4 → width 8/12/16)   zorch MerkleTree
  ▼
root (4 Goldilocks) ──▶ transcript.put(root)   zisk_zorch.transcript
```

## The conventions that make it byte-identical

- **Poseidon2 parameters** ([`../zisk_zorch/poseidon2/goldilocks.py`](../zisk_zorch/poseidon2/goldilocks.py)):
  pil2's external M4 is the HorizenLabs reference matrix, NOT the Plonky3 one
  zorch defaults to — every width passes its matrix explicitly (and therefore
  takes the generic fused-region route until zkx parameterizes its dedicated
  emitter's M4).
- **NTT domain order** ([`../zisk_zorch/commit/trace_commit.py`](../zisk_zorch/commit/trace_commit.py)):
  zk_dtypes' Goldilocks two-adic generator is Plonky3's; pil2's `W[32]`
  differs (`pil2 = zk^4168946053`). `extend` re-indexes rows into and out of
  the native NTT so the extended matrix lands in pil2's row order. A
  root-parameterized native NTT is the gather-free zkx follow-up.
- **Leaf hashing** ([`../zisk_zorch/commit/linear_hash.py`](../zisk_zorch/commit/linear_hash.py)):
  pil2's chained linear hash (zero-padded blocks, capacity chaining, <= 4
  shortcut) is NOT zorch's padding-free sponge, so it lives here and
  duck-types the leaf-hasher seam.
- **Tree** : one width-`4*arity` permutation hashes both leaves and nodes;
  incomplete levels complete with zero digests (zorch's k-ary MerkleTree
  stores the padded layers).
- **Transcript** ([`../zisk_zorch/transcript/transcript.py`](../zisk_zorch/transcript/transcript.py)):
  pil2's pending/out buffer discipline; 3-limb cubic challenges; 63-bit query
  index packing.

## What the next slices need

- **Openings / query phase**: `MerkleTreeGL::getGroupProof` layout over the
  k-ary digest layers; zorch's k-ary `open`/`reconstruct_root` exist, the
  pil2 proof serialization does not yet.
- **FRI fold**: needs the Goldilocks cubic extension x³ − x − 1 —
  zk_dtypes' `goldilocksx3` is u³ − 7, so a new dtype (or parameterized
  modulus) is a zk_dtypes prerequisite.
- **Stage 2 / Q / evals**: constraint evaluation on the extended domain,
  grand-sum witnesses, DEEP quotient — after the query phase closes the
  commit loop.
