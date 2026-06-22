"""Warm-GPU microbench for the inner-proof prover stages, on zkbench.

The baseline-vs-port comparison this repo exists for: the three legs of pil2's
`GENERATING_INNER_PROOFS` (24.6s native on RTX 5090, block 24654300), each timed
on the ZKX GPU plugin so the zkx-compiled path can be lined up against native
ZisK. One `zkbench.JaxBenchmark` yields an op per (stage, size), so a single run
emits one report keyed `stage_2p<n_bits>` covering the whole inner proof:

  extend   — stage-1 coset LDE (INTT+NTT), pil2's `extendPol`
  commit   — stage-1 linear-hash leaves + k-ary Poseidon2 Merkle, `merkelize`
  full     — stage-1 extend ∘ commit, the whole `extendAndMerkelize`
  quotient — stage-2 constraint eval + alpha-fold + zerofier divide
  divide   — stage-2 bare `compute_quotient` (composite * inv-zerofier), the
             pointwise tail of `quotient` isolated for reference
  fri      — FRI fold/commit chain (`prove`), the query phase excluded

zkbench owns warmup, timed iterations, device-memory peak, statistics, and
test-vector hashing. The jitted single-function stages also pass a `lower` thunk,
so zkbench times `lowered.compile()` as `compile_time` (the compile wall the
Poseidon2-fusion fix in zorch #264 / zkx #676 cut from ~540s to ~5s). `fri`'s
`prove` is a Python driver over jitted islands, not one jitted function, so it
reports warm latency only (no `--phase compile`).

`quotient`'s `eval_fn` is a parametric proxy (`--n_constraints` degree-`--degree`
column products over `--n_cols`) tunable to a target AIR — absolute is a proxy,
size scaling is real. Inputs are Montgomery-form `goldilocks_mont` per the
benchmark-field-standardization convention.

Run (on a GPU host, ZKX plugin resolved via the venv):

    PYTHONPATH=<zisk-zorch>:<zorch> CUDA_VISIBLE_DEVICES=<free> \\
        python -m zisk_zorch.bench_inner_proof \\
        --n_bits=20 --n_bits=21 --n_bits=22 --n_cols=64 --arity=2 -o report.json
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Iterable

import jax
import jax.numpy as jnp
import numpy as np
from zk_dtypes import goldilocks_mont as F
from zk_dtypes import goldilocksx3_mont as F3
from zkbench import BenchmarkConfig, BenchmarkOp, JaxBenchmark, compute_hash

from zisk_zorch.commit.trace_commit import extend, merkle_tree
from zisk_zorch.fri.prover import prove
from zisk_zorch.poseidon2.goldilocks import goldilocks_perm
from zisk_zorch.quotient.quotient import compute_quotient, quotient_from_constraints
from zisk_zorch.transcript.transcript import Transcript
from zorch.testkit.random_field import rand_field

_STAGES = ("extend", "commit", "full", "quotient", "divide", "fri")


def _rand_cubic(length: int, seed: int) -> jax.Array:
    """Canonical Goldilocks-cubic evals; 3 base limbs view as one cubic element.

    Built via numpy then `jnp.asarray`, NOT zorch's `rand_ext_field`: that bitcasts
    a jax array (`jnp_array.view(F3)`), which aborts on the zkx jax fork
    (`Check failed: IsArray()`); the numpy `.view` path is fork-safe.
    """
    ints = np.random.default_rng(seed).integers(0, 1 << 30, (length, 3), np.int64)
    return jnp.asarray(ints.astype(F).view(F3).reshape(length))


def _make_eval_fn(n_cols: int, n_constraints: int, degree: int) -> Callable:
    """`n_constraints` constraints in the trailing axis, each a degree-`degree`
    product of distinct columns — a field-mul proxy for an AIR's constraint
    expression."""
    if min(n_cols, n_constraints, degree) < 1:
        raise ValueError(
            f"n_cols/n_constraints/degree must be >= 1, got "
            f"{n_cols}/{n_constraints}/{degree}"
        )

    def eval_fn(t: jax.Array) -> jax.Array:
        cols = []
        for k in range(n_constraints):
            c = t[:, k % n_cols]
            for d in range(1, degree):
                c = c * t[:, (k * degree + d) % n_cols]
            cols.append(c)
        return jnp.stack(cols, axis=-1)  # (rows, n_constraints)

    return eval_fn


def _fold_steps(n_bits_ext: int, fold_bits: int, final_bits: int) -> list[int]:
    """Strictly-decreasing FRI layer schedule `nBitsExt -> ... -> final_bits`,
    folding by `fold_bits` per layer (the tail folds the remainder)."""
    steps = list(range(n_bits_ext, final_bits, -fold_bits))
    if steps[-1] != final_bits:
        steps.append(final_bits)
    return steps


def _first_array(result: object) -> jax.Array:
    """The first device array in a result (an op returns an array, a tuple, or a
    pytree commitment); hashing one representative array pins reproducibility."""
    leaves = [leaf for leaf in jax.tree_util.tree_leaves(result) if hasattr(leaf, "shape")]
    if not leaves:
        raise TypeError(f"no array leaf to hash in {type(result)}")
    return leaves[0]


def _hash_array(arr: jax.Array) -> str:
    """Hash a field array's raw bytes. zkbench's `compute_array_hash` casts to
    `<u4` (31-bit fields only) and can't represent 64-bit Goldilocks; the raw
    Montgomery limbs are deterministic, which is all an input/output hash needs."""
    return compute_hash(np.asarray(arr).tobytes())


class InnerProofBenchmark(JaxBenchmark):
    """One op per (stage, n_bits) across the inner proof, for `zkbench`."""

    def get_config(self) -> BenchmarkConfig:
        return BenchmarkConfig(
            implementation="zisk-zorch",
            version="0.1.0",
            default_iterations=20,
            default_warmup=3,
        )

    def add_custom_args(self, parser: argparse.ArgumentParser) -> None:
        parser.add_argument(
            "--n_bits", type=int, action="append",
            help="log2 base trace height N; repeat to sweep (default 20).",
        )
        parser.add_argument("--n_cols", type=int, default=64)
        parser.add_argument("--blowup_bits", type=int, default=1)
        parser.add_argument(
            "--arity", type=int, default=2,
            help="Merkle arity (2/3/4 -> Poseidon2 8/12/16); arity>=3 hits the "
            "merkle_commit power-of-two leaf-layer limit at scale, so default 2.",
        )
        parser.add_argument("--n_constraints", type=int, default=64)
        parser.add_argument("--degree", type=int, default=3)
        # Defaults match the production FRI schedule: the ZisK v1.0.0-alpha proving-key
        # starkStructs fold every inner-proof AIR by a uniform drop of 3 (factor 8)
        # down to nBits 5 (dominant [22,19,16,13,10,7,5] == _fold_steps(22, 3, 5)).
        # Drop-2-to-0 is a schedule no ZisK config uses — it adds tiny tail rounds
        # that over-state the per-round fold/commit compile the fri stage measures.
        parser.add_argument(
            "--fold_bits", type=int, default=3,
            help="bits folded per FRI layer (default 3 = production factor-8 drop).",
        )
        parser.add_argument(
            "--final_bits", type=int, default=5,
            help="log2 size of the final FRI polynomial (default 5, production).",
        )
        parser.add_argument(
            "--stages", type=str, default=",".join(_STAGES),
            help=f"comma list, subset of {_STAGES}.",
        )

    def get_ops(self, args: argparse.Namespace) -> Iterable[BenchmarkOp]:
        stages = [s for s in args.stages.split(",") if s]
        unknown = set(stages) - set(_STAGES)
        if unknown:
            raise ValueError(f"unknown stage(s) {sorted(unknown)}; pick from {_STAGES}")
        for n_bits in args.n_bits or [20]:
            yield from self._stage_ops(n_bits, stages, args)

    def _stage_ops(
        self, n_bits: int, stages: list[str], args: argparse.Namespace
    ) -> Iterable[BenchmarkOp]:
        blowup = 1 << args.blowup_bits
        n = 1 << n_bits
        n_ext = n * blowup
        meta = {"n_bits": n_bits, "n_cols": args.n_cols, "arity": args.arity}

        def op(name: str, fn: Callable, arg, lower: bool = True, hash_arg=None):
            """Assemble a BenchmarkOp. The output hash is taken lazily (one fn call
            at hash time, after the runtime phase) so the op is NOT pre-compiled
            here — otherwise zkbench's compile phase would hit a warm cache and
            report ~0. `lower` is attached only when `fn` is a single jitted fn."""
            return BenchmarkOp(
                name=f"{name}_2p{n_bits}",
                fn=lambda fn=fn, arg=arg: fn(arg),
                metadata={**meta, "stage": name},
                throughput_unit="rows/s",
                throughput_count=n_ext,
                input_hash=_hash_array(arg if hash_arg is None else hash_arg),
                output_hash_fn=lambda fn=fn, arg=arg: _hash_array(_first_array(fn(arg))),
                lower=(lambda fn=fn, arg=arg: fn.lower(arg)) if lower else None,
            )

        # Stage-1 inputs: a random trace, its LDE made device-resident so the
        # commit zone's wall time excludes the extend it depends on.
        if {"extend", "commit", "full"} & set(stages):
            trace = jax.block_until_ready(rand_field(0, (n, args.n_cols), F))
            extend_jit = jax.jit(lambda t: extend(t, blowup))
            mt = merkle_tree(args.arity)
            if "extend" in stages:
                yield op("extend", extend_jit, trace)
            if "commit" in stages:
                extended = jax.block_until_ready(extend_jit(trace))
                yield op("commit", jax.jit(lambda e: mt.commit(e)), extended,
                         hash_arg=extended)
            if "full" in stages:
                yield op("full", jax.jit(lambda t: mt.commit(extend(t, blowup))), trace)

        if "quotient" in stages:
            eval_fn = _make_eval_fn(args.n_cols, args.n_constraints, args.degree)
            alpha = jax.block_until_ready(_rand_cubic(args.n_constraints, 1))
            trace_ext = jax.block_until_ready(rand_field(0, (n_ext, args.n_cols), F))
            qfn = jax.jit(
                lambda t, eval_fn=eval_fn, alpha=alpha: quotient_from_constraints(
                    eval_fn, t, alpha, n_bits, args.blowup_bits
                )
            )
            yield op("quotient", qfn, trace_ext)

        if "divide" in stages:
            composite = jax.block_until_ready(_rand_cubic(n_ext, 2))
            dfn = jax.jit(lambda c: compute_quotient(c, n_bits, args.blowup_bits))
            yield op("divide", dfn, composite)

        if "fri" in stages:
            # Warm the memoized Poseidon2 perms (host-side M4/const analysis)
            # so the jit trace reuses them instead of rebuilding under trace.
            goldilocks_perm(8)
            goldilocks_perm(12)
            n_bits_ext = n_bits + args.blowup_bits
            steps = _fold_steps(n_bits_ext, args.fold_bits, args.final_bits)
            fri_pol = jax.block_until_ready(_rand_cubic(1 << n_bits_ext, 0))

            def fri_outputs(pol, steps=steps, arity=args.arity):
                # A fresh transcript per call keeps the squeezed challenges
                # deterministic; return the arrays so block_until_ready waits on
                # the fold/commit chain (FriProof is not a pytree).
                t = Transcript()  # width 12 == pil2 transcriptArity 3
                t.put(jnp.zeros((1,), F))  # stand-in for the pre-FRI proof state
                proof = prove(pol, steps, arity=arity, transcript=t)
                return (proof.final_pol, *proof.roots)

            # Jitted: the perm-memoize + device-bitcast seam make the fold loop
            # traceable as one function, so warm fri reflects compute, not the
            # per-call recompile (the fair box-4 number).
            fri_jit = jax.jit(fri_outputs)
            yield op("fri", fri_jit, fri_pol, lower=True)


if __name__ == "__main__":
    raise SystemExit(InnerProofBenchmark().run())
