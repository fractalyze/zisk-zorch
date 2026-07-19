"""Adapters binding ZisK's pil2-stark FRI to zorch's k-ary fold seam.

zorch owns the generic fold *orchestration* — `PreFoldKGroupCommitRound` (the
prover's commit→squeeze→fold round) and `verify_group_fold_chain` (the verifier's
per-layer fold check) — over the `KFoldableCode` seam (zorch#252 Option B). This
module supplies ZisK's half of that seam: the pil2-specific grouping, coset, and
transcript stay here while the loop shape comes from zorch.

Two adapters:

- `Pil2FriCode` — a `KFoldableCode` over pil2's coset FRI. `group_leaves` is
  pil2's `getTransposed` regroup (cubic→base limb expansion for the linear-hash
  leaf); `fold_group` / `fold_group_values` delegate to `fold` / `verify_fold`
  (coset Lagrange over `W[32]`/coset-7, the `fri_fold_k_values` kernel); the
  index maps generalize pil2's per-layer query modulus. The code is
  `steps`-parameterized, so each method reads its layer's `(prev, current)` bit
  sizes off the live codeword length — variable-drop schedules included.
- `Pil2SeamTranscript` — wraps ZisK's mutable pil2 `Transcript` (`.put` /
  `.get_field`) as zorch's functional `Transcript` (`.observe` / `.sample`).
  One `put` + one cubic `get_field` per layer is exactly the seam's one
  `observe` + one `sample`, so the Fiat-Shamir byte stream is unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass

import frx
import frx.numpy as fnp
from frx import Array
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3

from zisk_zorch.fri.fold import fold, intt, verify_fold
from zisk_zorch.transcript.transcript import Transcript


def _cubic_to_base(values: Array) -> Array:
    """View a cubic array as its Goldilocks limbs, the cubic axis expanded in
    place (`[c0, c1, c2]` -> three contiguous base lanes) — pil2's
    `FIELD_EXTENSION`-contiguous memory layout the linear hash leaves see.

    Device-native `bitcast_convert` keeps the reinterpret on-device so the FRI
    fold loop traces as one jitted function; the prior `np.asarray(...).view`
    host round-trip forced eager execution (re-compiling per `prove` call).
    `bitcast_convert` appends the size-3 limb axis, which the reshape folds back
    into the trailing axis to recover the contiguous limb layout."""
    base = frx.lax.bitcast_convert_type(values, F)
    return base.reshape(*values.shape[:-1], values.shape[-1] * 3)


def _base_to_cubic(values: Array) -> Array:
    """The inverse view: contiguous base limbs -> cubic elements (the 3 limbs are
    that element's coefficients, cf. golden `u64x3`)."""
    triples = values.reshape(*values.shape[:-1], -1, 3)
    return frx.lax.bitcast_convert_type(triples, F3)


@dataclass(frozen=True)
class Pil2FriCode:
    """`KFoldableCode` over pil2-stark's coset FRI (Goldilocksˣ³), parameterized
    by the layer `steps` (`steps[0] == nBitsExt`, strictly decreasing). Holds the
    pil2-specific grouping + coset construction the zorch fold seam drives over;
    it implements only the fold-seam methods the orchestration calls (no
    `LinearCode` encode surface — the FRI driver never re-encodes)."""

    steps: tuple[int, ...]

    def __post_init__(self) -> None:
        # Validate the step schedule once, where both the prover and verifier
        # construct the code — so neither entry point can index a malformed
        # schedule (arity is validated by `merkle_tree`, which both also call).
        steps = self.steps
        if len(steps) < 1:
            raise ValueError("steps must list at least the extended-domain size")
        if any(not 0 <= s <= 32 for s in steps):
            raise ValueError(
                f"each step must be a domain log size in [0, 32], got {list(steps)}"
            )
        if list(steps) != sorted(steps, reverse=True) or len(set(steps)) != len(steps):
            raise ValueError(f"steps must strictly decrease, got {list(steps)}")

    @property
    def n_bits_ext(self) -> int:
        return self.steps[0]

    @property
    def fold_factor(self) -> int:
        """The layer-0 fold factor `2^(steps[0]-steps[1])`. Meaningful as a single
        value only for a uniform-drop schedule; a variable-drop schedule's factor
        is per-layer (see `group_indices`), so the seam reads it off `steps`, not
        this attribute (which the round never consults)."""
        return 1 << (self.steps[0] - self.steps[1])

    def _level_of(self, length: int) -> int:
        """The layer index of a codeword of `length` entries: the position of
        `log2(length)` in `steps`. Lets the level-blind `fold_group` /
        `group_leaves` recover their `(prev, current)` sizes from the live
        codeword, so a per-layer (variable-drop) factor needs no level argument."""
        return self.steps.index(length.bit_length() - 1)

    def fold_group(self, codeword: Array, beta: Array) -> Array:
        """Fold layer `i`'s codeword (size `2^steps[i]`) to `2^steps[i+1]` at the
        cubic `beta` — the prover-side per-round fold (`PreFoldKGroupCommitRound`)."""
        i = self._level_of(codeword.shape[0])
        return fold(codeword, beta, self.n_bits_ext, self.steps[i], self.steps[i + 1])

    def group_leaves(self, codeword: Array) -> Array:
        """pil2's `getTransposed`: layer `i`'s codeword -> the
        `(2^steps[i+1], 2^(steps[i]-steps[i+1]) * 3)` base-limb matrix whose row
        `g` holds the `n_x` cubic entries `pol[j*2^steps[i+1] + g]` (the coset the
        next fold reads), cubic-expanded for the linear-hash leaf."""
        i = self._level_of(codeword.shape[0])
        cur_n = 1 << self.steps[i + 1]
        n_x = 1 << (self.steps[i] - self.steps[i + 1])
        # reshape(n_x, cur_n).T[g, j] = pol[j*cur_n + g]; the cubic view then
        # expands each entry to its 3 limbs -> (cur_n, n_x*3).
        rows = codeword.reshape(n_x, cur_n).T
        return _cubic_to_base(rows)

    def group_indices(self, positions: Array, level: int) -> tuple[Array, ...]:
        """The `2^(steps[level]-steps[level+1])` leaf indices of layer `level`'s
        group whose fold lands at `positions` in layer `level+1`. Group member `j`
        of landing row `g` is the strided index `g + j*2^steps[level+1]` — the
        column order `group_leaves` lays down."""
        stride = 1 << self.steps[level + 1]
        k = 1 << (self.steps[level] - self.steps[level + 1])
        return tuple(positions + j * stride for j in range(k))

    def fold_group_values(
        self, group: Array, beta: Array, positions: Array, level: int
    ) -> Array:
        """Fold opened cubic k-groups of layer `level` at `positions` (verifier
        side) — `verify_fold` per query, batched. `group` is `(Q, n_x)` cubic; the
        landing row `positions` picks each query's coset points."""
        prev, current = self.steps[level], self.steps[level + 1]
        return frx.vmap(
            lambda values, idx: verify_fold(
                values, beta, self.n_bits_ext, prev, current, idx
            )
        )(group, positions)

    def group_layer_positions(self, positions: Array, num_rounds: int) -> list[Array]:
        """Per-layer query leaf indices: layer `i`'s fold lands at
        `positions mod 2^steps[i+1]` in layer `i+1` (pil2's `query % 2^currentBits`
        opening index, one per committed layer)."""
        return [positions % (1 << self.steps[i + 1]) for i in range(num_rounds)]

    def check_final(self, final: Array, n_bits: int) -> Array:
        """pil2's terminal low-degree test (`stark_verify.hpp` L672-L691): INTT the
        in-clear final polynomial and assert every coefficient at or above the
        degree bound vanishes.

        `final` is the last layer's `2^steps[-1]` cubic evaluations the prover
        sends uncompressed; `n_bits` is the base trace log size. The full fold
        chain reduces `f`'s degree by the blowup factor, so a genuine FRI
        polynomial leaves at most `2^(steps[-1] - (steps[0] - n_bits))`
        coefficients — pil2's `init`. A prover that sends a higher-degree final
        pol leaves a nonzero coefficient above the bound and is rejected, closing
        the soundness gap a fold-consistency check alone leaves open.

        https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/stark_verify.hpp#L679-L698"""
        last_bits = self.steps[-1]
        n = 1 << last_bits
        if final.shape != (n,):
            raise ValueError(f"final must have shape {(n,)}, got {final.shape}")
        blowup_bits = self.steps[0] - n_bits
        if blowup_bits < 0:
            raise ValueError(
                f"n_bits {n_bits} exceeds the extended-domain size {self.steps[0]}"
            )
        # init: the count of coefficients a degree-bounded final pol may keep
        # (pil2 stark_verify.hpp L684). A blowup wider than the final layer
        # collapses the bound to the zero polynomial.
        init = 0 if blowup_bits > last_bits else 1 << (last_bits - blowup_bits)
        coeffs = intt(_cubic_to_base(final).reshape(n, 3), last_bits)  # (n, 3) base
        return fnp.all(coeffs[init:] == fnp.zeros((), F))


@dataclass(frozen=True)
class Pil2SeamTranscript:
    """ZisK's pil2 `Transcript` as a zorch `Transcript`. The wrapped transcript is
    mutable (pil2's pending/out buffer), so `observe`/`sample` mutate it and return
    `self` — threading the same object the way the seam threads its functional
    transcript. Challenges are cubic: one `get_field` squeezes 3 Goldilocks limbs."""

    inner: Transcript

    def observe(self, values: Array) -> "Pil2SeamTranscript":
        self.inner.put(values)
        return self

    def sample(self, n: int = 1) -> tuple["Pil2SeamTranscript", Array]:
        challenges = [
            _base_to_cubic(self.inner.get_field()).reshape(()) for _ in range(n)
        ]
        return self, challenges[0] if n == 1 else fnp.stack(challenges)

    def observe_and_sample(
        self, values: Array, n: int = 1
    ) -> tuple["Pil2SeamTranscript", Array]:
        return self.observe(values).sample(n)
