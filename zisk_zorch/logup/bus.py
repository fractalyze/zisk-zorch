"""pil2-stark's LogUp bus argument (`std_sum`) — the stage-2 cross-AIR bus.

Each row contributes `sum_i mult_i / den_i`, where `den_i` is the bus tuple
combined into one cubic value, and the committed `gsum` column is the running
prefix sum of those local terms (pil2's `calculateWitnessSTD(prod=false)` → hint
`gsum_col`). `gsum`'s last row is this instance's airgroup `gsum_result`, tied to
it by the `__L1__'` constraint. `quotient/` only folds the constraints those
columns satisfy.

Two α orders live here, and both are pil2's: `denominator` reads an explicit
tuple with `tuple_[0]` at the *highest* α power (the `fields`-crate reference the
golden pins), while `denominators` walks an `Interaction`'s values in *reverse*
and appends the native bus id at the low end (pil2's inline `gsum_e`).

zisk-zorch's counterpart to sp1-zorch's `sp1_zorch/logup_gkr/` — same argument,
different reduction — so `bus.py` sits where that package's `circuit.py` does.

std_sum driver: https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/gen_proof.hpp#L24-L65
gsum/im hints:  https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/hints.cpp
"""

from __future__ import annotations

import dataclasses
from typing import Any, Mapping, Sequence

import frx
import frx.numpy as fnp
from frx import Array
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3

from zisk_zorch.golden import embed

# pil2's global challenge order (proving-key `challengesMap`): α then γ, ahead
# of the quotient challenge at 2.
STD_ALPHA = 0
STD_GAMMA = 1


# `eq=False`: the challenge fields are device arrays, so a generated `__eq__`
# would return an array rather than a bool, and its paired `__hash__` would raise.
@dataclasses.dataclass(frozen=True, eq=False)
class LogUpBus:
    """The LogUp bus argument for one AIR: its bus interactions under the
    `std_alpha` / `std_gamma` challenges.

    `interactions` stays empty for the tuple-level primitives, which need only
    the challenges.
    """

    alpha: Array
    gamma: Array
    interactions: Sequence[Any] = ()

    @classmethod
    def from_chip(cls, chip, challenges: Mapping[int, Array]) -> LogUpBus:
        """The bus of a `rw_constraints` chip, its interactions in pil2's
        `std_sum` order (sends, then receives)."""
        return cls(
            alpha=challenges[STD_ALPHA],
            gamma=challenges[STD_GAMMA],
            interactions=[s.interaction for s in chip.get_sends()]
            + [r.interaction for r in chip.get_receives()],
        )

    def denominator(self, tuple_: Array) -> Array:
        """The denominator of one explicit `[..., T]` cubic tuple: Horner in α
        with `tuple_[..., 0]` the highest power, then `+ γ`."""
        den = tuple_[..., 0]
        for k in range(1, tuple_.shape[-1]):
            den = den * self.alpha + tuple_[..., k]
        return den + self.gamma

    def denominators(self, trace: Array) -> list[Array]:
        """pil2's inline `gsum_e + std_gamma`, per configured interaction:
        reverse-α-Horner over its `VirtualPairCol` tuple, `· α + kind_int` to
        append the native bus id at the low end, `+ γ`."""
        return [self._gsum_e(it, trace) + self.gamma for it in self.interactions]

    def _gsum_e(self, interaction, trace: Array) -> Array:
        vals = [self.eval_pair_col(v, trace) for v in interaction.values]
        den = vals[-1]
        for v in reversed(vals[:-1]):
            den = den * self.alpha + v
        return den * self.alpha + embed([str(interaction.kind)])

    @staticmethod
    def eval_pair_col(vpc, trace: Array) -> Array:
        """Materialize a rw `VirtualPairCol` on the base `trace` `(N, n_cols)`,
        embedded to `F3`: `const + Σ wᵢ·colᵢ + Σ wₖ·colₐ·col_b`.

        The bilinear part is not dead weight — arith's operation bus
        (`proves_operation`) carries `div·chunk` products, and its denominator
        only matches pil2's `gsum_e` with them. Tuples are read at the current
        row (the `is_pre` flag is unused; no exported ZisK tuple sets it).
        """
        n = trace.shape[0]
        acc = fnp.broadcast_to(_scalar(vpc.constant), (n,))
        for col, _is_pre, weight in vpc.column_weights:
            acc = acc + _scalar(weight) * trace[:, col]
        for col_a, _pre_a, col_b, _pre_b, weight in getattr(vpc, "column_products", ()):
            acc = acc + _scalar(weight) * trace[:, col_a] * trace[:, col_b]
        return acc.astype(F3)

    @staticmethod
    @frx.jit
    def grand_sum(numerators: Array, denominators: Array) -> Array:
        """The committed `gsum` column: the running prefix sum of each row's local
        term `sum_i numerator_i * denominator_i^-1`.

        Inputs are `[I, N]` cubic — interaction-major, one contiguous `[N]` column
        per interaction. Both that layout and this one-`fnp.sum` spelling are
        load-bearing; every alternative is byte-identical, so only the benchmark
        in development.md catches a regression.
        """
        return fnp.cumsum(fnp.sum(numerators / denominators, axis=0))

    @staticmethod
    def gsum_result(gsum: Array) -> Array:
        """This instance's airgroup export, modulo pil2's single-row direct
        update (handled by the caller)."""
        return gsum[-1]


def _scalar(value) -> Array:
    """A base-field scalar for a `VirtualPairCol` coefficient. The coefficient is
    canonical in [0, p) (rw stores −1 as p−1), so it value-converts straight to
    `F` with no reduction — like `embed`'s decimals."""
    return fnp.array(int(value), dtype=F)
