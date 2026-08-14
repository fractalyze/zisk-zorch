"""The Main continuation port against the proving key, on real block data.

The golden (`scripts/extract_continuation_fixture.py`) evaluates the key's
own `im_airval` hint SSA — every direct interaction's numerator and
denominator — over the 12 Main segments of a real eth-block capture. The
port is correct iff it reproduces those 96 fractions per segment from the
stage-1 air values alone: the denominator pins the tuple (bus id, slot
values, fold), the numerator pins the direction and selector. On top of
the per-interaction bijection, the boundary chain and the global
telescope check the property the predicate exists for — every value one
segment proves is what its successor assumes, and the once-per-proof
global tuples close both cycles."""

from __future__ import annotations

import collections
import pathlib

import frx.numpy as fnp
import numpy as np
from absl.testing import absltest
from zk_dtypes import goldilocksx3 as F3

from zisk_zorch.constraints.continuation import (
    MAIN_CONTINUATION_ID,
    MainSegmentAirValues,
    continuation_assumes,
    continuation_proves,
    direct_term,
    fold_denominator,
    global_updates,
    interactions,
)
from zisk_zorch.golden import load, u64x3

_GOLDEN = (
    pathlib.Path(__file__).parent / "testdata" / "golden" / "main_continuation.json"
)
_P = 0xFFFFFFFF00000001


def _scalar(words) -> fnp.ndarray:
    return u64x3([str(int(w)) for w in words]).reshape(())


def _limbs(x) -> tuple[str, ...]:
    flat = np.asarray(fnp.array(x, dtype=F3).reshape(1)).view(np.uint64)
    return tuple(str(w) for w in flat.reshape(-1)[:3])


class MainContinuationTest(absltest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        super().setUpClass()
        golden = load(_GOLDEN)
        cls.n_bits = golden["n_bits"]
        cls.segments = golden["segments"]
        cls.airvals = [
            MainSegmentAirValues.from_named(s["airvalues_named"]) for s in cls.segments
        ]

    def test_every_interaction_matches_the_key(self) -> None:
        """Bijection against the key's `im_airval` evaluations: each ported
        (denominator, numerator) fraction appears and the counts agree — so
        the tuples, directions, and selectors are the key's, not a
        plausible transcription. The golden carries the per-interaction
        fractions for one segment (the Σ in `test_direct_term_matches_the_key`
        pins the rest; this test exists to localize a failure)."""
        checked = 0
        for seg, av in zip(self.segments, self.airvals):
            if "interactions" not in seg:
                continue
            alpha, gamma = _scalar(seg["alpha"]), _scalar(seg["gamma"])
            ours = collections.Counter()
            for it in interactions(av, self.n_bits):
                den = _limbs(fold_denominator(it, alpha, gamma))
                num = _limbs(u64x3([str(it.numerator % _P), "0", "0"]).reshape(()))
                ours[(den, num)] += 1
            want = collections.Counter(
                (tuple(i["denominator"]), tuple(i["numerator"]))
                for i in seg["interactions"]
            )
            self.assertEqual(ours, want, f"{seg['instance']}: fraction multisets")
            checked += 1
        self.assertGreater(checked, 0, "no segment carries per-interaction fractions")

    def test_direct_term_matches_the_key(self) -> None:
        for seg, av in zip(self.segments, self.airvals):
            got = direct_term(
                av, self.n_bits, _scalar(seg["alpha"]), _scalar(seg["gamma"])
            )
            self.assertEqual(
                _limbs(got), tuple(seg["direct_term"]), f"{seg['instance']}: Σ num/D"
            )

    def test_boundary_chain(self) -> None:
        """Segment k's proves tuple is segment k+1's assumes tuple — the
        continuation carry, on the real block's 11 boundaries."""
        for k in range(len(self.airvals) - 1):
            self.assertEqual(
                continuation_proves(self.airvals[k]).values,
                continuation_assumes(self.airvals[k + 1]).values,
                f"boundary {k} -> {k + 1}",
            )

    def test_global_telescope(self) -> None:
        """Across the whole proof the continuation family cancels: every
        proves tuple (segments + the global boot row) pairs with exactly
        one assumes tuple (segments + the global end row)."""
        proves: collections.Counter = collections.Counter()
        assumes: collections.Counter = collections.Counter()
        for av in self.airvals:
            proves[continuation_proves(av).values] += 1
            assumes[continuation_assumes(av).values] += 1
        for it in global_updates():
            if it.bus_id != MAIN_CONTINUATION_ID:
                continue
            (proves if it.numerator > 0 else assumes)[it.values] += 1
        self.assertEqual(proves, assumes)

    def test_last_segment_flag(self) -> None:
        """`main_last_segment` is boolean everywhere and set exactly on
        the final segment (main.pil's airval constraint)."""
        flags = [av.main_last_segment for av in self.airvals]
        self.assertTrue(all(f in (0, 1) for f in flags))
        self.assertEqual(flags, [0] * (len(flags) - 1) + [1])


if __name__ == "__main__":
    absltest.main()
