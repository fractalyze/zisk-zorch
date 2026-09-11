"""Pins how `leg_phases.py` reads a leg out of a `-vv` run log: which lines are
a run's basic proofs on each stack, a wall being a union rather than a sum, the
overlap bound, and the guard against reading a bridged run as a native one.

The fixtures are shaped like real logs and trimmed to the lines the reader
looks at. They are synthetic on purpose: a real log cannot express the case
that matters most here — a bridged run whose `ZZ_LOG` was off, which is
indistinguishable from a native run except by how little of the leg its
`GEN_PROOF` spans explain."""

import contextlib
import io
import pathlib

from absl.testing import absltest

from bridge.bench import leg_phases

LEG_OPEN = "2026-09-11T02:00:00.000000Z proofman INFO: >>> GENERATING_INNER_PROOFS\n"


def leg_shut(ms: int) -> str:
    return (
        "2026-09-11T02:00:10.000000Z proofman INFO:"
        f" <<< GENERATING_INNER_PROOFS ({ms}ms)\n"
    )


def gen_proof(n: int, end: str, ms: int) -> str:
    return (
        f"2026-09-11T02:00:{end}Z proofman DEBUG: <<< GEN_PROOF_{n} [0:{n}] ({ms}ms)\n"
    )


def recursive(end: str, ms: int) -> str:
    return (
        f"2026-09-11T02:00:{end}Z proofman::recursion DEBUG:"
        f" <<< GEN_RECURSIVE_PROOF_Recursive1 [0:0] ({ms}ms)\n"
    )


def held(instance: int, at: float, total: float, waiting: float, kind="worker") -> str:
    return (
        f"[zz + {at:6.3f}] instance {instance} Air_n21 ({kind}):"
        f" {total:.3f} s, of which {waiting:.3f} s waiting for the client"
        " (0.000 s of uploads and reads done ahead)\n"
    )


# Two basic proofs that overlap: 1 s each, ending 0.5 s apart, so they sum to
# 2 s of spans over 1.5 s of wall.
NATIVE = (
    LEG_OPEN
    + gen_proof(0, "02.000000", 1000)
    + gen_proof(1, "02.500000", 1000)
    + recursive("04.000000", 1000)
    + leg_shut(3000)
)

# The same leg under the bridge: proofman's spans are milliseconds and the
# ZZ_LOG lines are the proves. Held = total - waiting, so instance 0 holds the
# client over [1.0, 2.0) and instance 1 over [2.0, 3.5).
BRIDGED = (
    LEG_OPEN
    + gen_proof(0, "02.000000", 4)
    + gen_proof(1, "02.500000", 7)
    + held(0, 2.0, 1.5, 0.5)
    + held(1, 3.5, 2.0, 0.5, kind="streamed")
    + recursive("04.000000", 1000)
    + leg_shut(4000)
)


class LegPhasesTest(absltest.TestCase):
    def test_native_basic_proofs_come_from_proofman(self):
        leg = leg_phases.Leg(NATIVE)
        self.assertFalse(leg.bridged)
        self.assertEqual(leg.n_basic, 2)
        self.assertEqual(leg.n_recursive, 1)

    def test_wall_is_a_union_and_spans_are_a_sum(self):
        leg = leg_phases.Leg(NATIVE)
        self.assertAlmostEqual(leg.basic_wall, 1.5)
        self.assertAlmostEqual(leg.basic_spans, 2.0)

    def test_bridged_basic_proofs_come_from_zz_log(self):
        leg = leg_phases.Leg(BRIDGED)
        self.assertTrue(leg.bridged)
        self.assertEqual(leg.n_basic, 2)
        # Held, not total: 1.0 s and 1.5 s, back to back and so 2.5 s of wall.
        self.assertAlmostEqual(leg.basic_wall, 2.5)
        self.assertAlmostEqual(leg.basic_spans, 2.5)

    def test_a_streamed_instance_is_counted(self):
        """A tally matching only `(worker)` drops the streamed instance and
        every figure built on the count is then quietly one prove short."""
        self.assertEqual(leg_phases.Leg(BRIDGED).n_basic, 2)
        one_worker = BRIDGED.replace("(streamed)", "(worker)")
        self.assertEqual(leg_phases.Leg(one_worker).n_basic, 2)

    def test_overlap_is_what_the_two_phases_cover_past_the_leg(self):
        # basic 1.5 + recursion 1.0 - leg 3.0 is negative: nothing is proven.
        self.assertAlmostEqual(leg_phases.Leg(NATIVE).overlap, 0.0)
        # basic 2.5 + recursion 1.0 - leg 4.0 leaves nothing either; a leg
        # shorter than the two phases is what makes the bound bite.
        tight = BRIDGED.replace(leg_shut(4000), leg_shut(3000))
        self.assertAlmostEqual(leg_phases.Leg(tight).overlap, 0.5)

    def test_remainder_moves_with_the_basic_phase(self):
        """The finding the report exists to stop anyone publishing: `leg -
        basic` is a residual, so a longer basic phase shrinks it with the
        recursion untouched."""
        short = leg_phases.Leg(NATIVE)
        longer = leg_phases.Leg(
            NATIVE.replace(
                gen_proof(1, "02.500000", 1000), gen_proof(1, "03.000000", 1000)
            )
        )
        self.assertGreater(longer.basic_wall, short.basic_wall)
        self.assertAlmostEqual(longer.recursive_wall, short.recursive_wall)
        self.assertLess(longer.remainder, short.remainder)

    def test_a_bridged_run_without_zz_log_is_flagged(self):
        no_zz = "".join(
            line + "\n" for line in BRIDGED.splitlines() if not line.startswith("[zz")
        )
        leg = leg_phases.Leg(no_zz)
        self.assertFalse(leg.bridged)
        self.assertTrue(leg.suspect)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            leg_phases.report(pathlib.Path("run.log"), leg)
        self.assertIn("WARNING", out.getvalue())

    def test_a_native_run_is_not_flagged(self):
        self.assertFalse(leg_phases.Leg(NATIVE).suspect)

    def test_a_log_without_a_leg_is_an_error(self):
        with self.assertRaises(ValueError):
            leg_phases.Leg(LEG_OPEN)


if __name__ == "__main__":
    absltest.main()
