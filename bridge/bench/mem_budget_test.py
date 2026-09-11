"""Pins how `mem_budget.py` reads a run's memory story: that the arena is read
rather than computed, that pil2's two refusal sentences are one verdict, that
a run whose clients disagree is an error rather than a number, and that the
walk table counts repeats per cell.

The fixtures are shaped like real logs and trimmed to the lines the reader
looks at. They are synthetic on purpose: the cases that matter most here are
the ones no run on this card produces on demand — clients with different
arenas, and pil2 refusing in the sentence that carries no figures."""

import contextlib
import io
import subprocess
import sys

from absl.testing import absltest

from bridge.bench import mem_budget

GIB = 1 << 30


def arena(gib: float) -> str:
    return (
        f"I0911 16:51:42.318202 4001412 gpu_helpers.cc:141] XLA backend"
        f" allocating {int(gib * GIB)} bytes on device 0 for BFCAllocator.\n"
    )


def pil2_sees(gb: float, basic=1, recursive=0) -> str:
    return (
        "[INFO] PilStark: Process rank 0: Using minimum memory"
        f" across 1 GPUs: {gb} GB\n"
        f"[INFO] PilStark: Using {basic} streams per GPU for basic proofs and"
        f" {recursive} streams per GPU for recursive proofs.\n"
    )


VERIFIED = "··· ✓ Vadcop Final proof was verified\n"

REFUSED_WITH_FIGURES = (
    "[ERROR]: GPU 0: Insufficient memory."
    " Need 12.904107 GB but only 12.827820 GB available\n"
)
REFUSED_WITHOUT = (
    "Error: Invalid configuration: Not enough GPU memory to run the proof\n"
)

CLIENT_OOM = (
    "2026-09-11T07:57:55.251173Z proofman::proofman ERROR: zisk-zorch bridge:"
    " instance 1: PJRT error in Event_Await: Out of memory while trying to"
    " allocate 1.88GiB.\n"
)

STATS = (
    "Limit:                        10.97GiB\n"
    "MaxInUse:                     10.30GiB\n"
    "MaxAllocSize:                  2.75GiB\n"
)


class ArenaTest(absltest.TestCase):
    def test_the_arena_is_the_bytes_the_run_allocated(self):
        """Not the fraction times the card: XLA's base is below the card's
        total, so a computed share is high in every cell."""
        run = mem_budget.Run(arena(11.60) + pil2_sees(18.572) + VERIFIED)
        self.assertAlmostEqual(run.share, 11.60, places=2)
        self.assertLen(run.arenas, 1)

    def test_two_clients_report_one_share(self):
        run = mem_budget.Run(arena(8.47) + arena(8.47) + pil2_sees(12.828))
        self.assertLen(run.arenas, 2)
        self.assertAlmostEqual(run.share, 8.47, places=2)

    def test_clients_that_disagree_are_an_error(self):
        """Every client of a run gets the same arena, so a log where they
        differ has been misread -- picking one would publish half a run."""
        run = mem_budget.Run(arena(8.47) + arena(11.60))
        with self.assertRaises(ValueError):
            _ = run.share

    def test_a_native_run_has_no_arena(self):
        run = mem_budget.Run(pil2_sees(30.9, basic=3, recursive=1) + VERIFIED)
        self.assertIsNone(run.share)
        self.assertEqual(run.outcome, "verified")


class OutcomeTest(absltest.TestCase):
    def test_the_shortfall_comes_from_the_refusal_itself(self):
        """Not from `Using minimum memory`, which pil2 prints on another line
        and a log carrying the refusal need not have at all."""
        run = mem_budget.Run(arena(8.47) * 2 + REFUSED_WITH_FIGURES)
        self.assertIsNone(run.pil2_sees)
        self.assertAlmostEqual(run.pil2_needs - run.pil2_available, 0.076287, places=5)

    def test_both_of_pil2s_sentences_are_a_refusal(self):
        with_figures = mem_budget.Run(
            arena(8.47) * 2 + pil2_sees(12.828) + REFUSED_WITH_FIGURES
        )
        without = mem_budget.Run(arena(8.78) * 2 + pil2_sees(12.199) + REFUSED_WITHOUT)
        self.assertEqual(with_figures.outcome, "pil2 refused")
        self.assertEqual(without.outcome, "pil2 refused")
        self.assertAlmostEqual(with_figures.pil2_needs, 12.904107)
        self.assertIsNone(without.pil2_needs)

    def test_a_client_oom_names_the_instance_and_the_size(self):
        run = mem_budget.Run(arena(10.98) + pil2_sees(19.2) + CLIENT_OOM + STATS)
        self.assertEqual(run.outcome, "client OOM (instance 1, 1.88 GiB)")
        self.assertEqual(run.stats["MaxAllocSize"], "2.75GiB")
        self.assertEqual(run.stats["MaxInUse"], "10.30GiB")

    def test_a_megabyte_request_is_not_read_as_gigabytes(self):
        oom = CLIENT_OOM.replace("1.88GiB", "832.03MiB")
        run = mem_budget.Run(arena(7.53) + oom)
        self.assertAlmostEqual(run.oom_gib, 832.03 / 1024, places=3)


class WalkTest(absltest.TestCase):
    def runs(self):
        return [
            mem_budget.Run(arena(11.60) + VERIFIED),
            mem_budget.Run(arena(11.60) + VERIFIED),
            mem_budget.Run(arena(10.66) + VERIFIED),
            mem_budget.Run(arena(10.66) + CLIENT_OOM),
            mem_budget.Run(arena(8.47) * 2 + REFUSED_WITH_FIGURES),
        ]

    def test_the_table_counts_repeats_per_cell(self):
        """A cell at the boundary is a race rather than a threshold, so the
        count is the result and a one-run cell is not a floor."""
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mem_budget.walk(self.runs())
        lines = out.getvalue().splitlines()
        self.assertIn("2/2", [ln.split()[-1] for ln in lines if "11.60" in ln][0])
        self.assertIn("1/2", [ln.split()[-1] for ln in lines if "10.66" in ln][0])
        self.assertIn("0/1", [ln.split()[-1] for ln in lines if "8.47" in ln][0])

    def test_the_table_keeps_client_counts_apart(self):
        """One client at 8.47 GiB and two at 8.47 GiB are different runs; a
        table that merged them would read as a floor nobody measured."""
        runs = [
            mem_budget.Run(arena(8.47) + VERIFIED),
            mem_budget.Run(arena(8.47) * 2 + REFUSED_WITH_FIGURES),
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mem_budget.walk(runs)
        cells = [ln for ln in out.getvalue().splitlines() if "8.47" in ln]
        self.assertLen(cells, 2)


class ScriptModeTest(absltest.TestCase):
    """docs/bridge.md "Memory budget" invokes this file by path, which puts its
    own directory on sys.path rather than the repo root."""

    def test_the_documented_invocation_runs(self):
        log = self.create_tempfile(
            "run.log", content=arena(11.60) + pil2_sees(18.572) + VERIFIED
        )
        done = subprocess.run(
            [sys.executable, "bridge/bench/mem_budget.py", log.full_path],
            capture_output=True,
            text=True,
        )
        self.assertEqual(done.returncode, 0, done.stderr)
        self.assertIn("11.60 GiB arena", done.stdout)


if __name__ == "__main__":
    absltest.main()
