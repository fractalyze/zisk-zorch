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
import pathlib
import subprocess
import sys

from absl.testing import absltest

from bridge.bench import mem_budget

GIB = 1 << 30


def arena(gib: float, allocator: str = "BFC") -> str:
    return (
        f"I0911 16:51:42.318202 4001412 gpu_helpers.cc:141] XLA backend"
        f" allocating {int(gib * GIB)} bytes on device 0 for {allocator}Allocator.\n"
    )


def pil2_sees(gb: float, basic=1, recursive=0) -> str:
    return (
        "[INFO] PilStark: Process rank 0: Using minimum memory"
        f" across 1 GPUs: {gb} GB\n"
        f"[INFO] PilStark: Using {basic} streams per GPU for basic proofs and"
        f" {recursive} streams per GPU for recursive proofs.\n"
    )


VERIFIED = "··· ✓ Vadcop Final proof was verified\n"
# What a prover built before 2026-09-08 prints instead -- the wording the
# #177/#188 walks on disk carry.
VERIFIED_OLD = "··· ✓ Proof verified successfully\n"


def exited(code: int) -> str:
    return f"exit={code}\n"


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


def mem_stats(
    client=0,
    in_use=1024,
    peak=10445,
    pool="-",
    peak_pool=11878,
    alloc=2816,
    limit=11878,
) -> str:
    return (
        f"[zz + 12.345] client {client} memory: in_use {in_use} MiB,"
        f" peak_in_use {peak} MiB, pool {pool} MiB, peak_pool {peak_pool} MiB,"
        f" largest_alloc {alloc} MiB, limit {limit} MiB\n"
    )


def took(instance: int, air: str) -> str:
    return f"[zz +  6.705] took instance {instance} {air}: 368 MB in 0.026 s\n"


STATS = (
    "Limit:                        10.97GiB\n"
    "MaxInUse:                     10.30GiB\n"
    "MaxAllocSize:                  2.75GiB\n"
)


class ArenaTest(absltest.TestCase):
    def test_the_arena_is_the_bytes_the_run_allocated(self):
        """Not the fraction times the card: XLA's base is below the card's
        total, so a computed share is high in every cell."""
        run = mem_budget.Run(arena(11.60) + pil2_sees(18.572) + VERIFIED + exited(0))
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

    def test_the_allocator_is_read_from_the_line_that_carries_the_arena(self):
        """The arm in a table comes from what the plugin says it built, not
        from the kind the sweep meant to set: `ZZ_ALLOCATOR` misspelled falls
        back to the default allocator and the run is otherwise identical."""
        bfc = mem_budget.Run(arena(11.60) + VERIFIED + exited(0))
        asyn = mem_budget.Run(arena(11.60, "CudaAsync") + VERIFIED + exited(0))
        self.assertEqual(bfc.allocator, "BFC")
        self.assertEqual(asyn.allocator, "CudaAsync")

    def test_clients_on_different_allocators_are_an_error(self):
        run = mem_budget.Run(arena(8.47) + arena(8.47, "CudaAsync"))
        with self.assertRaises(ValueError):
            _ = run.allocator

    def test_a_native_run_has_no_arena(self):
        run = mem_budget.Run(
            pil2_sees(30.9, basic=3, recursive=1) + VERIFIED + exited(0)
        )
        self.assertIsNone(run.share)
        self.assertIsNone(run.allocator)
        self.assertEqual(run.outcome, "verified")


class PeakTest(absltest.TestCase):
    """`ZZ_MEM_STATS=1` lines: what the allocator held against what was live in
    it, which a run that finished can state and a run that died cannot."""

    def test_the_last_line_of_a_client_carries_the_runs_peaks(self):
        log = arena(11.60) + mem_stats(peak=8000) + mem_stats(peak=10445) + exited(0)
        peak = mem_budget.Run(log).peaks[0]
        self.assertAlmostEqual(peak.in_use, 10445 / 1024, places=2)
        self.assertAlmostEqual(peak.held, 11878 / 1024, places=2)
        self.assertAlmostEqual(peak.largest_alloc, 2.75, places=2)

    def test_a_statistic_the_allocator_does_not_keep_is_not_zero(self):
        """An allocator with no pool of its own leaves `peak_pool` unset, and
        a reader that called that zero would report it holding less than was
        live in it."""
        run = mem_budget.Run(arena(7.53, "CudaAsync") + mem_stats(peak_pool="-"))
        self.assertIsNone(run.peaks[0].held)
        self.assertIsNotNone(run.peaks[0].in_use)

    def test_the_room_above_the_data_is_reported(self):
        log = arena(11.60) + mem_stats(peak=10445, peak_pool=11878) + exited(0)
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mem_budget.report(pathlib.Path("run.log"), mem_budget.Run(log))
        self.assertIn("held 11.60 GiB -- 1.40 GiB of room", out.getvalue())


class OutcomeTest(absltest.TestCase):
    def test_an_older_rescued_run_without_an_exit_line_is_a_pass(self):
        """The compound case, and the only one where the verify wording decides
        anything here: a pre-2026-09-08 run (`Proof verified successfully`)
        that rescued an out-of-memory, in a log with no `exit=` for the
        fallback to lean on. A reader knowing only the later wording sees the
        OOM, calls it an abort, and scores the cell a failure."""
        run = mem_budget.Run(arena(11.60) + CLIENT_OOM + VERIFIED_OLD)
        self.assertTrue(run.completed)
        self.assertStartsWith(run.outcome, "verified, 1.88 GiB rescued")

    def test_a_rescued_oom_is_not_an_abort(self):
        """The bridge catches a read-ahead upload's OOM and sends it under the
        slot, and Rust's panic hook has already printed the message. Only
        `exit=` tells that run from one that died."""
        run = mem_budget.Run(arena(11.60) + CLIENT_OOM + VERIFIED + exited(0))
        self.assertTrue(run.completed)
        self.assertStartsWith(run.outcome, "verified, 1.88 GiB rescued")

    def test_a_refusal_without_an_exit_line_is_not_a_pass(self):
        """A log cut before `run.sh` appended `exit=` has only its text, and a
        refusal carries no out-of-memory for the text fallback to catch."""
        run = mem_budget.Run(arena(8.47) * 2 + REFUSED_WITH_FIGURES)
        self.assertFalse(run.completed)
        self.assertEqual(run.outcome, "pil2 refused")

    def test_a_fatal_oom_is_an_abort(self):
        """Same text, no verify line, exit 134."""
        run = mem_budget.Run(arena(11.60) + CLIENT_OOM + exited(134))
        self.assertFalse(run.completed)
        self.assertEqual(run.outcome, "client OOM (instance 1, 1.88 GiB)")

    def test_an_abort_names_the_air_that_bound(self):
        """Which shape binds is not fixed across a ladder, so a cell that named
        only the instance id could not be compared with the cell above it."""
        log = arena(10.98) + took(1, "Main_n22") + took(9, "VirtualTableZisk0_n21")
        run = mem_budget.Run(
            log + CLIENT_OOM.replace("instance 1:", "instance 9:") + exited(134)
        )
        self.assertEqual(
            run.outcome, "client OOM (instance 9 VirtualTableZisk0_n21, 1.88 GiB)"
        )

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
            mem_budget.Run(arena(11.60) + VERIFIED + exited(0)),
            mem_budget.Run(arena(11.60) + VERIFIED_OLD + exited(0)),
            mem_budget.Run(arena(10.66) + VERIFIED + exited(0)),
            mem_budget.Run(arena(10.66) + CLIENT_OOM + exited(134)),
            mem_budget.Run(arena(8.47) * 2 + REFUSED_WITH_FIGURES + exited(1)),
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

    def test_the_table_keeps_allocators_apart(self):
        """The same arena under two kinds is two different statements -- one is
        a ceiling every allocation is placed inside, the other a claim the
        pool grows past -- so a merged cell would read as a floor nobody
        measured."""
        runs = [
            mem_budget.Run(arena(10.66) + CLIENT_OOM + exited(134)),
            mem_budget.Run(arena(10.66, "CudaAsync") + VERIFIED + exited(0)),
        ]
        out = io.StringIO()
        with contextlib.redirect_stdout(out):
            mem_budget.walk(runs)
        cells = [ln for ln in out.getvalue().splitlines() if "10.66" in ln]
        self.assertLen(cells, 2)
        self.assertIn("0/1", [c for c in cells if "BFC" in c][0])
        self.assertIn("1/1", [c for c in cells if "CudaAsync" in c][0])

    def test_the_table_keeps_client_counts_apart(self):
        """One client at 8.47 GiB and two at 8.47 GiB are different runs; a
        table that merged them would read as a floor nobody measured."""
        runs = [
            mem_budget.Run(arena(8.47) + VERIFIED + exited(0)),
            mem_budget.Run(arena(8.47) * 2 + REFUSED_WITH_FIGURES + exited(1)),
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
            "run.log",
            content=arena(11.60) + pil2_sees(18.572) + VERIFIED + exited(0),
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
