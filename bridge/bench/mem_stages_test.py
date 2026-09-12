"""What `mem_stages.py` must get right about a `ZZ_MEM_STAGES` log.

Every case here is a way the reader could produce a table that looks correct:
an inventory truncated at an abort, a peak attributed to the wrong stage, two
clients' blocks folded into one prove, a last reader read off the manifest
instead of off what the run actually ran. None of them would raise on their
own."""

import pathlib

from absl.testing import absltest

from bridge.bench import mem_stages


def stage(name, *, in_use, peak, pool="4000", rows=(), live=None):
    """One stage block as the bridge writes it."""
    live = sum(int(r[2]) for r in rows) if live is None else live
    count = sum(int(r[1]) for r in rows)
    out = [
        f"[zz +  1.000] mem stage {name}: in_use {in_use}, peak {peak},"
        f" pool {pool}, live {live} in {count} buffers"
    ]
    out += [f"[zz +  1.000] mem stage {name} buf {o} {n} {b}" for o, n, b in rows]
    return "\n".join(out)


def instance(index, air):
    return (
        f"[zz +  9.000] instance {index} {air} (worker):"
        " 1.000 s, of which 0.100 s waiting"
    )


CM1 = ("commit1/cm1_ext", "1", "2550136832")
TRACE = ("upload/trace", "1", "1275068416")


class EmittedLineTest(absltest.TestCase):
    """The other half of the contract in `src/memlog.rs`.

    Every other test here hand-writes the log format in `stage()` / `prog()`,
    so all of them would keep passing if the bridge changed what it emits --
    the reader would simply match nothing and report a run as having no
    blocks. These parse the exact bytes the Rust tests assert on."""

    LINES = (
        pathlib.Path(__file__).parent / "testdata" / "mem_stages_lines.txt"
    ).read_text()

    def test_the_reader_parses_the_lines_the_bridge_emits(self):
        stamped = "\n".join(f"[zz +  1.000] {line}" for line in self.LINES.splitlines())
        prove = mem_stages.proves(stamped + "\n" + instance(0, "Main_n22"))[0]
        quotient = prove.stage("quotient")
        self.assertEqual(quotient.in_use, 8990000000)
        self.assertEqual(quotient.peak, 9400000000)
        self.assertEqual(quotient.pool, 15150000000)
        self.assertEqual(quotient.rows, [("commit1/cm1_ext", 1, 2550136832)])
        self.assertEqual(
            [m.ran for m in prove.marks if m.ran == "commit2"], ["commit2"]
        )

    def test_a_line_pil2_spliced_is_still_read(self):
        # pil2 writes its `[TRACE]` lines to the same fd from C++ while the
        # bridge writes these from Rust, and the two splice mid-line: seen in
        # the field as `[zz + [TRACE] PilStark: ...` on one physical line and
        # `7.005] mem stage grind: ...` on the next. The reader used to
        # require the `[zz + <t>]` prefix, so it dropped that header and then
        # raised a bare KeyError on the `buf` lines under it -- one run in
        # fifteen unreadable, for a timestamp nothing here reads.
        spliced = "\n".join(
            [
                "[zz + [TRACE] PilStark: <-- STARK_COMMIT_STAGE_1 : 0.04 s",
                " 7.005] mem stage quotient: in_use 8990000000, peak 9400000000,"
                " pool 15150000000, live 8000000000 in 3 buffers",
                "[zz +  7.005[TRACE] PilStark: <-- CALCULATE_IM_POLS : 0.00 s",
                "] mem stage quotient buf commit1/cm1_ext 1 2550136832",
                instance(0, "Main_n22"),
            ]
        )
        quotient = mem_stages.proves(spliced)[0].stage("quotient")
        self.assertEqual(quotient.in_use, 8990000000)
        self.assertEqual(quotient.rows, [("commit1/cm1_ext", 1, 2550136832)])

    def test_a_spliced_run_line_still_counts_its_program(self):
        # `run` splices like the rest and fails more quietly than they do: a
        # dropped one leaves its program out of `programs`, and `readers_of`
        # then names an earlier program as a section's last reader with
        # nothing raised at all.
        manifest = {
            "programs": {
                "commit1": {"inputs": [{"name": "trace"}], "outputs": []},
                "commit2": {"inputs": [{"name": "trace"}], "outputs": []},
            }
        }
        spliced = "\n".join(
            [
                "[zz +  1.000] mem stage stage1: in_use 1, peak 1, pool 1,"
                " live 1 in 0 buffers",
                "[zz +  1.000]   run commit1: enqueue 0.02 ms",
                "[zz +  1.[TRACE] PilStark: <-- STARK_STEP_Q : 0.11 s",
                "100]   run commit2: enqueue 0.03 ms",
                instance(0, "Main_n22"),
            ]
        )
        prove = mem_stages.proves(spliced)[0]
        self.assertEqual(prove.programs, ["commit1", "commit2"])
        self.assertEqual(
            mem_stages.readers_of(manifest, "upload/trace", prove.programs),
            ["commit1", "commit2"],
        )

    def test_a_spliced_prog_line_still_counts_and_still_names_the_peak(self):
        # `mem prog` splices like the rest, and at ZZ_MEM_STAGES=2 it is the
        # *authoritative* program order -- `Prove.programs` prefers the marks
        # and falls back to the `run` lines only at level 1. So a dropped one
        # takes a program out of the order and out of `peak_program`, which is
        # what names the program the high-water rose across.
        spliced = "\n".join(
            [
                "[zz +  1.000] mem stage stage1: in_use 100, peak 100, pool 1,"
                " live 100 in 0 buffers",
                "[zz +  1.000] mem prog commit1: in_use 200, peak 200, live 200",
                "[zz +  1.[TRACE] PilStark: <-- STARK_STEP_Q : 0.11 s",
                "100] mem prog commit2: in_use 300, peak 300, live 300",
                instance(0, "Main_n22"),
            ]
        )
        prove = mem_stages.proves(spliced)[0]
        self.assertEqual(prove.programs, ["commit1", "commit2"])
        self.assertEqual(prove.peak_program, ("commit2", 100))

    def test_a_statistic_the_allocator_does_not_keep_reads_as_none(self):
        # The `done` line in the fixture carries `-` in every total, which is
        # what a pin without the readback emits. Parsing it as 0 would report
        # an allocator holding nothing.
        stamped = "\n".join(f"[zz +  1.000] {line}" for line in self.LINES.splitlines())
        done = mem_stages.proves(stamped + "\n" + instance(0, "Main_n22"))[0].stage(
            "done"
        )
        self.assertIsNone(done.in_use)
        self.assertIsNone(done.peak)
        self.assertIsNone(done.pool)


class ProvesTest(absltest.TestCase):
    def test_a_block_is_attributed_to_the_instance_line_that_follows_it(self):
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="100", rows=[TRACE]),
                stage("done", in_use="50", peak="100"),
                instance(0, "Main_n22"),
                stage("stage1", in_use="200", peak="300", rows=[CM1]),
                stage("done", in_use="60", peak="300"),
                instance(1, "Binary_n22"),
            ]
        )
        found = mem_stages.proves(log)
        self.assertEqual(
            [(p.air, p.index) for p in found], [("Main_n22", 0), ("Binary_n22", 1)]
        )
        self.assertEqual(
            found[1].stage("stage1").rows, [("commit1/cm1_ext", 1, 2550136832)]
        )

    def test_an_instance_with_no_blocks_does_not_shift_the_later_names(self):
        # The prove that carries blocks is the second instance. A reader that
        # advanced its instance cursor only on blocks would label it with the
        # first instance's AIR -- wrong, and silently so.
        log = "\n".join(
            [
                instance(0, "Rom_n22"),
                stage("stage1", in_use="200", peak="300", rows=[CM1]),
                stage("done", in_use="60", peak="300"),
                instance(1, "Main_n22"),
            ]
        )
        self.assertEqual(
            [(p.air, p.index) for p in mem_stages.proves(log)], [("Main_n22", 1)]
        )

    def test_two_clients_interleaving_is_an_error_not_a_merged_prove(self):
        # Both proves are mid-flight, so neither block set can be attributed.
        # Folding them together would report a live set twice the truth.
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="100", rows=[TRACE]),
                stage("stage1", in_use="200", peak="200", rows=[CM1]),
                instance(0, "Main_n22"),
            ]
        )
        with self.assertRaisesRegex(ValueError, "ZZ_CLIENTS=1"):
            mem_stages.proves(log)


class PeakStageTest(absltest.TestCase):
    def test_the_peak_stage_is_where_the_monotonic_peak_last_rose(self):
        # in_use at the boundaries is largest entering `evals`, but the peak
        # rose during `quotient` -- inside it, where no boundary can see it.
        # A reader that ranked the boundaries would name the wrong stage.
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="100"),
                stage("quotient", in_use="200", peak="200"),
                stage("evals", in_use="900", peak="990"),
                stage("done", in_use="10", peak="990"),
                instance(0, "Main_n22"),
            ]
        )
        # `quotient`, not `evals`: the rise is visible at the `evals`
        # boundary but happened in the stage that just closed.
        self.assertEqual(mem_stages.proves(log)[0].peak_stage, "quotient")

    def test_a_peak_inside_the_last_stage_is_still_attributed_to_it(self):
        # The `done` boundary exists for this case: without it the rise during
        # `openings` has no later mark to be read from, and the peak would be
        # charged to `evals` or to nothing.
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="100"),
                stage("evals", in_use="200", peak="200"),
                stage("openings", in_use="300", peak="300"),
                stage("done", in_use="20", peak="800"),
                instance(0, "Main_n22"),
            ]
        )
        self.assertEqual(mem_stages.proves(log)[0].peak_stage, "openings")

    def test_no_allocator_peak_names_no_stage_rather_than_the_first(self):
        # On a pin without the readback every peak is `-`. Naming a stage
        # anyway would publish an attribution the log cannot support.
        log = "\n".join(
            [
                stage("stage1", in_use="-", peak="-", pool="-"),
                stage("done", in_use="-", peak="-", pool="-"),
                instance(0, "Main_n22"),
            ]
        )
        prove = mem_stages.proves(log)[0]
        self.assertIsNone(prove.peak_stage)
        self.assertIsNone(prove.stage("stage1").unnamed)


def prog(name, *, in_use, peak, live="0"):
    """One per-program line as the bridge writes it at ZZ_MEM_STAGES=2."""
    return f"[zz +  1.000] mem prog {name}: in_use {in_use}, peak {peak}, live {live}"


class PeakProgramTest(absltest.TestCase):
    def test_the_rise_is_charged_to_the_program_it_happened_across(self):
        # The reading is taken after each program, so a peak higher at
        # `commit2` than at `commit1` rose while commit2 ran. Charging it to
        # the mark before would name the program that had just finished.
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="100"),
                prog("commit1", in_use="200", peak="200"),
                prog("logup", in_use="300", peak="300"),
                prog("commit2", in_use="400", peak="900"),
                stage("done", in_use="10", peak="900"),
                instance(0, "Main_n22"),
            ]
        )
        self.assertEqual(mem_stages.proves(log)[0].peak_program, ("commit2", 600))

    def test_without_the_per_program_level_no_program_is_named(self):
        # At level 1 there are only boundaries. Naming one of the dozen
        # programs inside a stage from them would be a guess.
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="100"),
                stage("done", in_use="10", peak="900"),
                instance(0, "Main_n22"),
            ]
        )
        prove = mem_stages.proves(log)[0]
        self.assertIsNone(prove.peak_program)
        self.assertEqual(prove.peak_stage, "stage1")

    def test_a_rise_after_a_stages_last_program_names_no_program(self):
        # The rise falls between `logup` and the boundary -- a download, the
        # transcript, the query draw. No program ran in it.
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="100"),
                prog("logup", in_use="200", peak="200"),
                stage("quotient", in_use="300", peak="900"),
                stage("done", in_use="10", peak="900"),
                instance(0, "Main_n22"),
            ]
        )
        self.assertIsNone(mem_stages.proves(log)[0].peak_program)

    def test_a_prove_that_did_not_set_the_high_water_names_no_stage(self):
        # Every prove after the binding one reports a flat peak. Reporting its
        # first stage as "the peak stage" would put the attribution on
        # whichever AIR happened to be proved second.
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="900"),
                stage("evals", in_use="200", peak="900"),
                stage("done", in_use="10", peak="900"),
                instance(0, "Main_n22"),
            ]
        )
        prove = mem_stages.proves(log)[0]
        self.assertIsNone(prove.peak_stage)
        self.assertIn(mem_stages.NO_PEAK, mem_stages.report(prove, None, finished=True))


def spec(name, dims, dtype="uint64"):
    return {"name": name, "dtype": dtype, "dims": dims}


class HeldPastLastReaderTest(absltest.TestCase):
    # Sized like VirtualTableZisk0_n21: 2^21 rows, 88 constants, 23 cm1
    # columns. `trace` is 2^21 x 23 x 8 here, which is what tells a co-resident
    # trace of another AIR from this one's.
    MANIFEST = {
        "quotient_chunks": [524288, 524288],
        "programs": {
            "commit1": {
                "inputs": [spec("trace", [1 << 21, 23])],
                "outputs": [spec("cm1_ext", [1 << 22, 23])],
            },
            "logup": {"inputs": [spec("const_base", [1 << 21, 88])], "outputs": []},
            "quotient": {
                "inputs": [
                    spec("cm1_ext", [1 << 22, 23]),
                    spec("rows", [524288], "int32"),
                ],
                "outputs": [],
            },
        },
    }
    CONST_BASE = (1 << 21) * 88 * 8
    CM1_EXT = (1 << 22) * 23 * 8
    TRACE = (1 << 21) * 23 * 8

    def _prove(self):
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="100"),
                prog("commit1", in_use="100", peak="100"),
                prog("logup", in_use="100", peak="100"),
                stage(
                    "quotient",
                    in_use="100",
                    peak="100",
                    rows=[
                        ("upload/const_base", "1", str(self.CONST_BASE)),
                        ("commit1/cm1_ext", "1", str(self.CM1_EXT)),
                        ("const_setup/const_setup_layers_0", "1", "134217728"),
                        # The next instance's trace, on the device under the
                        # default admission while this prove runs. A different
                        # AIR, so a size this manifest never declares.
                        ("upload/trace", "1", str(1248 * (1 << 20))),
                    ],
                ),
                prog("quotient", in_use="100", peak="100"),
                stage("done", in_use="0", peak="100"),
                instance(0, "VirtualTableZisk0_n21"),
            ]
        )
        return mem_stages.proves(log)[0]

    def test_a_section_whose_last_reader_already_ran_is_reported(self):
        # `const_base` is read by logup, which ran in stage1; it is still
        # bound in quotient. That is category (a) of the attribution, and it
        # is the whole reason for reading the run's program order rather than
        # the manifest's signatures.
        prove = self._prove()
        past = mem_stages.held_past_last_reader(
            prove, prove.stage("quotient"), self.MANIFEST
        )
        self.assertEqual(past, [("upload/const_base", self.CONST_BASE, "logup")])

    def test_a_co_resident_upload_is_not_charged_to_this_prove(self):
        # The next instance's trace is 1,248 MiB and its last reader,
        # `commit1`, ran in stage1 -- so by name and order alone it looks like
        # the largest section held past its reader in the run. It is another
        # instance's, waiting for a reader rather than outliving one, and
        # charging it here would put the biggest buffer in the workload into
        # category (a) and size a fix off it.
        prove = self._prove()
        origins = [
            o
            for o, _, _ in mem_stages.held_past_last_reader(
                prove, prove.stage("quotient"), self.MANIFEST
            )
        ]
        self.assertNotIn("upload/trace", origins)

    def test_one_row_holding_this_proves_copy_and_the_next_ones_counts_one(self):
        # Two buffers under one origin at this AIR's size: the prove binds one.
        self.assertEqual(
            mem_stages.own_bytes("upload/trace", 2, 2 * self.TRACE, self.MANIFEST),
            self.TRACE,
        )

    def test_every_quotient_row_window_belongs_to_this_prove(self):
        # The manifest says how many are uploaded; counting the extras as
        # another instance's understated the prove and invented co-residency.
        window = 524288 * 4
        self.assertEqual(
            mem_stages.own_bytes("upload/rows", 2, 2 * window, self.MANIFEST),
            2 * window,
        )

    def test_a_section_this_stage_still_reads_is_not_reported(self):
        # cm1_ext is read by the quotient, which runs in this stage. Calling
        # it dead would invent a lever out of a section in use.
        prove = self._prove()
        origins = [
            o
            for o, _, _ in mem_stages.held_past_last_reader(
                prove, prove.stage("quotient"), self.MANIFEST
            )
        ]
        self.assertNotIn("commit1/cm1_ext", origins)

    def test_an_unresolved_row_is_left_out_rather_than_assumed_dead(self):
        # The setup trees' layers bind under renamed names, so no reader
        # resolves. Counting them as held past their last reader would make
        # them the largest finding on the page, from a lookup failure.
        prove = self._prove()
        origins = [
            o
            for o, _, _ in mem_stages.held_past_last_reader(
                prove, prove.stage("quotient"), self.MANIFEST
            )
        ]
        self.assertNotIn("const_setup/const_setup_layers_0", origins)

    def test_without_the_per_program_level_nothing_is_claimed(self):
        # At level 1 no mark says which stage a reader ran in. Reporting
        # anything here would be a guess with a table's authority.
        log = "\n".join(
            [
                stage(
                    "stage1",
                    in_use="100",
                    peak="100",
                    rows=[("upload/const_base", "1", str(self.CONST_BASE))],
                ),
                stage("done", in_use="0", peak="100"),
                instance(0, "Main_n22"),
            ]
        )
        prove = mem_stages.proves(log)[0]
        self.assertEqual(
            mem_stages.held_past_last_reader(
                prove, prove.stage("stage1"), self.MANIFEST
            ),
            [],
        )


class UnnamedTest(absltest.TestCase):
    def test_what_the_allocator_holds_beyond_the_registry_is_the_difference(self):
        # The term the registry cannot see: XLA's allocations inside an
        # execution. It is reported, not reconciled away.
        log = "\n".join(
            [
                stage("stage1", in_use="3000000000", peak="3000000000", rows=[CM1]),
                stage("done", in_use="0", peak="3000000000"),
                instance(0, "Main_n22"),
            ]
        )
        first = mem_stages.proves(log)[0].stage("stage1")
        self.assertEqual(first.live, 2550136832)
        self.assertEqual(first.unnamed, 3000000000 - 2550136832)


class ReadersTest(absltest.TestCase):
    MANIFEST = {
        "programs": {
            "commit1": {
                "inputs": [{"name": "trace"}],
                "outputs": [{"name": "cm1_ext"}],
            },
            "logup": {"inputs": [{"name": "trace"}], "outputs": [{"name": "cm2"}]},
            "quotient": {"inputs": [{"name": "cm1_ext"}], "outputs": [{"name": "q"}]},
            "evals": {"inputs": [{"name": "cm1_ext"}], "outputs": [{"name": "evals"}]},
            "open_const": {"inputs": [{"name": "const_layers_0"}], "outputs": []},
        }
    }

    def test_the_last_reader_is_the_last_binder_the_run_actually_ran(self):
        # Read off the run's own program order, not off the manifest's
        # ordering: the manifest is a dict of signatures and says nothing
        # about when anything ran.
        order = ["commit1", "logup", "quotient", "evals"]
        self.assertEqual(
            mem_stages.readers_of(self.MANIFEST, "commit1/cm1_ext", order),
            ["quotient", "evals"],
        )
        self.assertEqual(
            mem_stages.readers_of(self.MANIFEST, "upload/trace", order),
            ["commit1", "logup"],
        )

    def test_a_program_that_did_not_run_is_not_a_reader(self):
        # A section released before a stage the run skipped has its last
        # reader earlier than the manifest alone would say.
        self.assertEqual(
            mem_stages.readers_of(
                self.MANIFEST, "commit1/cm1_ext", ["commit1", "quotient"]
            ),
            ["quotient"],
        )

    def test_a_renamed_layer_resolves_to_nothing_rather_than_to_a_guess(self):
        # `driver::set_fixed` binds `const_setup_layers_k` as `const_layers_k`,
        # so the output name is nobody's input. Returning empty keeps the
        # rename rule in the driver instead of copied into this reader, where
        # the two would drift.
        self.assertEqual(
            mem_stages.readers_of(
                self.MANIFEST,
                "const_setup/const_setup_layers_0",
                ["const_setup", "open_const"],
            ),
            [],
        )


class TruncationTest(absltest.TestCase):
    ABORTED = "\n".join(
        [
            stage("stage1", in_use="100", peak="100", rows=[TRACE]),
            instance(0, "Main_n22"),
            "PJRT error in run: Out of memory while trying to allocate 2.50GiB",
            "exit=1",
        ]
    )

    def test_strict_refuses_an_inventory_that_stops_at_an_abort(self):
        # The blocks before the abort are complete and look it; the stages
        # after it are simply absent. Quoting such a table as a prove's live
        # set is the error this flag exists to prevent.
        self.assertEqual(mem_stages.main([self._log(), "--strict"]), 1)

    def test_a_prove_that_died_before_its_instance_line_still_reports(self):
        # The shape of a real mid-prove abort: blocks, then nothing. The
        # instance line is written after a prove returns, so the one run whose
        # inventory anyone needs to look at is the one that never writes it.
        # Dropping the tail printed "no ZZ_MEM_STAGES blocks" for exactly that
        # log.
        log = "\n".join(
            [
                stage("stage1", in_use="100", peak="100", rows=[TRACE]),
                instance(0, "Rom_n22"),
                stage("stage1", in_use="200", peak="300", rows=[CM1]),
                stage("quotient", in_use="300", peak="400", rows=[CM1]),
                "PJRT error in run: Out of memory while trying to allocate 2.50GiB",
                "exit=1",
            ]
        )
        found = mem_stages.proves(log)
        self.assertEqual(len(found), 2)
        # The AIR stays unknown: the line that would have named it never came.
        self.assertEqual(found[1].air, "?")
        self.assertEqual([s.name for s in found[1].stages], ["stage1", "quotient"])

    def test_without_strict_the_table_says_the_run_did_not_finish(self):
        prove = mem_stages.proves(self.ABORTED)[0]
        self.assertIn("DID NOT FINISH", mem_stages.report(prove, None, finished=False))
        self.assertIn("[run finished]", mem_stages.report(prove, None, finished=True))

    def _log(self):
        path = self.create_tempfile("aborted.log", content=self.ABORTED)
        return path.full_path


if __name__ == "__main__":
    absltest.main()
