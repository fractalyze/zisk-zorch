"""What `mem_stages.py` must get right about a `ZZ_MEM_STAGES` log.

Every case here is a way the reader could produce a table that looks correct:
an inventory truncated at an abort, a peak attributed to the wrong stage, two
clients' blocks folded into one prove, a last reader read off the manifest
instead of off what the run actually ran. None of them would raise on their
own."""

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
        rose = mem_stages.proves(log)[0].peak_program
        self.assertEqual(rose[0], "(end of stage1)")

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


class HeldPastLastReaderTest(absltest.TestCase):
    MANIFEST = {
        "programs": {
            "commit1": {"inputs": [{"name": "trace"}], "outputs": []},
            "logup": {"inputs": [{"name": "const_base"}], "outputs": []},
            "quotient": {"inputs": [{"name": "cm1_ext"}], "outputs": []},
        }
    }

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
                        ("upload/const_base", "1", "1350565888"),
                        ("commit1/cm1_ext", "1", "771751936"),
                        ("const_setup/const_setup_layers_0", "1", "134217728"),
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
        self.assertEqual(past, [("upload/const_base", 1350565888, "logup")])

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
                    rows=[("upload/const_base", "1", "1350565888")],
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

    def test_without_strict_the_table_says_the_run_did_not_finish(self):
        prove = mem_stages.proves(self.ABORTED)[0]
        self.assertIn("DID NOT FINISH", mem_stages.report(prove, None, finished=False))
        self.assertIn("[run finished]", mem_stages.report(prove, None, finished=True))

    def _log(self):
        path = self.create_tempfile("aborted.log", content=self.ABORTED)
        return path.full_path


if __name__ == "__main__":
    absltest.main()
