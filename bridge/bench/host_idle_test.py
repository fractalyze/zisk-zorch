"""Pins how `host_idle.py` charges a bridge leg's device idle to host phases:
the span algebra underneath it, the rule that only the prove holding the
client explains the idle, and the two readers' filters.

The attribution cases are built here rather than cut from a capture — each
one is a handful of ranges saying exactly what it is about, which a real
capture cannot do without carrying thousands of rows to make one assertion.
The fixtures are the other half: a few real `nsys` rows, kept only to pin the
schema the readers parse. See testdata/README.md."""

import pathlib

from absl.testing import absltest, parameterized

from bridge.bench import host_idle

TRACE = pathlib.Path("bridge/bench/testdata/host_cuda_gpu_trace.csv")
NVTX = pathlib.Path("bridge/bench/testdata/host_nvtx_pushpop_trace.csv")
API = pathlib.Path("bridge/bench/testdata/host_cuda_api_trace.csv")

HOLDER, QUEUED, WORKER = "100", "200", "300"


def phase(name, start, end, tid=HOLDER, rid=None, parent=""):
    """One range instance, named the way the bridge opens it."""
    return host_idle.RangeRow(name, (start, end), tid, rid or f"{name}@{start}", parent)


class SpansTest(parameterized.TestCase):
    """The interval algebra the attribution is built on."""

    @parameterized.named_parameters(
        ("disjoint", [(0, 1), (2, 3)], [(0, 1), (2, 3)]),
        ("overlapping", [(0, 2), (1, 3)], [(0, 3)]),
        ("touching", [(0, 1), (1, 2)], [(0, 2)]),
        ("unsorted", [(2, 3), (0, 1)], [(0, 1), (2, 3)]),
        ("nested", [(0, 9), (3, 4)], [(0, 9)]),
    )
    def test_merge(self, spans, want):
        self.assertEqual(host_idle.merge(spans), want)

    @parameterized.named_parameters(
        ("holes", [(0, 10)], [(2, 4), (6, 8)], [(0, 2), (4, 6), (8, 10)]),
        ("all_covered", [(0, 10)], [(0, 20)], []),
        ("nothing_removed", [(0, 10)], [(20, 30)], [(0, 10)]),
        (
            "cursor_spans_several",
            [(0, 5), (10, 15)],
            [(1, 2), (11, 12)],
            [(0, 1), (2, 5), (10, 11), (12, 15)],
        ),
        ("edge_touching", [(0, 10)], [(0, 3), (7, 10)], [(3, 7)]),
    )
    def test_subtract(self, a, b, want):
        self.assertEqual(host_idle.subtract(a, b), want)

    @parameterized.named_parameters(
        ("partial", [(0, 10)], [(5, 15)], [(5, 10)]),
        ("none", [(0, 5)], [(5, 10)], []),
        ("several", [(0, 10)], [(1, 2), (3, 4)], [(1, 2), (3, 4)]),
    )
    def test_intersect(self, a, b, want):
        self.assertEqual(host_idle.intersect(a, b), want)

    def test_subtract_and_intersect_partition_the_time(self):
        # The report's reconciliation rests on this: what one cover takes
        # from another plus what it leaves is the whole of it.
        a, b = [(0, 100)], [(10, 20), (30, 90)]
        self.assertEqual(
            host_idle.covered(host_idle.intersect(a, b))
            + host_idle.covered(host_idle.subtract(a, b)),
            host_idle.covered(a),
        )


class TurnsTest(absltest.TestCase):
    """Which span of the leg each prove is answerable for."""

    def test_a_turn_runs_from_leaving_slot_wait_to_the_end_of_prove(self):
        rows = [phase("host/slot_wait", 0, 100), phase("host/prove", 110, 400)]
        self.assertEqual(host_idle.turns(rows)[HOLDER], [(100, 400)])

    def test_a_thread_that_never_proves_holds_no_turn(self):
        # proofman's proof workers only ever run `take`.
        rows = [phase("host/take/trace", 0, 100, tid=WORKER)]
        self.assertEqual(host_idle.turns(rows).get(WORKER, []), [])

    def test_two_proves_on_one_thread_pair_up_in_order(self):
        rows = [
            phase("host/slot_wait", 0, 10),
            phase("host/prove", 10, 20),
            phase("host/slot_wait", 20, 30),
            phase("host/prove", 30, 40),
        ]
        self.assertEqual(host_idle.turns(rows)[HOLDER], [(10, 20), (30, 40)])


class AttributionTest(absltest.TestCase):
    """Who the idle is charged to."""

    def test_only_the_holder_is_charged_and_the_split_is_exact(self):
        # One turn, 100-1000: the holder installs the key (100-400) and then
        # proves. A queued prove waits throughout and a proof worker unpacks
        # the next instance beside it. The device idles 100-400.
        rows = [
            phase("host/slot_wait", 0, 100),
            phase("host/fixed_install", 100, 400),
            phase("host/prove", 400, 1000),
            phase("host/slot_wait", 0, 1000, tid=QUEUED),
            phase("host/take/trace", 150, 350, tid=WORKER),
        ]
        idle = [(100, 400)]
        s = host_idle.shares(rows, idle)

        self.assertEqual(s.holding["host/fixed_install"], 300)
        # A queued prove's wait and a worker's unpack overlap the same idle
        # but cannot explain it; charging them restates the premise.
        self.assertEqual(s.holding.get("host/slot_wait", 0), 0)
        self.assertEqual(s.holding.get("host/take/trace", 0), 0)
        self.assertEqual(s.other["host/take/trace"], 200)
        # held + free is the whole idle, because a holder's phases tile its
        # turn — this is what makes the report a split, not a set of
        # overlapping upper bounds.
        self.assertEqual(sum(s.holding.values()), host_idle.covered(idle))

    def test_a_passed_turn_map_is_used_rather_than_recomputed(self):
        # The report derives the map once and hands it to both `shares` and
        # `api_idle`; if the parameter were ignored the sweep would run three
        # times and, worse, a caller could not correct it.
        rows = [phase("host/slot_wait", 0, 100), phase("host/prove", 100, 900)]
        narrowed = {HOLDER: [(100, 300)]}
        s = host_idle.shares(rows, [(100, 900)], narrowed)
        self.assertEqual(sum(s.holding.values()), 200)
        # Same rows, map derived internally: the whole turn is charged.
        self.assertEqual(
            sum(host_idle.shares(rows, [(100, 900)]).holding.values()), 800
        )

    def test_idle_outside_every_turn_is_charged_to_no_one(self):
        # The handover gap: one prove has released the client and the next
        # has not taken it. Only the second idle span is inside the turn.
        rows = [phase("host/slot_wait", 0, 500), phase("host/prove", 500, 900)]
        s = host_idle.shares(rows, [(100, 200), (600, 700)])
        self.assertEqual(sum(s.holding.values()), 100)

    def test_a_phase_is_not_charged_for_the_programs_nested_in_it(self):
        # Without the child subtraction `host/prove` would absorb the whole
        # prove and no stage would ever show anything.
        rows = [
            phase("host/slot_wait", 0, 0),
            phase("host/prove", 0, 1000, rid="outer"),
            phase("commit1", 200, 500, rid="inner", parent="outer"),
        ]
        s = host_idle.shares(rows, [(0, 1000)])
        self.assertEqual(s.holding["commit1"], 300)
        self.assertEqual(s.holding["host/prove"], 700)


class SeveralClientsTest(absltest.TestCase):
    """`ZZ_CLIENTS` defaults to 3, and the exact split assumes one."""

    def test_overlapping_turns_do_not_make_the_free_time_negative(self):
        # Two threads holding different slot mutexes at once. Both charge the
        # same idle nanoseconds, so the phase shares sum to more than the
        # idle; the union is what "client held" must report, or "client free"
        # prints negative with nothing flagging it.
        rows = [
            phase("host/slot_wait", 0, 100),
            phase("host/prove", 100, 500),
            phase("host/slot_wait", 0, 100, tid=QUEUED),
            phase("host/prove", 100, 500, tid=QUEUED),
        ]
        idle = [(100, 500)]
        s = host_idle.shares(rows, idle)
        by_tid = host_idle.turns(rows)
        inside = host_idle.covered(
            host_idle.intersect(
                idle,
                host_idle.merge([sp for spans in by_tid.values() for sp in spans]),
            )
        )
        # Charged twice over, but the union is the real occupied time.
        self.assertEqual(sum(s.holding.values()), 800)
        self.assertEqual(inside, 400)
        self.assertGreaterEqual(host_idle.covered(idle) - inside, 0)


class DriverCallTest(absltest.TestCase):
    """The second cut of the same idle: what the driver was doing."""

    def test_only_a_holders_calls_are_counted(self):
        # A module load on a queued thread runs while its prove waits; it
        # cannot be what the device is idle for.
        turns = {HOLDER: [(0, 1000)]}
        api = [
            host_idle.ApiRow("cuModuleLoadFatBinary", (100, 400), HOLDER),
            host_idle.ApiRow("cuModuleLoadFatBinary", (100, 400), QUEUED),
        ]
        idle, calls = host_idle.api_idle(api, turns, [(0, 1000)])
        self.assertEqual(idle["cuModuleLoadFatBinary"], 300)
        # The count divides the idle beside it, so it counts the same calls.
        self.assertEqual(calls["cuModuleLoadFatBinary"], 1)

    def test_a_call_that_contributed_no_idle_is_not_counted_either(self):
        # The report divides the ns by this count, so the two have to be of
        # the same population. A holder's `cuEventRecord` issued during
        # `host/upload_inputs`, before it took the slot, contributes no idle
        # — counting it would deflate the per-call cost it is quoted as.
        turns = {HOLDER: [(500, 1000)]}
        api = [
            host_idle.ApiRow("cuEventRecord", (0, 100), HOLDER),  # pre-slot
            host_idle.ApiRow("cuEventRecord", (600, 700), HOLDER),  # in the idle
        ]
        idle, calls = host_idle.api_idle(api, turns, [(600, 800)])
        self.assertEqual(calls["cuEventRecord"], 1)
        self.assertEqual(idle["cuEventRecord"], 100)

    def test_a_call_is_charged_only_where_the_device_was_actually_idle(self):
        # Half the call overlaps a running kernel, which costs the leg
        # nothing.
        turns = {HOLDER: [(0, 1000)]}
        api = [host_idle.ApiRow("cuModuleLoadFatBinary", (0, 200), HOLDER)]
        idle, _ = host_idle.api_idle(api, turns, [(100, 1000)])
        self.assertEqual(idle["cuModuleLoadFatBinary"], 100)

    def test_the_fixtures_calls_land_on_the_prove_that_made_them(self):
        rows = host_idle.read_ranges(NVTX)
        prove = next(r for r in rows if r.name == "host/prove")
        idle, calls = host_idle.api_idle(
            host_idle.read_api(API), host_idle.turns(rows), [prove.span]
        )
        # Three of the fixture's four calls are the holder's; the fourth is
        # another thread's module load.
        self.assertEqual(calls["cuModuleLoadFatBinary"], 1)
        self.assertGreater(idle["cuModuleLoadFatBinary"], idle["cuLaunchKernelEx"])


class ReadingTest(parameterized.TestCase):
    """The filters the two readers apply, against real `nsys` rows."""

    @parameterized.named_parameters(
        ("bridge_fusion", "loop_add_fusion", True),
        ("bridge_indexed", "sponge_hash_1", True),
        ("pil2_signature", "evalTwiddleFirstKernel(gl64_t *, ...)", False),
    )
    def test_owner(self, kernel, is_bridge):
        self.assertEqual(host_idle.owner_is_bridge(kernel), is_bridge)

    @parameterized.named_parameters(
        # nsys writes a default-domain range with a bare leading colon.
        ("bridge_phase", ":host/take/trace", "host/take/trace"),
        ("bridge_program", ":commit1", "commit1"),
        ("no_colon", "commit1", "commit1"),
        ("xlas_own", "TSL:XlaModule:#hlo_module=jit_fn#", None),
    )
    def test_phase_name(self, raw, want):
        self.assertEqual(host_idle.phase_name(raw), want)

    def test_only_the_bridges_compute_kernels_are_read(self):
        # The fixture holds three of the bridge's kernels beside two of
        # pil2's, two copies and a memset; counting any of those four would
        # make the leg look busy where it is not.
        self.assertLen(host_idle.read_kernels(TRACE), 3)

    def test_a_time_column_is_scaled_by_the_unit_in_its_header(self):
        # nsys picks the unit by capture length, so the reader has to scale
        # rather than assume nanoseconds.
        self.assertEqual(
            host_idle.column(["Start (us)"], "Start"), ("Start (us)", 1_000)
        )
        with self.assertRaises(ValueError):
            host_idle.column(["Start (bytes)"], "Start")

    def test_xlas_own_ranges_are_left_out(self):
        names = {row.name for row in host_idle.read_ranges(NVTX)}
        self.assertNotIn(None, names)
        self.assertFalse([n for n in names if n.startswith("TSL")])
        self.assertIn("host/fixed_install", names)

    def test_a_driver_call_keeps_its_thread(self):
        rows = host_idle.read_api(API)
        self.assertLen(rows, 4)
        self.assertLen({row.tid for row in rows}, 2)

    def test_a_range_keeps_the_thread_and_parent_the_report_needs(self):
        rows = {row.name: row for row in host_idle.read_ranges(NVTX)}
        # `take` runs on a proof worker; the prove's phases share one thread.
        self.assertNotEqual(rows["host/take/trace"].tid, rows["host/prove"].tid)
        self.assertEqual(rows["host/stage1"].parent_id, rows["host/prove"].range_id)


if __name__ == "__main__":
    absltest.main()
