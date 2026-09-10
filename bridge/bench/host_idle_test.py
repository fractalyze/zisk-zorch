"""Pins how `host_idle.py` charges a bridge leg's device idle to host phases:
the rule that only the prove holding the client explains the idle, and the two
readers' filters. The span algebra underneath it is shared with
`h2d_overlap.py` and pinned in nsys_trace_test.

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


class WithoutCallTest(absltest.TestCase):
    """Crossing the two cuts: what a phase keeps when a driver call goes.

    Sizing a bridge-side lever off the phase column alone over-states it by
    whatever the driver was doing inside that phase — which is how the leg's
    largest phase rows turned out to be graph instantiation wearing a phase's
    name."""

    def test_a_phase_that_is_all_driver_call_keeps_nothing(self):
        rows = [
            phase("host/slot_wait", 0, 100),
            phase("lev", 100, 400, parent="p"),
            phase("host/prove", 100, 1000, rid="p"),
        ]
        api = [host_idle.ApiRow("cuGraphInstantiateWithFlags", (100, 400), HOLDER)]
        rest = host_idle.without_call(
            api,
            rows,
            host_idle.turns(rows),
            [(100, 400)],
            "cuGraphInstantiateWithFlags",
        )
        self.assertEqual(rest["lev"], 0)

    def test_a_phase_the_call_never_entered_keeps_all_of_it(self):
        # The bridge's own host work is not a driver call, so a bump that
        # moves one leaves `host/fixed_install` exactly where it was.
        rows = [
            phase("host/slot_wait", 0, 100),
            phase("host/fixed_install", 100, 400),
            phase("host/prove", 400, 1000),
        ]
        api = [host_idle.ApiRow("cuGraphInstantiateWithFlags", (500, 600), HOLDER)]
        rest = host_idle.without_call(
            api,
            rows,
            host_idle.turns(rows),
            [(100, 400)],
            "cuGraphInstantiateWithFlags",
        )
        self.assertEqual(rest["host/fixed_install"], 300)

    def test_only_a_holders_calls_subtract(self):
        # Same rule the driver-call cut turns on: a queued thread's
        # instantiate runs beside the holder and explains none of its idle,
        # so subtracting it would make a lever look already spent.
        rows = [
            phase("host/slot_wait", 0, 100),
            phase("host/fixed_install", 100, 400),
            phase("host/prove", 400, 1000),
            phase("host/slot_wait", 0, 1000, tid=QUEUED),
        ]
        api = [host_idle.ApiRow("cuGraphInstantiateWithFlags", (100, 400), QUEUED)]
        rest = host_idle.without_call(
            api,
            rows,
            host_idle.turns(rows),
            [(100, 400)],
            "cuGraphInstantiateWithFlags",
        )
        self.assertEqual(rest["host/fixed_install"], 300)

    def test_a_partly_covered_phase_keeps_the_rest(self):
        rows = [
            phase("host/slot_wait", 0, 100),
            phase("constants", 100, 400, parent="p"),
            phase("host/prove", 100, 1000, rid="p"),
        ]
        api = [host_idle.ApiRow("cuGraphInstantiateWithFlags", (100, 250), HOLDER)]
        rest = host_idle.without_call(
            api,
            rows,
            host_idle.turns(rows),
            [(100, 400)],
            "cuGraphInstantiateWithFlags",
        )
        self.assertEqual(rest["constants"], 150)

    def test_the_phases_keep_the_held_idle_minus_what_the_call_took(self):
        # The two cuts are of one set of nanoseconds, so crossing them has to
        # close: whatever the call did not take is still charged to a phase.
        rows = [
            phase("host/slot_wait", 0, 100),
            phase("host/fixed_install", 100, 300),
            phase("constants", 300, 600, parent="p"),
            phase("host/prove", 300, 1000, rid="p"),
        ]
        idle = [(100, 600)]
        turns = host_idle.turns(rows)
        api = [host_idle.ApiRow("cuGraphInstantiateWithFlags", (350, 500), HOLDER)]
        took, _ = host_idle.api_idle(api, turns, idle)
        rest = host_idle.without_call(
            api, rows, turns, idle, "cuGraphInstantiateWithFlags"
        )
        held = sum(host_idle.shares(rows, idle, turns).holding.values())
        self.assertEqual(sum(rest.values()), held - took["cuGraphInstantiateWithFlags"])


class MinusCallFlagTest(absltest.TestCase):
    """The two ways `--minus-call` could answer instead of refusing.

    Both print something that looks like a result. The report already
    raises rather than guesses elsewhere — `column` on an unknown unit,
    `owner` on a memory operation — and this flag is the same shape: a
    wrong subtraction silently rescales the lever someone is about to
    build."""

    def test_it_is_refused_without_the_api_trace(self):
        # `report` returns before the driver-call cut when there is no API
        # CSV, so the flag would be dropped in silence and exit 0 — a table
        # the user asked for and did not get, with nothing said.
        with self.assertRaises(SystemExit) as cm:
            host_idle.main(
                ["", str(TRACE), str(NVTX), "--minus-call", "cuModuleLoadFatBinary"]
            )
        self.assertEqual(cm.exception.code, 2)

    def test_a_call_absent_from_the_capture_is_refused(self):
        # Subtracting a name nothing matches returns the phase column
        # unchanged, which reads as "this call is free". The driver-call
        # table above is --top-capped, so its absence there proves nothing.
        with self.assertRaisesRegex(ValueError, "no such call"):
            host_idle.main(
                ["", str(TRACE), str(NVTX), str(API), "--minus-call", "cuNoSuchThing"]
            )

    def test_a_call_that_cost_no_idle_is_answered_not_refused(self):
        # The other side of the line: a call really made, but never while
        # the device starved, is a legitimate question with the phase
        # column as its true answer — `cuModuleLoadFatBinary` under
        # ZZ_EAGER_MODULES=1 is exactly this. It must not raise.
        self.assertEqual(
            host_idle.main(
                [
                    "",
                    str(TRACE),
                    str(NVTX),
                    str(API),
                    "--minus-call",
                    "cuLaunchKernelEx",
                ]
            ),
            0,
        )


class ReadingTest(parameterized.TestCase):
    """The filters the two readers apply, against real `nsys` rows."""

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
