//! What a client holds on the device at each stage of a prove
//! (`ZZ_MEM_STAGES`), buffer by buffer.
//!
//! The question a memory unit asks is not how large the peak is — the
//! allocator's own statistics answer that — but which buffers make it up and
//! why each is still alive. Neither the allocator nor a card-level sample can
//! say: both report totals, and a total cannot be attributed to a lifetime.
//!
//! So the bridge reports what it knows. Every device buffer it creates passes
//! through `Artifact::upload_bytes` or `Artifact::run`, and both sites hold
//! the manifest `Spec` the buffer was made from: its exact byte size, its
//! name, and — for an output — the program that produced it. Registering
//! there and deregistering in `DeviceBuf::drop` makes the live set readable
//! at any instant, with no second copy of the driver's schedule to drift
//! against the real one (the stage names come from the driver itself, and the
//! order the programs actually ran in is in the log beside these lines).
//!
//! What the registry cannot see is what XLA allocates inside an execution —
//! a fusion's scratch, an extend's output while its input is still live. That
//! is the point of logging the allocator's own `bytes_in_use` on the same
//! line: the difference between it and the registry's total is precisely the
//! part of the peak the bridge does not name, and is a term in the
//! attribution rather than an error bar.
//!
//! Off by default. The registry costs a mutex per buffer create and drop,
//! which is nothing against a prove but is not free, and the lines land in a
//! log four readers in `bench/` parse.

use std::collections::HashMap;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};

/// `ZZ_MEM_STAGES` as a level: `1` reports the live set at each stage
/// boundary, `2` adds a line after every program.
fn level() -> u32 {
    static L: OnceLock<u32> = OnceLock::new();
    *L.get_or_init(|| level_from(std::env::var("ZZ_MEM_STAGES").ok().as_deref()))
}

/// The levels on their own, so the table in the tests can state them. Read
/// like `ZZ_LOG`: a number, or `1` for any other non-empty value.
fn level_from(value: Option<&str>) -> u32 {
    match value.map(str::trim) {
        None | Some("") | Some("0") => 0,
        Some(s) => s.parse().unwrap_or(1),
    }
}

/// Whether a prove reports its live device buffers stage by stage.
pub fn enabled() -> bool {
    level() >= 1
}

/// Whether every program is followed by a line of the allocator's totals
/// (`ZZ_MEM_STAGES=2`).
///
/// A stage boundary cannot see the high-water inside a stage, and on a wide
/// AIR most of the peak is inside one: the extend's output exists beside its
/// input, and a section stays bound until the stage that removes it ends. The
/// program lines bracket each execution, so the program the allocator's peak
/// rose across is named rather than inferred.
pub fn per_program() -> bool {
    level() >= 2
}

/// The allocator's totals after one program, for the per-program level.
pub fn report_program(name: &str, t: Totals) {
    let live: usize = snapshot().iter().map(|r| r.2).sum();
    zzlog!("{}", prog_line(name, t, live));
}

/// One live device buffer: where it came from and how large it is.
struct Entry {
    origin: String,
    bytes: usize,
}

/// The live set. `None` until the first buffer is recorded, so a run with the
/// reporting off never builds the map.
static LIVE: OnceLock<Mutex<HashMap<u64, Entry>>> = OnceLock::new();
static NEXT_ID: AtomicU64 = AtomicU64::new(0);
/// Set once a buffer has been dropped while its registry entry was gone,
/// which would make every later snapshot an overcount. Reported rather than
/// panicked on: a wrong inventory must say so, but not take the prove with it.
static LEAKED: AtomicBool = AtomicBool::new(false);

fn live() -> &'static Mutex<HashMap<u64, Entry>> {
    LIVE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// A buffer's entry in the live set, removed when the buffer is freed.
/// Only `record` makes one, so there is no token around that deregisters a
/// buffer it never registered.
pub struct Tag(u64);

impl Drop for Tag {
    fn drop(&mut self) {
        let mut m = live().lock().unwrap_or_else(|p| p.into_inner());
        if m.remove(&self.0).is_none() {
            LEAKED.store(true, Ordering::Relaxed);
        }
    }
}

/// Register a device buffer of `bytes` created by `origin`, or `None` when
/// the reporting is off. `origin` is `<program>/<output>` for a program's
/// output and `upload/<input>` for an upload — the manifest names, so a
/// reader can look each one up in the artifact's manifest.
pub fn record(origin: &str, bytes: usize) -> Option<Tag> {
    if !enabled() {
        return None;
    }
    let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
    let entry = Entry { origin: origin.to_string(), bytes };
    live().lock().unwrap_or_else(|p| p.into_inner()).insert(id, entry);
    Some(Tag(id))
}

/// The live set aggregated by origin: `(origin, count, bytes)`, largest
/// first. Buffers sharing an origin are the same section made more than once
/// — the quotient's chunks, a tree's digest layers under one program — and
/// each row's `bytes` is their total.
pub fn snapshot() -> Vec<(String, usize, usize)> {
    let m = live().lock().unwrap_or_else(|p| p.into_inner());
    let mut by_origin: HashMap<&str, (usize, usize)> = HashMap::new();
    for e in m.values() {
        let row = by_origin.entry(&e.origin).or_insert((0, 0));
        row.0 += 1;
        row.1 += e.bytes;
    }
    let mut rows: Vec<(String, usize, usize)> =
        by_origin.into_iter().map(|(o, (n, b))| (o.to_string(), n, b)).collect();
    // Largest first, then by name: a snapshot is read top-down for what
    // dominates it, and ties must not reorder between two runs of the same
    // prove or a diff of two inventories is noise.
    rows.sort_by(|a, b| b.2.cmp(&a.2).then_with(|| a.0.cmp(&b.0)));
    rows
}

/// Whether a buffer has been freed whose registry entry was already gone.
/// A snapshot taken after that is an overcount, so the reporting says so
/// instead of publishing a live set that is quietly too large.
pub fn leaked() -> bool {
    LEAKED.load(Ordering::Relaxed)
}

/// The allocator's own totals at the same instant, as bytes: what is live
/// now, the high-water since the client came up, and what the allocator holds
/// from the driver to place it all in.
///
/// `peak_in_use` is monotonic, so it does double duty: the stage during which
/// it last rose is the stage the peak is in, which a boundary snapshot on its
/// own cannot say.
///
/// Every field is optional because the whole readback is — a plugin need only
/// keep `bytes_in_use`, and the platform allocator keeps nothing.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Totals {
    pub in_use: Option<i64>,
    pub peak_in_use: Option<i64>,
    pub pool: Option<i64>,
}

/// A prove's host phase, reporting the client's live device buffers each time
/// it changes. Wraps `nvtx::Phase` rather than sitting beside it so the stage
/// names a capture is cut by and the stage names an inventory is indexed by
/// cannot drift apart — there is one list of them, in `driver::prove`.
pub struct Stage<F: Fn() -> Totals> {
    phase: crate::nvtx::Phase,
    totals: F,
}

impl<F: Fn() -> Totals> Stage<F> {
    /// Open the first stage. `totals` reads the client's allocator; it is a
    /// closure because the driver owns the session and this module does not
    /// depend on PJRT.
    pub fn start(name: &str, totals: F) -> Stage<F> {
        let stage = Stage { phase: crate::nvtx::Phase::start(name), totals };
        stage.report(name);
        stage
    }

    /// Close the open stage and open `name` in its place, reporting the live
    /// set at the boundary — before the new stage's first program, so a row
    /// that appears here was made by the stage just closed.
    pub fn set(&mut self, name: &str) {
        self.phase.set(name);
        self.report(name);
    }

    /// One boundary's line and its rows, under `name` rather than
    /// `self.name` so the drop can close the last stage as `done`.
    fn report(&self, name: &str) {
        if !enabled() {
            return;
        }
        let rows = snapshot();
        zzlog!("{}", stage_line(name, (self.totals)(), &rows, leaked()));
        for (origin, n, bytes) in rows {
            zzlog!("{}", buf_line(name, &origin, n, bytes));
        }
    }
}

/// A stage boundary's line, without the log stamp.
///
/// A pure function because it is a contract with `bench/mem_stages.py`, which
/// is the only thing that reads it: the tests below assert on the exact bytes
/// and the reader's tests parse the same bytes out of
/// `bench/testdata/mem_stages_lines.txt`, so a change to the wording here
/// fails on both sides instead of silently emptying the reader.
fn stage_line(name: &str, t: Totals, rows: &[(String, usize, usize)], leaked: bool) -> String {
    let live: usize = rows.iter().map(|r| r.2).sum();
    let count: usize = rows.iter().map(|r| r.1).sum();
    format!(
        "mem stage {name}: in_use {}, peak {}, pool {}, live {live} in {count} buffers{}",
        opt(t.in_use),
        opt(t.peak_in_use),
        opt(t.pool),
        if leaked { ", INVENTORY INCOMPLETE" } else { "" },
    )
}

/// One live-set row's line, without the log stamp.
fn buf_line(stage: &str, origin: &str, count: usize, bytes: usize) -> String {
    format!("mem stage {stage} buf {origin} {count} {bytes}")
}

/// A program's line, without the log stamp.
fn prog_line(name: &str, t: Totals, live: usize) -> String {
    format!(
        "mem prog {name}: in_use {}, peak {}, live {live}",
        opt(t.in_use),
        opt(t.peak_in_use),
    )
}

/// The boundary after the last stage. Without it the last stage has no
/// boundary on its far side, so nothing in the log carries the peak as it
/// stood when the stage closed — and the peak of a prove whose high-water is
/// in its final stage would be attributed to the stage before it, or to none.
impl<F: Fn() -> Totals> Drop for Stage<F> {
    fn drop(&mut self) {
        self.report("done");
    }
}

/// A byte count the allocator does not keep, as `-`. A zero would read as
/// "it held nothing", which is a different claim.
fn opt(bytes: Option<i64>) -> String {
    bytes.map_or_else(|| "-".to_string(), |b| b.to_string())
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The registry is process-wide, so the tests that touch it run under one
    /// lock and clean up after themselves. `enabled()` is a `OnceLock` read
    /// of the environment, which a test cannot set per-case; these exercise
    /// the parts below that gate instead, through `record_for_test`.
    static TEST_LOCK: Mutex<()> = Mutex::new(());

    /// `record` without the environment gate, so the registry's behaviour is
    /// testable in a build where `ZZ_MEM_STAGES` is unset.
    fn record_for_test(origin: &str, bytes: usize) -> Tag {
        let id = NEXT_ID.fetch_add(1, Ordering::Relaxed);
        live()
            .lock()
            .unwrap()
            .insert(id, Entry { origin: origin.to_string(), bytes });
        Tag(id)
    }

    #[test]
    fn the_reporting_is_off_unless_asked_for() {
        // Off by default: the registry costs a lock per buffer and the lines
        // land in a log the readers in bench/ parse.
        assert_eq!(level_from(None), 0);
        assert_eq!(level_from(Some("")), 0);
        assert_eq!(level_from(Some("0")), 0);
        assert_eq!(level_from(Some("1")), 1);
        assert_eq!(level_from(Some("yes")), 1);
    }

    #[test]
    fn the_per_program_lines_need_asking_for_separately() {
        // One line per program is ~34 more per prove than the boundaries, and
        // a run only wants them when the question is which program the peak
        // is inside. `1` must not turn them on by accident.
        assert_eq!(level_from(Some("2")), 2);
        assert!(level_from(Some("1")) >= 1 && level_from(Some("1")) < 2);
    }

    #[test]
    fn a_buffer_leaves_the_live_set_when_it_is_freed() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let held = record_for_test("commit1/cm1_ext", 2_550_136_832);
        {
            let _temporary = record_for_test("logup/gsum", 100);
            assert_eq!(snapshot().len(), 2);
        }
        // The point of the whole instrument: a section that is gone must stop
        // being counted at the next stage boundary, or a lifetime fix would
        // be invisible in the inventory it is measured by.
        let rows = snapshot();
        assert_eq!(rows, [("commit1/cm1_ext".to_string(), 1, 2_550_136_832)]);
        drop(held);
        assert!(snapshot().is_empty());
    }

    #[test]
    fn buffers_from_one_program_are_one_row_carrying_their_total() {
        let _lock = TEST_LOCK.lock().unwrap_or_else(|p| p.into_inner());
        let _a = record_for_test("const_setup/const_layers", 64);
        let _b = record_for_test("const_setup/const_layers", 32);
        let _c = record_for_test("upload/trace", 1024);
        // Largest first, so a snapshot is read top-down for what dominates
        // it; the quotient's chunks and a tree's layers arrive as one row.
        assert_eq!(
            snapshot(),
            [
                ("upload/trace".to_string(), 1, 1024),
                ("const_setup/const_layers".to_string(), 2, 96),
            ]
        );
    }

    #[test]
    fn a_statistic_the_allocator_does_not_keep_is_not_zero() {
        // `-` rather than 0: an allocator that keeps no total has reported
        // nothing, which is a different claim from reporting none.
        assert_eq!(opt(None), "-");
        assert_eq!(opt(Some(0)), "0");
        assert_eq!(opt(Some(2_952_790_016)), "2952790016");
    }

    /// The exact bytes `bench/mem_stages.py` parses, kept in a file both
    /// sides read. Asserting a literal here and hand-writing the same shape
    /// in the reader's tests is what let the two drift apart unnoticed.
    const LINES: &str = include_str!("../bench/testdata/mem_stages_lines.txt");

    fn line(prefix: &str) -> &'static str {
        LINES
            .lines()
            .find(|l| l.starts_with(prefix))
            .expect("testdata is missing a line the reader parses")
    }

    #[test]
    fn a_stage_boundary_emits_the_line_the_reader_parses() {
        let totals = Totals { in_use: Some(8_990_000_000), peak_in_use: Some(9_400_000_000), pool: Some(15_150_000_000) };
        let rows = vec![("commit1/cm1_ext".to_string(), 1, 2_550_136_832)];
        assert_eq!(stage_line("quotient", totals, &rows, false), line("mem stage quotient:"));
        assert_eq!(buf_line("quotient", "commit1/cm1_ext", 1, 2_550_136_832), line("mem stage quotient buf"));
        assert_eq!(prog_line("commit2", totals, 7_253_000_000), line("mem prog commit2:"));
    }

    #[test]
    fn the_boundary_after_the_last_stage_is_named_done() {
        // The peak attribution rests on this one: without a mark past
        // `openings` a high-water inside the final stage is charged to the
        // stage before it, and nothing else in the log carries it.
        assert_eq!(
            stage_line("done", Totals::default(), &[], false),
            line("mem stage done:")
        );
    }

    #[test]
    fn an_incomplete_inventory_says_so_on_the_line() {
        // A snapshot taken after a buffer was freed twice is an overcount.
        // The reader keys on this text to refuse the table.
        let said = stage_line("stage1", Totals::default(), &[], true);
        assert!(said.ends_with(", INVENTORY INCOMPLETE"), "{said}");
    }
}
