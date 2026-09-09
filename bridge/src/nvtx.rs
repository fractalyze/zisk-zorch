//! NVTX ranges around the programs a prove runs and around the host phases
//! between them. With the `nvtx` feature on, `nsys` attributes every kernel
//! to the program whose range launched it, which is what
//! `bridge/bench/nvtx_programs.py` aggregates and what the per-program tables
//! in docs/bridge.md "Profiling" are made of; the host phases are what
//! `bridge/bench/host_idle.py` charges the device's idle time to.
//!
//! Off by default and off in proofman builds: the feature links the CUDA
//! toolkit's NVTX interop library (`libnvtx3interop`), which nothing else in
//! the bridge needs. With it off every range here is inert — no name is even
//! built.

/// The prefix that tells a host phase from a program. `nsys` writes a range
/// that has a domain as `<domain>:<name>` and these are in the default
/// domain, so the prefix — not a domain — is what the bench scripts key on:
/// `nvtx_programs.py` leaves phases out of its per-program table, and
/// `host_idle.py` reports them by name.
#[cfg(any(feature = "nvtx", test))]
const HOST: &str = "host/";

/// An open NVTX range, closed where it goes out of scope. Without the
/// feature it is inert, so callers need no `cfg` of their own. The private
/// field is what keeps the pop in `drop` honest: only `push` can make a
/// `Range`, so there is no value around that pops a range it never opened.
pub struct Range(());

#[cfg(feature = "nvtx")]
#[link(name = "nvtx3interop")]
extern "C" {
    fn nvtxRangePushA(message: *const std::ffi::c_char) -> i32;
    fn nvtxRangePop() -> i32;
}

#[cfg(any(feature = "nvtx", test))]
thread_local! {
    /// Ranges this thread holds open. NVTX push/pop is a per-thread stack,
    /// so the invariant `Phase` has to keep — pop the open range before
    /// pushing the next — is observable here and nowhere else.
    static DEPTH: std::cell::Cell<usize> = const { std::cell::Cell::new(0) };
}

#[cfg(test)]
thread_local! {
    /// Every name this thread pushed. Without the feature there is no
    /// profiler to read the names back from, and the `host/` prefix is a
    /// contract two bench scripts key on — `host_idle.py` finds phases by
    /// it and `nvtx_programs.py` excludes them by it — so a test build
    /// records what was pushed and the tests below assert on that rather
    /// than on a literal.
    static PUSHED: std::cell::RefCell<Vec<String>> =
        const { std::cell::RefCell::new(Vec::new()) };
}

impl Range {
    pub fn push(name: &str) -> Range {
        #[cfg(any(feature = "nvtx", test))]
        DEPTH.with(|d| d.set(d.get() + 1));
        #[cfg(test)]
        PUSHED.with(|p| p.borrow_mut().push(name.to_string()));
        #[cfg(feature = "nvtx")]
        {
            // NVTX copies the message out of the call, so a temporary buffer
            // is enough. Dropping interior NULs rather than refusing the name
            // keeps every push paired with the pop in `drop`.
            let mut message: Vec<u8> = name.bytes().filter(|b| *b != 0).collect();
            message.push(0);
            unsafe { nvtxRangePushA(message.as_ptr().cast()) };
        }
        // Nothing reads the name in a build with neither the feature nor
        // the tests; the ranges are inert.
        #[cfg(not(any(feature = "nvtx", test)))]
        let _ = name;
        Range(())
    }

    /// A range for one of the bridge's host phases, under the `host/` prefix
    /// the bench scripts key on.
    pub fn host(name: &str) -> Range {
        #[cfg(any(feature = "nvtx", test))]
        let name = &format!("{HOST}{name}");
        Range::push(name)
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        #[cfg(any(feature = "nvtx", test))]
        DEPTH.with(|d| d.set(d.get() - 1));
        #[cfg(feature = "nvtx")]
        unsafe {
            nvtxRangePop();
        }
    }
}

/// The host phase a thread is in, as a range it replaces rather than nests:
/// a prove's phases are consecutive spans of one thread's time, so `nsys`
/// reports them as siblings and their idle shares add up instead of
/// containing one another. Dropping it closes the last phase.
pub struct Phase(Option<Range>);

impl Phase {
    pub fn start(name: &str) -> Phase {
        Phase(Some(Range::host(name)))
    }

    /// Close the open phase and open `name` in its place. The two steps are
    /// in this order because NVTX pops a thread's *innermost* range: pushing
    /// first would make `name` the range the pop closes, and every phase
    /// after it would be charged to the one before.
    pub fn set(&mut self, name: &str) {
        self.0 = None;
        self.0 = Some(Range::host(name));
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn depth() -> usize {
        DEPTH.with(|d| d.get())
    }

    /// The names this thread pushed. Each `#[test]` runs on a thread of its
    /// own, so the log starts empty.
    fn pushed() -> Vec<String> {
        PUSHED.with(|p| p.borrow().clone())
    }

    #[test]
    fn a_phase_replaces_its_predecessor_rather_than_nesting_inside_it() {
        assert_eq!(depth(), 0);
        let mut phase = Phase::start("take");
        assert_eq!(depth(), 1);
        phase.set("upload_inputs");
        // 2 here would mean the push ran before the pop, so `nsys` would
        // close "upload_inputs" where "take" was meant to end and charge
        // every later phase to its predecessor.
        assert_eq!(depth(), 1, "set must pop the open phase before pushing the next");
        drop(phase);
        assert_eq!(depth(), 0);
    }

    #[test]
    fn a_program_range_nests_inside_the_phase_that_runs_it() {
        let _phase = Phase::start("prove");
        let inner = Range::push("commit1");
        assert_eq!(depth(), 2);
        drop(inner);
        assert_eq!(depth(), 1);
    }

    #[test]
    fn a_phase_is_pushed_under_the_prefix_the_bench_scripts_key_on() {
        // Not an assertion about `HOST` — about what actually reaches NVTX.
        // `host_idle.py` finds phases by this prefix and `nvtx_programs.py`
        // excludes them by it, so dropping the prefixing would silently make
        // the first find nothing and the second double-count every kernel.
        let mut phase = Phase::start("take/trace");
        phase.set("fixed_install");
        drop(phase);
        assert_eq!(pushed(), ["host/take/trace", "host/fixed_install"]);
    }

    #[test]
    fn a_program_range_is_pushed_under_its_bare_name() {
        // The other half of the same contract: `nvtx_programs.py` counts
        // these, so a stray prefix here would empty its table.
        let _r = Range::push("commit1");
        assert_eq!(pushed(), ["commit1"]);
    }
}
