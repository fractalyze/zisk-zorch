//! NVTX ranges around the programs a prove runs. With the `nvtx` feature on,
//! `nsys` attributes every kernel to the program whose range launched it,
//! which is what `bridge/bench/nvtx_programs.py` aggregates and what the
//! per-program tables in docs/bridge.md "Profiling" are made of.
//!
//! Off by default and off in proofman builds: the feature links the CUDA
//! toolkit's NVTX interop library (`libnvtx3interop`), which nothing else in
//! the bridge needs.

/// An open NVTX range, closed where it goes out of scope. Without the
/// feature it is inert, so callers need no `cfg` of their own.
pub struct Range;

#[cfg(feature = "nvtx")]
#[link(name = "nvtx3interop")]
extern "C" {
    fn nvtxRangePushA(message: *const std::ffi::c_char) -> i32;
    fn nvtxRangePop() -> i32;
}

impl Range {
    #[cfg(feature = "nvtx")]
    pub fn push(name: &str) -> Range {
        // NVTX copies the message out of the call, so a temporary buffer is
        // enough. Dropping interior NULs rather than refusing the name keeps
        // every push paired with the pop in `drop`.
        let mut message: Vec<u8> = name.bytes().filter(|b| *b != 0).collect();
        message.push(0);
        unsafe { nvtxRangePushA(message.as_ptr().cast()) };
        Range
    }

    #[cfg(not(feature = "nvtx"))]
    pub fn push(_name: &str) -> Range {
        Range
    }
}

impl Drop for Range {
    fn drop(&mut self) {
        #[cfg(feature = "nvtx")]
        unsafe {
            nvtxRangePop();
        }
    }
}
