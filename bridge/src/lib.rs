//! pil2-proofman's `gen_proof` over zisk-zorch's exported artifacts: the
//! host half of genProof (transcript, challenges, query draw, the flat
//! proof), called from proofman's `gen_proof` wrapper in place of
//! `gen_proof_c`. Design, environment variables, gates and numbers:
//! `docs/bridge.md`.

/// Seconds since the bridge came up, for the `[zz +t]` log stamps.
pub fn uptime() -> f64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_secs_f64()
}

/// `ZZ_LOG` as a level: a number, 0 when unset or empty, 1 for any other
/// value (so `ZZ_LOG=1` and `ZZ_LOG=true` agree, and `ZZ_LOG=10` is not
/// below `2` the way a string comparison had it).
pub fn log_level() -> u32 {
    static L: OnceLock<u32> = OnceLock::new();
    *L.get_or_init(|| parse_log_level(std::env::var("ZZ_LOG").ok().as_deref()))
}

fn parse_log_level(v: Option<&str>) -> u32 {
    match v.map(str::trim) {
        None | Some("") | Some("0") => 0,
        Some(s) => s.parse().unwrap_or(1),
    }
}

/// `eprintln!` with the bridge's `[zz +seconds]` stamp, so the lines of
/// concurrent loads and proves can be ordered against proofman's own log.
macro_rules! zzlog {
    ($($arg:tt)*) => {
        eprintln!("[zz +{:7.3}] {}", $crate::uptime(), format!($($arg)*))
    };
}

pub mod ab;
pub mod artifact;
pub mod driver;
pub mod manifest;
pub mod nvtx;
pub mod transcript;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, OnceLock};
use std::time::Instant;


pub use driver::{FixedSections, InstanceInputs, ProveOutputs};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// One PJRT client's per-AIR drivers; proves on it run one at a time.
pub struct Slot {
    drivers: HashMap<String, driver::AirDriver>,
    /// Last use per AIR, for evicting resident fixed sections.
    last_used: HashMap<String, u64>,
    tick: u64,
}

/// Compiled artifacts per slot, filled by `preload` threads while proofman
/// is still computing witnesses, so a prove finds its programs ready.
struct Loaded {
    ready: HashMap<String, Arc<artifact::Artifact>>,
    /// Keys a preload thread is loading right now; a prove for one of these
    /// waits on the condvar instead of loading a second copy.
    in_flight: std::collections::HashSet<String>,
}

/// pil2's `StepsParams` inputs copied to owned words (unpacked, reduced),
/// so a prove can outlive `gen_proof` (proofman frees the instance when
/// it returns).
pub struct OwnedRequest {
    pub key: String,
    pub const_pols_path: String,
    pub custom_fixed_path: Option<String>,
    pub stream_id: Option<usize>,
    pub instance_id: u64,
    pub trace: Vec<u64>,
    pub publics: Vec<u64>,
    pub airvalues: Vec<u64>,
    pub proofvalues: Vec<u64>,
    pub global_challenge: Vec<u64>,
    pub proof_words: usize,
}

/// pil2's `StepsParams` host pointers, as canonical u64 words. Their
/// lengths follow from the artifact's manifest (the trace is
/// `2^nBits x cm1 width`, the value sections their packed widths), which
/// is why the bridge takes pointers and slices them itself.
#[derive(Clone, Copy)]
pub struct InputPtrs {
    pub trace: *const u64,
    pub publics: *const u64,
    pub airvalues: *const u64,
    pub proofvalues: *const u64,
    pub global_challenge: *const u64,
}

/// What a prove needs beyond the instance: which artifact, and where the
/// key's fixed sections are on disk (read once per slot and AIR).
pub struct ProveRequest<'a> {
    pub air: &'a str,
    pub n_bits: u32,
    /// pil2's `.const` (or `.const_gpu`, from which `.const` is derived).
    pub const_pols_path: &'a str,
    /// The custom-commit fixed file, when the AIR has one.
    pub custom_fixed_path: Option<&'a str>,
    pub inputs: InputPtrs,
    /// proofman's stream for a streamed instance; pins the slot.
    pub stream_id: Option<usize>,
    pub instance_id: u64,
    /// The trace's packing, when the witness library bit-packs rows:
    /// `(words per row, bits per column)`. pil2 unpacks on the device; the
    /// bridge unpacks on the host before uploading.
    pub packed: Option<(usize, &'a [u64])>,
}

/// Rows split across the host's cores: `f(first_row, rows_out)` fills the
/// output words of a contiguous row range (`row_words` words per row).
fn par_rows<F>(n_rows: usize, row_words: usize, f: F) -> Vec<u64>
where
    F: Fn(usize, &mut [u64]) + Sync,
{
    let mut out = vec![0u64; n_rows * row_words];
    if n_rows == 0 || row_words == 0 {
        return out;
    }
    let threads = host_threads();
    let rows_per = n_rows.div_ceil(threads).max(1);
    let results: Vec<std::thread::Result<()>> = std::thread::scope(|scope| {
        let handles: Vec<_> = out
            .chunks_mut(rows_per * row_words)
            .enumerate()
            .map(|(i, chunk)| {
                let f = &f;
                scope.spawn(move || f(i * rows_per, chunk))
            })
            .collect();
        handles.into_iter().map(|h| h.join()).collect()
    });
    // A worker's panic carries its message (an I/O fault names its file and
    // offset); re-raise that, not the scope's generic "a scoped thread
    // panicked", so the caller's catch_unwind reports the cause.
    for r in results {
        if let Err(payload) = r {
            std::panic::resume_unwind(payload);
        }
    }
    out
}

/// Proves admitted to a client's queue at once (two per client: one
/// proving, one with its uploads done ahead), so the instances proofman
/// has handed over do not all stage their traces on the device together.
/// This never blocks proofman's worker: `take` copies and returns, and the
/// prove thread waits here. Host-side backpressure is proofman's own
/// trace pool, which the fork's `gen_proof` keeps an instance's buffer in
/// until the bridge's completion callback (pil2's contract: the worker
/// returns at once, witness generation blocks on the pool).
#[derive(Default)]
pub struct Pending {
    count: Mutex<usize>,
    cv: std::sync::Condvar,
}

pub struct PendingPass(Arc<Pending>);

impl Pending {
    /// Wait until fewer than `cap` instances are pending, then count one in.
    pub fn acquire(self: &Arc<Self>, cap: usize) -> PendingPass {
        let mut n = self.count.lock().unwrap_or_else(|p| p.into_inner());
        while *n >= cap {
            n = self.cv.wait(n).unwrap_or_else(|p| p.into_inner());
        }
        *n += 1;
        PendingPass(self.clone())
    }

    /// Count one in when fewer than `cap` are pending, or `None` at once:
    /// for a caller that has a slower path of its own rather than a reason
    /// to wait (the fixed sections' read-ahead below).
    pub fn try_acquire(self: &Arc<Self>, cap: usize) -> Option<PendingPass> {
        let mut n = self.count.lock().unwrap_or_else(|p| p.into_inner());
        if *n >= cap {
            return None;
        }
        *n += 1;
        Some(PendingPass(self.clone()))
    }

    pub fn pending(&self) -> usize {
        *self.count.lock().unwrap_or_else(|p| p.into_inner())
    }
}

impl Drop for PendingPass {
    fn drop(&mut self) {
        *self.0.count.lock().unwrap_or_else(|p| p.into_inner()) -= 1;
        self.0.cv.notify_all();
    }
}

/// Threads for the host-side copies (`ZZ_HOST_THREADS`; default half the
/// cores, at most 8). They are memory-bound, so more buys little, and
/// they run beside proofman's own pools: taking every core stalls its
/// recursion witnesses, which synchronize their threads at barriers and
/// fall over under oversubscription (a 50 ms witness took 3 s).
fn host_threads() -> usize {
    static N: OnceLock<usize> = OnceLock::new();
    *N.get_or_init(|| {
        let cores = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(2);
        std::env::var("ZZ_HOST_THREADS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or((cores / 2).clamp(1, 8))
    })
}

/// A word reduced into `[0, p)`: pil2 reads a raw machine word as a residue.
#[inline]
fn reduce(w: u64) -> u64 {
    if w >= GOLDILOCKS_P {
        w - GOLDILOCKS_P
    } else {
        w
    }
}

/// One packed row (`src`, `words_per_row` words) widened into `out`, one
/// reduced word per column.
fn unpack_row(src: &[u64], bits: &[u64], out: &mut [u64]) {
    let (mut word_idx, mut bit_offset) = (0usize, 0u64);
    let mut word = src[0];
    for (c, nbits) in bits.iter().enumerate() {
        let nbits = *nbits;
        let bits_left = 64 - bit_offset;
        let val;
        if nbits <= bits_left {
            let mask = if nbits == 64 { !0u64 } else { (1u64 << nbits) - 1 };
            val = (word >> bit_offset) & mask;
            bit_offset += nbits;
            if bit_offset == 64 && word_idx + 1 < src.len() {
                word_idx += 1;
                word = src[word_idx];
                bit_offset = 0;
            }
        } else {
            let low = word >> bit_offset;
            word_idx += 1;
            word = src[word_idx];
            let high = word & ((1u64 << (nbits - bits_left)) - 1);
            val = (high << bits_left) | low;
            bit_offset = nbits - bits_left;
        }
        out[c] = reduce(val);
    }
}

/// pil2's `unpack_trace`: bit-fields packed little-endian across
/// `words_per_row` words per row, in column order, widened to one reduced
/// word per column. Rows are split across the host's cores: a Main trace
/// is over a gigabyte and this runs on proofman's proof worker.
pub fn unpack_trace(packed: &[u64], n_rows: usize, words_per_row: usize, bits: &[u64]) -> Vec<u64> {
    let n_cols = bits.len();
    par_rows(n_rows, n_cols, |first_row, out| {
        for (i, row_out) in out.chunks_mut(n_cols).enumerate() {
            let row = first_row + i;
            unpack_row(&packed[row * words_per_row..(row + 1) * words_per_row], bits, row_out);
        }
    })
}

/// `words` copied with every word reduced into `[0, p)`, across the host's
/// cores; the unpacked-trace path for AIRs whose witness is not bit-packed.
pub fn copy_canonical(words: &[u64]) -> Vec<u64> {
    const CHUNK: usize = 1 << 16;
    if words.len() <= CHUNK {
        return words.iter().map(|w| reduce(*w)).collect();
    }
    let n_chunks = words.len().div_ceil(CHUNK);
    let mut out = par_rows(n_chunks, CHUNK, |first, out| {
        let src = &words[first * CHUNK..];
        for (dst, w) in out.iter_mut().zip(src) {
            *dst = reduce(*w);
        }
    });
    out.truncate(words.len());
    out
}

/// How many AIRs' fixed sections may be on the device beyond the running
/// prove's. One: the read-ahead then costs the card a single key's worth
/// of memory however many proves queue up behind the slot.
const FIXED_AHEAD: usize = 1;

/// One client's read-ahead schedule for the fixed sections: which AIRs' are
/// on the device, and the permit that bounds how far ahead of the running
/// prove the next AIR's may go.
///
/// Residency is mirrored here rather than read off the slot because the
/// slot's lock is held for the whole of the running prove: a `try_lock`
/// there answers "not resident" in exactly the case the read-ahead exists
/// for, which would send every prove to re-read a key already on the card.
/// Both writers hold the slot lock, so the mirror only ever trails by the
/// window between a prove's glance and its own turn — and a stale glance
/// costs a read, never a wrong proof.
#[derive(Default)]
struct FixedAhead {
    resident: Mutex<std::collections::HashSet<String>>,
    permit: Arc<Pending>,
}

/// What a prove does with its AIR's fixed sections on the way to the slot.
enum AheadPlan {
    /// On the device already: read nothing, upload nothing.
    Resident,
    /// Read the key and upload it before the slot — this prove holds the
    /// client's read-ahead permit until the sections are installed.
    ReadAndUpload(PendingPass),
    /// Read the key, but leave the upload to the slot: another AIR's
    /// sections are already in flight beyond the running prove.
    ReadOnly,
}

impl FixedAhead {
    fn plan(&self, key: &str) -> AheadPlan {
        if self.resident.lock().unwrap_or_else(|p| p.into_inner()).contains(key) {
            return AheadPlan::Resident;
        }
        match self.permit.try_acquire(FIXED_AHEAD) {
            Some(pass) => AheadPlan::ReadAndUpload(pass),
            None => AheadPlan::ReadOnly,
        }
    }

    /// The sections are the running prove's now. The permit goes back here
    /// rather than when that prove ends, so the next prove reads its own
    /// ahead instead of waiting out a whole prove for a transfer it could
    /// have started.
    fn installed(&self, key: &str, permit: Option<PendingPass>) {
        drop(permit);
        self.resident.lock().unwrap_or_else(|p| p.into_inner()).insert(key.to_string());
    }

    fn evicted(&self, key: &str) {
        self.resident.lock().unwrap_or_else(|p| p.into_inner()).remove(key);
    }
}

/// An AIR's fixed sections on their way to a prove: the key's words, and
/// the device buffers when this prove got the client's one read-ahead.
struct AheadFixed {
    const_base: Vec<u64>,
    /// commitId -> the custom commit's base section
    customs: Vec<(usize, Vec<u64>)>,
    /// On the device already, or `None` for a prove that found the
    /// read-ahead taken and must upload under the slot.
    uploaded: Option<driver::UploadedFixed>,
    /// Held until the sections are installed on the driver.
    permit: Option<PendingPass>,
    /// Set when this prove held the permit and its own read-ahead upload did
    /// not go through, which is a different reason for uploading under the
    /// slot than never having had the permit.
    gave_way: bool,
    read_s: f64,
    upload_s: f64,
}

impl AheadFixed {
    /// The key's words, read from disk; nothing uploaded, no permit taken.
    fn read(m: &manifest::Manifest, req: &OwnedRequest) -> Result<AheadFixed, Error> {
        let t = Instant::now();
        let (const_base, customs) = read_fixed(m, &req.const_pols_path, req.custom_fixed_path.as_deref())?;
        Ok(AheadFixed {
            const_base,
            customs,
            uploaded: None,
            permit: None,
            gave_way: false,
            read_s: t.elapsed().as_secs_f64(),
            upload_s: 0.0,
        })
    }

    /// The key's words read *under* the slot, because the sections looked
    /// resident on the way in and were evicted before this prove landed.
    /// Nothing happened ahead, so the ahead timings stay zero: the caller's
    /// `slot_s` is what accounts for this read.
    fn read_under_slot(m: &manifest::Manifest, req: &OwnedRequest) -> Result<AheadFixed, Error> {
        let mut ahead = AheadFixed::read(m, req)?;
        ahead.read_s = 0.0;
        Ok(ahead)
    }

    /// Record a read-ahead upload's outcome, returning why it failed.
    ///
    /// On failure the permit goes back at once — nothing of this AIR's is on
    /// the card, so the next prove may still read ahead — and `uploaded`
    /// stays `None`, which is what sends the same words up under the slot.
    fn took_upload(&mut self, outcome: Result<driver::UploadedFixed, String>, took: f64) -> Option<String> {
        match outcome {
            Ok(uploaded) => {
                self.uploaded = Some(uploaded);
                self.upload_s = took;
                None
            }
            Err(why) => {
                self.permit = None;
                self.gave_way = true;
                Some(why)
            }
        }
    }

    fn sections(&self) -> FixedSections<'_> {
        FixedSections {
            const_base: &self.const_base,
            custom_base: self.customs.iter().map(|(id, w)| (*id, w.as_slice())).collect(),
            uploaded: self.uploaded.clone(),
        }
    }
}

/// Why an AIR's fixed sections went up where they did, for the `ZZ_LOG=2`
/// trace. Four ways in, and the arms are worth keeping distinct: a prove that
/// never held the permit and one whose own upload gave way both end up
/// uploading under the slot, but only the first is waiting on another AIR.
fn ahead_trace_note(uploaded_ahead: bool, looked_resident: bool, gave_way: bool) -> &'static str {
    match (uploaded_ahead, looked_resident, gave_way) {
        (true, _, _) => "",
        (_, true, _) => " (all under the slot: the sections were resident when this prove looked)",
        (_, _, true) => " (uploaded under the slot: this prove's read-ahead upload gave way)",
        _ => " (uploaded under the slot: another AIR's were already in flight)",
    }
}

/// Run a read-ahead upload, reporting **any** failure rather than raising it.
///
/// This upload is speculative: it runs while the prove ahead of this one
/// holds the client and is at its memory peak, so it is the allocation most
/// likely to fail and the one least worth failing a prove for. The same
/// words go up under the slot instead, once that prove's working set is
/// released — docs/bridge.md records a card with room for one family's
/// constants and not two.
///
/// Catching the unwind is what makes that fallback real. A full card reaches
/// us as a panic, not an error: xla-pjrt's `check` ends in `panic!("PJRT
/// error in {ctx}: {msg}")` and `Artifact::upload_bytes` hands back a `Buf`
/// rather than a `Result`, so the `Err` arm carries only our own spec
/// mismatches — which the slot's upload raises again rather than swallowing.
/// The panic starts in xla-pjrt's Rust after the FFI call has returned, so
/// nothing unwinds across the C boundary, and buffers uploaded before the
/// failing one drop normally on the way out. `prove_async` catches PJRT the
/// same way.
fn upload_ahead(
    upload: impl FnOnce() -> Result<driver::UploadedFixed, Error>,
) -> Result<driver::UploadedFixed, String> {
    match std::panic::catch_unwind(std::panic::AssertUnwindSafe(upload)) {
        Ok(Ok(uploaded)) => Ok(uploaded),
        Ok(Err(e)) => Err(e.to_string()),
        Err(p) => Err(p
            .downcast_ref::<String>()
            .cloned()
            .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
            .unwrap_or_else(|| "upload panicked".to_string())),
    }
}

pub struct Bridge {
    artifacts: PathBuf,
    cache: PathBuf,
    resident_airs: usize,
    /// One client per slot, reachable without the slot lock so a loader
    /// never waits on a prove's slot while holding the client's gate.
    clients: Vec<Arc<artifact::Client>>,
    slots: Vec<Mutex<Slot>>,
    loaded: Vec<(Mutex<Loaded>, std::sync::Condvar)>,
    preload: Mutex<PreloadQueue>,
    used: Mutex<std::collections::BTreeSet<String>>,
    next: AtomicUsize,
    /// Device-side admission, per client (see `Pending`).
    pending: Vec<Arc<Pending>>,
    /// The fixed sections' read-ahead schedule, per client (see `FixedAhead`).
    fixed_ahead: Vec<FixedAhead>,
    log: bool,
}

/// The preload work list and the count of workers draining it, kept under
/// one lock so a worker's "queue empty, I am leaving" and a caller's "queue
/// non-empty, enough workers running" cannot interleave: with the two
/// apart, an enqueue landing between a worker's empty pop and its exit saw
/// that worker as alive, spawned nothing, and the work sat there forever.
#[derive(Default)]
pub struct PreloadQueue {
    work: std::collections::VecDeque<(usize, String, bool)>,
    workers: usize,
}

impl PreloadQueue {
    /// Queue `work` (at the front when `first`) and return how many workers
    /// to spawn so `threads` are running; the count is taken now.
    pub fn push(&mut self, work: Vec<(usize, String, bool)>, first: bool, threads: usize) -> usize {
        if first {
            for w in work.into_iter().rev() {
                self.work.push_front(w);
            }
        } else {
            self.work.extend(work);
        }
        let spawn = threads.max(1).saturating_sub(self.workers);
        self.workers += spawn;
        spawn
    }

    /// The next item for a worker; `None` retires the worker in the same
    /// step, so a later `push` counts it gone.
    pub fn take(&mut self) -> Option<(usize, String, bool)> {
        let next = self.work.pop_front();
        if next.is_none() {
            self.workers -= 1;
        }
        next
    }

    pub fn workers(&self) -> usize {
        self.workers
    }
}

static BRIDGE: OnceLock<Option<Arc<Bridge>>> = OnceLock::new();

/// Per-AIR trace packing, keyed by (airgroup id, air id): words per row and
/// bits per column. proofman registers it once from its options; a prove
/// looks its AIR up.
static PACKED_INFO: OnceLock<HashMap<(usize, usize), (usize, Vec<u64>)>> = OnceLock::new();

pub fn set_packed_info(info: HashMap<(usize, usize), (usize, Vec<u64>)>) {
    let _ = PACKED_INFO.set(info);
}

pub fn packed_info_for(airgroup_id: usize, air_id: usize) -> Option<(usize, Vec<u64>)> {
    PACKED_INFO.get().and_then(|m| m.get(&(airgroup_id, air_id)).cloned())
}

impl Bridge {
    /// The process's bridge, created on first call; `None` when
    /// `ZZ_ARTIFACTS` is unset (pil2's own gen_proof runs). `n_streams` is
    /// the client count unless `ZZ_CLIENTS` overrides it.
    pub fn global(n_streams: usize) -> Option<Arc<Bridge>> {
        BRIDGE
            .get_or_init(|| {
                let dir = std::env::var("ZZ_ARTIFACTS").ok().filter(|s| !s.is_empty())?;
                let clients = std::env::var("ZZ_CLIENTS")
                    .ok()
                    .and_then(|s| s.parse::<usize>().ok())
                    .filter(|n| *n > 0)
                    .unwrap_or(n_streams.max(1));
                let bridge = Arc::new(Bridge::new(Path::new(&dir), clients));
                // Loads start now, under proofman's own initialization: by
                // default the AIRs the previous run used (a workload's AIR set
                // repeats), with ZZ_PRELOAD=all every AIR of the key, with
                // ZZ_PRELOAD=0 none until proofman's instance list arrives.
                match std::env::var("ZZ_PRELOAD").as_deref() {
                    Ok("0") => {}
                    Ok("all") => bridge.preload(bridge.all_keys()),
                    _ => bridge.preload_with(bridge.last_used(), true),
                }
                Some(bridge)
            })
            .clone()
    }

    /// The AIRs the previous run on these artifacts proved (`.last-used`).
    pub fn last_used(&self) -> Vec<String> {
        std::fs::read_to_string(self.artifacts.join(".last-used"))
            .map(|t| t.lines().filter(|l| !l.is_empty()).map(|l| l.to_string()).collect())
            .unwrap_or_default()
    }

    /// Record `key` as used by this run (for the next run's preload).
    fn note_used(&self, key: &str) {
        let mut used = self.used.lock().unwrap();
        if used.insert(key.to_string()) {
            let text: String = used.iter().map(|k| format!("{k}\n")).collect();
            let _ = std::fs::write(self.artifacts.join(".last-used"), text);
        }
    }

    /// Every `<Air>_n<nBits>` directory under the artifacts root.
    pub fn all_keys(&self) -> Vec<String> {
        let mut keys: Vec<String> = std::fs::read_dir(&self.artifacts)
            .map(|rd| {
                rd.filter_map(|e| e.ok())
                    .filter(|e| e.path().join("manifest.json").exists())
                    .map(|e| e.file_name().to_string_lossy().into_owned())
                    .collect()
            })
            .unwrap_or_default();
        keys.sort();
        keys
    }

    pub fn new(artifacts: &Path, clients: usize) -> Bridge {
        uptime();
        let log = log_level() >= 1;
        let fraction = match std::env::var("ZZ_MEMORY_FRACTION").ok().filter(|s| !s.is_empty()) {
            None => None,
            Some(text) => match text.parse::<f32>().ok().filter(|f| *f > 0.0 && *f < 1.0) {
                Some(f) => Some(f / clients as f32),
                None => {
                    // Silently allocating on demand here would let pil2 take
                    // the card first, the very thing the variable prevents.
                    eprintln!("[zz] ZZ_MEMORY_FRACTION={text:?} is not a fraction in (0, 1); the clients allocate on demand");
                    None
                }
            },
        };
        let client_handles: Vec<Arc<artifact::Client>> =
            (0..clients).map(|_| artifact::new_client(fraction)).collect();
        let slots = (0..clients)
            .map(|_| Mutex::new(Slot { drivers: HashMap::new(), last_used: HashMap::new(), tick: 0 }))
            .collect();
        if log {
            zzlog!(
                "bridge up: {clients} PJRT client(s){}, artifacts {}{}",
                match fraction {
                    Some(f) => format!(" each holding {:.0}% of the card", f * 100.0),
                    None => String::new(),
                },
                artifacts.display(),
                if ab::enabled() { ", A/B against pil2" } else { "" }
            );
        }
        let cache = std::env::var("ZZ_COMPILE_CACHE")
            .ok()
            .filter(|s| !s.is_empty())
            .map(PathBuf::from)
            .unwrap_or_else(|| artifacts.join(".pjrt-cache"));
        let resident_airs = std::env::var("ZZ_RESIDENT_AIRS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .filter(|n| *n > 0)
            .unwrap_or(1);
        let loaded = (0..clients)
            .map(|_| {
                (
                    Mutex::new(Loaded { ready: HashMap::new(), in_flight: std::collections::HashSet::new() }),
                    std::sync::Condvar::new(),
                )
            })
            .collect();
        Bridge {
            artifacts: artifacts.to_path_buf(),
            cache,
            resident_airs,
            clients: client_handles,
            slots,
            loaded,
            preload: Mutex::new(PreloadQueue::default()),
            used: Mutex::new(std::collections::BTreeSet::new()),
            next: AtomicUsize::new(0),
            pending: (0..clients).map(|_| Arc::new(Pending::default())).collect(),
            fixed_ahead: (0..clients).map(|_| FixedAhead::default()).collect(),
            log,
        }
    }

    pub fn artifacts(&self) -> &Path {
        &self.artifacts
    }

    /// How many proves can run at once — proofman spawns this many basic
    /// proof workers when the bridge is on, so pil2's stream count (which
    /// its GPU memory dictates) does not cap the bridge's concurrency.
    pub fn clients(&self) -> usize {
        self.slots.len()
    }

    /// Load an AIR's programs on slot `slot_idx` (from the cache when
    /// possible) unless another thread already has or is doing so.
    fn artifact(&self, slot_idx: usize, key: &str) -> Result<Arc<artifact::Artifact>, Error> {
        let (lock, cv) = &self.loaded[slot_idx];
        {
            let mut l = lock.lock().unwrap();
            loop {
                if let Some(a) = l.ready.get(key) {
                    return Ok(a.clone());
                }
                if !l.in_flight.contains(key) {
                    break;
                }
                l = cv.wait(l).unwrap();
            }
            l.in_flight.insert(key.to_string());
        }
        let client = self.clients[slot_idx].clone();
        let t = Instant::now();
        let dir = self.artifacts.join(key);
        let cache = self.cache.clone();
        // A plugin failure panics inside xla-pjrt; it must still clear the
        // in-flight mark or a prove waiting for this key waits forever.
        // (Each program takes the client's load gate on its own, inside
        // `compile_all`, so a prove waits for one program, not a whole AIR.)
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            artifact::Artifact::load(client.clone(), &dir, Some(&cache)).and_then(|a| a.compile_all(1).map(|_| a))
        }))
        .unwrap_or_else(|p| {
            Err(p
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                .unwrap_or_else(|| "load panicked".to_string())
                .into())
        });
        let mut l = lock.lock().unwrap();
        l.in_flight.remove(key);
        let art = match result {
            Ok(a) => Arc::new(a),
            Err(e) => {
                cv.notify_all();
                return Err(e);
            }
        };
        if self.log {
            zzlog!(
                "loaded {key} ({} programs, {} from the cache) in {:.1} s",
                art.manifest.programs.len(),
                art.cache_hits.load(Ordering::Relaxed),
                t.elapsed().as_secs_f64()
            );
        }
        l.ready.insert(key.to_string(), art.clone());
        cv.notify_all();
        Ok(art)
    }

    /// Start loading these AIRs' programs on every slot in the background
    /// (`ZZ_PRELOAD_THREADS`, default 6), so proves that come later find
    /// them ready. Keys are `<Air>_n<nBits>`.
    pub fn preload(self: &Arc<Self>, keys: Vec<String>) {
        self.preload_with(keys, false)
    }

    /// `preload`, and with `first` the keys go ahead of anything queued —
    /// the instance list, once proofman knows it, ahead of the rest of the
    /// key's AIRs.
    pub fn preload_with(self: &Arc<Self>, keys: Vec<String>, first: bool) {
        let work: Vec<(usize, String, bool)> =
            (0..self.slots.len()).flat_map(|s| keys.iter().map(move |k| (s, k.clone(), first))).collect();
        let threads = std::env::var("ZZ_PRELOAD_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(6usize).max(1);
        let spawn = self.preload.lock().unwrap_or_else(|p| p.into_inner()).push(work, first, threads);
        for _ in 0..spawn {
            let bridge = self.clone();
            std::thread::spawn(move || loop {
                let next = bridge.preload.lock().unwrap_or_else(|p| p.into_inner()).take();
                match next {
                    // Background keys (the whole proving key) load only until
                    // proving starts; requested keys always do.
                    Some((slot, _, false)) if bridge.clients[slot].started_proving() => continue,
                    Some((slot, key, _)) => {
                        if let Err(e) = bridge.artifact(slot, &key) {
                            zzlog!("preload {key}: {e}");
                        }
                    }
                    None => return,
                }
            });
        }
    }

    /// The `slot` index a request lands on (see `slot`).
    fn slot_index(&self, stream_id: Option<usize>, instance_id: u64) -> usize {
        let n = self.slots.len();
        match stream_id {
            Some(s) => s % n,
            None => {
                let start = self.next.fetch_add(1, Ordering::Relaxed) % n;
                for k in 0..n {
                    if self.slots[(start + k) % n].try_lock().is_ok() {
                        return (start + k) % n;
                    }
                }
                (instance_id as usize) % n
            }
        }
    }

    /// Copy the instance out of pil2's buffers (unpacked, reduced), ready to
    /// prove after `gen_proof` returns.
    ///
    /// # Safety
    /// `req.inputs` must point at buffers of the lengths the artifact's
    /// manifest implies, valid during this call.
    pub unsafe fn take(&self, req: &ProveRequest) -> Result<OwnedRequest, Error> {
        // This runs on proofman's proof worker, not on a prove thread, so its
        // phases sit beside a running prove's on the profile rather than
        // inside them (`bench/host_idle.py`).
        let mut phase = nvtx::Phase::start("take/manifest");
        let t0 = Instant::now();
        let key = format!("{}_n{}", req.air, req.n_bits);
        let m = manifest::Manifest::load(&self.artifacts.join(&key))?;
        let n = 1usize << m.n_bits;
        let p = &req.inputs;
        phase.set("take/trace");
        let trace = match req.packed {
            Some((words_per_row, bits)) => {
                if bits.len() != m.widths.cm1 {
                    return Err(format!("{key}: packing lists {} columns, cm1 has {}", bits.len(), m.widths.cm1).into());
                }
                let packed_bits: u64 = bits.iter().sum();
                if packed_bits > 64 * words_per_row as u64 {
                    return Err(format!("{key}: packing needs {packed_bits} bits per row, {words_per_row} words hold {}", 64 * words_per_row).into());
                }
                unpack_trace(std::slice::from_raw_parts(p.trace, n * words_per_row), n, words_per_row, bits)
            }
            None => copy_canonical(std::slice::from_raw_parts(p.trace, n * m.widths.cm1)),
        };
        phase.set("take/scalars");
        let proof_words = driver::AirDriver::proof_words_of(&m);
        if self.log {
            zzlog!(
                "took instance {} {key}: {} MB{} in {:.3} s",
                req.instance_id,
                (trace.len() * 8) >> 20,
                if req.packed.is_some() { " (unpacked)" } else { "" },
                t0.elapsed().as_secs_f64()
            );
        }
        Ok(OwnedRequest {
            key,
            const_pols_path: req.const_pols_path.to_string(),
            custom_fixed_path: req.custom_fixed_path.map(|s| s.to_string()),
            stream_id: req.stream_id,
            instance_id: req.instance_id,
            trace,
            publics: copy_canonical(std::slice::from_raw_parts(p.publics, m.n_publics)),
            airvalues: copy_canonical(std::slice::from_raw_parts(p.airvalues, manifest::Manifest::packed_width(&m.airvalues))),
            proofvalues: copy_canonical(std::slice::from_raw_parts(p.proofvalues, manifest::Manifest::packed_width(&m.proofvalues))),
            global_challenge: copy_canonical(std::slice::from_raw_parts(p.global_challenge, 3)),
            proof_words,
        })
    }

    /// Prove one instance; writes the flat proof into `proof_out` (which
    /// must hold the AIR's proof size) and returns the host-side products.
    ///
    /// # Safety
    /// `req.inputs` must point at buffers of the lengths the artifact's
    /// manifest implies, valid until this returns.
    pub unsafe fn prove(&self, req: &ProveRequest, proof_out: &mut [u64]) -> Result<ProveOutputs, Error> {
        let owned = self.take(req)?;
        self.prove_owned(&owned, proof_out)
    }

    /// `prove` on a thread of its own: `done` receives the flat proof (or
    /// the error) when it finishes. The bridge's client serializes proves,
    /// so this frees the caller's thread rather than the GPU.
    pub fn prove_async(self: &Arc<Self>, owned: OwnedRequest, done: Box<dyn FnOnce(Result<(Vec<u64>, ProveOutputs), String>) + Send + 'static>) {
        let bridge = self.clone();
        std::thread::spawn(move || {
            let mut proof = vec![0u64; owned.proof_words];
            // A PJRT failure surfaces as a panic inside xla-pjrt; turn it into
            // the error path so the caller can stop the process instead of
            // waiting forever on a completion that will not come.
            let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
                bridge.prove_owned(&owned, &mut proof).map_err(|e| e.to_string())
            }));
            let result = match result {
                Ok(Ok(out)) => Ok((proof, out)),
                Ok(Err(e)) => Err(e),
                Err(p) => Err(p
                    .downcast_ref::<String>()
                    .cloned()
                    .or_else(|| p.downcast_ref::<&str>().map(|s| s.to_string()))
                    .unwrap_or_else(|| "prove panicked".to_string())),
            };
            done(result);
        });
    }

    /// Carry out a `FixedAhead` plan before this prove takes the slot, so the
    /// transfers run while the prove ahead of it still has the client. A
    /// table AIR's `const_base` is 1.2-1.4 GB out of pageable host memory,
    /// and under the slot the client idles for it. `None` when the sections
    /// are already on the device and there is nothing to do.
    fn read_ahead_fixed(
        &self,
        plan: AheadPlan,
        art: &artifact::Artifact,
        req: &OwnedRequest,
    ) -> Result<Option<AheadFixed>, Error> {
        let permit = match plan {
            AheadPlan::Resident => return Ok(None),
            AheadPlan::ReadAndUpload(pass) => Some(pass),
            AheadPlan::ReadOnly => None,
        };
        let mut ahead = AheadFixed::read(&art.manifest, req)?;
        ahead.permit = permit;
        if ahead.permit.is_some() {
            let t = Instant::now();
            let uploaded = {
                let sections = ahead.sections();
                upload_ahead(|| driver::upload_fixed(art, &sections))
            };
            if let Some(why) = ahead.took_upload(uploaded, t.elapsed().as_secs_f64()) {
                if self.log {
                    zzlog!("{}: read-ahead upload gave way to the slot ({why})", req.key);
                }
            }
        }
        Ok(Some(ahead))
    }

    pub fn prove_owned(&self, req: &OwnedRequest, proof_out: &mut [u64]) -> Result<ProveOutputs, Error> {
        let mut phase = nvtx::Phase::start("artifact");
        let t0 = Instant::now();
        let key = req.key.clone();
        self.note_used(&key);
        let slot_idx = self.slot_index(req.stream_id, req.instance_id);
        let art = self.artifact(slot_idx, &key)?;
        // Two proves per client on the device at once: one running, one
        // with its uploads ahead (`ZZ_PENDING` overrides the count). Counted
        // per client: streamed instances pin their slot, so a bridge-wide
        // count would let every admission land on one client.
        let per_client = std::env::var("ZZ_PENDING").ok().and_then(|s| s.parse::<usize>().ok()).filter(|n| *n > 0).unwrap_or(2);
        phase.set("admit");
        let _admitted = self.pending[slot_idx].acquire(per_client);
        let mut inputs = InstanceInputs {
            trace: &req.trace,
            publics: &req.publics,
            airvalues: &req.airvalues,
            proofvalues: &req.proofvalues,
            global_challenge: &req.global_challenge,
            uploaded: None,
        };
        // Ahead of the slot, while another prove may have the client: the
        // instance's uploads (their transfers queue behind that prove's
        // work) and, unless the AIR's fixed sections are already on the
        // device, the key's files and their upload. A stale glance at
        // residency only costs a read.
        let t = Instant::now();
        phase.set("upload_inputs");
        let uploaded = driver::upload_inputs(&art, &inputs)?;
        let plan = self.fixed_ahead[slot_idx].plan(&key);
        phase.set("fixed_ahead");
        let prefetched = self.read_ahead_fixed(plan, &art, req)?;
        let ahead = t.elapsed().as_secs_f64();
        phase.set("slot_wait");
        let mut slot = self.slots[slot_idx].lock().unwrap_or_else(|p| p.into_inner());
        // No load may enter the plugin while this prove has work in flight.
        let _exclusive = self.clients[slot_idx].enter_prove();
        let waited = t0.elapsed().as_secs_f64();
        phase.set("resident");
        if !slot.drivers.contains_key(&key) {
            slot.drivers.insert(key.clone(), driver::AirDriver::new(art.clone()));
        }
        // Keep at most `resident_airs` AIRs' fixed sections on this client:
        // evict the least recently used before this prove needs its own.
        slot.tick += 1;
        let tick = slot.tick;
        slot.last_used.insert(key.clone(), tick);
        let resident: Vec<String> =
            slot.drivers.iter().filter(|(k, d)| d.has_fixed() && **k != key).map(|(k, _)| k.clone()).collect();
        if resident.len() + 1 > self.resident_airs {
            let mut by_age: Vec<(u64, String)> =
                resident.into_iter().map(|k| (slot.last_used.get(&k).copied().unwrap_or(0), k)).collect();
            by_age.sort();
            let evict = by_age.len() + 1 - self.resident_airs;
            for (_, k) in by_age.into_iter().take(evict) {
                slot.drivers.get_mut(&k).unwrap().drop_fixed();
                self.fixed_ahead[slot_idx].evicted(&k);
                if self.log {
                    zzlog!("released {k}'s fixed sections");
                }
            }
        }
        phase.set("fixed_install");
        let driver = slot.drivers.get_mut(&key).unwrap();
        if !driver.has_fixed() {
            let t = Instant::now();
            let looked_resident = prefetched.is_none();
            let mut ahead = match prefetched {
                Some(sections) => sections,
                None => AheadFixed::read_under_slot(&art.manifest, req)?,
            };
            let uploaded_ahead = ahead.uploaded.is_some();
            driver.set_fixed(&ahead.sections())?;
            self.fixed_ahead[slot_idx].installed(&key, ahead.permit.take());
            let slot_s = t.elapsed().as_secs_f64();
            if self.log {
                // The two columns are disjoint, so a run can sum them:
                // `slot_s` covers everything that happened here, and
                // `read_under_slot` left the ahead timings at zero for the
                // one arm whose read did.
                zzlog!(
                    "fixed sections for {key}: {slot_s:.2} s under the slot, {:.2} s ahead of it",
                    ahead.read_s + ahead.upload_s
                );
            }
            if artifact::trace_enabled() {
                let why = ahead_trace_note(uploaded_ahead, looked_resident, ahead.gave_way);
                zzlog!(
                    "  fixed sections {key} ahead: {:.2} s reading the key, {:.2} s uploading{why}",
                    ahead.read_s,
                    ahead.upload_s,
                );
            }
        } else {
            // Resident after all — the mirror trailed an install on another
            // prove. Drop the read-ahead's buffers, hand the permit back and
            // re-sync the view.
            drop(prefetched);
            self.fixed_ahead[slot_idx].installed(&key, None);
        }
        inputs.uploaded = Some(uploaded);
        if proof_out.len() != driver.proof_words() {
            return Err(format!(
                "{key}: proof buffer holds {} words, layout says {}",
                proof_out.len(),
                driver.proof_words()
            )
            .into());
        }
        let m = driver.manifest();
        if let Ok(dir) = std::env::var("ZZ_DUMP_INPUTS") {
            // The instance exactly as received, as a `zz_prove` case
            // directory (the fixed sections and `case.json` included), for
            // replaying it outside proofman.
            let d = std::path::PathBuf::from(dir).join(format!("{}_{}", req.instance_id, key));
            let _ = std::fs::create_dir_all(&d);
            let (const_base, customs) = read_fixed(m, &req.const_pols_path, req.custom_fixed_path.as_deref())?;
            let mut files: Vec<(String, &[u64])> = vec![
                ("trace".into(), inputs.trace),
                ("publics".into(), inputs.publics),
                ("airvalues".into(), inputs.airvalues),
                ("proofvalues".into(), inputs.proofvalues),
                ("global_challenge".into(), inputs.global_challenge),
                ("const_base".into(), &const_base),
            ];
            files.extend(customs.iter().map(|(id, w)| (format!("custom_base_{id}"), w.as_slice())));
            for (name, words) in files {
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                let _ = std::fs::write(d.join(format!("{name}.bin")), bytes);
            }
            let _ = std::fs::write(
                d.join("case.json"),
                format!("{{\"air\": \"{}\", \"n_bits\": {}}}\n", m.air, m.n_bits),
            );
        }
        // `prove` opens phases of its own inside this one, so what stays
        // here is only the transcript's own setup.
        phase.set("prove");
        let mut transcript = transcript::HostTranscript::new(&m.hash_family)?;
        let out = driver.prove(&inputs, &mut transcript, proof_out)?;
        if self.log {
            zzlog!(
                "instance {} {} ({}): {:.3} s, of which {:.3} s waiting for the client ({ahead:.3} s of uploads and reads done ahead)",
                req.instance_id,
                key,
                if req.stream_id.is_some() { "streamed" } else { "worker" },
                t0.elapsed().as_secs_f64(),
                waited
            );
        }
        Ok(out)
    }
}

const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

const TILE_HEIGHT: usize = 256;
const TILE_WIDTH: usize = 4;

/// pil2's device layout (`getBufferOffset`: column-major within 256x4
/// tiles, tiles of one column block contiguous down the rows) back to
/// row-major `(n_rows, n_cols)`. Row ranges go to separate cores; each
/// walks its tiles column by column so the reads stay contiguous.
pub fn tiled_to_row_major(tiled: &[u64], n_rows: usize, n_cols: usize) -> Vec<u64> {
    par_rows(n_rows, n_cols, |first_row, out| {
        let rows = out.len() / n_cols;
        for col in 0..n_cols {
            let block_y = col / TILE_WIDTH;
            let n_cols_block = (n_cols - TILE_WIDTH * block_y).min(TILE_WIDTH);
            let col_block = col % TILE_WIDTH;
            for i in 0..rows {
                let row = first_row + i;
                let block_x = row / TILE_HEIGHT;
                let row_block = row % TILE_HEIGHT;
                let src = block_y * TILE_WIDTH * n_rows + block_x * n_cols_block * TILE_HEIGHT + col_block * TILE_HEIGHT + row_block;
                out[i * n_cols + col] = tiled[src];
            }
        }
    })
}

/// `<stem>.const_gpu` (what proofman passes on GPU) -> `<stem>.const`, the
/// plain row-major section the exporter read.
pub fn plain_const_path(path: &str) -> String {
    path.strip_suffix("_gpu").unwrap_or(path).to_string()
}

/// The first `n_words` little-endian words of `path` after `skip_bytes`,
/// read straight into the word buffer (no byte-to-word pass) in parallel
/// slices. A longer file is fine: pil2's custom-commit fixed file carries
/// the extended section and its tree behind the base rows.
fn read_words(path: &str, skip_bytes: usize, n_words: usize) -> Result<Vec<u64>, Error> {
    use std::os::unix::fs::FileExt;
    let file = std::fs::File::open(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let len = file.metadata().map_err(|e| format!("cannot stat {path}: {e}"))?.len() as usize;
    let end = skip_bytes + n_words * 8;
    if len < end {
        return Err(format!("{path} is {len} bytes, expected at least {end}").into());
    }
    const SLICE: usize = 8 << 20;
    let n_slices = (n_words * 8).div_ceil(SLICE).max(1);
    let mut words = par_rows(n_slices, SLICE / 8, |first, out| {
        let offset = skip_bytes + first * SLICE;
        let want = (n_words * 8 - first * SLICE).min(out.len() * 8);
        // SAFETY: `out` is a live `[u64]`; its bytes are written, then read
        // back as little-endian words below.
        let bytes = unsafe { std::slice::from_raw_parts_mut(out.as_mut_ptr() as *mut u8, want) };
        if let Err(e) = file.read_exact_at(bytes, offset as u64) {
            // Reported by the length check on the way in; a read that fails
            // after that is an I/O fault, and zeros would prove a wrong key.
            panic!("cannot read {path} at {offset}: {e}");
        }
    });
    words.truncate(n_words);
    for w in &mut words {
        *w = u64::from_le(*w);
    }
    Ok(words)
}

/// Whether pil2 wrote the custom-commit fixed file in its tiled device
/// layout: its GPU backend does (`fromRowMajorToTiled`) and names the
/// constant file `.const_gpu`; the CPU backend writes row-major and names
/// it `.const`. The bridge serves both (proofman routes to it either way).
fn fixed_is_tiled(const_pols_path: &str) -> bool {
    const_pols_path.ends_with("_gpu")
}

/// The key's fixed sections as row-major words: the constants and each
/// custom commit's base section.
fn read_fixed(
    m: &manifest::Manifest,
    const_pols_path: &str,
    custom_fixed_path: Option<&str>,
) -> Result<(Vec<u64>, Vec<(usize, Vec<u64>)>), Error> {
    let n = 1usize << m.n_bits;
    let const_base = read_words(&plain_const_path(const_pols_path), 0, n * m.n_constants)?;
    let mut customs: Vec<(usize, Vec<u64>)> = Vec::new();
    for cc in &m.custom_commits {
        let path = custom_fixed_path
            .filter(|p| !p.is_empty())
            .ok_or_else(|| format!("{} has a custom commit but no fixed path", m.air))?;
        // pil2 skips a 32-byte Merkle-root header (one custom commit per
        // AIR); the GPU backend then copies the rest straight to the device,
        // so that file holds the section in the prover's tiled layout.
        let words = read_words(path, 32, n * cc.width)?;
        customs.push((cc.id, if fixed_is_tiled(const_pols_path) { tiled_to_row_major(&words, n, cc.width) } else { words }));
    }
    Ok((const_base, customs))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A small deterministic generator (xorshift), so the tests need no crate.
    fn rng(seed: u64) -> impl FnMut() -> u64 {
        let mut x = seed | 1;
        move || {
            x ^= x << 13;
            x ^= x >> 7;
            x ^= x << 17;
            x
        }
    }

    /// pil2's packing, written the slow way: each column's `bits` low bits
    /// appended little-endian across the row's words.
    fn pack_rows(rows: &[Vec<u64>], bits: &[u64], words_per_row: usize) -> Vec<u64> {
        let mut out = vec![0u64; rows.len() * words_per_row];
        for (r, row) in rows.iter().enumerate() {
            let dst = &mut out[r * words_per_row..(r + 1) * words_per_row];
            let mut pos = 0u64;
            for (c, nbits) in bits.iter().enumerate() {
                for b in 0..*nbits {
                    if (row[c] >> b) & 1 == 1 {
                        let p = pos + b;
                        dst[(p / 64) as usize] |= 1u64 << (p % 64);
                    }
                }
                pos += nbits;
            }
        }
        out
    }

    #[test]
    fn unpack_matches_the_packer_and_reduces() {
        let mut next = rng(7);
        let bits: Vec<u64> = vec![1, 64, 17, 3, 64, 40, 12, 64, 5, 31];
        let words_per_row = bits.iter().sum::<u64>().div_ceil(64) as usize;
        let n_rows = 1000;
        let rows: Vec<Vec<u64>> = (0..n_rows)
            .map(|r| {
                bits.iter()
                    .enumerate()
                    .map(|(c, b)| {
                        let mask = if *b == 64 { !0u64 } else { (1u64 << b) - 1 };
                        // A 64-bit column above the modulus on some rows.
                        if *b == 64 && (r + c) % 3 == 0 { GOLDILOCKS_P + (next() % 1000) } else { next() & mask }
                    })
                    .collect()
            })
            .collect();
        let packed = pack_rows(&rows, &bits, words_per_row);
        let got = unpack_trace(&packed, n_rows, words_per_row, &bits);
        let want: Vec<u64> = rows.iter().flatten().map(|w| reduce(*w)).collect();
        assert_eq!(got, want);
    }

    #[test]
    fn copy_canonical_reduces_large_and_small() {
        let small = vec![0, 1, GOLDILOCKS_P - 1, GOLDILOCKS_P, GOLDILOCKS_P + 5, u64::MAX];
        assert_eq!(copy_canonical(&small), vec![0, 1, GOLDILOCKS_P - 1, 0, 5, u64::MAX - GOLDILOCKS_P]);
        let mut next = rng(3);
        let big: Vec<u64> = (0..(3 << 16) + 17).map(|_| next()).collect();
        let want: Vec<u64> = big.iter().map(|w| reduce(*w)).collect();
        assert_eq!(copy_canonical(&big), want);
    }

    /// pil2's `getBufferOffset` for one element, the forward direction.
    fn tiled_offset(row: usize, col: usize, n_rows: usize, n_cols: usize) -> usize {
        let block_y = col / TILE_WIDTH;
        let n_cols_block = (n_cols - TILE_WIDTH * block_y).min(TILE_WIDTH);
        block_y * TILE_WIDTH * n_rows
            + (row / TILE_HEIGHT) * n_cols_block * TILE_HEIGHT
            + (col % TILE_WIDTH) * TILE_HEIGHT
            + row % TILE_HEIGHT
    }

    #[test]
    fn untile_inverts_the_device_layout() {
        for (n_rows, n_cols) in [(512, 7), (1024, 4), (256 * 40, 13)] {
            let row_major: Vec<u64> = (0..n_rows * n_cols as usize).map(|i| i as u64 * 31 + 1).collect();
            let mut tiled = vec![0u64; n_rows * n_cols];
            for row in 0..n_rows {
                for col in 0..n_cols {
                    tiled[tiled_offset(row, col, n_rows, n_cols)] = row_major[row * n_cols + col];
                }
            }
            assert_eq!(tiled_to_row_major(&tiled, n_rows, n_cols), row_major, "{n_rows}x{n_cols}");
        }
    }

    #[test]
    fn log_level_reads_numbers_and_flags_alike() {
        assert_eq!(parse_log_level(None), 0);
        assert_eq!(parse_log_level(Some("")), 0);
        assert_eq!(parse_log_level(Some("0")), 0);
        assert_eq!(parse_log_level(Some("1")), 1);
        assert_eq!(parse_log_level(Some(" 2 ")), 2);
        assert_eq!(parse_log_level(Some("10")), 10);
        assert_eq!(parse_log_level(Some("true")), 1);
    }

    #[test]
    fn a_worker_panic_keeps_its_message() {
        let caught = std::panic::catch_unwind(|| par_rows(64, 4, |first, _| if first > 0 { panic!("row {first} failed") }));
        let payload = caught.err().expect("par_rows should propagate the panic");
        let text = payload.downcast_ref::<String>().cloned().unwrap_or_default();
        assert!(text.starts_with("row ") && text.ends_with(" failed"), "got {text:?}");
    }

    #[test]
    fn admission_is_counted_per_slot() {
        // Streamed instances pin their slot, so the cap has to be per
        // client: filling one slot must not admit anything extra there, and
        // must not hold the other slot back.
        let slots: Vec<Arc<Pending>> = (0..2).map(|_| Arc::new(Pending::default())).collect();
        let a = slots[0].acquire(2);
        let _b = slots[0].acquire(2);
        let (tx, rx) = std::sync::mpsc::channel();
        let s0 = slots[0].clone();
        let waiter = std::thread::spawn(move || {
            let _c = s0.acquire(2);
            tx.send(()).unwrap();
        });
        assert!(rx.recv_timeout(std::time::Duration::from_millis(200)).is_err(), "a third prove got onto slot 0");
        let _other = slots[1].acquire(2);
        assert_eq!(slots[1].pending(), 1, "slot 1 was held back by slot 0's cap");
        drop(a);
        rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();
        waiter.join().unwrap();
    }

    #[test]
    fn one_air_reads_its_fixed_sections_ahead_of_the_running_prove() {
        // `FixedAhead::plan` is the schedule `prove_owned` follows on the way
        // to the slot, in the order it follows it.
        let client = FixedAhead::default();
        // Nothing on the device: the first prove reads its key and uploads it
        // while the prove ahead of it still holds the client.
        let first = client.plan("Main_n22");
        assert!(matches!(first, AheadPlan::ReadAndUpload(_)), "the first prove did not take the read-ahead");
        // A second AIR queued behind it reads, but its upload waits for the
        // slot: one key's sections beyond the running prove's, however many
        // proves queue up.
        assert!(matches!(client.plan("Rom_n22"), AheadPlan::ReadOnly), "a second AIR's sections went up beside the first's");

        // The permit comes back when the sections are installed, not when
        // that prove ends — otherwise the next prove waits out a whole prove
        // for a transfer it could have started.
        let AheadPlan::ReadAndUpload(permit) = first else { unreachable!() };
        client.installed("Main_n22", Some(permit));
        assert!(matches!(client.plan("Rom_n22"), AheadPlan::ReadAndUpload(_)), "the permit did not come back at the install");

        // An AIR whose sections are on the device reads nothing at all. This
        // is the case a `try_lock` on the slot cannot see: the running prove
        // holds that lock for its whole duration, so residency has to be
        // readable without it.
        assert!(matches!(client.plan("Main_n22"), AheadPlan::Resident), "a resident AIR re-read its key");
        client.evicted("Main_n22");
        assert!(!matches!(client.plan("Main_n22"), AheadPlan::Resident), "an evicted AIR still looked resident");
    }

    #[test]
    fn a_full_card_sends_the_read_ahead_upload_under_the_slot_instead_of_failing_the_prove() {
        // The upload runs while the prove ahead of this one is at its memory
        // peak, so it is the allocation most likely to fail. PJRT reports a
        // full card by panicking (xla-pjrt's `check`), not by returning an
        // error, so the panic is the arm that matters.
        let quiet = std::panic::take_hook();
        std::panic::set_hook(Box::new(|_| {}));
        let oom = upload_ahead(|| panic!("PJRT error in BufferFromHostBuffer: Out of memory while trying to allocate 3.00GiB"));
        let spec = upload_ahead(|| Err("const_base: 8 words for a [2, 2] buffer of 4".into()));
        std::panic::set_hook(quiet);
        // `UploadedFixed` holds device handles and has no `Debug`, so match
        // rather than unwrap.
        let Err(oom_why) = oom else {
            panic!("a full card escaped the read-ahead and would have failed the prove")
        };
        assert_eq!(oom_why, "PJRT error in BufferFromHostBuffer: Out of memory while trying to allocate 3.00GiB");
        assert!(spec.is_err(), "an upload error escaped the read-ahead");
    }

    #[test]
    fn the_trace_names_the_reason_the_sections_went_up_under_the_slot() {
        // A prove that never held the permit and one whose own upload gave
        // way both upload under the slot; only the first waits on another
        // AIR, and the trace has twice been caught saying otherwise.
        assert_eq!(ahead_trace_note(true, false, false), "", "an upload ahead of the slot was explained at all");
        assert!(
            ahead_trace_note(false, false, false).contains("another AIR's"),
            "a prove that never got the permit was not credited to the AIR ahead of it"
        );
        assert!(
            ahead_trace_note(false, false, true).contains("this prove's read-ahead upload gave way"),
            "a prove whose own upload gave way was blamed on another AIR"
        );
        assert!(
            ahead_trace_note(false, true, false).contains("resident when this prove looked"),
            "a prove that read under the slot was not told apart"
        );
    }

    #[test]
    fn a_failed_read_ahead_upload_hands_the_permit_back() {
        let client = FixedAhead::default();
        let AheadPlan::ReadAndUpload(permit) = client.plan("Main_n22") else {
            panic!("the first prove did not take the read-ahead")
        };
        let mut ahead = AheadFixed {
            const_base: Vec::new(),
            customs: Vec::new(),
            uploaded: None,
            permit: Some(permit),
            gave_way: false,
            read_s: 0.0,
            upload_s: 0.0,
        };
        let why = ahead.took_upload(Err("PJRT error in BufferFromHostBuffer: Out of memory".into()), 1.5);
        assert!(why.is_some(), "the failure was not reported to the caller");
        assert!(ahead.uploaded.is_none(), "a failed upload still claimed to have uploaded");
        assert_eq!(ahead.upload_s, 0.0, "a failed upload was billed to the ahead column");
        assert_eq!(client.permit.pending(), 0, "a failed read-ahead upload kept the permit");
        // With nothing uploaded the prove carries on: `set_fixed` uploads the
        // same words once the slot is its own and the card has room.
        assert!(ahead.sections().uploaded.is_none());
    }

    #[test]
    fn the_read_ahead_permit_is_counted_per_client() {
        let (one, two) = (FixedAhead::default(), FixedAhead::default());
        let held = one.plan("Main_n22");
        assert!(matches!(held, AheadPlan::ReadAndUpload(_)));
        assert!(matches!(two.plan("Main_n22"), AheadPlan::ReadAndUpload(_)), "one client's read-ahead held another's back");
        drop(held);
        assert_eq!(one.permit.pending(), 0, "the read-ahead permit leaked");
    }

    #[test]
    fn a_push_after_the_last_worker_leaves_spawns_again() {
        let mut q = PreloadQueue::default();
        assert_eq!(q.push(vec![(0, "A".into(), true)], true, 2), 2);
        assert!(q.take().is_some());
        // Both workers find the queue empty and retire, in the same step as
        // their empty pop.
        assert!(q.take().is_none());
        assert!(q.take().is_none());
        assert_eq!(q.workers(), 0);
        // The race the reviewer described: work arriving as the last
        // worker leaves. Counted under the same lock, the push sees no
        // workers and asks for a full set again.
        assert_eq!(q.push(vec![(0, "B".into(), true)], true, 2), 2);
        assert_eq!(q.take(), Some((0, "B".into(), true)));
        // A push while workers are alive spawns only the shortfall.
        assert_eq!(q.push(vec![(1, "C".into(), false)], false, 3), 1);
        assert_eq!(q.workers(), 3);
    }

    /// The queue keeps its callers' priority and order: a requested batch
    /// goes ahead of queued background work, and within the batch the order
    /// the caller gave survives. Both callers choose that order for their
    /// own reasons — `Bridge::global` passes the `.last-used` file, the
    /// proofman fork its instance list — so a queue that reordered them
    /// would be substituting its own order for theirs.
    #[test]
    fn a_requested_batch_goes_ahead_of_background_work_in_the_order_given() {
        let mut q = PreloadQueue::default();
        // The rest of the proving key, queued as background work.
        q.push(vec![(0, "Keccakf_n17".into(), false), (0, "Sha256f_n18".into(), false)], false, 1);
        // The run's own AIRs arrive after it and go to the front. Pushing
        // each key to the front on its own would flip the batch.
        q.push(vec![(0, "Main_n22".into(), true), (0, "Rom_n22".into(), true)], true, 1);
        assert_eq!(q.take(), Some((0, "Main_n22".into(), true)));
        assert_eq!(q.take(), Some((0, "Rom_n22".into(), true)));
        assert_eq!(q.take(), Some((0, "Keccakf_n17".into(), false)));
        assert_eq!(q.take(), Some((0, "Sha256f_n18".into(), false)));
    }

    #[test]
    fn read_words_skips_the_header_and_spans_slices() {
        let dir = std::env::temp_dir().join(format!("zz-read-words-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("fixed.bin");
        // Header of 32 bytes, then more words than the caller asks for,
        // and enough of them to cross the 8 MB read slices.
        let n = (9 << 20) / 8 + 5;
        let words: Vec<u64> = (0..n as u64).map(|i| i.wrapping_mul(0x9E37_79B9_7F4A_7C15)).collect();
        let mut bytes = vec![0xAAu8; 32];
        bytes.extend(words.iter().flat_map(|w| w.to_le_bytes()));
        std::fs::write(&path, &bytes).unwrap();
        let p = path.to_str().unwrap();
        assert_eq!(read_words(p, 32, n - 3).unwrap(), words[..n - 3]);
        assert_eq!(read_words(p, 0, 4).unwrap(), vec![0xAAAA_AAAA_AAAA_AAAA; 4]);
        assert!(read_words(p, 32, n + 1).is_err());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
