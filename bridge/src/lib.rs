//! pil2-proofman's `gen_proof` over zisk-zorch's exported artifacts.
//!
//! The device half of genProof is the per-AIR StableHLO programs
//! `zisk_zorch/export` writes; this crate is the host half, called from
//! proofman's `gen_proof` wrapper in place of `gen_proof_c`: it uploads the
//! instance, drives the programs through PJRT in the schedule's order,
//! sponges the transcript, and writes the flat proof pil2 reads back.
//!
//! Concurrency: one PJRT client per slot, one prove per client at a time.
//! proofman proves streamed instances from `n_streams` threads at once and
//! the rest from per-stream workers, so a slot per stream keeps those
//! proves overlapping on the card (one client serializes its executions).
//!
//! Environment:
//!   ZZ_ARTIFACTS    directory of `<Air>_n<nBits>/` exports; unset = bridge off
//!   XLA_PJRT_PLUGIN the frx/jax CUDA PJRT plugin .so (xla-pjrt reads it)
//!   ZZ_CLIENTS      PJRT clients (default: the stream count proofman passes)
//!   ZZ_PRELOAD      what loads at creation: the previous run's AIRs (default,
//!                   `<ZZ_ARTIFACTS>/.last-used`), `all`, or `0` (only the
//!                   instance list, once proofman has it); ZZ_PRELOAD_THREADS=6
//!   ZZ_MEMORY_FRACTION  share of the card the clients claim up front, split
//!                   evenly (unset: allocate on demand). pil2 sizes its own
//!                   stream buffers from what is free at init, so this is
//!                   what keeps it from taking the whole card first.
//!   ZZ_AB=1         prove through pil2 too and compare (see `ab`)
//!   ZZ_RESIDENT_AIRS  AIRs whose fixed sections stay on a client at once
//!                   (default 1, least recently used evicted); a table AIR's
//!                   sections run to gigabytes and every AIR of a block
//!                   cannot stay resident beside pil2's buffers
//!   ZZ_COMPILE_CACHE  directory of serialized executables (default
//!                   `<ZZ_ARTIFACTS>/.pjrt-cache`); a miss compiles and stores
//!   ZZ_LOG=1        per-instance timing on stderr (`ZZ_LOG=2` per program),
//!                   each line stamped with the seconds since the bridge came up

/// Seconds since the bridge came up, for the `[zz +t]` log stamps.
pub fn uptime() -> f64 {
    static T0: OnceLock<Instant> = OnceLock::new();
    T0.get_or_init(Instant::now).elapsed().as_secs_f64()
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
    let threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, 32);
    let rows_per = n_rows.div_ceil(threads).max(1);
    std::thread::scope(|scope| {
        for (i, chunk) in out.chunks_mut(rows_per * row_words).enumerate() {
            let f = &f;
            scope.spawn(move || f(i * rows_per, chunk));
        }
    });
    out
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

pub struct Bridge {
    artifacts: PathBuf,
    cache: PathBuf,
    resident_airs: usize,
    /// One client per slot, reachable without the slot lock so a loader
    /// never waits on a prove's slot while holding the client's gate.
    clients: Vec<Arc<artifact::Client>>,
    slots: Vec<Mutex<Slot>>,
    loaded: Vec<(Mutex<Loaded>, std::sync::Condvar)>,
    preload_queue: Mutex<std::collections::VecDeque<(usize, String, bool)>>,
    used: Mutex<std::collections::BTreeSet<String>>,
    preload_threads: AtomicUsize,
    next: AtomicUsize,
    log: bool,
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
        let log = std::env::var("ZZ_LOG").map(|v| v != "0" && !v.is_empty()).unwrap_or(false);
        let fraction = std::env::var("ZZ_MEMORY_FRACTION")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|f| *f > 0.0 && *f < 1.0)
            .map(|f| f / clients as f32);
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
            preload_queue: Mutex::new(std::collections::VecDeque::new()),
            used: Mutex::new(std::collections::BTreeSet::new()),
            preload_threads: AtomicUsize::new(0),
            next: AtomicUsize::new(0),
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
            artifact::Artifact::load(client.clone(), &dir, Some(&cache)).and_then(|a| a.compile_all().map(|_| a))
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
        {
            let mut q = self.preload_queue.lock().unwrap();
            if first {
                for w in work.into_iter().rev() {
                    q.push_front(w);
                }
            } else {
                q.extend(work);
            }
        }
        let threads = std::env::var("ZZ_PRELOAD_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(6usize).max(1);
        let running = self.preload_threads.fetch_add(0, Ordering::SeqCst);
        for _ in running..threads {
            self.preload_threads.fetch_add(1, Ordering::SeqCst);
            let bridge = self.clone();
            std::thread::spawn(move || loop {
                let next = bridge.preload_queue.lock().unwrap().pop_front();
                match next {
                    // Background keys (the whole proving key) load only until
                    // proving starts; requested keys always do.
                    Some((slot, _, false)) if bridge.clients[slot].started_proving() => continue,
                    Some((slot, key, _)) => {
                        if let Err(e) = bridge.artifact(slot, &key) {
                            zzlog!("preload {key}: {e}");
                        }
                    }
                    None => {
                        bridge.preload_threads.fetch_sub(1, Ordering::SeqCst);
                        return;
                    }
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
                let start = self.next.load(Ordering::Relaxed) % n;
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
        let t0 = Instant::now();
        let key = format!("{}_n{}", req.air, req.n_bits);
        let m = manifest::Manifest::load(&self.artifacts.join(&key))?;
        let n = 1usize << m.n_bits;
        let p = &req.inputs;
        let trace = match req.packed {
            Some((words_per_row, bits)) => {
                if bits.len() != m.widths.cm1 {
                    return Err(format!("{key}: packing lists {} columns, cm1 has {}", bits.len(), m.widths.cm1).into());
                }
                unpack_trace(std::slice::from_raw_parts(p.trace, n * words_per_row), n, words_per_row, bits)
            }
            None => copy_canonical(std::slice::from_raw_parts(p.trace, n * m.widths.cm1)),
        };
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

    pub fn prove_owned(&self, req: &OwnedRequest, proof_out: &mut [u64]) -> Result<ProveOutputs, Error> {
        let t0 = Instant::now();
        let key = req.key.clone();
        self.note_used(&key);
        let slot_idx = self.slot_index(req.stream_id, req.instance_id);
        let art = self.artifact(slot_idx, &key)?;
        let mut slot = self.slots[slot_idx].lock().unwrap_or_else(|p| p.into_inner());
        // No load may enter the plugin while this prove has work in flight.
        let _exclusive = self.clients[slot_idx].enter_prove();
        let waited = t0.elapsed().as_secs_f64();
        if !slot.drivers.contains_key(&key) {
            slot.drivers.insert(key.clone(), driver::AirDriver::new(art));
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
                if self.log {
                    zzlog!("released {k}'s fixed sections");
                }
            }
        }
        let driver = slot.drivers.get_mut(&key).unwrap();
        if !driver.has_fixed() {
            let t = Instant::now();
            let read_s = load_fixed(driver, &req.const_pols_path, req.custom_fixed_path.as_deref())?;
            if self.log {
                zzlog!(
                    "fixed sections for {key} in {:.2} s ({read_s:.2} s reading the key, the rest upload and setup)",
                    t.elapsed().as_secs_f64()
                );
            }
        }
        if proof_out.len() != driver.proof_words() {
            return Err(format!(
                "{key}: proof buffer holds {} words, layout says {}",
                proof_out.len(),
                driver.proof_words()
            )
            .into());
        }
        let m = driver.manifest();
        let inputs = InstanceInputs {
            trace: &req.trace,
            publics: &req.publics,
            airvalues: &req.airvalues,
            proofvalues: &req.proofvalues,
            global_challenge: &req.global_challenge,
        };
        if let Ok(dir) = std::env::var("ZZ_DUMP_INPUTS") {
            // The instance exactly as received, for replaying it outside proofman.
            let d = std::path::PathBuf::from(dir).join(format!("{}_{}", req.instance_id, key));
            let _ = std::fs::create_dir_all(&d);
            for (name, words) in [
                ("trace", inputs.trace),
                ("publics", inputs.publics),
                ("airvalues", inputs.airvalues),
                ("proofvalues", inputs.proofvalues),
                ("global_challenge", inputs.global_challenge),
            ] {
                let bytes: Vec<u8> = words.iter().flat_map(|w| w.to_le_bytes()).collect();
                let _ = std::fs::write(d.join(format!("{name}.bin")), bytes);
            }
        }
        let mut transcript = transcript::HostTranscript::new(&m.hash_family)?;
        let out = driver.prove(&inputs, &mut transcript, proof_out)?;
        if self.log {
            zzlog!(
                "instance {} {} ({}): {:.3} s, of which {:.3} s waiting for the client",
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

/// Read the key's fixed sections and hand them to the driver; returns the
/// seconds spent reading (the rest is upload and the setup programs).
fn load_fixed(driver: &mut driver::AirDriver, const_pols_path: &str, custom_fixed_path: Option<&str>) -> Result<f64, Error> {
    let t = Instant::now();
    let m = driver.manifest().clone();
    let n = 1usize << m.n_bits;
    let const_base = read_words(&plain_const_path(const_pols_path), 0, n * m.n_constants)?;
    let mut customs: Vec<(usize, Vec<u64>)> = Vec::new();
    for cc in &m.custom_commits {
        let path = custom_fixed_path
            .filter(|p| !p.is_empty())
            .ok_or_else(|| format!("{} has a custom commit but no fixed path", m.air))?;
        // pil2 skips a 32-byte Merkle-root header (one custom commit per AIR)
        // and copies the rest straight to the device, so the section sits in
        // the prover's tiled layout rather than row-major.
        customs.push((cc.id, tiled_to_row_major(&read_words(path, 32, n * cc.width)?, n, cc.width)));
    }
    let read_s = t.elapsed().as_secs_f64();
    let fixed = FixedSections {
        const_base: &const_base,
        custom_base: customs.iter().map(|(id, w)| (*id, w.as_slice())).collect(),
    };
    driver.set_fixed(&fixed)?;
    Ok(read_s)
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
