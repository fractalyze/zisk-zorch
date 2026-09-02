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
//!   ZZ_MEMORY_FRACTION  share of the card the clients claim up front, split
//!                   evenly (unset: allocate on demand). pil2 sizes its own
//!                   stream buffers from what is free at init, so this is
//!                   what keeps it from taking the whole card first.
//!   ZZ_AB=1         prove through pil2 too and compare (see `ab`)
//!   ZZ_RESIDENT_AIRS  AIRs whose fixed sections stay on a client at once
//!                   (default 2, least recently used evicted); a table AIR's
//!                   sections run to gigabytes and every AIR of a block
//!                   cannot stay resident beside pil2's buffers
//!   ZZ_COMPILE_CACHE  directory of serialized executables (default
//!                   `<ZZ_ARTIFACTS>/.pjrt-cache`); a miss compiles and stores
//!   ZZ_LOG=1        per-instance timing on stderr

pub mod ab;
pub mod artifact;
pub mod driver;
pub mod manifest;
pub mod transcript;

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, OnceLock};
use std::time::Instant;

use xla_pjrt::Session;

pub use driver::{FixedSections, InstanceInputs, ProveOutputs};

pub type Error = Box<dyn std::error::Error + Send + Sync>;

/// One PJRT client and its per-AIR drivers.
pub struct Slot {
    session: Arc<Session>,
    drivers: HashMap<String, driver::AirDriver>,
    /// Last use per AIR, for evicting resident fixed sections.
    last_used: HashMap<String, u64>,
    tick: u64,
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

/// pil2's `unpack_trace`: bit-fields packed little-endian across
/// `words_per_row` words per row, in column order, widened to one word
/// per column.
pub fn unpack_trace(packed: &[u64], n_rows: usize, words_per_row: usize, bits: &[u64]) -> Vec<u64> {
    let n_cols = bits.len();
    let mut out = vec![0u64; n_rows * n_cols];
    for row in 0..n_rows {
        let src = &packed[row * words_per_row..(row + 1) * words_per_row];
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
                if bit_offset == 64 && word_idx + 1 < words_per_row {
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
            out[row * n_cols + c] = val;
        }
    }
    out
}

pub struct Bridge {
    artifacts: PathBuf,
    cache: PathBuf,
    resident_airs: usize,
    slots: Vec<Mutex<Slot>>,
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
                Some(Arc::new(Bridge::new(Path::new(&dir), clients)))
            })
            .clone()
    }

    pub fn new(artifacts: &Path, clients: usize) -> Bridge {
        let log = std::env::var("ZZ_LOG").map(|v| v != "0" && !v.is_empty()).unwrap_or(false);
        let fraction = std::env::var("ZZ_MEMORY_FRACTION")
            .ok()
            .and_then(|s| s.parse::<f32>().ok())
            .filter(|f| *f > 0.0 && *f < 1.0)
            .map(|f| f / clients as f32);
        let slots = (0..clients)
            .map(|_| {
                Mutex::new(Slot {
                    session: artifact::new_session(fraction),
                    drivers: HashMap::new(),
                    last_used: HashMap::new(),
                    tick: 0,
                })
            })
            .collect();
        if log {
            eprintln!(
                "[zz] bridge up: {clients} PJRT client(s){}, artifacts {}{}",
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
            .unwrap_or(2);
        Bridge { artifacts: artifacts.to_path_buf(), cache, resident_airs, slots, next: AtomicUsize::new(0), log }
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

    /// A streamed instance runs on its stream's slot; any other takes the
    /// first idle slot, or waits on a fixed one so a burst of workers still
    /// spreads across clients.
    fn slot(&self, stream_id: Option<usize>, instance_id: u64) -> MutexGuard<'_, Slot> {
        let n = self.slots.len();
        if let Some(s) = stream_id {
            return self.slots[s % n].lock().unwrap();
        }
        let start = self.next.fetch_add(1, Ordering::Relaxed) % n;
        for k in 0..n {
            if let Ok(g) = self.slots[(start + k) % n].try_lock() {
                return g;
            }
        }
        self.slots[(instance_id as usize) % n].lock().unwrap()
    }

    /// Prove one instance; writes the flat proof into `proof_out` (which
    /// must hold the AIR's proof size) and returns the host-side products.
    ///
    /// # Safety
    /// `req.inputs` must point at buffers of the lengths the artifact's
    /// manifest implies, valid until this returns.
    pub unsafe fn prove(&self, req: &ProveRequest, proof_out: &mut [u64]) -> Result<ProveOutputs, Error> {
        let t0 = Instant::now();
        let mut slot = self.slot(req.stream_id, req.instance_id);
        let key = format!("{}_n{}", req.air, req.n_bits);
        if !slot.drivers.contains_key(&key) {
            let t = Instant::now();
            let art = artifact::Artifact::load(slot.session.clone(), &self.artifacts.join(&key), Some(&self.cache))?;
            art.compile_all()?;
            let n = art.manifest.programs.len();
            let hits = art.cache_hits.load(Ordering::Relaxed);
            slot.drivers.insert(key.clone(), driver::AirDriver::new(art));
            if self.log {
                eprintln!(
                    "[zz] loaded {key} ({n} programs, {hits} from the cache) in {:.1} s",
                    t.elapsed().as_secs_f64()
                );
            }
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
                    eprintln!("[zz] released {k}'s fixed sections");
                }
            }
        }
        let driver = slot.drivers.get_mut(&key).unwrap();
        if !driver.has_fixed() {
            let t = Instant::now();
            load_fixed(driver, req.const_pols_path, req.custom_fixed_path)?;
            if self.log {
                eprintln!("[zz] fixed sections for {key} in {:.1} s", t.elapsed().as_secs_f64());
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
        let n = 1usize << m.n_bits;
        let p = &req.inputs;
        // pil2's Goldilocks accepts any u64 as a representative (its arithmetic
        // reduces lazily) and a witness holds raw machine words, so a trace
        // word above the modulus is that word's residue to pil2; our kernels
        // read storage as canonical, so reduce on the way in.
        let unpacked;
        let trace_words: &[u64] = match req.packed {
            Some((words_per_row, bits)) => {
                if bits.len() != m.widths.cm1 {
                    return Err(format!("{key}: packing lists {} columns, cm1 has {}", bits.len(), m.widths.cm1).into());
                }
                unpacked = unpack_trace(std::slice::from_raw_parts(p.trace, n * words_per_row), n, words_per_row, bits);
                &unpacked
            }
            None => std::slice::from_raw_parts(p.trace, n * m.widths.cm1),
        };
        let trace = canonical(trace_words);
        let publics = canonical(std::slice::from_raw_parts(p.publics, m.n_publics));
        let airvalues = canonical(std::slice::from_raw_parts(p.airvalues, manifest::Manifest::packed_width(&m.airvalues)));
        let proofvalues = canonical(std::slice::from_raw_parts(p.proofvalues, manifest::Manifest::packed_width(&m.proofvalues)));
        let global_challenge = canonical(std::slice::from_raw_parts(p.global_challenge, 3));
        let inputs = InstanceInputs {
            trace: &trace,
            publics: &publics,
            airvalues: &airvalues,
            proofvalues: &proofvalues,
            global_challenge: &global_challenge,
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
            eprintln!(
                "[zz] instance {} {} ({}): {:.3} s",
                req.instance_id,
                key,
                if req.stream_id.is_some() { "streamed" } else { "worker" },
                t0.elapsed().as_secs_f64()
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
/// row-major `(n_rows, n_cols)`.
pub fn tiled_to_row_major(tiled: &[u64], n_rows: usize, n_cols: usize) -> Vec<u64> {
    let mut out = vec![0u64; n_rows * n_cols];
    for col in 0..n_cols {
        let block_y = col / TILE_WIDTH;
        let n_cols_block = (n_cols - TILE_WIDTH * block_y).min(TILE_WIDTH);
        let col_block = col % TILE_WIDTH;
        for row in 0..n_rows {
            let block_x = row / TILE_HEIGHT;
            let row_block = row % TILE_HEIGHT;
            let src = block_y * TILE_WIDTH * n_rows + block_x * n_cols_block * TILE_HEIGHT + col_block * TILE_HEIGHT + row_block;
            out[row * n_cols + col] = tiled[src];
        }
    }
    out
}

/// Every word reduced into `[0, p)`. Borrows when nothing needs reducing
/// (the common case for a table's small values); copies otherwise.
fn canonical(words: &[u64]) -> std::borrow::Cow<'_, [u64]> {
    if words.iter().all(|w| *w < GOLDILOCKS_P) {
        std::borrow::Cow::Borrowed(words)
    } else {
        std::borrow::Cow::Owned(words.iter().map(|w| if *w >= GOLDILOCKS_P { w - GOLDILOCKS_P } else { *w }).collect())
    }
}

/// `<stem>.const_gpu` (what proofman passes on GPU) -> `<stem>.const`, the
/// plain row-major section the exporter read.
pub fn plain_const_path(path: &str) -> String {
    path.strip_suffix("_gpu").unwrap_or(path).to_string()
}

/// The first `n_words` little-endian words of `path` after `skip_bytes`.
/// A longer file is fine: pil2's custom-commit fixed file carries the
/// extended section and its tree behind the base rows.
fn read_words(path: &str, skip_bytes: usize, n_words: usize) -> Result<Vec<u64>, Error> {
    let bytes = std::fs::read(path).map_err(|e| format!("cannot read {path}: {e}"))?;
    let end = skip_bytes + n_words * 8;
    if bytes.len() < end {
        return Err(format!("{path} is {} bytes, expected at least {end}", bytes.len()).into());
    }
    Ok(bytes[skip_bytes..end]
        .chunks_exact(8)
        .map(|c| u64::from_le_bytes(c.try_into().unwrap()))
        .collect())
}

fn load_fixed(driver: &mut driver::AirDriver, const_pols_path: &str, custom_fixed_path: Option<&str>) -> Result<(), Error> {
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
    let fixed = FixedSections {
        const_base: &const_base,
        custom_base: customs.iter().map(|(id, w)| (*id, w.as_slice())).collect(),
    };
    driver.set_fixed(&fixed)
}
