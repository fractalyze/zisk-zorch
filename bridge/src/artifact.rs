//! One exported AIR on one PJRT session: the manifest, the programs
//! compiled on first use, and execution by manifest name with inputs
//! bound by name — `zisk_zorch/export/runtime.py`'s `Artifact`.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

use xla_pjrt::{Executable, Session, SessionOptions};

use crate::manifest::{Manifest, ProgramInfo, Spec};
use crate::Error;

/// A device buffer that frees itself with its last owner. `xla_pjrt::Buffer`
/// deliberately does not drop (it has no session handle); this pairs it
/// with the session that can release it.
pub struct DeviceBuf {
    session: Arc<Session>,
    buf: Option<xla_pjrt::Buffer>,
}

impl DeviceBuf {
    fn raw(&self) -> &xla_pjrt::Buffer {
        self.buf.as_ref().expect("buffer already released")
    }
}

impl Drop for DeviceBuf {
    fn drop(&mut self) {
        if let Some(b) = self.buf.take() {
            unsafe { self.session.free_buffer(b) };
        }
    }
}

pub type Buf = Arc<DeviceBuf>;

/// `ZZ_LOG=2`: per-program and per-phase timing on stderr.
pub fn trace_enabled() -> bool {
    crate::log_level() >= 2
}
/// Inputs bound by manifest name.
pub type Env = HashMap<String, Buf>;

/// A bridge client. Without a `memory_fraction` the allocator grows on
/// demand and stays out of everyone's way; with one it claims that share
/// of the card up front, so a co-tenant that sizes itself from free memory
/// (pil2's stream buffers) leaves it room.
/// One bridge client: the PJRT session plus the gate between loading and
/// proving. A deserialization while an execution is in flight on the same
/// client wedges both (the load synchronizes the device; the execution's
/// completion needs what the load holds), while deserializations alongside
/// each other are fine. So loads take the gate shared, per program
/// (`Artifact::executable`), and a prove takes it exclusively for its
/// whole run (`Bridge::prove_owned`).
pub struct Client {
    pub session: Arc<Session>,
    gate: Gate,
}

/// The load/prove gate, on its own so it can be exercised without a
/// plugin: loads enter shared, a prove enters alone, and a prove that is
/// waiting holds new loads back.
#[derive(Default)]
pub struct Gate {
    state: Mutex<GateState>,
    cv: std::sync::Condvar,
}

#[derive(Default)]
struct GateState {
    loads_in_flight: usize,
    proving: bool,
    /// Proves waiting to enter: new loads hold back for them, so a prove
    /// waits at most for the loads already in flight.
    proves_waiting: usize,
    started_proving: bool,
}

/// Releases its side of the gate on drop.
pub struct GatePass<'a> {
    gate: &'a Gate,
    load: bool,
}

impl Drop for GatePass<'_> {
    fn drop(&mut self) {
        let mut g = self.gate.state.lock().unwrap_or_else(|p| p.into_inner());
        if self.load {
            g.loads_in_flight -= 1;
        } else {
            g.proving = false;
        }
        self.gate.cv.notify_all();
    }
}

impl Gate {
    /// Wait until no prove is running or waiting, then count this load in.
    pub fn enter_load(&self) -> GatePass<'_> {
        let mut g = self.state.lock().unwrap_or_else(|p| p.into_inner());
        while g.proving || g.proves_waiting > 0 {
            g = self.cv.wait(g).unwrap_or_else(|p| p.into_inner());
        }
        g.loads_in_flight += 1;
        GatePass { gate: self, load: true }
    }

    /// Wait until no load is in flight and no other prove runs, then own the
    /// plugin for one prove.
    pub fn enter_prove(&self) -> GatePass<'_> {
        let mut g = self.state.lock().unwrap_or_else(|p| p.into_inner());
        g.proves_waiting += 1;
        while g.proving || g.loads_in_flight > 0 {
            g = self.cv.wait(g).unwrap_or_else(|p| p.into_inner());
        }
        g.proves_waiting -= 1;
        g.proving = true;
        g.started_proving = true;
        GatePass { gate: self, load: false }
    }

    /// Whether a prove has ever run on this client (the background preload
    /// of the whole key stops then; only requested AIRs load afterwards).
    pub fn started_proving(&self) -> bool {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).started_proving
    }
}

impl Client {
    pub fn enter_load(&self) -> GatePass<'_> {
        self.gate.enter_load()
    }

    pub fn enter_prove(&self) -> GatePass<'_> {
        self.gate.enter_prove()
    }

    pub fn started_proving(&self) -> bool {
        self.gate.started_proving()
    }
}

/// Apply `f` to every item, `threads` at a time, and return the first error.
///
/// Threads share one queue rather than taking a slice each, because the items
/// here are programs whose compile times differ by more than an order of
/// magnitude -- a static split leaves one thread holding every slow one. A
/// panic in a worker propagates, since callers wrap this in `catch_unwind` to
/// clear their in-flight marks.
///
/// Every item is attempted whatever the thread count, and the first error is
/// the one returned: a failure part-way through must leave the same set
/// compiled at one thread as at eight, or the preload path (which runs at one)
/// would fill the cache differently from a warm.
fn each_parallel<T: Send + Sync>(
    items: Vec<T>,
    threads: usize,
    f: impl Fn(&T) -> Result<(), Error> + Sync,
) -> Result<(), Error> {
    if threads <= 1 {
        return items.iter().fold(Ok(()), |first, item| match (first, f(item)) {
            (Ok(()), Err(e)) => Err(e),
            (first, _) => first,
        });
    }
    let queue = std::sync::Mutex::new(items.iter());
    let mut first = Ok(());
    std::thread::scope(|scope| {
        let workers: Vec<_> = (0..threads)
            .map(|_| {
                scope.spawn(|| loop {
                    let next = queue.lock().unwrap_or_else(|p| p.into_inner()).next();
                    match next {
                        Some(item) => f(item)?,
                        None => return Ok(()),
                    }
                })
            })
            .collect();
        for worker in workers {
            match worker.join() {
                Ok(Err(e)) if first.is_ok() => first = Err(e),
                Err(panic) => std::panic::resume_unwind(panic),
                _ => {}
            }
        }
    });
    first
}

pub fn new_client(memory_fraction: Option<f32>) -> Arc<Client> {
    Arc::new(Client {
        session: new_session(memory_fraction),
        gate: Gate::default(),
    })
}

/// Whether the plugin loads an executable's modules into the CUDA context as
/// it is deserialized rather than on its first execution.
///
/// It pays when the loads land somewhere other than a prove slot, which is
/// what `ZZ_PRELOAD` arranges, so it follows that by default. `ZZ_EAGER_MODULES`
/// overrides either way — a measurement has to vary this without also varying
/// what gets preloaded, or the two changes land in one number.
///
/// Off means sending no option at all, not `false`: PJRT rejects a create
/// option a plugin does not know, so a plugin built before
/// fractalyze/xla#664 fails client creation on the key whatever its value.
fn eager_module_loads() -> Option<bool> {
    // The same ZZ_PRELOAD the bridge acts on in `Bridge::global`; clients are
    // built before that runs, so it is read here too.
    eager_module_loads_from(
        std::env::var("ZZ_EAGER_MODULES").ok().as_deref(),
        std::env::var("ZZ_PRELOAD").ok().as_deref(),
    )
}

/// The decision on its own, so the table in the tests can state it.
///
/// `0` is the off spelling for both variables, as it is for `ZZ_PRELOAD`; an
/// empty value is not a value at all but an unset one, which is how every
/// other variable here reads it (`filter(|s| !s.is_empty())` in `Bridge::global`).
fn eager_module_loads_from(eager: Option<&str>, preload: Option<&str>) -> Option<bool> {
    let on = match eager.filter(|s| !s.is_empty()) {
        Some("0") => false,
        Some(_) => true,
        None => preload.filter(|s| !s.is_empty()) != Some("0"),
    };
    on.then_some(true)
}

pub fn new_session(memory_fraction: Option<f32>) -> Arc<Session> {
    let eager = eager_module_loads();
    if crate::log_level() >= 1 {
        // A run's own log has to say which way this went: two runs that differ
        // only by this option are otherwise indistinguishable after the fact.
        // Once per run, not per client -- the value is read from the
        // environment, so every client of a run reports the same thing.
        static SAID: std::sync::Once = std::sync::Once::new();
        SAID.call_once(|| zzlog!("eager module loads {}", if eager.is_some() { "on" } else { "off" }));
    }
    let options = SessionOptions {
        preallocate: Some(memory_fraction.is_some()),
        memory_fraction,
        eager_load_executable_modules: eager,
    };
    let session = Arc::new(unsafe { Session::with_options(options) });
    if memory_fraction.is_some() {
        // The plugin builds its allocator pool on the client's first device
        // use, not at creation; claim the share now so a co-tenant that
        // measures free memory next (pil2 at init) sees it gone.
        let word = [0u8; 8];
        let buf = unsafe { session.input_buffer(&word, &[1], xla_pjrt::sys::PJRT_Buffer_Type_U64) };
        unsafe { session.free_buffer(buf) };
    }
    session
}

/// A hash of the PJRT plugin the executables are serialized by (its path,
/// size and modification time), folded into the cache key. Zero when
/// `XLA_PJRT_PLUGIN` is unset.
fn plugin_identity() -> u64 {
    static ID: std::sync::OnceLock<u64> = std::sync::OnceLock::new();
    *ID.get_or_init(|| {
        let Some(path) = std::env::var_os("XLA_PJRT_PLUGIN") else { return 0 };
        let mut key = path.to_string_lossy().into_owned().into_bytes();
        if let Ok(meta) = std::fs::metadata(&path) {
            key.extend(meta.len().to_le_bytes());
            if let Ok(t) = meta.modified().and_then(|t| t.duration_since(std::time::UNIX_EPOCH).map_err(std::io::Error::other)) {
                key.extend(t.as_secs().to_le_bytes());
            }
        }
        fnv1a64(&key)
    })
}

/// Process-wide exclusion per cache file, so the clients of one bridge do
/// not compile (and write) the same entry at once.
struct CacheEntryLock(PathBuf);

impl CacheEntryLock {
    fn state() -> &'static (Mutex<std::collections::HashSet<PathBuf>>, std::sync::Condvar) {
        static S: std::sync::OnceLock<(Mutex<std::collections::HashSet<PathBuf>>, std::sync::Condvar)> = std::sync::OnceLock::new();
        S.get_or_init(Default::default)
    }

    fn take(path: &Path) -> CacheEntryLock {
        let (busy, cv) = Self::state();
        let mut set = busy.lock().unwrap_or_else(|p| p.into_inner());
        while set.contains(path) {
            set = cv.wait(set).unwrap_or_else(|p| p.into_inner());
        }
        set.insert(path.to_path_buf());
        CacheEntryLock(path.to_path_buf())
    }
}

impl Drop for CacheEntryLock {
    fn drop(&mut self) {
        let (busy, cv) = Self::state();
        busy.lock().unwrap_or_else(|p| p.into_inner()).remove(&self.0);
        cv.notify_all();
    }
}

pub struct Artifact {
    pub manifest: Manifest,
    dir: PathBuf,
    client: Arc<Client>,
    session: Arc<Session>,
    exes: Mutex<HashMap<String, Arc<Executable>>>,
    /// Serialized executables, keyed by the bytecode's hash: a compile costs
    /// seconds per program, a load milliseconds. Plugin-version specific —
    /// drop the directory with the plugin.
    cache: Option<PathBuf>,
    pub cache_hits: std::sync::atomic::AtomicUsize,
}

/// FNV-1a over the bytecode: a cache key, not a security boundary.
fn fnv1a64(bytes: &[u8]) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in bytes {
        h ^= *b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

impl Artifact {
    pub fn load(client: Arc<Client>, dir: &Path, cache: Option<&Path>) -> Result<Artifact, Error> {
        let cache = match cache {
            Some(c) => {
                let sub = c.join(dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
                match std::fs::create_dir_all(&sub) {
                    Ok(()) => Some(sub),
                    Err(e) => {
                        // The cache is an optimization: a read-only or
                        // shared artifacts directory proves without it.
                        zzlog!("cannot create {}: {e}; compiling without the executable cache", sub.display());
                        None
                    }
                }
            }
            None => None,
        };
        Ok(Artifact {
            manifest: Manifest::load(dir)?,
            dir: dir.to_path_buf(),
            session: client.session.clone(),
            client,
            exes: Mutex::new(HashMap::new()),
            cache,
            cache_hits: std::sync::atomic::AtomicUsize::new(0),
        })
    }

    pub fn session(&self) -> &Arc<Session> {
        &self.session
    }

    pub fn dir(&self) -> &Path {
        &self.dir
    }

    fn executable(&self, name: &str) -> Result<Arc<Executable>, Error> {
        if let Some(exe) = self.exes.lock().unwrap().get(name) {
            return Ok(exe.clone());
        }
        let info = self.manifest.program(name)?;
        let path = self.dir.join(&info.file);
        let code = std::fs::read(&path).map_err(|e| format!("cannot read {}: {e}", path.display()))?;
        // Keyed by the bytecode and the plugin: an executable serialized by
        // one plugin build is not one another accepts.
        let cached = self.cache.as_ref().map(|c| c.join(format!("{name}-{:016x}.pjrt", fnv1a64(&code) ^ plugin_identity())));
        // One program at a time through the gate: a prove that arrives
        // mid-load waits for this program, not for the AIR's whole set.
        let _loading = self.client.enter_load();
        // One compile per cache entry across clients: a second client for
        // the same AIR waits here and then finds the first one's file.
        let _entry = cached.as_ref().map(|p| CacheEntryLock::take(p));
        let t = std::time::Instant::now();
        let mut exe = None;
        if let Some(p) = &cached {
            if let Ok(bytes) = std::fs::read(p) {
                let read_ms = t.elapsed().as_secs_f64() * 1e3;
                // A stale entry (another plugin build, another GPU) makes
                // the plugin fail its load, which xla-pjrt reports as a
                // panic; that is a cache miss, not a fatal error.
                match std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| unsafe { self.session.deserialize_and_load(&bytes) })) {
                    Ok(e) => {
                        self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                        if trace_enabled() {
                            zzlog!(
                                "  load {name}: {} KB, read {read_ms:.1} ms, deserialize {:.1} ms",
                                bytes.len() / 1024,
                                t.elapsed().as_secs_f64() * 1e3 - read_ms
                            );
                        }
                        exe = Some(e);
                    }
                    Err(_) => {
                        zzlog!("cached executable {} rejected by the plugin; recompiling {name}", p.display());
                        let _ = std::fs::remove_file(p);
                    }
                }
            }
        }
        let exe = match exe {
            Some(e) => e,
            None => {
                let exe = unsafe { self.session.compile(&code) };
                if let Some(p) = &cached {
                    let bytes = unsafe { self.session.serialize(&exe) };
                    static SEQ: std::sync::atomic::AtomicUsize = std::sync::atomic::AtomicUsize::new(0);
                    let tmp = p.with_extension(format!(
                        "tmp{}-{}",
                        std::process::id(),
                        SEQ.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
                    ));
                    if std::fs::write(&tmp, bytes).and_then(|_| std::fs::rename(&tmp, p)).is_err() {
                        let _ = std::fs::remove_file(&tmp);
                    }
                }
                exe
            }
        };
        let exe = Arc::new(exe);
        self.exes.lock().unwrap().insert(name.to_string(), exe.clone());
        Ok(exe)
    }

    /// Compile (or load from the cache) every program, `threads` at a time.
    ///
    /// One thread is the default everywhere a prove might be waiting: each
    /// program takes the client's load gate on its own, so a prove arriving
    /// mid-load waits for one program rather than for `threads` of them, and
    /// that is the latency guarantee the per-program gate exists for. Warming
    /// a single AIR is the case that wants more — nothing is proving, and the
    /// serial loop is otherwise about an hour and a half for ~34 programs.
    pub fn compile_all(&self, threads: usize) -> Result<(), Error> {
        each_parallel(self.manifest.programs.keys().cloned().collect(), threads, |name| {
            self.executable(name).map(|_| ())
        })
    }

    /// Host words -> a device buffer shaped and typed by `spec`. Field words
    /// are canonical u64 storage; a cubic element is three consecutive words.
    pub fn upload_words(&self, words: &[u64], spec: &Spec) -> Result<Buf, Error> {
        let want = spec.elems() * spec.words_per_elem();
        if words.len() != want {
            return Err(format!("{}: {} words for a {:?} buffer of {want}", spec.name, words.len(), spec.dims).into());
        }
        if spec.elem_bytes() == 4 {
            return Err(format!("{}: 32-bit inputs upload through upload_i32", spec.name).into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(words.as_ptr() as *const u8, words.len() * 8) };
        Ok(self.upload_bytes(bytes, spec))
    }

    pub fn upload_i32(&self, values: &[i32], spec: &Spec) -> Result<Buf, Error> {
        if values.len() != spec.elems() || spec.elem_bytes() != 4 {
            return Err(format!("{}: {} values for a {:?} buffer", spec.name, values.len(), spec.dims).into());
        }
        let bytes = unsafe { std::slice::from_raw_parts(values.as_ptr() as *const u8, values.len() * 4) };
        Ok(self.upload_bytes(bytes, spec))
    }

    fn upload_bytes(&self, bytes: &[u8], spec: &Spec) -> Buf {
        let t = std::time::Instant::now();
        let buf = unsafe { self.session.input_buffer(bytes, &spec.dims, spec.buffer_type()) };
        if trace_enabled() && bytes.len() >= 1 << 20 {
            zzlog!("  upload {}: {} MB, {:.2} ms", spec.name, bytes.len() >> 20, t.elapsed().as_secs_f64() * 1e3);
        }
        Arc::new(DeviceBuf { session: self.session.clone(), buf: Some(buf) })
    }

    /// A device buffer's words on the host (32-bit outputs widened).
    pub fn download_words(&self, buf: &Buf, spec: &Spec) -> Result<Vec<u64>, Error> {
        let t = std::time::Instant::now();
        let bytes = unsafe { self.session.buffer_to_host(buf.raw()) };
        if trace_enabled() {
            zzlog!("  download {}: {} KB, {:.2} ms (waits for the work before it)", spec.name, bytes.len() / 1024, t.elapsed().as_secs_f64() * 1e3);
        }
        if bytes.len() != spec.elems() * spec.elem_bytes() {
            return Err(format!("{}: device buffer is {} bytes, spec says {}", spec.name, bytes.len(), spec.elems() * spec.elem_bytes()).into());
        }
        Ok(match spec.elem_bytes() {
            4 => bytes.chunks_exact(4).map(|c| u32::from_le_bytes(c.try_into().unwrap()) as u64).collect(),
            _ => bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect(),
        })
    }

    /// Execute `name` with its inputs bound from `env`; outputs in manifest order.
    pub fn run(&self, name: &str, env: &Env) -> Result<Vec<Buf>, Error> {
        let info = self.manifest.program(name)?;
        let mut args: Vec<&xla_pjrt::Buffer> = Vec::with_capacity(info.inputs.len());
        for spec in &info.inputs {
            let buf = env.get(&spec.name).ok_or_else(|| format!("{name}: missing input {}", spec.name))?;
            args.push(buf.raw());
        }
        let exe = self.executable(name)?;
        let t = std::time::Instant::now();
        let outs = {
            let _nvtx = crate::nvtx::Range::push(name);
            unsafe { self.session.run_buffers_to_device(&exe, &args, info.outputs.len()) }
        };
        if trace_enabled() {
            zzlog!("  run {name}: enqueue {:.2} ms", t.elapsed().as_secs_f64() * 1e3);
        }
        Ok(outs
            .into_iter()
            .map(|b| Arc::new(DeviceBuf { session: self.session.clone(), buf: Some(b) }))
            .collect())
    }

    /// `run`, with the outputs stored into `env` under their manifest names.
    /// A `rename` of `(from, to)` rewrites output names carrying the `from`
    /// prefix: the setup programs name their digest layers after themselves,
    /// the opening programs expect the tree's name.
    pub fn run_into(&self, name: &str, env: &mut Env, rename: Option<(&str, &str)>) -> Result<(), Error> {
        let outs = self.run(name, env)?;
        let info: &ProgramInfo = self.manifest.program(name)?;
        for (spec, buf) in info.outputs.iter().zip(outs) {
            let mut key = spec.name.clone();
            if let Some((from, to)) = rename {
                if let Some(rest) = key.strip_prefix(from) {
                    key = format!("{to}{rest}");
                }
            }
            env.insert(key, buf);
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{each_parallel, eager_module_loads_from, CacheEntryLock, Gate};
    use std::sync::mpsc;
    use std::time::Duration;

    #[test]
    fn each_parallel_runs_every_item_once_on_every_thread_count() {
        for threads in [1, 2, 8, 64] {
            let seen = std::sync::Mutex::new(Vec::new());
            let items: Vec<usize> = (0..50).collect();
            each_parallel(items, threads, |i| {
                seen.lock().unwrap().push(*i);
                Ok(())
            })
            .unwrap();
            let mut got = seen.into_inner().unwrap();
            got.sort_unstable();
            assert_eq!(got, (0..50).collect::<Vec<_>>(), "threads={threads}");
        }
    }

    #[test]
    fn each_parallel_reports_a_failure_and_still_drains_the_queue() {
        // Pinned at 1 as well as above it: the bridge's own preload runs the
        // serial path, so a failure there has to leave the cache in the same
        // state a warm at eight threads would.
        for threads in [1, 8] {
            let ran = std::sync::atomic::AtomicUsize::new(0);
            let err = each_parallel((0..4).collect(), threads, |i| {
                ran.fetch_add(1, std::sync::atomic::Ordering::SeqCst);
                if *i == 2 {
                    return Err("program 2 failed".into());
                }
                Ok(())
            })
            .unwrap_err();
            assert_eq!(err.to_string(), "program 2 failed", "threads={threads}");
            // The other three still ran: one failure must not strand the rest,
            // since the caller retries nothing.
            assert_eq!(ran.load(std::sync::atomic::Ordering::SeqCst), 4, "threads={threads}");
        }
    }

    #[test]
    fn each_parallel_returns_the_first_failure_when_several_fail() {
        for threads in [1, 8] {
            let err = each_parallel(vec![0, 1, 2], threads, |i| match i {
                0 => Ok(()),
                _ => Err(format!("item {i} failed").into()),
            })
            .unwrap_err();
            // At one thread "first" is positional; above it, whichever worker
            // lost the race -- so only assert it is one of the two failures.
            assert!(
                ["item 1 failed", "item 2 failed"].contains(&err.to_string().as_str()),
                "threads={threads} got {err}"
            );
        }
    }

    #[test]
    fn eager_module_loads_follow_the_preload_unless_overridden() {
        // Something preloads, so the loads happen off the prove path and may
        // as well pull the modules across with them.
        assert_eq!(eager_module_loads_from(None, None), Some(true));
        assert_eq!(eager_module_loads_from(None, Some("all")), Some(true));
        // Nothing preloads: every load is already inside a prove slot.
        assert_eq!(eager_module_loads_from(None, Some("0")), None);
        // The override moves this one thing on its own, which is what lets a
        // run measure it without also changing what gets preloaded.
        assert_eq!(eager_module_loads_from(Some("0"), None), None);
        assert_eq!(eager_module_loads_from(Some("1"), Some("0")), Some(true));
        // An empty value is unset, not "on": `ZZ_EAGER_MODULES=` falls through
        // to the preload the same way an absent one does, and an empty
        // `ZZ_PRELOAD` is the default (preloading) rather than `0`.
        assert_eq!(eager_module_loads_from(Some(""), Some("0")), None);
        assert_eq!(eager_module_loads_from(Some(""), None), Some(true));
        assert_eq!(eager_module_loads_from(None, Some("")), Some(true));
    }

    #[test]
    fn one_compile_per_cache_entry_at_a_time() {
        let path = std::path::Path::new("/nonexistent/zz-test/commit1-0.pjrt");
        let held = CacheEntryLock::take(path);
        let (tx, rx) = mpsc::channel();
        let second = std::thread::spawn(move || {
            let _l = CacheEntryLock::take(std::path::Path::new("/nonexistent/zz-test/commit1-0.pjrt"));
            tx.send(()).unwrap();
        });
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err(), "second compile of one entry ran concurrently");
        // A different entry is independent.
        let _other = CacheEntryLock::take(std::path::Path::new("/nonexistent/zz-test/commit2-0.pjrt"));
        drop(held);
        rx.recv_timeout(Duration::from_secs(5)).unwrap();
        second.join().unwrap();
    }

    #[test]
    fn a_prove_waits_for_loads_in_flight_and_holds_new_loads_back() {
        let gate = std::sync::Arc::new(Gate::default());
        let load = gate.enter_load();
        let (tx, rx) = mpsc::channel();
        let g = gate.clone();
        let prover = std::thread::spawn(move || {
            let pass = g.enter_prove();
            tx.send("proving").unwrap();
            // Hold the prove until told to finish.
            std::thread::sleep(Duration::from_millis(300));
            drop(pass);
        });
        assert!(rx.recv_timeout(Duration::from_millis(200)).is_err(), "prove entered under a load");
        // With a prove waiting, a new load must not slip in first.
        let (ltx, lrx) = mpsc::channel();
        let g = gate.clone();
        let loader = std::thread::spawn(move || {
            let _pass = g.enter_load();
            ltx.send("loaded").unwrap();
        });
        assert!(lrx.recv_timeout(Duration::from_millis(100)).is_err(), "load entered ahead of a waiting prove");
        drop(load);
        assert_eq!(rx.recv_timeout(Duration::from_secs(5)).unwrap(), "proving");
        // The new load waits for the prove to finish.
        assert_eq!(lrx.recv_timeout(Duration::from_secs(5)).unwrap(), "loaded");
        prover.join().unwrap();
        loader.join().unwrap();
        assert!(gate.started_proving());
    }
}
