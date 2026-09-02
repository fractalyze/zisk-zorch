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
/// Inputs bound by manifest name.
pub type Env = HashMap<String, Buf>;

/// A bridge client. Without a `memory_fraction` the allocator grows on
/// demand and stays out of everyone's way; with one it claims that share
/// of the card up front, so a co-tenant that sizes itself from free memory
/// (pil2's stream buffers) leaves it room.
pub fn new_session(memory_fraction: Option<f32>) -> Arc<Session> {
    let options = match memory_fraction {
        Some(f) => SessionOptions { preallocate: Some(true), memory_fraction: Some(f) },
        None => SessionOptions { preallocate: Some(false), memory_fraction: None },
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

pub struct Artifact {
    pub manifest: Manifest,
    dir: PathBuf,
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
    pub fn load(session: Arc<Session>, dir: &Path, cache: Option<&Path>) -> Result<Artifact, Error> {
        let cache = match cache {
            Some(c) => {
                let sub = c.join(dir.file_name().map(|n| n.to_string_lossy().into_owned()).unwrap_or_default());
                std::fs::create_dir_all(&sub).map_err(|e| format!("cannot create {}: {e}", sub.display()))?;
                Some(sub)
            }
            None => None,
        };
        Ok(Artifact {
            manifest: Manifest::load(dir)?,
            dir: dir.to_path_buf(),
            session,
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
        let cached = self.cache.as_ref().map(|c| c.join(format!("{name}-{:016x}.pjrt", fnv1a64(&code))));
        let exe = match cached.as_ref().and_then(|p| std::fs::read(p).ok()) {
            Some(bytes) => {
                self.cache_hits.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                unsafe { self.session.deserialize_and_load(&bytes) }
            }
            None => {
                let exe = unsafe { self.session.compile(&code) };
                if let Some(p) = &cached {
                    let bytes = unsafe { self.session.serialize(&exe) };
                    let tmp = p.with_extension(format!("tmp{}", std::process::id()));
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

    /// Compile every program now rather than on first use.
    pub fn compile_all(&self) -> Result<(), Error> {
        let names: Vec<String> = self.manifest.programs.keys().cloned().collect();
        for name in names {
            self.executable(&name)?;
        }
        Ok(())
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
        let buf = unsafe { self.session.input_buffer(bytes, &spec.dims, spec.buffer_type()) };
        Arc::new(DeviceBuf { session: self.session.clone(), buf: Some(buf) })
    }

    /// A device buffer's words on the host (32-bit outputs widened).
    pub fn download_words(&self, buf: &Buf, spec: &Spec) -> Result<Vec<u64>, Error> {
        let bytes = unsafe { self.session.buffer_to_host(buf.raw()) };
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
        let outs = unsafe { self.session.run_buffers_to_device(&exe, &args, info.outputs.len()) };
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
