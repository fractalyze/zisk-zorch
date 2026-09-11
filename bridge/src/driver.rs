//! genProof's schedule over one artifact, step for step the same as
//! `zisk_zorch/export/replay.py`: one `AirDriver` per (session, artifact),
//! keeping the key's fixed sections resident so a prove uploads only the
//! instance.

use std::collections::HashMap;
use std::sync::Arc;

use crate::artifact::{Artifact, Buf, Env};
use crate::manifest::{Manifest, StageOnly};
use crate::transcript::{HostTranscript, DIGEST};
use crate::Error;

/// The proving key's base-domain fixed sections, as row-major u64 words.
pub struct FixedSections<'a> {
    /// (2^nBits, nConstants)
    pub const_base: &'a [u64],
    /// commitId -> (2^nBits, width)
    pub custom_base: HashMap<usize, &'a [u64]>,
    /// The same sections already on the device (`upload_fixed`), when the
    /// caller could upload them while another prove had the client.
    pub uploaded: Option<UploadedFixed>,
}

/// A key's fixed sections on the device, uploaded before the prove that
/// needs them took the client.
#[derive(Clone)]
pub struct UploadedFixed {
    const_base: Buf,
    /// commitId -> the base section
    custom_base: HashMap<usize, Buf>,
}

/// Upload the key's fixed sections, no more: a prove queued behind another
/// on the same client can do this before its turn, so the slot pays only
/// the setup programs over them. A table AIR's `const_base` is 1.2-1.4 GB
/// read from pageable host memory, and under the slot the client idles for
/// it. Running the setup programs ahead too does not fit — `logup` reads
/// `const_base` through the prove, so a second AIR's sections would have to
/// live beside the running prove's whole working set.
pub fn upload_fixed(art: &Artifact, fixed: &FixedSections) -> Result<UploadedFixed, Error> {
    let m = &art.manifest;
    let const_base = art.upload_words(fixed.const_base, &m.program("const_setup")?.inputs[0])?;
    let mut custom_base = HashMap::new();
    for cc in &m.custom_commits {
        let words = fixed
            .custom_base
            .get(&cc.id)
            .ok_or_else(|| format!("upload_fixed: custom commit {} section missing", cc.id))?;
        let spec = &m.program(&format!("custom_setup_{}", cc.id))?.inputs[0];
        custom_base.insert(cc.id, art.upload_words(words, spec)?);
    }
    Ok(UploadedFixed { const_base, custom_base })
}

/// One `StepsParams` worth of host inputs, as canonical u64 words.
pub struct InstanceInputs<'a> {
    /// (2^nBits, cm1 width)
    pub trace: &'a [u64],
    pub publics: &'a [u64],
    /// dumped packing
    pub airvalues: &'a [u64],
    /// dumped packing
    pub proofvalues: &'a [u64],
    pub global_challenge: &'a [u64],
    /// The same sections already on the device (`upload_inputs`), when the
    /// caller could upload them while another prove had the client.
    pub uploaded: Option<Uploaded>,
}

pub struct Uploaded {
    pub trace: Buf,
    pub publics: Buf,
    pub airvalues: Buf,
    pub proofvalues: Buf,
}

/// Upload an instance's sections. Separate from `prove` so a prove queued
/// behind another on the same client can have its uploads (a gigabyte for
/// Main, DMA'd straight out of pageable host memory: the plugin stages a
/// transfer through pinned memory only below `staging_threshold_bytes`,
/// which defaults to 1 GiB) done before its turn. The caller hands the
/// result back on `InstanceInputs::uploaded`, which `prove` takes: the
/// buffers belong to that prove, and a second handle kept anywhere else
/// would outlive the releases below and hold the trace to the last opening.
pub fn upload_inputs(art: &Artifact, inp: &InstanceInputs) -> Result<Uploaded, Error> {
    let m = &art.manifest;
    let spec = |prog: &str, input: &str| -> Result<crate::manifest::Spec, Error> {
        m.program(prog)?.input(input).cloned().ok_or_else(|| format!("{prog}: no input {input}").into())
    };
    Ok(Uploaded {
        publics: art.upload_words(inp.publics, &spec("logup", "publics")?)?,
        airvalues: art.upload_words(inp.airvalues, &spec("logup", "airvalues")?)?,
        proofvalues: art.upload_words(inp.proofvalues, &spec("logup", "proofvalues")?)?,
        trace: art.upload_words(inp.trace, &spec("commit1", "trace")?)?,
    })
}

#[derive(Clone, Debug, Default)]
pub struct ProveOutputs {
    /// (n, 3)
    pub airgroupvalues: Vec<u64>,
    /// dumped packing, after the stage-2 hints
    pub airvalues: Vec<u64>,
    pub nonce: u64,
}

pub struct AirDriver {
    artifact: Arc<Artifact>,
    fixed: Option<Env>,
}

struct TreeOpening {
    rows: Vec<u64>,
    paths: Vec<u64>,
    last_level: Vec<u64>,
}

/// `proof_serializer._n_siblings`: the path levels the wire carries.
fn n_siblings(n_bits: u32, arity: usize, llv: u32) -> usize {
    let per_level = (arity as f64).log2();
    (n_bits as f64 / per_level).ceil() as usize - llv as usize
}

/// `proof_serializer._values3`: dumped packing -> three words per entry.
fn push_values3(out: &mut Vec<u64>, words: &[u64], stages: &[StageOnly]) {
    for (off, width) in Manifest::value_offsets(stages) {
        for k in 0..3 {
            out.push(if k < width { words[off + k] } else { 0 });
        }
    }
}

/// Let go of one tree once its openings are on the wire: the extended
/// section it was built over and every digest layer above it. Nothing later
/// in a prove reads either.
fn release_tree(env: &mut Env, section: &str, layers_prefix: &str) {
    env.remove(section);
    env.retain(|name, _| !name.starts_with(layers_prefix));
}

impl AirDriver {
    pub fn new(artifact: Arc<Artifact>) -> AirDriver {
        AirDriver { artifact, fixed: None }
    }

    pub fn manifest(&self) -> &Manifest {
        &self.artifact.manifest
    }

    pub fn artifact(&self) -> &Artifact {
        &self.artifact
    }

    pub fn has_fixed(&self) -> bool {
        self.fixed.is_some()
    }

    /// Release the resident fixed sections (the compiled programs stay);
    /// the next prove runs the setup programs again.
    pub fn drop_fixed(&mut self) {
        self.fixed = None;
    }

    /// Run the setup programs over the key's fixed sections, uploading them
    /// first unless the caller already did (`upload_fixed`).
    pub fn set_fixed(&mut self, fixed: &FixedSections) -> Result<(), Error> {
        let art = &self.artifact;
        let m = &art.manifest;
        let mut env = Env::new();
        art.run_into("constants", &mut env, None)?;
        let up = match &fixed.uploaded {
            Some(u) => u.clone(),
            None => upload_fixed(art, fixed)?,
        };
        env.insert("const_base".into(), up.const_base);
        art.run_into("const_setup", &mut env, Some(("const_setup_layers_", "const_layers_")))?;
        for cc in &m.custom_commits {
            let buf = up
                .custom_base
                .get(&cc.id)
                .ok_or_else(|| format!("set_fixed: custom commit {} was not uploaded", cc.id))?;
            let prog = format!("custom_setup_{}", cc.id);
            env.insert(format!("custom_base_{}", cc.id), buf.clone());
            let from = format!("{prog}_layers_");
            let to = format!("custom_layers_{}_", cc.id);
            art.run_into(&prog, &mut env, Some((&from, &to)))?;
        }
        // The quotient's row windows are fixed per AIR: resident, so the
        // chunks enqueue back to back instead of each waiting on an upload
        // queued behind the previous chunk.
        if m.quotient_chunks.len() > 1 {
            let mut start = 0i32;
            for (k, size) in m.quotient_chunks.iter().enumerate() {
                let spec = m.program(&format!("quotient_{size}"))?.input("rows").cloned().ok_or("quotient: no rows input")?;
                let rows: Vec<i32> = (start..start + *size as i32).collect();
                env.insert(format!("rows_{k}"), art.upload_i32(&rows, &spec)?);
                start += *size as i32;
            }
        }
        self.fixed = Some(env);
        Ok(())
    }

    /// The flat wire proof's length in u64 words.
    pub fn proof_words(&self) -> usize {
        Self::proof_words_of(&self.artifact.manifest)
    }

    pub fn proof_words_of(m: &Manifest) -> usize {
        let per_level = (m.arity - 1) * DIGEST;
        let llv = m.last_level_verification;
        // pil2's `getProofSize`: a last level only when one is verified.
        let last_level = if llv > 0 { m.arity.pow(llv) * DIGEST } else { 0 };
        let tree_block = |width: usize, n_bits: u32| {
            m.n_queries * (width + n_siblings(n_bits, m.arity, llv) * per_level) + last_level
        };
        let mut n = 0;
        n += m.airgroupvalues.len() * 3;
        n += m.airvalues.len() * 3;
        n += 3 * DIGEST;
        n += m.ev_map_size * 3;
        n += tree_block(m.n_constants, m.n_bits_ext);
        for cc in &m.custom_commits {
            n += tree_block(cc.width, m.n_bits_ext);
        }
        n += tree_block(m.widths.cm1, m.n_bits_ext);
        n += tree_block(m.widths.cm2, m.n_bits_ext);
        n += tree_block(m.widths.qsec, m.n_bits_ext);
        let rounds = m.steps.len() - 1;
        n += rounds * DIGEST;
        for i in 0..rounds {
            let n_x = 1usize << (m.steps[i] - m.steps[i + 1]);
            n += tree_block(n_x * 3, m.steps[i + 1]);
        }
        n += (1usize << m.steps[rounds]) * 3;
        n + 1
    }

    fn open_tree(&self, name: &str, width: usize, n_bits: u32, env: &Env, positions: &Buf) -> Result<TreeOpening, Error> {
        let art = &self.artifact;
        let m = &art.manifest;
        let prog = format!("open_{name}");
        let mut penv = env.clone();
        penv.insert("positions".into(), positions.clone());
        let outs = art.run(&prog, &penv)?;
        let info = m.program(&prog)?;
        let flat = art.download_words(&outs[0], &info.outputs[0])?;
        let last_level = art.download_words(&outs[1], &info.outputs[1])?;
        let nq = info.outputs[0].dims[0] as usize;
        let row_words = info.outputs[0].dims[1] as usize;
        let per_level = (m.arity - 1) * DIGEST;
        let n_levels = (row_words - width) / per_level;
        let n_sib = n_siblings(n_bits, m.arity, m.last_level_verification);
        if n_sib > n_levels {
            return Err(format!("{prog}: {n_levels} path levels, wire wants {n_sib}").into());
        }
        let mut rows = Vec::with_capacity(nq * width);
        let mut paths = Vec::with_capacity(nq * n_sib * per_level);
        for q in 0..nq {
            let row = &flat[q * row_words..(q + 1) * row_words];
            rows.extend_from_slice(&row[..width]);
            paths.extend_from_slice(&row[width..width + n_sib * per_level]);
        }
        Ok(TreeOpening { rows, paths, last_level })
    }

    /// Prove one instance through the artifacts, writing `proof_words()`
    /// words into `proof_out`.
    pub fn prove(&self, mut inp: InstanceInputs, transcript: &mut HostTranscript, proof_out: &mut [u64]) -> Result<ProveOutputs, Error> {
        let fixed = self.fixed.as_ref().ok_or("prove: set_fixed first")?;
        let art = &self.artifact;
        let m = &art.manifest;
        let nbe = m.n_bits_ext;
        let in_spec = |prog: &str, input: &str| -> Result<crate::manifest::Spec, Error> {
            m.program(prog)?.input(input).cloned().ok_or_else(|| format!("{prog}: no input {input}").into())
        };
        let out_spec = |prog: &str, output: &str| -> Result<crate::manifest::Spec, Error> {
            m.program(prog)?.output(output).cloned().ok_or_else(|| format!("{prog}: no output {output}").into())
        };
        let squeeze = |t: &mut HostTranscript, challenges: &mut Vec<u64>, stage: u32| {
            for id in m.stage_challenge_ids(stage) {
                let c = t.get_field();
                challenges[id * 3..id * 3 + 3].copy_from_slice(&c);
            }
        };

        // One phase per stage of the schedule. The program ranges nest inside
        // them, so what a phase keeps to itself is the host work between two
        // enqueues — the downloads, the transcript, the query draw — which is
        // what `bench/host_idle.py` charges the device's idle time to.
        let mut phase = crate::nvtx::Phase::start("stage1");
        // Scalars ride PACKED, as the instance dumped them; the stage-2 hints
        // rewrite the air values below and every later program reads those.
        let mut env = fixed.clone();
        // Taken rather than cloned: from here the env holds the only handle
        // to each uploaded section, so removing one below actually frees it.
        let up = match inp.uploaded.take() {
            Some(u) => u,
            None => upload_inputs(art, &inp)?,
        };
        env.insert("publics".into(), up.publics);
        env.insert("airvalues".into(), up.airvalues);
        env.insert("proofvalues".into(), up.proofvalues);
        env.insert("trace".into(), up.trace);
        // Non-recursive schedule: the seed already binds root1 through the
        // contributions phase, so root1 itself is never absorbed, and the
        // stage-2 challenges are known before commit1 runs. Everything up to
        // commit2 is enqueued before the first wait: an upload or download
        // here would sit behind the queued work, idling the device for one
        // enqueue latency per stage.
        transcript.put(inp.global_challenge);
        let mut challenges = vec![0u64; m.challenges.len() * 3];
        squeeze(transcript, &mut challenges, 2);
        env.insert("challenges".into(), art.upload_words(&challenges, &in_spec("logup", "challenges")?)?);
        if m.witness_calc {
            let trace = art.run("witness_calc", &env)?.remove(0);
            env.insert("trace".into(), trace);
        }
        art.run_into("commit1", &mut env, None)?;
        art.run_into("logup", &mut env, None)?;
        // Nothing after logup reads the base trace, and nothing after
        // commit2 the base cm2: a gigabyte or more each on a wide AIR,
        // released before the quotient's peak (the plugin defers the free
        // until the enqueued work is done). `Artifact::run` binds every
        // input by name, so an export that gained a later reader fails on
        // the missing bind rather than proving against a buffer that is
        // gone.
        env.remove("trace");
        art.run_into("commit2", &mut env, None)?;
        env.remove("cm2");
        phase.set("stage2");
        let mut result = ProveOutputs::default();
        result.airvalues = art.download_words(&env["airvalues"], &out_spec("logup", "airvalues")?)?;
        let root2 = art.download_words(&env["root2"], &out_spec("commit2", "root2")?)?;
        transcript.put(&root2);
        for (i, (off, _)) in Manifest::value_offsets(&m.airvalues).into_iter().enumerate() {
            if m.airvalues[i].stage == 2 {
                transcript.put(&result.airvalues[off..off + 3]);
            }
        }
        result.airgroupvalues = vec![0u64; m.airgroupvalues.len() * 3];
        if let Some(buf) = env.get("airgroupvalue") {
            let agv = art.download_words(buf, &out_spec("logup", "airgroupvalue")?)?;
            let idx = m.airgroupvalue_index.ok_or("manifest: airgroupvalue without an index")?;
            result.airgroupvalues[idx * 3..idx * 3 + 3].copy_from_slice(&agv);
        }

        squeeze(transcript, &mut challenges, m.n_stages + 1);
        phase.set("quotient");
        env.insert("challenges".into(), art.upload_words(&challenges, &in_spec("logup", "challenges")?)?);
        let single = m.quotient_chunks.len() == 1;
        let qprog0 = if single { "quotient".to_string() } else { format!("quotient_{}", m.quotient_chunks[0]) };
        env.insert("airgroupvalues".into(), art.upload_words(&result.airgroupvalues, &in_spec(&qprog0, "airgroupvalues")?)?);
        let mut qenv = Env::new();
        if single {
            qenv.insert("q_0".into(), art.run("quotient", &env)?.remove(0));
        } else {
            for (k, size) in m.quotient_chunks.iter().enumerate() {
                let prog = format!("quotient_{size}");
                let rows = env.get(&format!("rows_{k}")).cloned().ok_or("set_fixed: quotient row windows missing")?;
                env.insert("rows".into(), rows);
                qenv.insert(format!("q_{k}"), art.run(&prog, &env)?.remove(0));
            }
            env.remove("rows");
        }
        let outs = art.run("quotient_commit", &qenv)?;
        drop(qenv);
        for (spec, buf) in m.program("quotient_commit")?.outputs.iter().zip(outs) {
            env.insert(spec.name.clone(), buf);
        }
        let rootq = art.download_words(&env["rootq"], &out_spec("quotient_commit", "rootq")?)?;
        transcript.put(&rootq);

        squeeze(transcript, &mut challenges, m.n_stages + 2);
        phase.set("evals");
        let xi_id = m.challenge_id("std_xi")?;
        env.insert("xi".into(), art.upload_words(&challenges[xi_id * 3..xi_id * 3 + 3], &in_spec("lev", "xi")?)?);
        let lev = art.run("lev", &env)?.remove(0);
        env.insert("lev".into(), lev);
        let evals_buf = art.run("evals", &env)?.remove(0);
        env.remove("lev");
        env.insert("evals".into(), evals_buf);
        let evals = art.download_words(&env["evals"], &out_spec("evals", "evals")?)?;
        transcript.absorb_section(&evals, m.hash_commits);
        squeeze(transcript, &mut challenges, m.n_stages + 3);
        phase.set("deep");
        let vf1 = m.challenge_id("std_vf1")?;
        let vf2 = m.challenge_id("std_vf2")?;
        env.insert("vf1".into(), art.upload_words(&challenges[vf1 * 3..vf1 * 3 + 3], &in_spec("deep", "vf1")?)?);
        env.insert("vf2".into(), art.upload_words(&challenges[vf2 * 3..vf2 * 3 + 3], &in_spec("deep", "vf2")?)?);
        let mut codeword = art.run("deep", &env)?.remove(0);

        phase.set("fri");
        let rounds = m.steps.len() - 1;
        let mut fri_roots = Vec::with_capacity(rounds * DIGEST);
        let mut fri_layers: Vec<Env> = Vec::with_capacity(rounds);
        for i in 0..rounds {
            let commit = format!("fri_commit_{i}");
            let fold = format!("fri_fold_{i}");
            let mut fenv = Env::new();
            fenv.insert("codeword".into(), codeword.clone());
            let mut layer = Env::new();
            let outs = art.run(&commit, &fenv)?;
            for (spec, buf) in m.program(&commit)?.outputs.iter().zip(outs) {
                layer.insert(spec.name.clone(), buf);
            }
            let root_name = format!("fri_root_{i}");
            let root = art.download_words(&layer[&root_name], &out_spec(&commit, &root_name)?)?;
            transcript.put(&root);
            fri_roots.extend_from_slice(&root);
            let beta = transcript.get_field();
            fenv.insert("beta".into(), art.upload_words(&beta, &in_spec(&fold, "beta")?)?);
            codeword = art.run(&fold, &fenv)?.remove(0);
            fri_layers.push(layer);
        }
        phase.set("fri_final");
        let mut fenv = Env::new();
        fenv.insert("codeword".into(), codeword);
        let final_buf = art.run("fri_final", &fenv)?.remove(0);
        let final_pol = art.download_words(&final_buf, &out_spec("fri_final", "final_pol")?)?;
        transcript.absorb_section(&final_pol, m.hash_commits);
        phase.set("grind");
        let challenge = transcript.get_field();
        let mut genv = Env::new();
        genv.insert("challenge".into(), art.upload_words(&challenge, &in_spec("grind", "challenge")?)?);
        let nonce_buf = art.run("grind", &genv)?.remove(0);
        result.nonce = art.download_words(&nonce_buf, &out_spec("grind", "nonce")?)?[0];
        let positions = {
            let mut seeded = transcript.fresh();
            seeded.put(&challenge);
            seeded.put(&[result.nonce]);
            seeded.get_permutations(m.n_queries, nbe)
        };
        let pos_ext = art.upload_words(&positions, &in_spec("open_cm1", "positions")?)?;

        phase.set("openings");
        // The wire, in `proof2pointer` order (`proof_serializer.serialize_proof`).
        let mut proof: Vec<u64> = Vec::with_capacity(self.proof_words());
        push_values3(&mut proof, &result.airgroupvalues, &m.airgroupvalues);
        push_values3(&mut proof, &result.airvalues, &m.airvalues);
        proof.extend(art.download_words(&env["root1"], &out_spec("commit1", "root1")?)?);
        proof.extend(root2);
        proof.extend(rootq);
        proof.extend(evals);
        let push_tree = |proof: &mut Vec<u64>, t: TreeOpening| {
            proof.extend(t.rows);
            proof.extend(t.paths);
            proof.extend(t.last_level);
        };
        // The key's trees belong to the resident set and stay: the next prove
        // of this AIR reads them and a setup program is what it costs to
        // rebuild them. The stage trees are this prove's own, and each goes
        // as its openings reach the wire — otherwise a wide AIR carries
        // `cm1_ext` and its digest layers through every later opening,
        // beside the next prove's uploads.
        push_tree(&mut proof, self.open_tree("const", m.n_constants, nbe, &env, &pos_ext)?);
        for cc in &m.custom_commits {
            push_tree(&mut proof, self.open_tree(&format!("custom_{}", cc.id), cc.width, nbe, &env, &pos_ext)?);
        }
        push_tree(&mut proof, self.open_tree("cm1", m.widths.cm1, nbe, &env, &pos_ext)?);
        release_tree(&mut env, "cm1_ext", "cm1_layers_");
        push_tree(&mut proof, self.open_tree("cm2", m.widths.cm2, nbe, &env, &pos_ext)?);
        release_tree(&mut env, "cm2_ext", "cm2_layers_");
        push_tree(&mut proof, self.open_tree("qsec", m.widths.qsec, nbe, &env, &pos_ext)?);
        release_tree(&mut env, "qsec", "qsec_layers_");
        proof.extend(fri_roots);
        // `into_iter`: each round's leaves and layers go at the end of its
        // own iteration, not at the end of the prove.
        for (i, layer) in fri_layers.into_iter().enumerate() {
            let leaf_bits = m.steps[i + 1];
            let n_x = 1usize << (m.steps[i] - leaf_bits);
            let mask = (1u64 << leaf_bits) - 1;
            let folded: Vec<u64> = positions.iter().map(|p| p & mask).collect();
            let name = format!("fri_{i}");
            let pos = art.upload_words(&folded, &in_spec(&format!("open_{name}"), "positions")?)?;
            push_tree(&mut proof, self.open_tree(&name, n_x * 3, leaf_bits, &layer, &pos)?);
        }
        proof.extend(final_pol);
        proof.push(result.nonce);
        if proof.len() != self.proof_words() || proof_out.len() < proof.len() {
            return Err(format!("proof is {} words, layout says {}, buffer holds {}", proof.len(), self.proof_words(), proof_out.len()).into());
        }
        proof_out[..proof.len()].copy_from_slice(&proof);
        Ok(result)
    }
}
