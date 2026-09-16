//! The exported artifact's manifest (`zisk_zorch/export/export_air.py`):
//! the schedule facts and, per program, every input and output by name,
//! dtype and shape in parameter order. The driver binds buffers by these
//! names, so a program's signature lives here, not in the Rust.
//!
//! An artifact is one SHAPE, not one set of constants. An aggregation
//! family's AIRs share a starkinfo, so one `recursive1` artifact serves
//! every basic AIR's `recursive1` key and the manifest says nothing about
//! which. What is on a client is therefore identified by the const file,
//! not by the artifact: residency, the read-ahead and proofman's plan all
//! key on `const_pols_path` (`lib.rs::fixed_key`), which for a basic AIR
//! is one-to-one with the artifact and needs no mapping either way.

use std::collections::BTreeMap;
use std::path::Path;

use serde::Deserialize;

use crate::Error;

#[derive(Clone, Debug, Deserialize)]
pub struct Spec {
    pub name: String,
    /// goldilocks | goldilocksx3 | uint64 | uint32 | int32
    pub dtype: String,
    pub dims: Vec<i64>,
}

impl Spec {
    pub fn elems(&self) -> usize {
        self.dims.iter().map(|d| *d as usize).product()
    }

    /// Host u64 words per element (the cubic dtype carries three).
    pub fn words_per_elem(&self) -> usize {
        if self.dtype == "goldilocksx3" {
            3
        } else {
            1
        }
    }

    pub fn elem_bytes(&self) -> usize {
        match self.dtype.as_str() {
            "goldilocks" | "uint64" => 8,
            "goldilocksx3" => 24,
            "uint32" | "int32" => 4,
            other => panic!("manifest: unknown dtype {other}"),
        }
    }

    pub fn buffer_type(&self) -> xla_pjrt::sys::PJRT_Buffer_Type {
        use xla_pjrt::sys::*;
        match self.dtype.as_str() {
            "goldilocks" => PJRT_Buffer_Type_GOLDILOCKS,
            "goldilocksx3" => PJRT_Buffer_Type_GOLDILOCKSX3,
            "uint64" => PJRT_Buffer_Type_U64,
            "uint32" => PJRT_Buffer_Type_U32,
            "int32" => PJRT_Buffer_Type_S32,
            other => panic!("manifest: unknown dtype {other}"),
        }
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ProgramInfo {
    pub file: String,
    pub inputs: Vec<Spec>,
    pub outputs: Vec<Spec>,
}

impl ProgramInfo {
    pub fn input(&self, name: &str) -> Option<&Spec> {
        self.inputs.iter().find(|s| s.name == name)
    }
    pub fn output(&self, name: &str) -> Option<&Spec> {
        self.outputs.iter().find(|s| s.name == name)
    }
}

#[derive(Clone, Debug, Deserialize)]
pub struct ChallengeInfo {
    pub id: usize,
    pub name: Option<String>,
    pub stage: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct CustomCommitInfo {
    pub id: usize,
    pub width: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct StageOnly {
    pub stage: u32,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Widths {
    pub cm1: usize,
    pub cm2: usize,
    pub qsec: usize,
}

#[derive(Clone, Debug, Deserialize)]
pub struct Manifest {
    pub air: String,
    pub artifact_version: u32,
    pub n_bits: u32,
    pub n_bits_ext: u32,
    pub hash_family: String,
    pub arity: usize,
    pub steps: Vec<u32>,
    pub n_queries: usize,
    pub pow_bits: u32,
    pub hash_commits: bool,
    pub last_level_verification: u32,
    pub n_stages: u32,
    pub n_publics: usize,
    pub n_constants: usize,
    pub widths: Widths,
    pub custom_commits: Vec<CustomCommitInfo>,
    pub ev_map_size: usize,
    pub opening_points: Vec<i64>,
    pub challenges: Vec<ChallengeInfo>,
    pub airvalues: Vec<StageOnly>,
    pub airgroupvalues: Vec<StageOnly>,
    pub proofvalues: Vec<StageOnly>,
    pub airgroupvalue_index: Option<usize>,
    pub quotient_chunks: Vec<usize>,
    /// The row windows `deep` is dispatched over, in domain order. Unlike
    /// `quotient_chunks` there is no whole-domain program behind it: the
    /// batch is always windowed, because a dispatch boundary is the only
    /// thing that divides its temporaries.
    pub deep_chunks: Vec<usize>,
    /// The row windows `evals` is dispatched over, in domain order, counted
    /// in BASE rows — the domain its sum reduces over. Always windowed, for
    /// `deep_chunks`' reason.
    pub evals_chunks: Vec<usize>,
    /// Clients the export was sized for (`ZISK_CLIENTS`): the quotient's
    /// window count is compiled in, so an artifact is sized for a client
    /// count rather than adapting to one.
    pub clients: usize,
    pub witness_calc: bool,
    /// The aggregation schedule (`gen_proof.hpp`'s `recursive`): the
    /// transcript seeds from the circuit's verkey, the publics and root1
    /// instead of the contributions phase's global challenge. A manifest
    /// without the field is a basic AIR, which is not recursive.
    #[serde(default)]
    pub recursive: bool,
    pub programs: BTreeMap<String, ProgramInfo>,
}

impl Manifest {
    pub fn load(dir: &Path) -> Result<Manifest, Error> {
        let path = dir.join("manifest.json");
        let text = std::fs::read_to_string(&path)
            .map_err(|e| format!("manifest: cannot read {}: {e}", path.display()))?;
        let m: Manifest =
            serde_json::from_str(&text).map_err(|e| format!("manifest: {}: {e}", path.display()))?;
        if m.artifact_version != 1 {
            return Err(format!("manifest: unsupported artifact_version {}", m.artifact_version).into());
        }
        Ok(m)
    }

    pub fn program(&self, name: &str) -> Result<&ProgramInfo, Error> {
        self.programs.get(name).ok_or_else(|| format!("manifest: no program {name}").into())
    }

    pub fn challenge_id(&self, name: &str) -> Result<usize, Error> {
        self.challenges
            .iter()
            .find(|c| c.name.as_deref() == Some(name))
            .map(|c| c.id)
            .ok_or_else(|| format!("manifest: no challenge named {name}").into())
    }

    pub fn stage_challenge_ids(&self, stage: u32) -> Vec<usize> {
        self.challenges.iter().filter(|c| c.stage == stage).map(|c| c.id).collect()
    }

    /// `pil2.value_offsets`: each packed value-section entry's start and
    /// width. pil2 packs a first-stage value as one word and any later one
    /// as a cubic triple.
    pub fn value_offsets(stages: &[StageOnly]) -> Vec<(usize, usize)> {
        let mut out = Vec::with_capacity(stages.len());
        let mut off = 0;
        for s in stages {
            let width = if s.stage == 1 { 1 } else { 3 };
            out.push((off, width));
            off += width;
        }
        out
    }

    pub fn packed_width(stages: &[StageOnly]) -> usize {
        Self::value_offsets(stages).last().map(|(off, w)| off + w).unwrap_or(0)
    }
}
