//! pil2's Fiat-Shamir transcript for the host half of the schedule —
//! `fields::Transcript` (the Rust twin of `TranscriptGL`, width 16) under
//! the key's Poseidon family. The bridge byte-matches pil2 by running
//! pil2's own transcript, not by re-implementing it.

use fields::{Goldilocks, PrimeField64, TranscriptP1_16, TranscriptP2_16};

use crate::Error;

pub const DIGEST: usize = 4;

pub enum HostTranscript {
    Poseidon1(TranscriptP1_16<Goldilocks>),
    Poseidon2(TranscriptP2_16<Goldilocks>),
}

impl HostTranscript {
    pub fn new(hash_family: &str) -> Result<Self, Error> {
        match hash_family {
            "Poseidon1" => Ok(Self::Poseidon1(TranscriptP1_16::new())),
            "Poseidon2" => Ok(Self::Poseidon2(TranscriptP2_16::new())),
            other => Err(format!("unknown hash family {other}").into()),
        }
    }

    /// A fresh transcript of the same family — the inner sponge of a
    /// hashed absorb, and the grinding-seeded query draw.
    pub fn fresh(&self) -> Self {
        match self {
            Self::Poseidon1(_) => Self::Poseidon1(TranscriptP1_16::new()),
            Self::Poseidon2(_) => Self::Poseidon2(TranscriptP2_16::new()),
        }
    }

    pub fn put(&mut self, words: &[u64]) {
        let elems: Vec<Goldilocks> = words.iter().map(|w| Goldilocks::new(*w)).collect();
        match self {
            Self::Poseidon1(t) => t.put(&elems),
            Self::Poseidon2(t) => t.put(&elems),
        }
    }

    /// One cubic challenge.
    pub fn get_field(&mut self) -> [u64; 3] {
        let mut out = [Goldilocks::new(0); 3];
        match self {
            Self::Poseidon1(t) => t.get_field(&mut out),
            Self::Poseidon2(t) => t.get_field(&mut out),
        }
        [out[0].as_canonical_u64(), out[1].as_canonical_u64(), out[2].as_canonical_u64()]
    }

    /// Flush and read the state's first `DIGEST` lanes (pil2 `getState`).
    pub fn get_state(&mut self) -> [u64; DIGEST] {
        let state = match self {
            Self::Poseidon1(t) => t.get_state(),
            Self::Poseidon2(t) => t.get_state(),
        };
        let mut out = [0u64; DIGEST];
        for (o, s) in out.iter_mut().zip(state.iter()) {
            *o = s.as_canonical_u64();
        }
        out
    }

    pub fn get_permutations(&mut self, n: usize, n_bits: u32) -> Vec<u64> {
        match self {
            Self::Poseidon1(t) => t.get_permutations(n as u64, n_bits as u64),
            Self::Poseidon2(t) => t.get_permutations(n as u64, n_bits as u64),
        }
    }

    /// pil2's section absorb: the raw words, or under a `hashCommits`
    /// stark struct their `calculateHash` digest (a fresh transcript's
    /// flushed state).
    pub fn absorb_section(&mut self, words: &[u64], hashed: bool) {
        if hashed {
            let mut inner = self.fresh();
            inner.put(words);
            let digest = inner.get_state();
            self.put(&digest);
        } else {
            self.put(words);
        }
    }
}
