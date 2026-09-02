//! The in-process A/B byte-gate. Under `ZZ_AB=1` proofman keeps proving
//! through pil2's own `gen_proof_c` and additionally runs the bridge into
//! a scratch buffer per instance; pil2's proof lands asynchronously
//! through the stream collectors, so the comparison happens once the
//! basic phase has collected every proof (`compare` from proofman).

use std::collections::BTreeMap;
use std::sync::Mutex;

static RECORDED: Mutex<BTreeMap<usize, Vec<u64>>> = Mutex::new(BTreeMap::new());
/// Verdicts taken at the moment pil2's proof was consumed: (differing words, first index).
static VERDICTS: Mutex<BTreeMap<usize, (usize, usize)>> = Mutex::new(BTreeMap::new());

fn diff(ours: &[u64], theirs: &[u64]) -> (usize, usize) {
    let n = ours.len().max(theirs.len());
    let bad: Vec<usize> = (0..n).filter(|i| ours.get(*i) != theirs.get(*i)).collect();
    (bad.len(), bad.first().copied().unwrap_or(0))
}

/// Compare pil2's proof for `instance_id` against the recorded bridge proof
/// the moment proofman consumes it (the slot is emptied for the recursion
/// witness right after, so this is the last point both exist).
pub fn check(instance_id: usize, theirs: &[u64]) {
    let ours = RECORDED.lock().unwrap().remove(&instance_id);
    let verdict = match ours {
        Some(ours) => diff(&ours, theirs),
        None => (usize::MAX, 0),
    };
    VERDICTS.lock().unwrap().insert(instance_id, verdict);
}

pub fn enabled() -> bool {
    std::env::var("ZZ_AB").map(|v| v != "0" && !v.is_empty()).unwrap_or(false)
}

pub fn record(instance_id: usize, proof: Vec<u64>) {
    RECORDED.lock().unwrap().insert(instance_id, proof);
}

/// Outcome of one comparison pass.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Report {
    pub identical: usize,
    pub mismatched: Vec<(usize, usize, usize)>, // (instance, differing words, first index)
    pub missing: Vec<usize>,
}

/// Compare every recorded bridge proof against `theirs(instance_id)`, the
/// proof pil2 produced, and drain the record.
pub fn compare(theirs: impl Fn(usize) -> Option<Vec<u64>>) -> Report {
    let mut report = Report::default();
    for (id, (words, first)) in std::mem::take(&mut *VERDICTS.lock().unwrap()) {
        if words == usize::MAX {
            report.missing.push(id);
        } else if words == 0 {
            report.identical += 1;
        } else {
            report.mismatched.push((id, words, first));
        }
    }
    // Proofs never consumed through `check` (no recursion ran) are still in
    // their slots.
    let recorded: Vec<(usize, Vec<u64>)> = std::mem::take(&mut *RECORDED.lock().unwrap()).into_iter().collect();
    for (id, ours) in recorded {
        match theirs(id) {
            None => report.missing.push(id),
            Some(pil2) => {
                let (words, first) = diff(&ours, &pil2);
                if words == 0 {
                    report.identical += 1;
                } else {
                    report.mismatched.push((id, words, first));
                }
            }
        }
    }
    report
}

/// `ZZ_DUMP_PROOFS=<dir>`: write `instance_id`'s proof as little-endian
/// words to `<dir>/<instance_id>.bin` — from whichever prover produced it.
pub fn dump(instance_id: usize, proof: &[u64]) {
    if let Ok(dir) = std::env::var("ZZ_DUMP_PROOFS") {
        if dir.is_empty() {
            return;
        }
        let _ = std::fs::create_dir_all(&dir);
        let bytes: Vec<u8> = proof.iter().flat_map(|w| w.to_le_bytes()).collect();
        let _ = std::fs::write(format!("{dir}/{instance_id}.bin"), bytes);
    }
}
