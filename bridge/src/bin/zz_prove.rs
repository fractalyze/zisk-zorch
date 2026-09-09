//! Prove one dumped case through the artifacts and compare with the
//! Python replay's proof — the bridge's byte-gate, runnable without
//! proofman.
//!
//!   zz_prove <artifacts-dir> <case-dir> [--repeat N]
//!   zz_prove --warm <artifacts-dir> [<Air>_n<nBits> ...]   # compile + cache only
//!
//! A case directory comes from `python -m zisk_zorch.export.cases` and
//! holds raw little-endian u64 files (`trace.bin`, `publics.bin`,
//! `airvalues.bin`, `proofvalues.bin`, `global_challenge.bin`,
//! `const_base.bin`, `custom_base_<id>.bin`, `expected_proof.bin`) plus
//! `case.json` naming the AIR and its height. `--repeat` re-proves the
//! warm driver to time a prove without the compile.

use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use zisk_zorch_bridge::artifact::{new_client, Artifact};
use zisk_zorch_bridge::driver::{AirDriver, FixedSections, InstanceInputs};
use zisk_zorch_bridge::transcript::HostTranscript;

/// Compile threads per AIR worker: one worker per AIR up to `threads`, and the
/// budget left over spread one per worker so all of it is used at any ratio.
///
/// Only one worker can be busy with a given AIR, so spawning `threads` workers
/// over fewer AIRs leaves the surplus exiting immediately; folding them inside
/// instead is what makes a single-AIR warm -- a plugin bisect -- parallel.
fn warm_split(threads: usize, dirs: usize) -> Vec<usize> {
    let across = threads.min(dirs).max(1);
    let (base, extra) = (threads / across, threads % across);
    (0..across).map(|worker| (base + usize::from(worker < extra)).max(1)).collect()
}

fn words(path: &Path) -> Vec<u64> {
    let bytes = std::fs::read(path).unwrap_or_else(|e| panic!("cannot read {}: {e}", path.display()));
    bytes.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("usage: zz_prove <artifacts-dir> <case-dir> [--repeat N]");
        std::process::exit(2);
    }
    if args[1] == "--warm" {
        // Compile (and cache) every program of the named AIR directories,
        // or of all of them, without proving anything.
        let artifacts = Path::new(&args[2]);
        let cache = std::env::var("ZZ_COMPILE_CACHE")
            .ok()
            .filter(|s| !s.is_empty())
            .map(std::path::PathBuf::from)
            .unwrap_or_else(|| artifacts.join(".pjrt-cache"));
        let dirs: Vec<std::path::PathBuf> = if args.len() > 3 {
            args[3..].iter().map(|a| artifacts.join(a)).collect()
        } else {
            let mut d: Vec<_> = std::fs::read_dir(artifacts)
                .unwrap()
                .filter_map(|e| e.ok().map(|e| e.path()))
                .filter(|p| p.join("manifest.json").exists())
                .collect();
            d.sort();
            d
        };
        // ZZ_WARM_THREADS=N loads that many AIRs at once on the one client:
        // the probe for how much concurrent deserialization the plugin takes.
        let threads: usize = std::env::var("ZZ_WARM_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
        // Spare threads go inside the AIRs rather than idling: only one worker
        // per AIR can be busy, so `threads` workers over fewer AIRs would leave
        // the rest exiting immediately. Warming one AIR -- what a plugin bisect
        // does -- otherwise runs a single thread through ~34 compiles however
        // high ZZ_WARM_THREADS is set. The remainder is spread one per worker
        // so the whole budget is used at any ratio, not only at a multiple.
        let split = warm_split(threads, dirs.len());
        let client = new_client(None);
        let queue = std::sync::Arc::new(std::sync::Mutex::new(dirs));
        let t0 = Instant::now();
        let handles: Vec<_> = split
            .into_iter()
            .map(|within| {
                let client = client.clone();
                let queue = queue.clone();
                let cache = cache.clone();
                std::thread::spawn(move || loop {
                    let dir = match queue.lock().unwrap().pop() {
                        Some(d) => d,
                        None => return,
                    };
                    let t = Instant::now();
                    let art = Artifact::load(client.clone(), &dir, Some(&cache)).unwrap();
                    art.compile_all(within).unwrap();
                    eprintln!(
                        "warm {} ({} programs, {} from the cache) in {:.1} s",
                        dir.display(),
                        art.manifest.programs.len(),
                        art.cache_hits.load(std::sync::atomic::Ordering::Relaxed),
                        t.elapsed().as_secs_f64()
                    );
                })
            })
            .collect();
        for h in handles {
            h.join().unwrap();
        }
        eprintln!("warm total {:.1} s with {threads} thread(s)", t0.elapsed().as_secs_f64());
        // ZZ_WARM_HOLD=<s>: stay alive with the executables loaded (to read
        // their device footprint off nvidia-smi).
        if let Some(secs) = std::env::var("ZZ_WARM_HOLD").ok().and_then(|s| s.parse::<u64>().ok()) {
            std::thread::sleep(std::time::Duration::from_secs(secs));
        }
        return;
    }
    let artifacts = Path::new(&args[1]);
    let case = Path::new(&args[2]);
    let repeat: usize = args
        .iter()
        .position(|a| a == "--repeat")
        .and_then(|i| args.get(i + 1))
        .and_then(|n| n.parse().ok())
        .unwrap_or(0);

    let meta: serde_json::Value =
        serde_json::from_str(&std::fs::read_to_string(case.join("case.json")).expect("case.json")).unwrap();
    let air = meta["air"].as_str().unwrap();
    let n_bits = meta["n_bits"].as_u64().unwrap();
    let dir = artifacts.join(format!("{air}_n{n_bits}"));

    let cache = std::env::var("ZZ_COMPILE_CACHE")
        .ok()
        .filter(|s| !s.is_empty())
        .map(std::path::PathBuf::from)
        .unwrap_or_else(|| artifacts.join(".pjrt-cache"));
    let t = Instant::now();
    let client = new_client(None);
    let art = Artifact::load(client, &dir, Some(&cache)).unwrap();
    // Nothing is proving here, so a cold case compiles as wide as it is told to.
    let threads: usize = std::env::var("ZZ_WARM_THREADS").ok().and_then(|s| s.parse().ok()).unwrap_or(1);
    art.compile_all(threads).unwrap();
    let m = art.manifest.clone();
    eprintln!(
        "loaded {} ({} programs, {} from the cache) in {:.1} s",
        dir.display(),
        m.programs.len(),
        art.cache_hits.load(std::sync::atomic::Ordering::Relaxed),
        t.elapsed().as_secs_f64()
    );

    let const_base = words(&case.join("const_base.bin"));
    let customs: Vec<(usize, Vec<u64>)> = m
        .custom_commits
        .iter()
        .map(|cc| (cc.id, words(&case.join(format!("custom_base_{}.bin", cc.id)))))
        .collect();
    let fixed = FixedSections {
        const_base: &const_base,
        custom_base: customs.iter().map(|(id, w)| (*id, w.as_slice())).collect::<HashMap<_, _>>(),
        uploaded: None,
    };
    let mut driver = AirDriver::new(std::sync::Arc::new(art));
    let t = Instant::now();
    driver.set_fixed(&fixed).unwrap();
    eprintln!("fixed sections in {:.2} s", t.elapsed().as_secs_f64());

    let trace = words(&case.join("trace.bin"));
    let publics = words(&case.join("publics.bin"));
    let airvalues = words(&case.join("airvalues.bin"));
    let proofvalues = words(&case.join("proofvalues.bin"));
    let global_challenge = words(&case.join("global_challenge.bin"));
    let inputs = InstanceInputs {
        trace: &trace,
        publics: &publics,
        airvalues: &airvalues,
        proofvalues: &proofvalues,
        global_challenge: &global_challenge,
        uploaded: None,
    };
    let mut proof = vec![0u64; driver.proof_words()];
    let expected_path = case.join("expected_proof.bin");
    // Every prove is compared, not just the last: a prove that comes out wrong
    // only sometimes is invisible to a gate that overwrites the buffer
    // `repeat` times and checks what is left.
    let expected = expected_path.exists().then(|| words(&expected_path));
    let mut wrong = Vec::new();
    for i in 0..=repeat {
        let t = Instant::now();
        let mut transcript = HostTranscript::new(&m.hash_family).unwrap();
        let out = driver.prove(&inputs, &mut transcript, &mut proof).unwrap();
        let label = if i == 0 { "proved" } else { "warm prove" };
        eprintln!("{label} {:.3} s (nonce {})", t.elapsed().as_secs_f64(), out.nonce);
        let Some(expected) = expected.as_ref() else { continue };
        if expected.len() != proof.len() {
            eprintln!("MISMATCH: proof has {} words, expected {}", proof.len(), expected.len());
            std::process::exit(1);
        }
        let diff: Vec<usize> = (0..proof.len()).filter(|k| proof[*k] != expected[*k]).collect();
        if !diff.is_empty() {
            eprintln!(
                "MISMATCH on prove {}: {} of {} words differ, first at {:?}",
                i + 1,
                diff.len(),
                proof.len(),
                &diff[..diff.len().min(8)]
            );
            wrong.push(i + 1);
        }
    }
    if expected.is_some() {
        if wrong.is_empty() {
            println!("byte-identical: {} words, {} prove(s)", proof.len(), repeat + 1);
        } else {
            eprintln!("{} of {} proves wrong: {:?}", wrong.len(), repeat + 1, wrong);
            std::process::exit(1);
        }
    } else {
        std::fs::write(case.join("bridge_proof.bin"), unsafe {
            std::slice::from_raw_parts(proof.as_ptr() as *const u8, proof.len() * 8)
        })
        .unwrap();
        println!("wrote bridge_proof.bin ({} words)", proof.len());
    }
}

#[cfg(test)]
mod tests {
    use super::warm_split;

    #[test]
    fn warm_split_uses_the_whole_budget_at_any_ratio() {
        // The recipe's own case: 11 threads over 6 AIRs ran 6 one-wide and
        // idled 5 before the remainder was spread.
        assert_eq!(warm_split(11, 6), [2, 2, 2, 2, 2, 1]);
        // A bisect: every thread goes inside the one AIR.
        assert_eq!(warm_split(11, 1), [11]);
        // More AIRs than threads: one worker each, none spare to fold in.
        assert_eq!(warm_split(4, 11), [1, 1, 1, 1]);
        assert_eq!(warm_split(1, 1), [1]);
        for (threads, dirs) in [(11, 6), (11, 1), (4, 11), (7, 3), (1, 9), (64, 5)] {
            let split = warm_split(threads, dirs);
            assert_eq!(split.len(), threads.min(dirs).max(1), "{threads}/{dirs}");
            if dirs <= threads {
                assert_eq!(split.iter().sum::<usize>(), threads, "{threads}/{dirs}");
            }
        }
    }

    #[test]
    fn warm_split_never_returns_a_zero_width_worker() {
        // `threads` is env-parsed, so 0 reaches here; a zero would spawn a
        // worker that compiles nothing and the warm would silently do nothing.
        assert_eq!(warm_split(0, 4), [1]);
        assert!(warm_split(0, 0).iter().all(|w| *w >= 1));
    }
}
