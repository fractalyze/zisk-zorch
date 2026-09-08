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
        let client = new_client(None);
        let queue = std::sync::Arc::new(std::sync::Mutex::new(dirs));
        let t0 = Instant::now();
        let handles: Vec<_> = (0..threads)
            .map(|_| {
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
                    art.compile_all().unwrap();
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
    art.compile_all().unwrap();
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
    };
    let art = std::sync::Arc::new(art);
    // The base sections belong to one prove, so every prove below uploads its
    // own — the same per-prove cost the bridge pays.
    let upload_base = || {
        let t = Instant::now();
        let base = zisk_zorch_bridge::driver::upload_fixed(&art, &fixed).unwrap();
        (base, t.elapsed().as_secs_f64())
    };
    let mut driver = AirDriver::new(art.clone());
    let t = Instant::now();
    let (base, _) = upload_base();
    driver.set_fixed(&base).unwrap();
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
    let t = Instant::now();
    let mut transcript = HostTranscript::new(&m.hash_family).unwrap();
    let out = driver.prove(base, &inputs, &mut transcript, &mut proof).unwrap();
    eprintln!("proved in {:.3} s (nonce {})", t.elapsed().as_secs_f64(), out.nonce);
    for _ in 0..repeat {
        let (base, upload_s) = upload_base();
        let t = Instant::now();
        let mut transcript = HostTranscript::new(&m.hash_family).unwrap();
        driver.prove(base, &inputs, &mut transcript, &mut proof).unwrap();
        eprintln!("warm prove {:.3} s ({upload_s:.3} s of it the base sections)", t.elapsed().as_secs_f64());
    }

    let expected_path = case.join("expected_proof.bin");
    if expected_path.exists() {
        let expected = words(&expected_path);
        if expected.len() != proof.len() {
            eprintln!("MISMATCH: proof has {} words, expected {}", proof.len(), expected.len());
            std::process::exit(1);
        }
        let diff: Vec<usize> = (0..proof.len()).filter(|i| proof[*i] != expected[*i]).collect();
        if diff.is_empty() {
            println!("byte-identical: {} words", proof.len());
        } else {
            eprintln!("MISMATCH: {} of {} words differ, first at {:?}", diff.len(), proof.len(), &diff[..diff.len().min(8)]);
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
