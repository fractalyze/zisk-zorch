//! Rust driver for the pil2-reference per-stage dumps.
//!
//! Runs one real pil2-proofman `genProof` in-process with the pinned fork's
//! `PIL2_DUMP_DIR` hooks armed, so each stage buffer lands on disk exactly as
//! the prover computed it. The `verify_*` runnables then byte-match an
//! assembled zisk-zorch stage against those buffers.
//!
//! The stages this repo gates (DEEP, FRI fold, evals, grand sum) have no entry
//! point outside `genProof` — pil2's host wrappers each take a fully-built
//! `SetupCtx`/`StepsParams`/`MerkleTreeGL` prover context — so observing them
//! means proving for real. See `README.md` for the pin and the recipe.
//!
//! ```text
//! pil2-dump --proving-key <dir> --witness-lib <so> --public-inputs <json>
//!           --out <dump dir> [--gpu]
//! ```
//!
//! The proving key and witness lib are pil2 toolchain build artifacts (see
//! README); this crate drives the prove, it does not build them.

use std::path::PathBuf;

use fields::Goldilocks;
use proofman::ProofMan;
use proofman_common::{ProofOptions, ProofmanOptions, VerboseMode};

struct Args {
    proving_key: PathBuf,
    witness_lib: PathBuf,
    public_inputs: Option<PathBuf>,
    out: PathBuf,
    gpu: bool,
}

fn parse_args() -> Args {
    let (mut proving_key, mut witness_lib, mut public_inputs, mut out) = (None, None, None, None);
    let mut gpu = false;
    let mut it = std::env::args().skip(1);
    while let Some(arg) = it.next() {
        let mut next = |what: &str| PathBuf::from(it.next().unwrap_or_else(|| panic!("{what} needs a path")));
        match arg.as_str() {
            "--proving-key" => proving_key = Some(next("--proving-key")),
            "--witness-lib" => witness_lib = Some(next("--witness-lib")),
            "--public-inputs" => public_inputs = Some(next("--public-inputs")),
            "--out" => out = Some(next("--out")),
            "--gpu" => gpu = true,
            other => panic!("unknown argument {other}"),
        }
    }
    Args {
        proving_key: proving_key.expect("--proving-key is required"),
        witness_lib: witness_lib.expect("--witness-lib is required"),
        public_inputs,
        out: out.expect("--out is required"),
        gpu,
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args = parse_args();
    std::fs::create_dir_all(&args.out)?;

    // The hooks read this at each dump point; setting it before the prove is
    // what arms them. An unset variable makes every hook a no-op, which is how
    // the pinned fork stays byte-identical to upstream for ordinary proves.
    std::env::set_var("PIL2_DUMP_DIR", &args.out);

    let mut options = ProofmanOptions::default();
    options.no_aggregation();
    if args.gpu {
        options.gpu();
    }
    options.verbose_mode(VerboseMode::Info);

    let proofman = ProofMan::<Goldilocks>::new(args.proving_key, options)?;
    proofman.set_barrier();
    // `verify_proofs` on: a dump is only a reference if the proof it came from
    // is valid, and the hooks must not have perturbed the prove.
    proofman.generate_proof(
        args.witness_lib,
        args.public_inputs,
        None,
        VerboseMode::Info,
        ProofOptions::new(false, false, true, false, true, false),
    )?;

    let written = std::fs::read_dir(&args.out)?.count();
    println!("wrote {written} stage buffers to {}", args.out.display());
    Ok(())
}
