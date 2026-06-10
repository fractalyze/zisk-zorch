//! Golden-vector generator for zisk-zorch's byte-match tests.
//!
//! Links pil2-proofman v0.15.0's `fields` crate — the same code the ZisK
//! prover runs — and emits JSON fixtures under zisk_zorch/**/testdata/golden/.
//! Deterministic (fixed splitmix64 seeds): regeneration is a no-op unless the
//! reference pin changes.
//!
//! All field elements are serialized as canonical-u64 decimal strings; JSON
//! numbers cannot carry 64-bit values exactly.

use fields::{
    intt_tiny, linear_hash_seq, partial_merkle_tree, poseidon2_hash, verify_mt, Field, Goldilocks,
    PrimeField64, Poseidon12, Poseidon16, Poseidon2Constants, Poseidon4, Poseidon8, Transcript,
};
use serde_json::{json, Value};
use std::fs;
use std::path::Path;

const GOLDILOCKS_P: u128 = 0xFFFF_FFFF_0000_0001;

fn splitmix64(state: &mut u64) -> u64 {
    *state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
    let mut z = *state;
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

fn rand_fe(state: &mut u64) -> Goldilocks {
    Goldilocks::new((splitmix64(state) as u128 % GOLDILOCKS_P) as u64)
}

fn ser(values: &[Goldilocks]) -> Vec<String> {
    values.iter().map(|v| v.as_canonical_u64().to_string()).collect()
}

fn pow(base: Goldilocks, mut exp: u64) -> Goldilocks {
    let mut acc = Goldilocks::new(1);
    let mut b = base;
    while exp > 0 {
        if exp & 1 == 1 {
            acc *= b;
        }
        b *= b;
        exp >>= 1;
    }
    acc
}

fn permutation_cases<C: Poseidon2Constants<W>, const W: usize>(seed: u64) -> Value {
    let mut cases = Vec::new();

    let iota: Vec<Goldilocks> = (0..W as u64).map(Goldilocks::new).collect();
    let mut input = [Goldilocks::ZERO; W];
    input.copy_from_slice(&iota);
    cases.push(json!({"input": ser(&input), "output": ser(&poseidon2_hash::<Goldilocks, C, W>(&input))}));

    let mut state = seed;
    for _ in 0..3 {
        let mut input = [Goldilocks::ZERO; W];
        for x in input.iter_mut() {
            *x = rand_fe(&mut state);
        }
        cases.push(json!({"input": ser(&input), "output": ser(&poseidon2_hash::<Goldilocks, C, W>(&input))}));
    }
    json!({"width": W, "cases": cases})
}

fn linear_hash_cases<C: Poseidon2Constants<W>, const W: usize>(seed: u64) -> Value {
    // Lengths probe every linear_hash regime: the <=4 copy-without-permute
    // shortcut, exactly one rate block, a partial last block, and multi-block
    // chaining (where the previous capacity feeds back into the state).
    let rate = C::RATE as u64;
    let lens = [
        1,
        3,
        4,
        5,
        rate - 1,
        rate,
        rate + 1,
        2 * rate,
        2 * rate + 3,
        5 * rate + 1,
    ];
    let mut state = seed;
    let mut cases = Vec::new();
    for &len in lens.iter() {
        let input: Vec<Goldilocks> = (0..len).map(|_| rand_fe(&mut state)).collect();
        let out = linear_hash_seq::<Goldilocks, C, W>(&input);
        cases.push(json!({"input": ser(&input), "output": ser(&out)}));
    }
    json!({"width": W, "rate": C::RATE, "cases": cases})
}

fn merkle_root<const W: usize, C: Poseidon2Constants<W>>(
    rows: &[Vec<Goldilocks>],
    arity: u64,
) -> [Goldilocks; 4] {
    let mut digests = Vec::with_capacity(rows.len() * 4);
    for row in rows {
        let h = linear_hash_seq::<Goldilocks, C, W>(row);
        digests.extend_from_slice(&h[..4]);
    }
    partial_merkle_tree::<Goldilocks, C, W>(&digests, rows.len() as u64, arity)
}

fn merkle_cases(seed: u64) -> Value {
    // Leaf counts include non-multiples of the arity at intermediate levels
    // (e.g. 2^5 with arity 4 hits the per-level zero-padding path at the top).
    let heights = [1u64, 2, 4, 6, 8, 32, 64];
    let n_cols = 9u64; // > one rate block for width 8 — exercises leaf chaining
    let mut out = Vec::new();
    for &arity in &[2u64, 3, 4] {
        let mut state = seed ^ arity;
        for &height in &heights {
            let rows: Vec<Vec<Goldilocks>> =
                (0..height).map(|_| (0..n_cols).map(|_| rand_fe(&mut state)).collect()).collect();
            let root = match arity {
                2 => merkle_root::<8, Poseidon8>(&rows, arity),
                3 => merkle_root::<12, Poseidon12>(&rows, arity),
                4 => merkle_root::<16, Poseidon16>(&rows, arity),
                _ => unreachable!(),
            };
            let flat: Vec<String> = rows.iter().flat_map(|r| ser(r)).collect();
            out.push(json!({
                "arity": arity,
                "height": height,
                "n_cols": n_cols,
                "rows": flat,
                "root": ser(&root),
            }));
        }
    }
    json!({"cases": out})
}

/// One query-phase case: rebuild the padded digest levels (the shape
/// `partial_merkle_tree`'s cursor stores), extract each query's sibling path
/// in `MerkleTreeGL::genMerkleProof` order — per level the (arity-1) group
/// digests with the node's own slot skipped — and emit the flat
/// `getGroupProof` array [row..., mp levels...].
/// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.15.0/pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp#L145-L175
///
/// Self-checked twice against the reference: the rebuilt root must equal
/// `partial_merkle_tree`, and every extracted path must pass `verify_mt`.
fn merkle_proof_case<const W: usize, C: Poseidon2Constants<W>>(
    rows: &[Vec<Goldilocks>],
    arity: u64,
) -> Value {
    let height = rows.len() as u64;
    let n_cols = rows[0].len() as u64;

    // Leaf digests, then fold; each stored level is zero-padded to a multiple
    // of the arity so boundary openings read their zero siblings like any node.
    let mut levels: Vec<Vec<Goldilocks>> = Vec::new();
    let mut level: Vec<Goldilocks> = rows
        .iter()
        .flat_map(|row| linear_hash_seq::<Goldilocks, C, W>(row)[..4].to_vec())
        .collect();
    while level.len() > 4 {
        while (level.len() / 4) % arity as usize != 0 {
            level.extend_from_slice(&[Goldilocks::ZERO; 4]);
        }
        levels.push(level.clone());
        let mut next = Vec::with_capacity(level.len() / arity as usize);
        for group in level.chunks(W) {
            let mut input = [Goldilocks::ZERO; W];
            input.copy_from_slice(group);
            next.extend_from_slice(&poseidon2_hash::<Goldilocks, C, W>(&input)[..4]);
        }
        level = next;
    }
    levels.push(level);

    let root = levels.last().unwrap()[..4].to_vec();
    let ref_root = partial_merkle_tree::<Goldilocks, C, W>(
        &levels[0][..(height * 4) as usize],
        height,
        arity,
    );
    assert_eq!(root, ref_root, "level fold diverged from partial_merkle_tree");

    let mut indices = vec![0, height / 2, height - 1];
    indices.dedup();
    let mut queries = Vec::new();
    for &index in &indices {
        let mut mp: Vec<Vec<Goldilocks>> = Vec::new();
        let mut idx = index;
        for lvl in &levels[..levels.len() - 1] {
            let pos = idx % arity;
            let group = (idx - pos) as usize;
            let mut siblings = Vec::new();
            for i in 0..arity as usize {
                if i as u64 == pos {
                    continue;
                }
                siblings.extend_from_slice(&lvl[(group + i) * 4..(group + i + 1) * 4]);
            }
            mp.push(siblings);
            idx /= arity;
        }
        assert!(
            verify_mt::<Goldilocks, C, W>(&root, &[], &mp, index, &rows[index as usize], arity, 0),
            "verify_mt rejected the extracted path (arity {arity}, height {height}, index {index})"
        );
        let mut proof = rows[index as usize].clone();
        for siblings in &mp {
            proof.extend_from_slice(siblings);
        }
        queries.push(json!({"index": index, "proof": ser(&proof)}));
    }

    json!({
        "arity": arity,
        "height": height,
        "n_cols": n_cols,
        "rows": rows.iter().flat_map(|r| ser(r)).collect::<Vec<_>>(),
        "root": ser(&root),
        "queries": queries,
    })
}

fn merkle_proof_cases(seed: u64) -> Value {
    // Heights hit every proof regime: 1 (empty path), 2 (single level), 6/8
    // (leaf-level padding for arity 4 / 3), 32 (interior padding: arity 4
    // leaves a 2-node top level). Indices probe both group boundaries.
    let heights = [1u64, 2, 6, 8, 32];
    let n_cols = 9u64; // > one rate block for width 8 — exercises leaf chaining
    let mut out = Vec::new();
    for &arity in &[2u64, 3, 4] {
        let mut state = seed ^ arity;
        for &height in &heights {
            let rows: Vec<Vec<Goldilocks>> =
                (0..height).map(|_| (0..n_cols).map(|_| rand_fe(&mut state)).collect()).collect();
            out.push(match arity {
                2 => merkle_proof_case::<8, Poseidon8>(&rows, arity),
                3 => merkle_proof_case::<12, Poseidon12>(&rows, arity),
                4 => merkle_proof_case::<16, Poseidon16>(&rows, arity),
                _ => unreachable!(),
            });
        }
    }
    json!({"cases": out})
}

fn transcript_case<C: Poseidon2Constants<W>, const W: usize>(seed: u64) -> Value {
    let mut state = seed;
    let mut t = Transcript::<Goldilocks, C, W>::new();
    let mut steps = Vec::new();

    // The scripted sequence mirrors a stage boundary: absorb a root-sized
    // batch, draw an extension challenge (3 limbs), absorb more (forces a
    // pending flush mid-stream), draw single elements, then query indices.
    let put1: Vec<Goldilocks> = (0..4).map(|_| rand_fe(&mut state)).collect();
    t.put(&put1);
    steps.push(json!({"op": "put", "values": ser(&put1)}));

    let mut challenge = [Goldilocks::ZERO; 3];
    t.get_field(&mut challenge);
    steps.push(json!({"op": "get_field", "output": ser(&challenge)}));

    let put2: Vec<Goldilocks> = (0..(2 * W as u64)).map(|_| rand_fe(&mut state)).collect();
    t.put(&put2);
    steps.push(json!({"op": "put", "values": ser(&put2)}));

    let singles: Vec<Goldilocks> = (0..5).map(|_| t.get_fields1()).collect();
    steps.push(json!({"op": "get_fields1_x5", "output": ser(&singles)}));

    let perms = t.get_permutations(8, 10);
    steps.push(json!({
        "op": "get_permutations",
        "n": 8,
        "n_bits": 10,
        "output": perms.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
    }));

    let st = t.get_state();
    steps.push(json!({"op": "get_state", "output": ser(&st)}));

    json!({"width": W, "steps": steps})
}

/// pil2-stark `extendPol`: INTT the evaluations, scale coefficient i by
/// SHIFT^i, evaluate on the blown-up domain. The extension is computed here
/// by naive per-point evaluation so the golden does not inherit any NTT
/// convention from the harness itself — only `intt_tiny` (the reference's
/// own inverse NTT) and the schoolbook Horner sum.
fn lde_case(n_bits: usize, blowup_bits: usize, n_cols: usize, seed: u64) -> Value {
    let n = 1usize << n_bits;
    let n_ext = n << blowup_bits;
    let mut state = seed;

    let evals: Vec<Goldilocks> = (0..n * n_cols).map(|_| rand_fe(&mut state)).collect();

    let mut coeffs = evals.clone();
    intt_tiny(&mut coeffs, n_bits, n_cols);

    // intt_tiny must return plain coefficient order: Horner at w^j has to
    // reproduce the input evaluations, or the LDE golden would silently
    // encode a permuted-coefficient convention.
    let w_n = Goldilocks::new(Goldilocks::W[n_bits]);
    for j in 0..n {
        let x = pow(w_n, j as u64);
        for c in 0..n_cols {
            let mut acc = Goldilocks::ZERO;
            for i in (0..n).rev() {
                acc = acc * x + coeffs[i * n_cols + c];
            }
            assert_eq!(acc, evals[j * n_cols + c], "intt_tiny convention mismatch at ({j},{c})");
        }
    }

    let shift = Goldilocks::new(Goldilocks::SHIFT);
    let w_ext = Goldilocks::new(Goldilocks::W[n_bits + blowup_bits]);
    let mut extended = vec![Goldilocks::ZERO; n_ext * n_cols];
    for j in 0..n_ext {
        let x = shift * pow(w_ext, j as u64);
        for c in 0..n_cols {
            let mut acc = Goldilocks::ZERO;
            // Horner over the n coefficients of column c.
            for i in (0..n).rev() {
                acc = acc * x + coeffs[i * n_cols + c];
            }
            extended[j * n_cols + c] = acc;
        }
    }

    json!({
        "n_bits": n_bits,
        "blowup_bits": blowup_bits,
        "n_cols": n_cols,
        "evals": ser(&evals),
        "coeffs": ser(&coeffs),
        "extended": ser(&extended),
    })
}

/// The full stage-1 pipeline on one small matrix: extendPol then leaf-hash
/// every extended row and fold the k-ary tree — the byte-match target for
/// zisk_zorch.commit.trace_commit.
fn stage1_case(n_bits: usize, blowup_bits: usize, n_cols: usize, arity: u64, seed: u64) -> Value {
    let lde = lde_case(n_bits, blowup_bits, n_cols, seed);
    let extended: Vec<Goldilocks> = lde["extended"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| Goldilocks::new(v.as_str().unwrap().parse().unwrap()))
        .collect();
    let n_ext = 1usize << (n_bits + blowup_bits);
    let rows: Vec<Vec<Goldilocks>> =
        (0..n_ext).map(|r| extended[r * n_cols..(r + 1) * n_cols].to_vec()).collect();
    let root = match arity {
        2 => merkle_root::<8, Poseidon8>(&rows, arity),
        3 => merkle_root::<12, Poseidon12>(&rows, arity),
        4 => merkle_root::<16, Poseidon16>(&rows, arity),
        _ => unreachable!(),
    };
    json!({
        "arity": arity,
        "lde": lde,
        "root": ser(&root),
    })
}

fn write(path: &str, value: Value) {
    let path = Path::new("..").join(path);
    fs::create_dir_all(path.parent().unwrap()).unwrap();
    fs::write(&path, serde_json::to_string_pretty(&value).unwrap()).unwrap();
    println!("wrote {}", path.display());
}

fn main() {
    write(
        "zisk_zorch/poseidon2/testdata/golden/permutation.json",
        json!({
            "widths": [
                permutation_cases::<Poseidon4, 4>(0xA0),
                permutation_cases::<Poseidon8, 8>(0xA1),
                permutation_cases::<Poseidon12, 12>(0xA2),
                permutation_cases::<Poseidon16, 16>(0xA3),
            ]
        }),
    );
    write(
        "zisk_zorch/commit/testdata/golden/linear_hash.json",
        json!({
            "widths": [
                linear_hash_cases::<Poseidon8, 8>(0xB1),
                linear_hash_cases::<Poseidon12, 12>(0xB2),
                linear_hash_cases::<Poseidon16, 16>(0xB3),
            ]
        }),
    );
    write("zisk_zorch/commit/testdata/golden/merkle_root.json", merkle_cases(0xC0));
    write("zisk_zorch/commit/testdata/golden/merkle_proof.json", merkle_proof_cases(0xC1));
    write(
        "zisk_zorch/transcript/testdata/golden/transcript.json",
        json!({
            "widths": [
                transcript_case::<Poseidon8, 8>(0xD1),
                transcript_case::<Poseidon12, 12>(0xD2),
                transcript_case::<Poseidon16, 16>(0xD3),
            ]
        }),
    );
    write(
        "zisk_zorch/commit/testdata/golden/lde.json",
        json!({
            "cases": [
                lde_case(3, 2, 3, 0xE1),
                lde_case(4, 1, 1, 0xE2),
                lde_case(5, 3, 2, 0xE3),
            ]
        }),
    );
    write(
        "zisk_zorch/commit/testdata/golden/stage1_commit.json",
        json!({
            "cases": [
                stage1_case(3, 2, 5, 2, 0xF1),
                stage1_case(3, 2, 5, 3, 0xF2),
                stage1_case(3, 2, 5, 4, 0xF3),
                stage1_case(5, 1, 9, 4, 0xF4),
            ]
        }),
    );
}
