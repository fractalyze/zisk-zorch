//! Golden-vector generator for zisk-zorch's byte-match tests.
//!
//! Links pil2-proofman v0.18.0's `fields` crate — the same code the ZisK
//! prover runs — and emits JSON fixtures under zisk_zorch/**/testdata/golden/.
//! Deterministic (fixed splitmix64 seeds): regeneration is a no-op unless the
//! reference pin changes.
//!
//! All field elements are serialized as canonical-u64 decimal strings; JSON
//! numbers cannot carry 64-bit values exactly.

use fields::{
    intt_tiny, linear_hash_seq, partial_merkle_tree, poseidon2_hash, verify_fold, verify_mt,
    CubicExtensionField, Field, Goldilocks, PrimeField64, Poseidon12, Poseidon16,
    Poseidon2Constants, Poseidon4, Poseidon8, Transcript,
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
    // Lengths probe every linear_hash regime: short single-block rows
    // (v0.15.0's <=4 unhashed shortcut is gone in v0.18.0 — they permute),
    // exactly one rate block, a partial last block, and multi-block chaining
    // (where the previous capacity feeds back into the state).
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
/// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/merkleTree/merkleTreeGL.cpp#L145-L175
///
/// Self-checked twice against the reference: the rebuilt root must equal
/// `partial_merkle_tree`, and every extracted path must pass `verify_mt`.
/// Rebuild the zero-padded digest levels `partial_merkle_tree`'s cursor stores:
/// `levels[0]` is the leaf digests, `levels.last()` the 4-element root. Each
/// level is padded to a multiple of the arity so boundary openings read their
/// zero siblings like any node.
fn build_levels<const W: usize, C: Poseidon2Constants<W>>(
    rows: &[Vec<Goldilocks>],
    arity: u64,
) -> Vec<Vec<Goldilocks>> {
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
    levels
}

/// One query's sibling path in `MerkleTreeGL::genMerkleProof` order: per level
/// (leaf-first) the (arity-1) group digests with the node's own slot skipped.
fn build_mp(levels: &[Vec<Goldilocks>], index: u64, arity: u64) -> Vec<Vec<Goldilocks>> {
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
    mp
}

/// Flat `getGroupProof` array for one opening: the committed row, then each
/// level's sibling group.
fn group_proof(row: &[Goldilocks], mp: &[Vec<Goldilocks>]) -> Vec<Goldilocks> {
    let mut proof = row.to_vec();
    for siblings in mp {
        proof.extend_from_slice(siblings);
    }
    proof
}

fn merkle_proof_case<const W: usize, C: Poseidon2Constants<W>>(
    rows: &[Vec<Goldilocks>],
    arity: u64,
) -> Value {
    let height = rows.len() as u64;
    let n_cols = rows[0].len() as u64;

    let levels = build_levels::<W, C>(rows, arity);
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
        let mp = build_mp(&levels, index, arity);
        assert!(
            verify_mt::<Goldilocks, C, W>(&root, &[], &mp, index, &rows[index as usize], arity, 0),
            "verify_mt rejected the extracted path (arity {arity}, height {height}, index {index})"
        );
        let proof = group_proof(&rows[index as usize], &mp);
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

/// The extended-domain points `x[i] = SHIFT · W[nBitsExt]^i` (pil2 `computeX`,
/// natural order) — the abscissae the boundary zerofiers below are built over.
fn coset_x(n_bits: usize, blowup_bits: usize) -> Vec<Goldilocks> {
    let n_ext = 1usize << (n_bits + blowup_bits);
    let w_ext = Goldilocks::new(Goldilocks::W[n_bits + blowup_bits]);
    let mut x = vec![Goldilocks::ZERO; n_ext];
    let mut cur = Goldilocks::new(Goldilocks::SHIFT);
    for xi in x.iter_mut() {
        *xi = cur;
        cur = cur * w_ext;
    }
    x
}

/// pil2-stark `buildZHInv` (setup_ctx.hpp): the inverse zerofier 1/(x^N − 1) on
/// the blown-up coset, the divisor stage-2's quotient `Q = C / Z_H` multiplies
/// by. On the coset SHIFT·<W[nBitsExt]>, `x^N = SHIFT^N · W[blowupBits]^j` takes
/// only `2^blowupBits` distinct values, so the inverse is that period tiled
/// across the extended domain (natural order).
fn every_row_zi(n_bits: usize, blowup_bits: usize) -> Vec<Goldilocks> {
    let n_ext = 1usize << (n_bits + blowup_bits);
    let extend = 1usize << blowup_bits;
    let sn = pow(Goldilocks::new(Goldilocks::SHIFT), 1u64 << n_bits);
    let w_ext = Goldilocks::new(Goldilocks::W[blowup_bits]);
    let one = Goldilocks::new(1);

    let mut zi = vec![Goldilocks::ZERO; n_ext];
    let mut w = one;
    for i in 0..extend {
        zi[i] = (sn * w - one).inverse();
        w = w * w_ext;
    }
    for i in extend..n_ext {
        zi[i] = zi[i % extend];
    }
    zi
}

/// Byte-match target for zisk_zorch.quotient.zerofier.inv_zerofier (everyRow).
fn zerofier_inv_case(n_bits: usize, blowup_bits: usize) -> Value {
    json!({
        "n_bits": n_bits,
        "blowup_bits": blowup_bits,
        "zi": ser(&every_row_zi(n_bits, blowup_bits)),
    })
}

/// pil2-stark `buildOneRowZerofierInv`: the firstRow (rowIndex 0) / lastRow
/// (rowIndex N) boundary divisor `1/((x − W[nBits]^rowIndex) · ZiEveryRow)`.
/// Byte-match target for inv_one_row_zerofier.
fn one_row_zerofier_case(n_bits: usize, blowup_bits: usize, row_index: u64) -> Value {
    let x = coset_x(n_bits, blowup_bits);
    let zi_h = every_row_zi(n_bits, blowup_bits);
    let root = pow(Goldilocks::new(Goldilocks::W[n_bits]), row_index);
    let zi: Vec<Goldilocks> =
        (0..x.len()).map(|i| ((x[i] - root) * zi_h[i]).inverse()).collect();
    json!({
        "n_bits": n_bits,
        "blowup_bits": blowup_bits,
        "row_index": row_index,
        "zi": ser(&zi),
    })
}

/// pil2-stark `buildFrameZerofierInv`: the everyFrame divisor — the product
/// `∏ (x − root_j)` over the first `offsetMin` and last `offsetMax` row roots.
/// (Despite the pil2 name it stores the product, not its inverse.) Byte-match
/// target for inv_frame_zerofier.
fn frame_zerofier_case(
    n_bits: usize,
    blowup_bits: usize,
    offset_min: u64,
    offset_max: u64,
) -> Value {
    let n = 1u64 << n_bits;
    let w_n = Goldilocks::new(Goldilocks::W[n_bits]);
    let x = coset_x(n_bits, blowup_bits);

    let mut roots = Vec::new();
    for i in 0..offset_min {
        roots.push(pow(w_n, i));
    }
    for i in 0..offset_max {
        roots.push(pow(w_n, n - i - 1));
    }
    let zi: Vec<Goldilocks> = x
        .iter()
        .map(|xi| roots.iter().fold(Goldilocks::new(1), |acc, r| acc * (*xi - *r)))
        .collect();
    json!({
        "n_bits": n_bits,
        "blowup_bits": blowup_bits,
        "offset_min": offset_min,
        "offset_max": offset_max,
        "zi": ser(&zi),
    })
}

type Ef = CubicExtensionField<Goldilocks>;

fn ef_zero() -> Ef {
    CubicExtensionField { value: [Goldilocks::ZERO; 3] }
}

fn rand_ef(state: &mut u64) -> Ef {
    CubicExtensionField { value: [rand_fe(state), rand_fe(state), rand_fe(state)] }
}

fn ser_ef(vals: &[Ef]) -> Vec<String> {
    let flat: Vec<Goldilocks> = vals.iter().flat_map(|e| e.value).collect();
    ser(&flat)
}

/// pil2 std_sum LogUp denominator: Horner in `std_alpha` over the bus tuple,
/// then `+ std_gamma` as the final additive bus separator (tuple[0] carries the
/// highest alpha power). The byte-match target for zisk_zorch ... bus_denominator.
fn bus_denominator_ref(tuple: &[Ef], alpha: Ef, gamma: Ef) -> Ef {
    let mut den = tuple[0];
    for t in &tuple[1..] {
        den = den * alpha + *t;
    }
    den + gamma
}

/// pil2 `gsum_col` (additive grand-sum): the running prefix sum of each row's
/// LogUp local term `Σ_i numerator_i · denominator_i^{-1}`. Row 0 is the raw
/// local term (the loop accumulates from zero). Byte-match target for grand_sum.
fn grand_sum_ref(numerators: &[Vec<Ef>], denominators: &[Vec<Ef>]) -> Vec<Ef> {
    let mut gsum = Vec::with_capacity(numerators.len());
    let mut acc = ef_zero();
    for r in 0..numerators.len() {
        let mut local = ef_zero();
        for i in 0..numerators[r].len() {
            local = local + numerators[r][i] * denominators[r][i].inverse();
        }
        acc = acc + local;
        gsum.push(acc);
    }
    gsum
}

fn bus_denominator_case(tuple_width: usize, seed: u64) -> Value {
    let mut state = seed;
    let tuple: Vec<Ef> = (0..tuple_width).map(|_| rand_ef(&mut state)).collect();
    let alpha = rand_ef(&mut state);
    let gamma = rand_ef(&mut state);
    let den = bus_denominator_ref(&tuple, alpha, gamma);
    json!({
        "tuple_width": tuple_width,
        "tuple": ser_ef(&tuple),
        "alpha": ser_ef(&[alpha]),
        "gamma": ser_ef(&[gamma]),
        "den": ser_ef(&[den]),
    })
}

fn grand_sum_case(n: usize, n_interactions: usize, seed: u64) -> Value {
    let mut state = seed;
    let numerators: Vec<Vec<Ef>> =
        (0..n).map(|_| (0..n_interactions).map(|_| rand_ef(&mut state)).collect()).collect();
    let denominators: Vec<Vec<Ef>> =
        (0..n).map(|_| (0..n_interactions).map(|_| rand_ef(&mut state)).collect()).collect();
    let gsum = grand_sum_ref(&numerators, &denominators);
    let flat_num: Vec<Ef> = numerators.into_iter().flatten().collect();
    let flat_den: Vec<Ef> = denominators.into_iter().flatten().collect();
    json!({
        "n": n,
        "n_interactions": n_interactions,
        "numerators": ser_ef(&flat_num),
        "denominators": ser_ef(&flat_den),
        "gsum": ser_ef(&gsum),
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

/// One FRI fold step: collapse a cubic-valued codeword over the previous
/// coset domain (size 2^prevBits) to the next layer (size 2^currentBits) at a
/// cubic challenge. Each output group `g` interpolates the nX = 2^(prevBits -
/// currentBits) codeword entries strided by pol2N = 2^currentBits — its coset
/// `shift_eff * w(prevBits)^(g + j*pol2N)` — and evaluates that line at the
/// challenge. Truth is the reference `verify_fold` itself (the same INTT + coset
/// rescale + Horner the C++ `FRI::fold` runs per group), called once per output.
/// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/fri/fri.hpp#L36-L113
/// One fold step over the whole codeword: each output group `g` reads the nX =
/// 2^(prevBits - currentBits) entries strided by pol2N = 2^currentBits and runs
/// the reference `verify_fold` (the INTT + coset rescale + Horner the C++
/// `FRI::fold` applies per group) at the cubic challenge.
fn fri_fold_step(
    pol: &[Goldilocks],
    challenge: [Goldilocks; 3],
    n_bits_ext: u64,
    prev_bits: u64,
    current_bits: u64,
) -> Vec<Goldilocks> {
    let cur_n = 1usize << current_bits;
    let n_x = 1usize << (prev_bits - current_bits);
    let pol2n = cur_n;
    let mut folded = Vec::with_capacity(cur_n * 3);
    for g in 0..cur_n {
        // ppar[j] = pol[j * pol2N + g] — the nX entries the fold reads for group g.
        let mut vals = Vec::with_capacity(n_x * 3);
        for j in 0..n_x {
            let idx = j * pol2n + g;
            vals.extend_from_slice(&pol[idx * 3..idx * 3 + 3]);
        }
        let out = verify_fold(
            n_bits_ext,
            current_bits,
            prev_bits,
            CubicExtensionField { value: challenge },
            g as u64,
            &vals,
        );
        folded.extend_from_slice(&out);
    }
    folded
}

fn fri_fold_case(n_bits_ext: u64, prev_bits: u64, current_bits: u64, seed: u64) -> Value {
    let prev_n = 1usize << prev_bits;
    let mut state = seed;
    // The codeword is one cubic column: prev_n elements, 3 Goldilocks limbs each.
    let pol: Vec<Goldilocks> = (0..prev_n * 3).map(|_| rand_fe(&mut state)).collect();
    let challenge = [rand_fe(&mut state), rand_fe(&mut state), rand_fe(&mut state)];

    let folded = fri_fold_step(&pol, challenge, n_bits_ext, prev_bits, current_bits);

    json!({
        "n_bits_ext": n_bits_ext,
        "prev_bits": prev_bits,
        "current_bits": current_bits,
        "pol": ser(&pol),
        "challenge": ser(&challenge),
        "folded": ser(&folded),
    })
}

/// pil2's `getTransposed`: regroup a degree-`2^currentBits` cubic codeword into
/// `2^nextBits` rows of `2^(currentBits - nextBits)` cubic entries, row `i`
/// holding the strided coset `pol[j*2^nextBits + i]` the next fold will read.
/// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/fri/fri.hpp#L126-L143
fn fri_transpose(pol: &[Goldilocks], current_bits: u64, next_bits: u64) -> Vec<Vec<Goldilocks>> {
    let w = 1usize << next_bits;
    let h = 1usize << (current_bits - next_bits);
    let mut rows = vec![Vec::with_capacity(h * 3); w];
    for (i, row) in rows.iter_mut().enumerate() {
        for j in 0..h {
            let fi = j * w + i;
            row.extend_from_slice(&pol[fi * 3..fi * 3 + 3]);
        }
    }
    rows
}

/// The full FRI prover loop (`gen_proof.hpp` STARK_FRI_FOLDING / QUERIES) over
/// one random cubic FRI polynomial `f`: fold the layer chain, commit each
/// intermediate layer's regrouped k-ary tree, drive challenges through the pil2
/// transcript (a fixed seed absorb stands in for the proof state preceding FRI),
/// send the final polynomial in clear, and open every layer at each query.
/// Self-checked: each layer root equals `partial_merkle_tree` and every opening
/// passes `verify_mt`.
/// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/gen_proof.hpp#L235-L282
fn fri_prove_case<const W: usize, C: Poseidon2Constants<W>>(
    n_bits_ext: u64,
    steps: &[u64],
    arity: u64,
    queries: &[u64],
    seed: u64,
) -> Value {
    let n_steps = steps.len();
    assert_eq!(steps[0], n_bits_ext, "steps[0] must be nBitsExt");

    let mut state = seed;
    let mut pol: Vec<Goldilocks> = (0..(1usize << n_bits_ext) * 3).map(|_| rand_fe(&mut state)).collect();
    let init_pol = pol.clone();

    // The transcript width follows transcriptArity = 3 (width 12); the merkle
    // tree arity is independent. A fixed absorb seeds it deterministically.
    let mut t = Transcript::<Goldilocks, Poseidon12, 12>::new();
    let seed_absorb: Vec<Goldilocks> = (0..4).map(|_| rand_fe(&mut state)).collect();
    t.put(&seed_absorb);

    let mut challenge = [Goldilocks::ZERO; 3];
    let mut roots: Vec<Vec<Goldilocks>> = Vec::new();
    let mut layer_rows: Vec<Vec<Vec<Goldilocks>>> = Vec::new();
    let mut layer_levels: Vec<Vec<Vec<Goldilocks>>> = Vec::new();
    let mut layer_leaf_bits: Vec<u64> = Vec::new();

    for step in 0..n_steps {
        let current_bits = steps[step];
        if step > 0 {
            // step 0's fold is a no-op (prevBits == currentBits == nBitsExt).
            pol = fri_fold_step(&pol, challenge, n_bits_ext, steps[step - 1], current_bits);
        }
        if step < n_steps - 1 {
            let next_bits = steps[step + 1];
            let rows = fri_transpose(&pol, current_bits, next_bits);
            let levels = build_levels::<W, C>(&rows, arity);
            let root = levels.last().unwrap()[..4].to_vec();
            let height = rows.len() as u64;
            let ref_root = partial_merkle_tree::<Goldilocks, C, W>(
                &levels[0][..(height * 4) as usize],
                height,
                arity,
            );
            assert_eq!(root, ref_root, "FRI layer {step} root diverged from partial_merkle_tree");
            t.put(&root); // addTranscript(root, HASH_SIZE)
            roots.push(root);
            layer_rows.push(rows);
            layer_levels.push(levels);
            layer_leaf_bits.push(next_bits);
        } else {
            t.put(&pol); // addTranscriptGL(final pol)
        }
        t.get_field(&mut challenge);
    }
    let final_pol = pol;

    let mut query_json = Vec::new();
    for &q in queries {
        let mut layers = Vec::new();
        for s in 0..(n_steps - 1) {
            let li = q % (1u64 << layer_leaf_bits[s]);
            let mp = build_mp(&layer_levels[s], li, arity);
            assert!(
                verify_mt::<Goldilocks, C, W>(&roots[s], &[], &mp, li, &layer_rows[s][li as usize], arity, 0),
                "FRI layer {s} opening rejected (query {q}, index {li})"
            );
            let proof = group_proof(&layer_rows[s][li as usize], &mp);
            layers.push(json!({"index": li, "proof": ser(&proof)}));
        }
        query_json.push(json!({"query": q, "layers": layers}));
    }

    json!({
        "n_bits_ext": n_bits_ext,
        "steps": steps,
        "arity": arity,
        "seed": ser(&seed_absorb),
        "init_pol": ser(&init_pol),
        "roots": roots.iter().map(|r| ser(r)).collect::<Vec<_>>(),
        "final_pol": ser(&final_pol),
        "queries": query_json,
    })
}

/// Dispatch the FRI prover golden on the merkle tree arity (2/3/4 -> width 8/12/16).
fn fri_prove(n_bits_ext: u64, steps: &[u64], arity: u64, queries: &[u64], seed: u64) -> Value {
    match arity {
        2 => fri_prove_case::<8, Poseidon8>(n_bits_ext, steps, arity, queries, seed),
        3 => fri_prove_case::<12, Poseidon12>(n_bits_ext, steps, arity, queries, seed),
        4 => fri_prove_case::<16, Poseidon16>(n_bits_ext, steps, arity, queries, seed),
        _ => unreachable!(),
    }
}

/// pil2-stark's terminal FRI low-degree test (`stark_verify.hpp` L672-L691): INTT
/// the in-clear final polynomial and assert every coefficient at or above the
/// degree bound `init = 2^(lastBits - blowupBits)` vanishes (`blowupBits =
/// nBitsExt - nBits`). The case plants a random polynomial of degree `< init`,
/// evaluates it on the order-`2^lastBits` subgroup by schoolbook Horner at
/// `W[lastBits]` (the ground truth, as in `lde_case`) for a genuine low-degree
/// `final_low`, and a `final_high` with one extra coefficient at index `init`
/// (degree exactly `init`) — so the port's `check_final` must accept the first
/// and reject the second. `intt_tiny` cross-checks that the constructed evals
/// invert to exactly the planted coefficients, pinning the INTT convention the
/// port mirrors. Only `steps[0]` (nBitsExt) and `steps[-1]` (lastBits) drive the
/// test; the interior steps just exercise the schedule validation.
/// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/stark_verify.hpp#L672-L691
fn fri_final_case(steps: &[u64], n_bits: u64, seed: u64) -> Value {
    let n_bits_ext = steps[0];
    let last_bits = steps[steps.len() - 1] as usize;
    let n = 1usize << last_bits;
    let blowup_bits = n_bits_ext - n_bits;
    let init = if blowup_bits > last_bits as u64 {
        0
    } else {
        1usize << (last_bits as u64 - blowup_bits)
    };
    assert!(init >= 1 && init < n, "pick params with 1 <= init < n for a meaningful test");

    let mut state = seed;
    // Schoolbook eval of the `n` coefficients (cubic, row-major) at w^j.
    let w = Goldilocks::new(Goldilocks::W[last_bits]);
    let eval = |coeffs: &[Goldilocks]| -> Vec<Goldilocks> {
        let mut out = vec![Goldilocks::ZERO; n * 3];
        for j in 0..n {
            let x = pow(w, j as u64);
            for c in 0..3 {
                let mut acc = Goldilocks::ZERO;
                for k in (0..n).rev() {
                    acc = acc * x + coeffs[k * 3 + c];
                }
                out[j * 3 + c] = acc;
            }
        }
        out
    };

    // A genuine low-degree final pol: coefficients < init, zero above.
    let mut low = vec![Goldilocks::ZERO; n * 3];
    for k in 0..init {
        for c in 0..3 {
            low[k * 3 + c] = rand_fe(&mut state);
        }
    }
    let final_low = eval(&low);

    // Convention guard: intt_tiny must invert the evals back to the plant.
    let mut recovered = final_low.clone();
    intt_tiny(&mut recovered, last_bits, 3);
    assert_eq!(recovered, low, "intt_tiny convention mismatch");

    // A high-degree final pol: one nonzero coefficient exactly at the bound.
    let mut high = low.clone();
    for c in 0..3 {
        // Re-roll until nonzero so the degree is genuinely `init`, not below it.
        loop {
            let v = rand_fe(&mut state);
            if v != Goldilocks::ZERO {
                high[init * 3 + c] = v;
                break;
            }
        }
    }
    let final_high = eval(&high);

    json!({
        "steps": steps,
        "n_bits": n_bits,
        "init": init,
        "final_low": ser(&final_low),
        "final_high": ser(&final_high),
    })
}

/// pil2's FRI query-position derivation (`gen_proof.hpp` post-fold query phase):
/// absorb the final polynomial into the running transcript, squeeze the cubic
/// grinding-seed challenge, then seed a FRESH transcript with `challenge ++ nonce`
/// and read `nQueries` positions of `nBitsExt` bits via `getPermutations`.
///
/// The proof-of-work search that finds `nonce` (`Poseidon2GoldilocksGrinding`) is
/// not exported by the v0.18.0 `fields` crate, so it has no golden; `nonce` is a
/// fixed input and the goldenable derivation around it is what this pins. The
/// `seed_absorb` + discarded squeeze stand in for the fold loop's trailing
/// observe/sample, so the derivation runs against a realistic mid-transcript state.
/// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/gen_proof.hpp#L235-L282
fn query_sample_case<C: Poseidon2Constants<W>, const W: usize>(
    final_bits: u64,
    n_queries: u64,
    n_bits_ext: u64,
    nonce: u64,
    seed: u64,
) -> Value {
    let mut state = seed;
    let mut t = Transcript::<Goldilocks, C, W>::new();

    // Stand-in for the fold loop's tail: one root-sized absorb + one squeeze.
    let seed_absorb: Vec<Goldilocks> = (0..4).map(|_| rand_fe(&mut state)).collect();
    t.put(&seed_absorb);
    let mut pre = [Goldilocks::ZERO; 3];
    t.get_field(&mut pre);

    // addTranscriptGL(transcript, friPol, (1 << finalBits) * FIELD_EXTENSION).
    let final_pol: Vec<Goldilocks> =
        (0..(1u64 << final_bits) * 3).map(|_| rand_fe(&mut state)).collect();
    t.put(&final_pol);
    let mut challenge = [Goldilocks::ZERO; 3];
    t.get_field(&mut challenge); // the grinding-seed challenge

    // Fresh transcript seeded with challenge ++ nonce, then getPermutations.
    let mut tp = Transcript::<Goldilocks, C, W>::new();
    tp.put(&challenge);
    tp.put(&[Goldilocks::new(nonce)]); // (Goldilocks::Element *)&nonce, canonical
    let positions = tp.get_permutations(n_queries, n_bits_ext);

    json!({
        "width": W,
        "seed_absorb": ser(&seed_absorb),
        "pre_challenge": ser(&pre),
        "final_pol": ser(&final_pol),
        "nonce": nonce.to_string(),
        "n_queries": n_queries,
        "n_bits_ext": n_bits_ext,
        "challenge": ser(&challenge),
        "positions": positions.iter().map(|p| p.to_string()).collect::<Vec<_>>(),
    })
}

/// pil2-stark's FRI grinding / proof-of-work (`Poseidon2GoldilocksGrinding::grinding`):
/// the smallest `nonce` whose width-4 Poseidon2 permutation of `challenge ++ nonce`
/// has `pow_bits` leading zero bits — its first output lane's canonical u64 is
/// `< 1 << (64 - pow_bits)`. The C++ search parallelizes the scan over OMP chunks;
/// any valid nonce passes the verifier, so the smallest (ascending) nonce is the
/// deterministic one the prover commits and the verifier re-checks.
///
/// Grinding is not in the `fields` crate (it ships only the verify-side predicate),
/// so this standalone case is the golden the Python port byte-matches. The
/// permutation is `poseidon2_hash::<_, Poseidon4, 4>` — the same width-4 permutation
/// `permutation.json` already pins. `image` is the predicate value (output lane 0),
/// cross-checked so a wrong permutation can't pass on the nonce alone.
/// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cpp#L172-L222
fn grinding_case(pow_bits: u32, seed: u64) -> Value {
    let mut state = seed;
    let challenge = [rand_fe(&mut state), rand_fe(&mut state), rand_fe(&mut state)];
    let level = 1u64 << (64 - pow_bits);

    let mut nonce = 0u64;
    let image = loop {
        let input = [challenge[0], challenge[1], challenge[2], Goldilocks::new(nonce)];
        let img = poseidon2_hash::<Goldilocks, Poseidon4, 4>(&input)[0].as_canonical_u64();
        if img < level {
            break img;
        }
        nonce += 1;
    };

    json!({
        "pow_bits": pow_bits,
        "challenge": ser(&challenge),
        "nonce": nonce.to_string(),
        "image": image.to_string(),
    })
}

/// pil2-stark `computeLEv`: the Lagrange-evaluation vector for opening the
/// committed polynomials at `xiChallenge` and its row shifts. For each opening
/// offset `p`, the per-index value is the geometric series `g^k` (k in [0, N))
/// of `g = xiChallenge * w(nBits)^p * shift^{-1}` (negative `p` inverts the
/// root power), then an INTT over the base domain N gives the coefficient row.
/// The output is row-major `LEv[(k*nOpen + i)]` cubic, matching the C++
/// `LEv[(k*openingPoints.size() + i)*FIELD_EXTENSION]` layout fed to
/// `NTT_Goldilocks::INTT(LEv, LEv, N, FIELD_EXTENSION * nOpen)`.
///
/// Truth is the reference `intt_tiny` itself (the same inverse NTT the C++
/// `computeLEv` runs); the consumer reproduces it without an NTT via the
/// geometric-series closed form, byte-identical by IDFT uniqueness.
/// https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/starks.hpp#L243-L279
fn compute_lev_case(n_bits: usize, opening_points: &[i64], seed: u64) -> Value {
    let n = 1usize << n_bits;
    let n_open = opening_points.len();

    let mut state = seed;
    let xi = [rand_fe(&mut state), rand_fe(&mut state), rand_fe(&mut state)];
    let xi_c = CubicExtensionField { value: xi };

    let w = Goldilocks::new(Goldilocks::W[n_bits]);
    let shift_inv = Goldilocks::new(Goldilocks::SHIFT).inverse();

    // xisShifted[i] = xiChallenge * w^|p| (inverted for p < 0) * shift^{-1}.
    let xis_shifted: Vec<CubicExtensionField<Goldilocks>> = opening_points
        .iter()
        .map(|&p| {
            let mut wp = pow(w, p.unsigned_abs());
            if p < 0 {
                wp = wp.inverse();
            }
            (xi_c * wp) * shift_inv
        })
        .collect();

    // LEv[k][i] = xisShifted[i]^k, row-major (k, i, limb), then INTT over N.
    let mut lev = vec![Goldilocks::ZERO; n * n_open * 3];
    for k in 0..n {
        for (i, g) in xis_shifted.iter().enumerate() {
            let v = g.pow(k as u64);
            let base = (k * n_open + i) * 3;
            lev[base..base + 3].copy_from_slice(&v.value);
        }
    }
    intt_tiny(&mut lev, n_bits, 3 * n_open);

    json!({
        "n_bits": n_bits,
        "opening_points": opening_points,
        "xi": ser(&xi),
        "lev": ser(&lev),
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
        "zisk_zorch/quotient/testdata/golden/zerofier_inv.json",
        json!({
            "every_row": [
                zerofier_inv_case(3, 1),
                zerofier_inv_case(3, 2),
                zerofier_inv_case(4, 2),
            ],
            "one_row": [
                one_row_zerofier_case(3, 2, 0),
                one_row_zerofier_case(3, 2, 1),
                one_row_zerofier_case(3, 2, 8),
                one_row_zerofier_case(4, 1, 5),
            ],
            "frame": [
                frame_zerofier_case(3, 2, 1, 1),
                frame_zerofier_case(4, 1, 2, 1),
            ],
        }),
    );
    write(
        "zisk_zorch/quotient/testdata/golden/gsum.json",
        json!({
            "denominator": [
                bus_denominator_case(2, 0x5A01),
                bus_denominator_case(3, 0x5A02),
                bus_denominator_case(5, 0x5A03),
            ],
            "grand_sum": [
                grand_sum_case(8, 1, 0x5B01),
                grand_sum_case(8, 3, 0x5B02),
                grand_sum_case(16, 2, 0x5B03),
            ],
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
    write(
        "zisk_zorch/fri/testdata/golden/fri_fold.json",
        json!({
            "cases": [
                // First step (prevBits == nBitsExt, no shift squaring), nX = 4.
                fri_fold_case(5, 5, 3, 0x101),
                // Later step (prevBits < nBitsExt, shift squared twice), nX = 4.
                fri_fold_case(5, 3, 1, 0x102),
                // Smallest fold, nX = 2.
                fri_fold_case(4, 4, 3, 0x103),
                // Wide single fold, nX = 32.
                fri_fold_case(7, 7, 2, 0x104),
                // A two-step chain at a larger domain: 6 -> 4 -> 2, nX = 4 each.
                fri_fold_case(6, 6, 4, 0x105),
                fri_fold_case(6, 4, 2, 0x106),
            ]
        }),
    );
    write(
        "zisk_zorch/fri/testdata/golden/fri_prove.json",
        json!({
            "cases": [
                // Two FRI trees, arity 4: heights 8 (irregular arity-4 padding)
                // and 2; each fold nX = 4; final pol of 2 cubic elements.
                fri_prove(5, &[5, 3, 1], 4, &[0, 3, 17, 31], 0x201),
                // Single FRI tree, arity 2 (historical single-sibling path):
                // one fold 2^4 -> 2^2, leaf row 4 cubic = 3 linear-hash blocks.
                fri_prove(4, &[4, 2], 2, &[0, 1, 9, 15], 0x202),
                // Three-layer chain, arity 3: 6 -> 4 -> 2 -> 0, two trees plus a
                // length-1 final pol; query indices probe group boundaries.
                fri_prove(6, &[6, 4, 2, 0], 3, &[0, 5, 40, 63], 0x203),
                // Production fold factor 8 (uniform drop 3), arity 4: 6 -> 3 -> 0.
                // The real ZisK FRI schedules fold by 8 (recursive2 [20,17,14,
                // 11,8,5], vadcop_final_compressed [19,16,13,10]); the cases above
                // only fold by 4 (drop 2), a factor no ZisK config uses.
                fri_prove(6, &[6, 3, 0], 4, &[0, 7, 33, 63], 0x204),
                // Production fold factor 16 (uniform drop 4), arity 2: 8 -> 4 -> 0.
                // Matches vadcop_final [21,17,13,9,5]; nX = 16 cubic per group is
                // the widest fold-group regroup the chain commits.
                fri_prove(8, &[8, 4, 0], 2, &[0, 15, 100, 255], 0x205),
            ]
        }),
    );
    write(
        "zisk_zorch/fri/testdata/golden/fri_final.json",
        json!({
            "cases": [
                // nBitsExt 5, nBits 3 -> blowup 2; last layer 2^3, bound init = 2.
                fri_final_case(&[5, 3], 3, 0x601),
                // Constant final pol: last layer == blowup (2), so init = 1 — the
                // production shape where the final pol collapses to one coeff.
                fri_final_case(&[6, 4, 2], 4, 0x602),
                // Wider final layer (2^4) with bound init = 4; multi-step schedule.
                fri_final_case(&[7, 4], 5, 0x603),
            ]
        }),
    );
    write(
        "zisk_zorch/fri/testdata/golden/query_sample.json",
        json!({
            "cases": [
                // arity 3 (width 12): 2-element final pol, 4 queries over nBitsExt 5.
                query_sample_case::<Poseidon12, 12>(1, 4, 5, 0x4142, 0x401),
                // arity 2 (width 8): length-1 final pol, queries straddle the
                // 63-bit element boundary (8 * 4 bits is small, but exercises width 8).
                query_sample_case::<Poseidon8, 8>(0, 6, 4, 0x9, 0x402),
                // arity 4 (width 16): 4-element final pol, wide nBitsExt 6,
                // 8 queries force a second squeezed element in getPermutations.
                query_sample_case::<Poseidon16, 16>(2, 8, 6, 0xABCDEF, 0x403),
            ]
        }),
    );
    write(
        "zisk_zorch/fri/testdata/golden/grinding.json",
        json!({
            "cases": [
                // Ascending pow_bits widen the required zero prefix; small values
                // keep the search short (~2^pow_bits tries) while still crossing a
                // squeezed-element worth of leading zeros.
                grinding_case(4, 0x501),
                grinding_case(8, 0x502),
                grinding_case(12, 0x503),
            ]
        }),
    );
    write(
        "zisk_zorch/evals/testdata/golden/compute_lev.json",
        json!({
            "cases": [
                // Single opening point at the row itself (p = 0).
                compute_lev_case(3, &[0], 0x301),
                // Current + next row (p = 0, 1) — the common STARK opening set.
                compute_lev_case(4, &[0, 1], 0x302),
                // Includes a negative offset (previous row), nBits = 5.
                compute_lev_case(5, &[-1, 0, 1], 0x303),
            ]
        }),
    );
}
