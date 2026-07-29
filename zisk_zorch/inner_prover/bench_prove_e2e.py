"""Time the composed inner-proof pipeline on a real pil2-proofman dump.

The end-to-end counterpart of `verify_inner_proof`: the same real-witness
instance, but each stage is *timed* — production compute functions run in
proof order with the dump's transcript challenges (replay), and every stage
with a dumped reference byte-gates its output before its timing is trusted.
The bracket matches pil2's per-instance `GEN_PROOF` timer: settled stage-1
witness in, proof sections out. Excluded on both sides: witness computation,
setup/key loading, and (ours) jit compilation — one warmup iteration
compiles, the reported number is the median of the warm repetitions.

Stages and their references:

- stage-1 commit (`commit_trace`) — root1
- stage-2 witness (hint num/den -> im_single, gsum, ImPol) — cm2_base
- stage-2 commit (`commit_trace`) — root2
- quotient (`cexp_ref` SSA interpreter over the composite cExp) — q_ext
- quotient commit (`merkle_tree` over the 3-limb codeword) — rootQ
- evals (`compute_lev` + committed-column sums) — evals
- DEEP (multi-opening double-Horner) — deep_f
- FRI fold+commit chain (`fri.prover.prove`, own transcript betas) — untimed
  byte-gate lives in `verify_inner_proof` (real betas); here it is timed only
- queries (grind + query squeeze + tree openings) — timed only

`--backend=device` runs every stage on the accelerator; it needs a frx with
the xla#335 fix (>= `0.10.1.dev20260728045935`) and carries the xla#340
rotation workaround (`_defuse_rotations`). `--backend=hybrid` keeps the two
expression-interpreter stages CPU-pinned — the correct mode for wheels that
still miscompile them (xla#334); the pin's device round-trips are part of
the reported time. `--backend=cpu` pins the whole pipeline to CPU.

Run (repo root on PYTHONPATH):
    XLA_FLAGS=--xla_gpu_experimental_max_unroll_factor=1 \
    python -m zisk_zorch.inner_prover.bench_prove_e2e --dump=<dir> \
        --instance=ag0_air0_inst0 --starkinfo=<...starkinfo.json> \
        --backend=device --reps=5
"""

from __future__ import annotations

import argparse
import copy
import json
import pathlib
import time
from types import SimpleNamespace

import frx
import frx.numpy as fnp
import numpy as np
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zorch.utils.field import join_coeffs, split_coeffs

from zisk_zorch.commit.trace_commit import commit_trace, merkle_tree
from zisk_zorch.evals.lev import compute_lev
from zisk_zorch.fri.prover import FriProof, _Layer, prove
from zisk_zorch.fri.queries import sample_query_positions
from zisk_zorch.poseidon2.goldilocks import goldilocks_perm
from zisk_zorch.quotient.cexp_ref import _run_block
from zisk_zorch.quotient.zerofier import inv_zerofier
from zisk_zorch.transcript.transcript import Transcript

P = 0xFFFFFFFF00000001


def _u64(path: pathlib.Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint64)


def _block(x):
    """Block until every array in a pytree is computed."""
    for leaf in frx.tree_util.tree_leaves(x):
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()
    return x


def _defuse_rotations(code, first_free_id):
    """Redirect every rotated (`prime` != 0) cm/const/custom operand in `code`
    to a fresh synthetic cm id, returning the rewritten copy plus the
    (new_id, type, commit_id, id, prime) list to materialize per call.

    The GPU backend miscompiles the rotation when it is fused into a larger
    cubic graph — one wrong limb at the wrap row, value-dependent
    (fractalyze/xla#340; still present in frx `dev20260729002119`). A
    rotation materialized as its own tiny kernel is exact, so the fused
    graph reads a pre-rolled column instead of rolling inline. Remove this
    pass when #340 closes."""
    code = copy.deepcopy(code)
    rotations = {}
    for op in code:
        for s in op["src"]:
            if s.get("type") in ("cm", "const", "custom") and s.get("prime"):
                key = (s["type"], s.get("commitId"), s["id"], s["prime"])
                if key not in rotations:
                    rotations[key] = first_free_id + len(rotations)
                s.update(type="cm", id=rotations[key], prime=0)
                s.pop("commitId", None)
    return code, [(v, *k) for k, v in rotations.items()]


def _materialize_rotations(env, rotations, extend):
    """The per-call half of `_defuse_rotations`: roll each rotated source
    column eagerly (its own kernel) into the synthetic cm slots."""
    if not rotations:
        return env
    cm = dict(env["cm"])
    for new_id, typ, commit_id, src_id, prime in rotations:
        src = (
            env["custom"][(commit_id, src_id)] if typ == "custom" else env[typ][src_id]
        )
        cm[new_id] = fnp.roll(src, -(prime * extend))
    return dict(env, cm=cm)


def _timed(fn, reps):
    """One compiling warmup call, then `reps` timed calls -> (result, median s)."""
    result = _block(fn())
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        _block(fn())
        times.append(time.perf_counter() - t0)
    return result, float(np.median(times))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dump", type=pathlib.Path, required=True)
    ap.add_argument("--instance", required=True)
    ap.add_argument("--starkinfo", type=pathlib.Path, required=True)
    ap.add_argument("--backend", choices=["hybrid", "cpu", "device"], default="hybrid")
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()

    cpu = frx.devices("cpu")[0]
    if args.backend == "cpu" and any(d.platform != "cpu" for d in frx.devices()):
        raise SystemExit(
            "--backend=cpu requires FRX_PLATFORMS=cpu so every stage, not "
            "just the pinned ones, lowers for the host"
        )

    si = json.loads(args.starkinfo.read_text())
    assert si["nStages"] == 2 and len(si["boundaries"]) == 1
    ss = si["starkStruct"]
    nb, nbe = ss["nBits"], ss["nBitsExt"]
    ne, stride = 1 << nbe, 1 << (nbe - nb)
    arity = ss["merkleTreeArity"]
    steps = [s["nBits"] for s in ss["steps"]]
    n_cols = si["mapSectionsN"]["cm1"]
    cm2_cols = si["mapSectionsN"]["cm2"]
    cmp_map, ev_map, openings = si["cmPolsMap"], si["evMap"], si["openingPoints"]
    pre = lambda name: args.dump / f"{args.instance}_{name}.bin"
    print(
        f"instance {args.instance}: N=2^{nb} ext=2^{nbe} cm1={n_cols} "
        f"cm2={cm2_cols} arity={arity} backend={args.backend} reps={args.reps}"
    )

    # --- inputs the GEN_PROOF bracket starts from (settled witness + key) ---
    trace = fnp.array(_u64(pre("trace_post")).astype(F).reshape(1 << nb, n_cols))
    const_ext = _u64(pre("const_ext")).astype(F).reshape(ne, si["nConstants"])
    custom_ext = {
        ci: _u64(pre(f"custom{ci}_ext"))
        .astype(F)
        .reshape(ne, si["mapSectionsN"][c["name"] + "0"])
        for ci, c in enumerate(si.get("customCommits", []))
    }
    const_base = np.fromfile(
        args.starkinfo.parent
        / args.starkinfo.name.replace(".starkinfo.json", ".const"),
        dtype=np.uint64,
    ).reshape(1 << nb, si["nConstants"])
    ei = json.loads(
        (
            args.starkinfo.parent
            / args.starkinfo.name.replace("starkinfo", "expressionsinfo")
        ).read_text()
    )
    cexp_code = next(e for e in ei["expressionsCode"] if e["expId"] == si["cExpId"])[
        "code"
    ]
    exps = {e["expId"]: e["code"] for e in ei["expressionsCode"]}
    hints = {h["name"]: h for h in ei["hintsInfo"]}

    def _hint_exp(hint, field):
        v = next(f for f in hints[hint]["fields"] if f["name"] == field)
        return exps[v["values"][0]["id"]]

    im_pol_exp = next(p["expId"] for p in cmp_map if p["stage"] == 2 and p.get("imPol"))

    # Replayed transcript state: challenges as the native prove drew them.
    chals = _u64(pre("challenges")).astype(F).view(F3).reshape(-1)
    name_of = {c.get("name", str(i)): i for i, c in enumerate(si["challengesMap"])}
    scalar = lambda n: fnp.array(chals[name_of[n] : name_of[n] + 1])[0].reshape(())
    z, vf1, vf2 = scalar("std_xi"), scalar("std_vf1"), scalar("std_vf2")

    publics = _u64(pre("publics"))

    def _cubic_scalar(words):
        return fnp.array(np.asarray(words, dtype=np.uint64).astype(F).view(F3))[0]

    def _values_env(path, vmap):
        words = _u64(path)
        out, off = {}, 0
        for i, v in enumerate(vmap):
            if v["stage"] == 1:
                out[i] = _cubic_scalar([words[off], 0, 0])
                off += 1
            else:
                out[i] = _cubic_scalar(words[off : off + 3])
                off += 3
        return out

    scalar_env = {
        "challenges": {i: fnp.array(chals[i : i + 1])[0] for i in range(len(chals))},
        "publics": {i: _cubic_scalar([publics[i], 0, 0]) for i in range(len(publics))},
        "airvalues": _values_env(pre("airvalues"), si["airValuesMap"]),
        "airgroupvalues": _values_env(pre("airgroupvalues"), si["airgroupValuesMap"]),
        "proofvalues": _values_env(pre("proofvalues"), si["proofValuesMap"]),
    }
    zi = inv_zerofier(nb, nbe - nb)

    timings: list[tuple[str, float]] = []
    gates_ok = True

    # Every stage callable is jitted ONCE and reused across the timed reps —
    # a fresh `frx.jit(...)` per call misses the executable cache and times
    # the recompile, not the runtime. The two sponge permutations must be
    # memoized host-side before any jit traces them (their M4 analysis cannot
    # see a tracer).
    goldilocks_perm(12)
    mt = merkle_tree(arity)

    def commit_fn(t):
        c = commit_trace(t, blowup=stride, arity=arity)
        return c.extended, c.root, c.digest_layers

    commit_jit = frx.jit(commit_fn)

    def gate(name, got_u64, want_path):
        nonlocal gates_ok
        ok = bool(np.array_equal(np.asarray(got_u64), _u64(want_path)))
        gates_ok &= ok
        print(("OK       " if ok else "MISMATCH ") + name)

    # --- 1. stage-1 commit -------------------------------------------------
    (ext1, root1, layers1), t = _timed(lambda: commit_jit(trace), args.reps)
    c1 = SimpleNamespace(extended=ext1, root=root1, digest_layers=layers1)
    timings.append(("stage-1 commit", t))
    gate("stage-1 commit root", np.asarray(c1.root).astype(np.uint64), pre("root1"))

    # --- 2. stage-2 witness (CPU-pinned: fractalyze/xla#334) ---------------
    env_base = dict(
        scalar_env,
        cm={i: trace[:, i] for i in range(n_cols)},
        const={
            i: fnp.array(const_base[:, i].astype(F)) for i in range(si["nConstants"])
        },
        custom={},
        zi={},
    )

    free_id = len(cmp_map)
    num_code, num_rots = _defuse_rotations(_hint_exp("im_col", "numerator"), free_id)
    den_code, den_rots = _defuse_rotations(
        _hint_exp("im_col", "denominator"), free_id + len(num_rots)
    )
    impol_code, impol_rots = _defuse_rotations(
        exps[im_pol_exp], free_id + len(num_rots) + len(den_rots)
    )
    s2_rots = num_rots + den_rots + impol_rots

    def stage2_cols(e):
        num = _run_block(num_code, e, 1)
        den = _run_block(den_code, e, 1)
        im_single = num / den
        gsum = fnp.cumsum(im_single)
        im_pol = _run_block(impol_code, e, 1)
        return gsum, im_single, im_pol

    s2_jit = frx.jit(stage2_cols)

    def run_stage2():
        # `device` trusts the (rotation-defused) GPU lowering; `hybrid` keeps
        # the historical CPU pin for wheels still carrying the miscompile.
        e = _materialize_rotations(env_base, s2_rots, 1)
        if args.backend == "device":
            return s2_jit(e)
        with frx.default_device(cpu):
            return s2_jit(e)

    s2, t = _timed(run_stage2, args.reps)
    timings.append(("stage-2 witness", t))
    cm2 = fnp.concatenate(
        [split_coeffs(col) for col in s2], axis=1
    )  # (N, 9): gsum | im_single | ImPol — pil2's cm2 column order
    gate(
        "stage-2 witness columns",
        np.asarray(cm2).astype(np.uint64).reshape(-1),
        pre("cm2_base"),
    )

    # --- 3. stage-2 commit -------------------------------------------------
    (ext2, root2, layers2), t = _timed(lambda: commit_jit(cm2), args.reps)
    c2 = SimpleNamespace(extended=ext2, root=root2, digest_layers=layers2)
    timings.append(("stage-2 commit", t))
    gate("stage-2 commit root", np.asarray(c2.root).astype(np.uint64), pre("root2"))

    # --- 4. quotient (CPU-pinned: fractalyze/xla#334) ----------------------
    def _ext_col(i):
        pm = cmp_map[i]
        base = {1: c1, 2: c2}[pm["stage"]].extended
        if pm["dim"] == 1:
            return base[:, pm["stagePos"]]
        return join_coeffs(base[:, pm["stagePos"] : pm["stagePos"] + 3], F3)

    # Stage-3 (quotient) columns don't exist yet — the cExp reads only
    # committed stages 1-2; evals picks the quotient up after its commit.
    env_ext = dict(
        scalar_env,
        cm={
            i: _ext_col(i) for i in range(len(cmp_map)) if cmp_map[i]["stage"] in (1, 2)
        },
        const={
            i: fnp.array(const_ext[:, i].astype(F)) for i in range(si["nConstants"])
        },
        custom={
            (ci, j): fnp.array(buf[:, j])
            for ci, buf in custom_ext.items()
            for j in range(buf.shape[1])
        },
        zi={0: zi},
    )

    cexp_defused, cexp_rots = _defuse_rotations(cexp_code, free_id + len(s2_rots))
    q_jit = frx.jit(lambda e: _run_block(cexp_defused, e, stride))

    def run_quotient():
        e = _materialize_rotations(env_ext, cexp_rots, stride)
        if args.backend == "device":
            return q_jit(e)
        with frx.default_device(cpu):
            return q_jit(e)

    q, t = _timed(run_quotient, args.reps)
    timings.append(("quotient", t))
    gate(
        "quotient q = cExp/Z_H",
        np.asarray(split_coeffs(q)).reshape(-1),
        pre("q_ext"),
    )

    # --- 5. quotient commit ------------------------------------------------
    q_matrix = split_coeffs(q)
    qc_jit = frx.jit(lambda m: mt.commit(m))

    (root_q, layers_q), t = _timed(lambda: qc_jit(q_matrix), args.reps)
    timings.append(("quotient commit", t))
    gate("quotient commit root", np.asarray(root_q).astype(np.uint64), pre("rootQ"))

    # --- 6. evals ----------------------------------------------------------
    def entry_col(e):
        if e["type"] == "cm":
            pm = cmp_map[e["id"]]
            if pm["stage"] == si["nStages"] + 1:
                return q if pm["dim"] == 3 else q_matrix[:, pm["stagePos"]]
            return env_ext["cm"][e["id"]]
        if e["type"] == "const":
            return env_ext["const"][e["id"]]
        return env_ext["custom"][(e["commitId"], e["id"])]

    cols = [entry_col(e) for e in ev_map]

    # cols/consts enter as jit arguments: closure-captured arrays lower as
    # in-graph constants (#67's crash class, and they'd be folded out of the
    # timing). `lev_constants` is compute_lev's eager prologue for exactly
    # this fused-zone use.
    from zisk_zorch.evals.lev import lev_constants

    lev_consts = lev_constants(list(openings), nb)

    def evals_fn(z, cols, consts):
        lev = compute_lev(z, list(openings), nb, consts=consts)
        return fnp.stack(
            [
                fnp.sum(lev[:, e["openingPos"]] * col[::stride])
                for e, col in zip(ev_map, cols)
            ]
        )

    evals_jit = frx.jit(evals_fn)

    def run_evals():
        return evals_jit(z, cols, lev_consts)

    got_evals, t = _timed(run_evals, args.reps)
    timings.append(("evals", t))
    gate(
        "evals",
        np.asarray(split_coeffs(got_evals)).reshape(-1),
        pre("evals"),
    )

    # --- 7. DEEP -----------------------------------------------------------
    from zisk_zorch.quotient.zerofier import _coset_points, _root

    g = int(np.asarray(_root(nb)))
    domain = _coset_points(nb, nbe - nb)

    def deep(cols, evals_arr, z, vf1, vf2, domain):
        accs, dens = [], []
        for k, prime in enumerate(openings):
            acc = None
            for i, e in enumerate(ev_map):
                if e["openingPos"] != k:
                    continue
                term = cols[i] - evals_arr[i]
                acc = term if acc is None else acc * vf2 + term
            xi = z * fnp.array(np.uint64(pow(g, prime % (1 << nb), P)).astype(F))
            accs.append(acc)
            dens.append(domain - xi)
        # One reciprocal for all opening groups (Montgomery batch inversion):
        # each 1/den costs a full Fermat chain, and the chain dominates this
        # stage — prefix-multiply, invert the product once, unwind.
        prefix = [dens[0]]
        for d in dens[1:]:
            prefix.append(prefix[-1] * d)
        inv_all = fnp.ones_like(prefix[-1]) / prefix[-1]
        invs = [None] * len(dens)
        for k in range(len(dens) - 1, 0, -1):
            invs[k] = inv_all * prefix[k - 1]
            inv_all = inv_all * dens[k]
        invs[0] = inv_all
        fri = None
        for acc, inv in zip(accs, invs):
            group = acc * inv
            fri = group if fri is None else fri * vf1 + group
        return fri

    evals_arr = got_evals
    deep_jit = frx.jit(deep)
    deep_f, t = _timed(
        lambda: deep_jit(cols, evals_arr, z, vf1, vf2, domain), args.reps
    )
    timings.append(("DEEP", t))
    gate(
        "DEEP polynomial",
        np.asarray(split_coeffs(deep_f)).reshape(-1),
        pre("deep_f"),
    )

    # --- 8. FRI fold+commit chain (own transcript betas; fold byte-gate ----
    # with the real betas lives in verify_inner_proof) ----------------------
    # FriProof/_Layer are not pytrees, so the jit boundary carries their
    # arrays and both are rebuilt outside for the query phase.
    def fri_outputs(pol):
        t = Transcript()
        t.put(fnp.zeros((1,), F))
        proof, layers = prove(pol, steps, arity=arity, transcript=t)
        return (
            proof.final_pol,
            proof.roots,
            [l.matrix for l in layers],
            [l.digest_layers for l in layers],
        )

    fri_jit = frx.jit(fri_outputs)
    (final_pol, fri_roots, fri_mats, fri_digests), t = _timed(
        lambda: fri_jit(deep_f), args.reps
    )
    fri_proof = FriProof(roots=fri_roots, final_pol=final_pol)
    fri_layers = [
        _Layer(mt, m, d, steps[i + 1])
        for i, (m, d) in enumerate(zip(fri_mats, fri_digests))
    ]
    timings.append(("FRI (fold+commit)", t))

    # --- 9. queries (grind + squeeze + tree openings) ----------------------
    from zisk_zorch.commit.openings import group_proof
    from zisk_zorch.fri.queries import (
        _GRIND_PERM,
        _GRIND_WIDTH,
        _grind_level,
        grinding_seed_challenge,
    )
    from zisk_zorch.transcript.transcript import _canonical

    n_queries, pow_bits = ss["nQueries"], ss["powBits"]

    # The production query phase is host-paced: `_GRIND_CHUNK` is 256 (a CPU
    # chunk size — pow 16 costs ~256 device round-trips), `get_permutations`
    # squeezes and bit-unpacks in Python, and the per-tree openings re-vmap
    # (recompile) on every call. Same math, device-paced: the transcript is a
    # registered pytree so seed and squeeze trace under jit, the grind hashes
    # 2^17 nonces per launch, the bit unpack vectorizes, and every opening
    # callable is jitted once. Byte-checked below against the production
    # `sample_query_positions` before any timing is trusted.
    def seed_fn(t, pol):
        return grinding_seed_challenge(t, pol), t

    seed_jit = frx.jit(seed_fn)

    grind_chunk = 1 << 17

    def grind_images_fn(challenge, base):
        nonces = base + fnp.arange(grind_chunk, dtype=np.uint64)
        chal = fnp.broadcast_to(challenge, (grind_chunk, _GRIND_WIDTH - 1))
        states = fnp.concatenate([chal, nonces.astype(F)[:, None]], axis=1)
        return frx.vmap(_GRIND_PERM.permute)(states)[:, 0]

    grind_jit = frx.jit(grind_images_fn)

    def grind_fast(challenge):
        level = np.uint64(_grind_level(pow_bits))
        base = 0
        while True:
            hits = np.nonzero(
                _canonical(grind_jit(challenge, np.uint64(base))) < level
            )[0]
            if hits.size:
                return base + int(hits[0])
            base += grind_chunk

    bits_per = 63
    n_fields = (n_queries * nbe - 1) // bits_per + 1

    def squeeze_fn(challenge, nonce_fe):
        perm = Transcript(12)
        perm.put(challenge)
        perm.put(nonce_fe)
        return fnp.stack([perm.get_fields1() for _ in range(n_fields)])

    squeeze_jit = frx.jit(squeeze_fn)

    def positions_fast(challenge, nonce):
        fields = _canonical(
            squeeze_jit(challenge, fnp.array(np.array([nonce], dtype=np.uint64), F))
        )
        bits = (
            (
                np.asarray(fields, dtype=np.uint64)[:, None]
                >> np.arange(bits_per, dtype=np.uint64)
            )
            & np.uint64(1)
        ).reshape(-1)[: n_queries * nbe]
        return bits.reshape(n_queries, nbe) @ (
            np.uint64(1) << np.arange(nbe, dtype=np.uint64)
        )

    def open_fn(matrix, digest_layers, idx):
        return frx.vmap(lambda i: group_proof(mt, matrix, digest_layers, i))(idx)

    open_jit = frx.jit(open_fn)

    def run_queries():
        transcript = Transcript()
        transcript.put(fnp.zeros((1,), F))
        challenge, _ = seed_jit(transcript, fri_proof.final_pol)
        nonce = grind_fast(challenge)
        positions = positions_fast(challenge, nonce)
        idx_ext = fnp.asarray(positions & np.uint64(ne - 1))
        trace_open = open_jit(c1.extended, c1.digest_layers, idx_ext)
        quotient_open = open_jit(q_matrix, layers_q, idx_ext)
        fri_open = [
            open_jit(
                layer.matrix,
                layer.digest_layers,
                fnp.asarray(positions % np.uint64(1 << layer.leaf_bits)),
            )
            for layer in fri_layers
        ]
        return trace_open, quotient_open, fri_open

    # Byte-check the fast path against the production query phase once.
    _t_ref = Transcript()
    _t_ref.put(fnp.zeros((1,), F))
    _want_pos, _want_nonce = sample_query_positions(
        _t_ref,
        fri_proof.final_pol,
        pow_bits=pow_bits,
        n_queries=n_queries,
        n_bits_ext=nbe,
    )
    _t_fast = Transcript()
    _t_fast.put(fnp.zeros((1,), F))
    _chal, _ = seed_jit(_t_fast, fri_proof.final_pol)
    _got_nonce = grind_fast(_chal)
    _got_pos = positions_fast(_chal, _got_nonce)
    _ok = _got_nonce == _want_nonce and np.array_equal(
        _got_pos, np.asarray(_want_pos, dtype=np.uint64)
    )
    gates_ok &= _ok
    print(("OK       " if _ok else "MISMATCH ") + "query grind + positions")

    _, t = _timed(run_queries, args.reps)
    timings.append(("queries", t))

    # --- composed wall: the GEN_PROOF-equivalent bracket -------------------
    # One pass through every stage in proof order with no per-stage syncs —
    # frx dispatch is async, so independent kernel launches pipeline exactly
    # as the native prover's stream does; the only mid-pipeline host syncs
    # are the ones the protocol forces (the grind and the query-position
    # squeeze read device bytes). The per-stage rows above each pay a full
    # device sync, so their sum over-counts this wall.
    def run_all():
        e1, r1, l1 = commit_jit(trace)
        s2c = s2_jit(_materialize_rotations(env_base, s2_rots, 1))
        cm2_w = fnp.concatenate([split_coeffs(c) for c in s2c], axis=1)
        e2, r2, _l2 = commit_jit(cm2_w)
        env_q = dict(
            env_ext,
            cm={
                i: _ext_col_from(e1, e2, i)
                for i in range(len(cmp_map))
                if cmp_map[i]["stage"] in (1, 2)
            },
        )
        qq = q_jit(_materialize_rotations(env_q, cexp_rots, stride))
        qm = split_coeffs(qq)
        rq, lq = qc_jit(qm)
        ev = evals_jit(z, cols, lev_consts)
        df = deep_jit(cols, ev, z, vf1, vf2, domain)
        fp_, _roots, mats_, digs_ = fri_jit(df)
        transcript = Transcript()
        transcript.put(fnp.zeros((1,), F))
        challenge, _ = seed_jit(transcript, fp_)
        nonce = grind_fast(challenge)
        positions = positions_fast(challenge, nonce)
        idx = fnp.asarray(positions & np.uint64(ne - 1))
        outs = [
            open_jit(e1, l1, idx),
            open_jit(qm, lq, idx),
            *(
                open_jit(m, d, fnp.asarray(positions % np.uint64(1 << s)))
                for m, d, s in zip(mats_, digs_, steps[1:])
            ),
        ]
        return (r1, r2, rq, ev, df, fp_, outs)

    def _ext_col_from(e1, e2, i):
        pm = cmp_map[i]
        base = {1: e1, 2: e2}[pm["stage"]]
        if pm["dim"] == 1:
            return base[:, pm["stagePos"]]
        return join_coeffs(base[:, pm["stagePos"] : pm["stagePos"] + 3], F3)

    _, wall = _timed(run_all, args.reps)

    # --- report ------------------------------------------------------------
    total = sum(t for _, t in timings)
    print()
    for name, t in timings:
        print(f"{name:24s} {t * 1e3:9.2f} ms")
    print(f"{'TOTAL (stage sum)':24s} {total * 1e3:9.2f} ms")
    print(f"{'COMPOSED WALL':24s} {wall * 1e3:9.2f} ms")
    print(
        "byte-gates: "
        + ("ALL OK" if gates_ok else "FAILED — timings above are UNTRUSTED")
    )
    return 0 if gates_ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
