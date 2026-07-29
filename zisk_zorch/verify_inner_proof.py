"""Byte-match every zisk-zorch inner-proof stage against a real pil2-proofman
genProof dump.

Consumes a ``PIL2_DUMP_DIR`` capture (the ``dump/per-stage-genproof`` patch on
pil2-proofman): one AIR instance's witness, stage buffers, roots, and
transcript-derived challenges from an actual prove. Gates, in proof order:

- stage-1 commit root from the settled witness (``commit_trace``)
- stage-2 witness columns from cm1 + the proving key alone (hint num/den
  expressions -> ``im_single``, its running sum -> ``gsum``, and the
  materialized-denominator ``ImPol``), then the stage-2 extension and root
- quotient ``q = cExp/Z_H`` via the generalized SSA interpreter
  (``quotient.cexp_ref``) against the prover's raw ``q`` section
- evals over every wrapped opening point (``compute_lev`` + committed columns)
- the DEEP polynomial in pil2's multi-opening double-Horner form
- the FRI fold chain, layer by layer, with the real per-layer betas

Field arithmetic is exact, so each gate is equal-or-wrong. The two expression-
interpreter gates are CPU-pinned: the frx GPU backend miscompiles those
specific large fused graphs (fractalyze/xla#334; the similarly-sized DEEP
graph is unaffected, so the trigger is graph shape, not op count alone).

Assumes a two-stage AIR with a single everyRow boundary (asserted) — the
ZisK / fibonacci-square shape.

Argless it runs on the committed capture under
``testdata/fibsq_specifiedranges/`` (260 KB of a real fibonacci-square prove);
an AIR without a LogUp intermediate simply skips the hint-chaining gate.
``tools/pil2-dump`` regenerates that capture, or a larger one for scale:

    python -m zisk_zorch.verify_inner_proof --dump=<dir> \
        --instance=ag0_air0_inst0 --starkinfo=<starkinfo.json>
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import frx.numpy as fnp
from zk_dtypes import goldilocks as F, goldilocksx3 as F3

from zorch.utils.field import split_coeffs

import frx

from zisk_zorch.commit.trace_commit import commit_trace
from zisk_zorch.evals.lev import compute_lev
from zisk_zorch.fri.fold import fold
from zisk_zorch.quotient.zerofier import _coset_points, _root

P = 0xFFFFFFFF00000001


def _u64(path: pathlib.Path) -> np.ndarray:
    return np.fromfile(path, dtype=np.uint64)


def _cubic(words: np.ndarray):
    return fnp.array(words.astype(F).view(F3).reshape(-1))


# The committed fixture: one real pil2-proofman genProof of fibonacci-square's
# SpecifiedRanges AIR (164 KB — small enough to commit, real enough to gate).
# Regenerate with tools/pil2-dump; point --dump at a bigger capture for scale.
_FIXTURE_DIR = pathlib.Path(__file__).parent / "testdata" / "fibsq_specifiedranges"
_FIXTURE_INSTANCE = "ag0_air2_inst5"
_FIXTURE_STARKINFO = _FIXTURE_DIR / "SpecifiedRanges.starkinfo.json"


def main() -> int:
    dump, instance, starkinfo = _FIXTURE_DIR, _FIXTURE_INSTANCE, _FIXTURE_STARKINFO
    for a in sys.argv[1:]:
        if a.startswith("--dump="):
            dump = pathlib.Path(a.split("=", 1)[1])
        elif a.startswith("--instance="):
            instance = a.split("=", 1)[1]
        elif a.startswith("--starkinfo="):
            starkinfo = pathlib.Path(a.split("=", 1)[1])
    assert dump and instance and starkinfo, "--dump/--instance/--starkinfo required"
    si = json.loads(starkinfo.read_text())
    assert si["nStages"] == 2, "buffer keys assume a two-stage AIR (quotient = cm3)"
    assert len(si["boundaries"]) == 1, "only the everyRow zerofier is wired"
    ss = si["starkStruct"]
    nb, nbe = ss["nBits"], ss["nBitsExt"]
    arity = ss["merkleTreeArity"]
    steps = [s["nBits"] for s in ss["steps"]]
    n_cols = si["mapSectionsN"]["cm1"]
    pre = lambda name: dump / f"{instance}_{name}.bin"
    print(f"instance {instance}: N=2^{nb} ext=2^{nbe} cm1={n_cols} arity={arity}")

    ok_all = True

    # trace_post is the settled stage-1 witness: hint-computed columns fill
    # asynchronously during STEP_1, so the pre-commit dump can be incomplete.
    trace_path = pre("trace_post") if pre("trace_post").exists() else pre("trace")
    trace_words = _u64(trace_path)
    assert trace_words.size == (1 << nb) * n_cols, "trace size mismatch"
    trace = fnp.array(trace_words.astype(F).reshape(1 << nb, n_cols))
    got_root = np.asarray(
        commit_trace(trace, blowup=1 << (nbe - nb), arity=arity).root
    ).astype(np.uint64)
    want_root = _u64(pre("root1"))
    ok = bool(np.array_equal(got_root, want_root))
    ok_all &= ok
    print(("OK       " if ok else "MISMATCH ") + "stage-1 commit root (real witness)")

    pol = _cubic(_u64(pre("fri_layer0")))
    deep_f = _cubic(_u64(pre("deep_f")))
    ok = bool(
        np.array_equal(np.asarray(split_coeffs(pol)), np.asarray(split_coeffs(deep_f)))
    )
    ok_all &= ok
    print(("OK       " if ok else "MISMATCH ") + "fri_layer0 == DEEP polynomial")
    for k in range(1, len(steps)):
        beta = _cubic(_u64(pre(f"fri_beta{k - 1}")))[0].reshape(())
        pol = fold(pol, beta, nbe, steps[k - 1], steps[k])
        want = _u64(pre(f"fri_layer{k}"))
        got = np.asarray(split_coeffs(pol)).reshape(-1)
        ok = bool(np.array_equal(got, want))
        ok_all &= ok
        print(
            ("OK       " if ok else "MISMATCH ")
            + f"fri fold layer {k} (nBits={steps[k]}, real beta)"
        )
        if not ok:
            break

    # -- multi-opening evals + DEEP against the dumped extended sections --
    ne = 1 << nbe
    stride = 1 << (nbe - nb)
    cmp_map = si["cmPolsMap"]
    ev_map = si["evMap"]
    openings = si["openingPoints"]
    stage_cols = {1: n_cols, 2: si["mapSectionsN"].get("cm2", 0)}
    bufs = {
        ("cm", 1): _u64(pre("cm1_ext")).astype(F).reshape(ne, n_cols),
        ("const", 0): _u64(pre("const_ext")).astype(F).reshape(ne, si["nConstants"]),
    }
    if stage_cols[2]:
        bufs[("cm", 2)] = _u64(pre("cm2_ext")).astype(F).reshape(ne, stage_cols[2])
    bufs[("cm", 3)] = (
        _u64(pre("quotient_cm"))
        .astype(F)
        .reshape(ne, si["mapSectionsN"]["cm" + str(si["nStages"] + 1)])
    )
    n_customs = len(si.get("customCommits", []))
    for ci in range(n_customs):
        name = si["customCommits"][ci]["name"] + "0"
        bufs[("custom", ci)] = (
            _u64(pre(f"custom{ci}_ext")).astype(F).reshape(ne, si["mapSectionsN"][name])
        )

    def entry_col(e):
        """The evMap entry's committed column over the extended domain, cubic
        entries joined from their three contiguous gl64 lanes."""
        if e["type"] == "cm":
            pm = cmp_map[e["id"]]
            buf = bufs[("cm", pm["stage"])]
            if pm["dim"] == 1:
                return fnp.array(buf[:, pm["stagePos"]])
            lanes = np.ascontiguousarray(buf[:, pm["stagePos"] : pm["stagePos"] + 3])
            return fnp.array(lanes.view(F3).reshape(ne))
        if e["type"] == "const":
            return fnp.array(bufs[("const", 0)][:, e["id"]])
        return fnp.array(bufs[("custom", e["commitId"])][:, e["id"]])

    chals = _u64(pre("challenges")).astype(F).view(F3).reshape(-1)
    name_of = {c.get("name", str(i)): i for i, c in enumerate(si["challengesMap"])}
    z = fnp.array(chals[name_of["std_xi"] : name_of["std_xi"] + 1])[0].reshape(())
    vf1 = fnp.array(chals[name_of["std_vf1"] : name_of["std_vf1"] + 1])[0].reshape(())
    vf2 = fnp.array(chals[name_of["std_vf2"] : name_of["std_vf2"] + 1])[0].reshape(())

    lev = compute_lev(z, list(openings), nb)
    cols = [entry_col(e) for e in ev_map]
    got_evals = fnp.stack(
        [
            fnp.sum(lev[:, e["openingPos"]] * col[::stride])
            for e, col in zip(ev_map, cols)
        ]
    )
    want_evals = _u64(pre("evals"))
    ok = bool(
        np.array_equal(np.asarray(split_coeffs(got_evals)).reshape(-1), want_evals)
    )
    ok_all &= ok
    print(
        ("OK       " if ok else "MISMATCH ")
        + f"evals ({len(ev_map)} openings over {len(openings)} points)"
    )

    g = int(np.asarray(_root(nb)))
    evals_arr = fnp.array(want_evals.astype(F).view(F3).reshape(-1))
    domain = _coset_points(nb, nbe - nb)

    def deep(cols, evals_arr, z, vf1, vf2, domain):
        """pil2 computeFRIExpression semantics: vf2-Horner within an opening
        group (evMap order), one reciprocal per group, vf1-Horner across
        groups in openingPoints order."""
        fri = None
        for k, prime in enumerate(openings):
            acc = None
            for i, e in enumerate(ev_map):
                if e["openingPos"] != k:
                    continue
                term = cols[i] - evals_arr[i]
                acc = term if acc is None else acc * vf2 + term
            xi = z * fnp.array(np.uint64(pow(g, prime % (1 << nb), P)).astype(F))
            group = acc / (domain - xi)
            fri = group if fri is None else fri * vf1 + group
        return fri

    got_f = frx.jit(deep)(cols, evals_arr, z, vf1, vf2, domain)
    want_f = _u64(pre("deep_f"))
    ok = bool(np.array_equal(np.asarray(split_coeffs(got_f)).reshape(-1), want_f))
    ok_all &= ok
    print(
        ("OK       " if ok else "MISMATCH ")
        + f"DEEP polynomial (multi-opening, {len(openings)} groups)"
    )

    if stage_cols[2] and pre("cm2_base").exists():
        cm2_base = fnp.array(
            _u64(pre("cm2_base")).astype(F).reshape(1 << nb, stage_cols[2])
        )
        c2 = commit_trace(cm2_base, blowup=1 << (nbe - nb), arity=arity)
        ok = bool(
            np.array_equal(
                np.asarray(c2.extended).astype(np.uint64),
                np.asarray(bufs[("cm", 2)]).astype(np.uint64),
            )
        )
        ok_all &= ok
        print(("OK       " if ok else "MISMATCH ") + "stage-2 extension == cm2_ext")
        ok = bool(
            np.array_equal(np.asarray(c2.root).astype(np.uint64), _u64(pre("root2")))
        )
        ok_all &= ok
        print(
            ("OK       " if ok else "MISMATCH ")
            + "stage-2 commit root (real witness-STD columns)"
        )

    # -- quotient: interpret the proving key's composite cExp on the dump --
    from zisk_zorch.quotient.cexp_ref import _run_block
    from zisk_zorch.quotient.zerofier import inv_zerofier

    ei = json.loads(
        (
            starkinfo.parent / starkinfo.name.replace("starkinfo", "expressionsinfo")
        ).read_text()
    )
    cexp_code = next(e for e in ei["expressionsCode"] if e["expId"] == si["cExpId"])[
        "code"
    ]

    def _cubic_scalar(words):
        return fnp.array(np.asarray(words, dtype=np.uint64).astype(F).view(F3))[0]

    def _values_env(path, vmap):
        """pil2 packs stage-1 values as one word, stage>=2 as three."""
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

    publics = _u64(pre("publics"))
    env = {
        "cm": {
            i: (
                entry_col({"type": "cm", "id": i})
                if cmp_map[i]["dim"] == 3
                else fnp.array(
                    bufs[("cm", cmp_map[i]["stage"])][:, cmp_map[i]["stagePos"]]
                )
            )
            for i in range(len(cmp_map))
        },
        "const": {
            i: fnp.array(bufs[("const", 0)][:, i]) for i in range(si["nConstants"])
        },
        "custom": {
            (ci, j): fnp.array(bufs[("custom", ci)][:, j])
            for ci in range(n_customs)
            for j in range(bufs[("custom", ci)].shape[1])
        },
        "challenges": {i: fnp.array(chals[i : i + 1])[0] for i in range(len(chals))},
        "publics": {i: _cubic_scalar([publics[i], 0, 0]) for i in range(len(publics))},
        "airvalues": _values_env(pre("airvalues"), si["airValuesMap"]),
        "airgroupvalues": _values_env(pre("airgroupvalues"), si["airgroupValuesMap"]),
        "proofvalues": _values_env(pre("proofvalues"), si["proofValuesMap"]),
        "zi": {0: inv_zerofier(nb, nbe - nb)},
    }
    # env enters as a jit argument: closure-captured arrays lower as in-graph
    # constants, which crashes the GPU compiler on the zerofier coset (#67).
    # CPU-pinned: the GPU backend miscompiles THIS fused graph (its 81-op
    # prefix diverges from the exact reference, the 80-op prefix matches;
    # CPU matches everywhere — fractalyze/xla#334). Unpin when #334 closes.
    with frx.default_device(frx.devices("cpu")[0]):
        got_q = frx.jit(lambda e: _run_block(cexp_code, e, stride))(env)
    want_q = _u64(pre("q_ext"))
    ok = bool(np.array_equal(np.asarray(split_coeffs(got_q)).reshape(-1), want_q))
    ok_all &= ok
    print(
        ("OK       " if ok else "MISMATCH ")
        + f"quotient q = cExp/Z_H ({len(cexp_code)} SSA ops, real interpreter output)"
    )

    # -- stage-2 hint chaining: reproduce cm2's columns from cm1 + the key --
    exps = {e["expId"]: e["code"] for e in ei["expressionsCode"]}
    const_base = np.fromfile(
        starkinfo.parent / starkinfo.name.replace(".starkinfo.json", ".const"),
        dtype=np.uint64,
    ).reshape(1 << nb, si["nConstants"])
    trace_np = np.asarray(trace).astype(np.uint64)
    env_base = dict(
        env,
        cm={i: fnp.array(trace_np[:, i].astype(F)) for i in range(n_cols)},
        const={
            i: fnp.array(const_base[:, i].astype(F)) for i in range(si["nConstants"])
        },
        zi={},
    )
    hints = {h["name"]: h for h in ei["hintsInfo"]}

    def _hint_exp(hint, field):
        v = next(f for f in hints[hint]["fields"] if f["name"] == field)
        return exps[v["values"][0]["id"]]

    # An AIR whose stage-2 carries no LogUp intermediate — a range check, say —
    # has no `im_col` hint and no imPol column, so there is nothing here to
    # chain. Report the skip rather than dying on the lookup.
    im_pol_exp = next(
        (p["expId"] for p in cmp_map if p["stage"] == 2 and p.get("imPol")), None
    )
    has_im_chaining = "im_col" in hints and im_pol_exp is not None

    def stage2_cols(e):
        num = _run_block(_hint_exp("im_col", "numerator"), e, 1)
        den = _run_block(_hint_exp("im_col", "denominator"), e, 1)
        im_single = num / den
        gsum = fnp.cumsum(im_single)
        im_pol = _run_block(exps[im_pol_exp], e, 1)
        return gsum, im_single, im_pol

    if not has_im_chaining:
        print("SKIP     stage-2 hint chaining (AIR has no LogUp im column)")
        return 0 if ok_all else 1

    # CPU-pinned like the quotient: this fused graph (division's inverse
    # chain included) also miscompiles on GPU — ImPol came back wrong while
    # the exact reference and CPU agree (fractalyze/xla#334).
    with frx.default_device(frx.devices("cpu")[0]):
        got_cols = frx.jit(stage2_cols)(env_base)
    assert pre("cm2_base").exists(), "stage-2 hint gate needs the cm2_base dump"
    cm2_np = _u64(pre("cm2_base")).reshape(1 << nb, stage_cols[2])
    ok = True
    for name, got_col, lo in [
        ("gsum", got_cols[0], 0),
        ("im_single", got_cols[1], 3),
        ("ImPol", got_cols[2], 6),
    ]:
        want_col = np.ascontiguousarray(cm2_np[:, lo : lo + 3]).reshape(-1)
        this = bool(
            np.array_equal(np.asarray(split_coeffs(got_col)).reshape(-1), want_col)
        )
        ok &= this
        print(
            ("OK       " if this else "MISMATCH ")
            + f"stage-2 witness column {name} (hint expressions)"
        )
    ok_all &= ok

    print("inner-proof byte-match: " + ("ALL COVERED LINKS OK" if ok_all else "FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
