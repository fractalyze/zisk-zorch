"""Byte-match zisk-zorch stages against a real pil2-proofman genProof dump.

Consumes a ``PIL2_DUMP_DIR`` capture (the ``dump/per-stage-genproof`` patch on
pil2-proofman): one AIR instance's witness trace, stage buffers, roots, and
transcript-derived challenges from an actual prove. Two links are fully
byte-gated today:

- **stage-1**: ``commit_trace`` on the dumped witness must reproduce the real
  prover's stage-1 root — the assembled extend∘leaf-hash∘merkelize path on a
  real (not synthetic) trace.
- **FRI**: the production ``fri.fold`` chained over the dumped per-layer betas
  must reproduce every dumped layer, and layer 0 must equal the dumped DEEP
  polynomial.

The remaining stages report SKIP with the reason: stage-2 needs the witness-STD
hint machinery, quotient needs this AIR's cExp extraction, and evals/DEEP are
multi-opening here (``openingPoints [-1..3]``) — the wired single-opening flow
does not cover them yet. Those become gates as the pipeline grows; a mismatch
in the covered links localizes with the per-stage ``verify_*`` runnables.

Run: python -m zisk_zorch.verify_inner_proof --dump=<dir> \
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


def main() -> int:
    dump = instance = starkinfo = None
    for a in sys.argv[1:]:
        if a.startswith("--dump="):
            dump = pathlib.Path(a.split("=", 1)[1])
        elif a.startswith("--instance="):
            instance = a.split("=", 1)[1]
        elif a.startswith("--starkinfo="):
            starkinfo = pathlib.Path(a.split("=", 1)[1])
    assert dump and instance and starkinfo, "--dump/--instance/--starkinfo required"
    si = json.loads(starkinfo.read_text())
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

    for stage, why in [
        (
            "stage-2 hint computation",
            "cm2 columns from cm1+challenges need the"
            " expressions interpreter (same class as quotient cExp)",
        ),
        ("quotient", "needs this AIR's cExp extraction (in-repo gate: cexp_ref)"),
    ]:
        print(f"SKIP     {stage}: {why}")

    print("inner-proof byte-match: " + ("ALL COVERED LINKS OK" if ok_all else "FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
