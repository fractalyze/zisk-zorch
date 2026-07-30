"""The inner-proof byte-gates: one per stage against a real pil2-proofman
genProof dump.

Each gate runs the production code path on the capture's real inputs and
compares byte-exactly against the section the native prover dumped
(``capture.Capture`` owns the bundle format, so gates read protocol objects,
never files). Field arithmetic is exact, so each gate is equal-or-wrong.
``GATES`` lists them in proof order; ``verify_inner_proof`` is the runnable
that drives them over a bundle.

The expression-interpreter gates — and the composed full-prove gate, whose
roles run the same graphs — are CPU-pinned: the pinned frx wheel's GPU
backend miscompiles their fused graphs (fractalyze/xla#334).

Assumes a two-stage AIR with a single everyRow boundary (asserted) — the
ZisK / fibonacci-square shape.
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zorch.pcs.deep import open_columns

from zisk_zorch.commit.trace_commit import commit_trace
from zisk_zorch.evals.lev import compute_lev
from zisk_zorch.fri.fold import fold
from zisk_zorch.fri.queries import grind_is_valid, query_positions_for
from zisk_zorch.fri.seam import Pil2FriCode
from zisk_zorch.inner_prover.capture import Capture, P, cubic, limbs
from zisk_zorch.inner_prover.prover import Pil2InnerProver
from zisk_zorch.inner_prover.schedule import _fri_layer_root, replay_challenges
from zisk_zorch.quotient.cexp_ref import _run_block
from zisk_zorch.quotient.gsum import grand_sum
from zisk_zorch.quotient.zerofier import _coset_points, _root
from zisk_zorch.transcript.transcript import DIGEST, Transcript
from zisk_zorch.types import InnerWitness


def _check(ok: bool, label: str) -> bool:
    print(("OK       " if ok else "MISMATCH ") + label)
    return bool(ok)


def _match(label: str, got, want) -> bool:
    """One byte compare with its verdict line, got/want printed on mismatch —
    the composed gate's values are self-derived, so a bare MISMATCH would
    leave nothing to diff against the dump. ``array_equal``, so a shape
    divergence reads as a mismatch rather than broadcasting away."""
    got, want = np.asarray(got), np.asarray(want)
    ok = bool(np.array_equal(got, want))
    print(("OK       " if ok else "MISMATCH ") + label)
    if not ok:
        print(f"  got:  {got}")
        print(f"  want: {want}")
    return ok


def _on_cpu(fn, arg):
    """Run a jitted expression-interpreter graph on the host: the pinned frx
    GPU backend miscompiles these fused graphs (fractalyze/xla#334). The env
    must enter as a jit argument — closure-captured arrays lower as in-graph
    constants, which crashes the compiler on the zerofier coset (#67)."""
    with frx.default_device(frx.devices("cpu")[0]):
        return frx.jit(fn)(arg)


def verify_transcript_schedule(cap: Capture) -> bool:
    """The Fiat-Shamir spine: replay pil2's absorb/squeeze schedule from the
    capture and byte-compare every squeezed challenge. Equality transitively
    gates the absorbs the dump does not carry (FRI layer roots, hashed
    evals)."""
    ok = True
    for label, got, want in replay_challenges(cap):
        ok &= bool(np.array_equal(got, want))
    return _check(ok, "transcript schedule (every squeezed challenge)")


def verify_stage1_commit(cap: Capture) -> bool:
    """pil2 STEP_1: extend-and-merkelize the settled witness -> ``root1``."""
    commitment = commit_trace(cap.trace, blowup=cap.stride, arity=cap.arity)
    got = np.asarray(commitment.root).astype(np.uint64)
    return _check(
        np.array_equal(got, cap.u64("root1")), "stage-1 commit root (real witness)"
    )


def verify_stage2_witness(cap: Capture) -> bool | None:
    """pil2 CALCULATE_WITNESS_STD + IM_POLS: reproduce cm2's columns from cm1
    and the proving key alone — the ``im_col`` hint's num/den expressions give
    ``im_single``, ``grand_sum`` its running sum, and ``ImPol`` materializes
    the denominator. ``None`` = the AIR has nothing to gate here."""
    if cap.im_col_exps is None:
        print("SKIP     stage-2 hint chaining (AIR has no LogUp im column)")
        return None
    num_code, den_code, im_pol_code = cap.im_col_exps

    def stage2_cols(env):
        num = _run_block(num_code, env, 1)
        den = _run_block(den_code, env, 1)
        gsum = grand_sum(num[:, None], den[:, None])
        return gsum, num / den, _run_block(im_pol_code, env, 1)

    got = _on_cpu(stage2_cols, cap.base_env)
    assert cap.path("cm2_base").exists(), "stage-2 hint gate needs the cm2_base dump"
    cm2 = cap.u64("cm2_base").reshape(cap.n, cap.cm2_cols)
    ok = True
    for name, col, lo in [
        ("gsum", got[0], 0),
        ("im_single", got[1], 3),
        ("ImPol", got[2], 6),
    ]:
        want = np.ascontiguousarray(cm2[:, lo : lo + 3]).reshape(-1)
        ok &= _check(
            np.array_equal(limbs(col), want),
            f"stage-2 witness column {name} (hint expressions)",
        )
    return ok


def verify_stage2_commit(cap: Capture) -> bool | None:
    """pil2 STEP_2 commit: extend-and-merkelize the dumped witness-STD
    columns -> ``cm2_ext`` and ``root2``."""
    if not cap.cm2_cols or not cap.path("cm2_base").exists():
        return None
    cm2_base = fnp.array(cap.u64("cm2_base").astype(F).reshape(cap.n, cap.cm2_cols))
    commitment = commit_trace(cm2_base, blowup=cap.stride, arity=cap.arity)
    ok = _check(
        np.array_equal(
            np.asarray(commitment.extended).astype(np.uint64),
            np.asarray(cap.bufs[("cm", 2)]).astype(np.uint64),
        ),
        "stage-2 extension == cm2_ext",
    )
    return ok & _check(
        np.array_equal(np.asarray(commitment.root).astype(np.uint64), cap.u64("root2")),
        "stage-2 commit root (real witness-STD columns)",
    )


def verify_quotient(cap: Capture) -> bool:
    """pil2 CALCULATE_QUOTIENT: interpret the key's composite cExp over the
    extended sections -> the prover's raw ``q`` section."""
    code = cap.cexp_code
    got = _on_cpu(lambda env: _run_block(code, env, cap.stride), cap.cexp_env)
    return _check(
        np.array_equal(limbs(got), cap.u64("q_ext")),
        f"quotient q = cExp/Z_H ({len(code)} SSA ops, real interpreter output)",
    )


def verify_evals(cap: Capture) -> bool:
    """pil2 STEP_EVALS via the production opening path: ``compute_lev``'s
    weight matrix (not zorch's ``compute_lagrange_basis``, quadratic in the
    base domain) drives ``open_columns``, whose base/extension column split
    re-permutes to evMap order for the byte compare."""
    lev = compute_lev(cap.challenge("std_xi"), list(cap.openings), cap.nb)
    is_base = [col.dtype == F for col in cap.opened_columns]
    order = [i for i, b in enumerate(is_base) if b] + [
        i for i, b in enumerate(is_base) if not b
    ]
    base = [cap.opened_columns[i] for i in order if is_base[i]]
    ext = [cap.opened_columns[i] for i in order[len(base) :]]
    got_split = open_columns(
        fnp.stack(base, axis=1) if base else fnp.zeros((cap.ne, 0), F),
        fnp.stack(ext, axis=1) if ext else fnp.zeros((cap.ne, 0), F3),
        lev,
        [cap.ev_map[i]["openingPos"] for i in order],
        stride=cap.stride,
    )
    got = fnp.zeros_like(got_split).at[np.array(order)].set(got_split)
    return _check(
        np.array_equal(limbs(got), cap.u64("evals")),
        f"evals ({len(cap.ev_map)} openings over {len(cap.openings)} points)",
    )


def verify_deep(cap: Capture) -> bool:
    """pil2 computeFRIExpression: vf2-Horner within an opening group (evMap
    order), one reciprocal per group, vf1-Horner across groups. zorch's
    ``deep_composition`` batches with one challenge's powers, so it cannot
    express this two-challenge form — the loop stays local."""
    g = int(np.asarray(_root(cap.nb)))
    z = cap.challenge("std_xi")
    vf1, vf2 = cap.challenge("std_vf1"), cap.challenge("std_vf2")
    evals_arr = cubic(cap.u64("evals"))
    domain = _coset_points(cap.nb, cap.nbe - cap.nb)

    def deep(cols, evals_arr, z, vf1, vf2, domain):
        fri = None
        for k, prime in enumerate(cap.openings):
            acc = None
            for i, e in enumerate(cap.ev_map):
                if e["openingPos"] != k:
                    continue
                term = cols[i] - evals_arr[i]
                acc = term if acc is None else acc * vf2 + term
            xi = z * fnp.array(np.uint64(pow(g, prime % cap.n, P)).astype(F))
            group = acc / (domain - xi)
            fri = group if fri is None else fri * vf1 + group
        return fri

    got = frx.jit(deep)(cap.opened_columns, evals_arr, z, vf1, vf2, domain)
    return _check(
        np.array_equal(limbs(got), cap.u64("deep_f")),
        f"DEEP polynomial (multi-opening, {len(cap.openings)} groups)",
    )


def verify_fri_chain(cap: Capture) -> bool:
    """pil2 STEP_FRI folds: chain the production ``fold`` down the layer
    schedule with the real per-layer betas, gating every layer."""
    pol = cubic(cap.u64("fri_layer0"))
    ok = _check(
        np.array_equal(limbs(pol), limbs(cubic(cap.u64("deep_f")))),
        "fri_layer0 == DEEP polynomial",
    )
    for k in range(1, len(cap.steps)):
        beta = cubic(cap.u64(f"fri_beta{k - 1}"))[0].reshape(())
        pol = fold(pol, beta, cap.nbe, cap.steps[k - 1], cap.steps[k])
        layer_ok = _check(
            np.array_equal(limbs(pol), cap.u64(f"fri_layer{k}")),
            f"fri fold layer {k} (nBits={cap.steps[k]}, real beta)",
        )
        ok &= layer_ok
        if not layer_ok:
            break
    return ok


def verify_full_prove(cap: Capture) -> bool:
    """The composed gate: run the pil2-mode `Pil2InnerProver` from the
    capture's trace and statement alone and byte-compare every stage seam —
    the per-stage gates above each pin one link with dumped inputs; this one
    proves the links compose, challenges included.

    Each stage's checks fire the moment `prove_stages` yields it, and a
    mismatch stops the drive loop — the remaining stages' compute is never
    paid, and the first bad seam is the last line printed. The final
    positions/nonce compare is the transitively-strongest link: the query
    draw seeds off the last squeeze, so equality means every absorb in the
    whole schedule matched the native prove. CPU-pinned like the interpreter
    gates (the roles' jit zones hit the same #334 graphs).
    """
    prover = Pil2InnerProver(cap.pil2_key)
    ss = cap.si["starkStruct"]
    transcript = Transcript(DIGEST * ss.get("transcriptArity", 3))
    want_ch = cap.u64("challenges").reshape(-1, 3)

    def u64(a) -> np.ndarray:
        return np.asarray(a).astype(np.uint64)

    def check_commit(commitment) -> bool:
        return _match("full prove: root1", u64(commitment.root), cap.u64("root1"))

    def check_stage2(stage2) -> bool:
        claim = stage2.reduced_claim
        ids = sorted(claim.challenges)
        ok = _match(
            "full prove: stage-2 challenges",
            np.stack([limbs(claim.challenges[i]) for i in ids]),
            want_ch[ids],
        )
        ok &= _match(
            "full prove: cm2 witness-STD columns",
            u64(stage2.reduction_proof.matrix).reshape(-1),
            cap.u64("cm2_base"),
        )
        ok &= _match("full prove: root2", u64(claim.root2), cap.u64("root2"))
        agv = claim.airgroupvalues
        return ok & _match(
            "full prove: airgroup values (gsum result)",
            np.concatenate([limbs(agv[i]) for i in sorted(agv)]),
            cap.u64("airgroupvalues"),
        )

    def check_quotient(quotient) -> bool:
        ids = sorted(quotient.reduced_claim.challenges)
        ok = _match(
            f"full prove: challenges through stage {cap.si['nStages'] + 1}",
            np.stack([limbs(quotient.reduced_claim.challenges[i]) for i in ids]),
            want_ch[ids],
        )
        ok &= _match(
            "full prove: quotient q",
            limbs(quotient.reduction_proof.codeword),
            cap.u64("q_ext"),
        )
        return ok & _match(
            "full prove: rootQ", u64(quotient.reduction_proof.root), cap.u64("rootQ")
        )

    def check_opening(opening) -> bool:
        proof = opening.reduction_proof
        ok = _match("full prove: evals", limbs(proof.evals), cap.u64("evals"))
        code = Pil2FriCode(tuple(cap.steps))
        ok &= _match(
            f"full prove: {len(proof.fri.roots)} FRI layer roots",
            np.stack([u64(root) for root in proof.fri.roots]),
            np.stack(
                [
                    u64(_fri_layer_root(cap, code, cap.u64(f"fri_layer{k}")))
                    for k in range(len(proof.fri.roots))
                ]
            ),
        )
        ok &= _match(
            "full prove: FRI final polynomial",
            limbs(proof.fri.final_pol),
            cap.u64(f"fri_layer{len(cap.steps) - 1}"),
        )
        # The dump carries no nonce/positions; both re-derive from the dumped
        # grinding seed (the last squeezed beta), which the equality then gates.
        seed = fnp.array(cap.u64(f"fri_beta{len(cap.steps) - 1}").astype(F))
        want_positions = query_positions_for(
            seed,
            transcript.width,
            proof.nonce,
            n_queries=ss["nQueries"],
            n_bits_ext=cap.nbe,
        )
        ok &= _check(
            grind_is_valid(seed, proof.nonce, ss["powBits"]),
            "full prove: grinding nonce",
        )
        return ok & _match(
            "full prove: query positions", proof.positions, want_positions
        )

    checks = {
        "commit": check_commit,
        "stage2": check_stage2,
        "quotient": check_quotient,
        "opening": check_opening,
    }
    ok = True
    with frx.default_device(frx.devices("cpu")[0]):
        for name, result in prover.prove_stages(
            cap.pil2_claim(), InnerWitness(cap.trace), transcript
        ):
            ok &= checks[name](result)
            if not ok:
                print(
                    "MISMATCH  full prove: fail-fast — skipping the " "remaining stages"
                )
                break
    return ok


# Proof order: a failure localizes to the first stage whose inputs went bad.
GATES = (
    verify_transcript_schedule,
    verify_stage1_commit,
    verify_stage2_witness,
    verify_stage2_commit,
    verify_quotient,
    verify_evals,
    verify_deep,
    verify_fri_chain,
    verify_full_prove,
)
