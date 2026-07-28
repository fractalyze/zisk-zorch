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

from zisk_zorch.commit.trace_commit import commit_trace
from zisk_zorch.fri.fold import fold


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

    for stage, why in [
        ("stage-2 commit", "witness-STD hint machinery not wired"),
        ("quotient", "needs this AIR's cExp extraction (in-repo gate: cexp_ref)"),
        ("evals/DEEP", "multi-opening (openingPoints span) not wired"),
    ]:
        print(f"SKIP     {stage}: {why}")

    print("inner-proof byte-match: " + ("ALL COVERED LINKS OK" if ok_all else "FAILED"))
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())
