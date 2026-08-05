"""pil2 genProof's Fiat-Shamir schedule, replayable against a capture.

The schedule is the spine every stage hangs off: seed with the
contributions-phase global challenge, absorb each commitment root as it is
produced, squeeze each ``challengesMap`` entry at its stage boundary, and
close with the FRI layer roots and the grinding seed. Squeezing hashes
everything absorbed before it, so challenge equality against a capture
transitively gates the absorbs the dump does not carry (FRI layer roots,
hashed evals).

Schedule source: ``gen_proof.hpp`` on the pinned pil2-proofman fork
(https://github.com/fractalyze/pil2-proofman/blob/11999a69/pil2-stark/src/starkpil/gen_proof.hpp).
"""

from __future__ import annotations

from collections.abc import Iterator

import frx.numpy as fnp
import numpy as np
from zk_dtypes import goldilocks as F

from zisk_zorch.commit.trace_commit import merkle_tree
from zisk_zorch.fri.seam import Pil2FriCode
from zisk_zorch.harness.capture import cubic
from zisk_zorch.harness.pil2 import (
    absorb_stage2_airvalues,
    absorb_words,
    stage_challenge_ids,
    transcript_width,
)
from zisk_zorch.transcript.transcript import Transcript, absorb_section


def _fri_layer_root(cap, code, layer_words: np.ndarray):
    """The layer's tree root, regrouped exactly as the FRI prover commits it."""
    from zorch.pcs.fold import to_base_field

    leaves = to_base_field(code.group_leaves(cubic(layer_words)))
    root, _ = merkle_tree(cap.arity, cap.hash_family).commit(leaves)
    return root


def replay_challenges(cap) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Drive a fresh transcript through the whole prove schedule using the
    capture's sections, yielding ``(label, got, want)`` for every squeezed
    challenge. All-equal means every absorb matched the native prove too.

    The final yield is the grinding seed — the challenge the query phase
    feeds to ``sample_query_positions``."""
    ss = cap.si["starkStruct"]
    width = transcript_width(ss)
    hash_commits = ss.get("hashCommits", False)
    want = cap.u64("challenges").reshape(-1, 3)
    n_stages = cap.si["nStages"]

    def squeeze(t: Transcript, stage: int) -> Iterator:
        for i in stage_challenge_ids(cap.si["challengesMap"], stage):
            got = np.asarray(t.get_field()).astype(np.uint64)
            name = cap.si["challengesMap"][i].get("name", str(i))
            yield f"challenge {name} (stage {stage})", got, want[i]

    t = Transcript(width, cap.hash_family)
    # No root1 absorb: the contributions phase derives the global challenge
    # from every instance's stage-1 root, so the transcript arrives already
    # bound to it (root1 rides inside the seed, not the schedule).
    absorb_words(t, cap.u64("global_challenge"))
    yield from squeeze(t, 2)

    absorb_words(t, cap.u64("root2"))
    absorb_stage2_airvalues(t, cap.u64("airvalues"), cap.si["airValuesMap"])
    yield from squeeze(t, n_stages + 1)

    absorb_words(t, cap.u64("rootQ"))
    yield from squeeze(t, n_stages + 2)

    absorb_section(t, fnp.array(cap.u64("evals").astype(F)), hashed=hash_commits)
    yield from squeeze(t, n_stages + 3)

    code = Pil2FriCode(tuple(cap.steps))
    for k in range(len(cap.steps)):
        if k < len(cap.steps) - 1:
            t.put(_fri_layer_root(cap, code, cap.u64(f"fri_layer{k}")))
        else:
            last = fnp.array(cap.u64(f"fri_layer{k}").astype(F))
            absorb_section(t, last, hashed=hash_commits)
        got = np.asarray(t.get_field()).astype(np.uint64)
        label = "grinding seed" if k == len(cap.steps) - 1 else f"fri beta {k}"
        yield label, got, cap.u64(f"fri_beta{k}")
