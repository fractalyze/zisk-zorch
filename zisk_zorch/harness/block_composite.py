"""The block composite — every instance of a block proved in one process
from a self-derived seed.

pil2's block schedule, over the harness roles: commit every instance's
trace and hash its contribution (`contributions.instance_contribution`),
derive the global challenge from the aggregate, prove every instance's
composed schedule seeded with it, then bind the block with the
cross-instance global-constraints check over OUR airgroup values.

Two deliberate v1 shapes:

- Instances re-commit inside their prove (phase 3) rather than carrying
  phase 1's extended sections — 38 resident LDEs exceed device memory at
  block scale, and a commit is ~10-40x cheaper than its prove. Overlapping
  the two phases is a scheduling lever, not a correctness one.
- The witness arrives through the `WitnessSource` seam; the capture bundle
  is the current implementation, and #115's rw intake swaps in behind the
  same protocol. The seed and the block binding are already self-derived.
"""

from __future__ import annotations

import gc
import pathlib
from dataclasses import dataclass, replace

import numpy as np

from zisk_zorch.harness.contributions import (
    aggregate_contributions,
    global_challenge,
    instance_contribution,
    stage1_values,
)
from zisk_zorch.harness.global_constraints import (
    aggregate_airgroupvalues,
    check_global_constraints,
)
from zisk_zorch.harness.pil2 import (
    limbs,
    release_device_sections,
    stage_challenge_ids,
    transcript_width,
)
from zisk_zorch.harness.pil2_prover import Pil2InnerProver
from zisk_zorch.harness.proof_serializer import serialize_proof
from zisk_zorch.harness.staging import TraceStager
from zisk_zorch.harness.witness_source import WitnessSource
from zisk_zorch.shape_cache import release_shape_caches
from zisk_zorch.transcript.transcript import Transcript
from zisk_zorch.types import InnerWitness


@dataclass(frozen=True)
class InstanceResult:
    """One instance's prove outcome: the airgroup-value words its LogUp
    stage settled, and the wire proof's path when emission was on. Stage
    results are deliberately not retained — observe them live via
    `prove_block`'s `on_stage` (38 instances' stage buffers exceed both
    host and device memory)."""

    instance: str
    airgroupvalues: np.ndarray
    wire_proof: pathlib.Path | None = None


def _airgroupvalue_words(logup_claim) -> np.ndarray:
    (value,) = logup_claim.airgroupvalues.values()
    return limbs(value).astype(np.uint64)


def _dumped_airgroupvalues(claim) -> np.ndarray:
    """A claim's airgroup values in their dumped packing: contiguous limb
    triples in id order (the byte-gates compare exactly this form)."""
    agv = claim.airgroupvalues
    if not agv:
        return np.zeros(0, dtype=np.uint64)
    return np.concatenate([limbs(agv[i]) for i in sorted(agv)]).astype(np.uint64)


def emit_wire_proof(
    src: WitnessSource, out_dir: pathlib.Path, claim, opening
) -> pathlib.Path:
    """Serialize one instance's own-prove wire proof — `claim` is the
    quotient-bound claim (the three roots and the values), `opening` the
    `Pil2OpeningProof` carrying `WireOpenings`."""
    proof = serialize_proof(
        src.si,
        airgroup_values=_dumped_airgroupvalues(claim),
        air_values=np.asarray(claim.airvalues, dtype=np.uint64),
        roots=[
            np.asarray(r).astype(np.uint64)
            for r in (claim.trace_root, claim.root2, claim.quotient_root)
        ],
        evals=limbs(opening.evals),
        const_opening=opening.wire.const,
        custom_openings=opening.wire.customs,
        stage_openings=opening.wire.stages,
        fri_roots=[np.asarray(r).astype(np.uint64) for r in opening.fri.roots],
        fri_openings=opening.wire.fri,
        final_pol=limbs(opening.fri.final_pol),
        nonce=opening.nonce,
    )
    path = out_dir / f"{src.instance}_proof.npy"
    np.save(path, proof)
    return path


def prove_block(
    sources: list[tuple[str, WitnessSource, np.ndarray]],
    *,
    global_info: dict,
    global_constraints: list[dict],
    on_stage=None,
    emit_dir: pathlib.Path | None = None,
    provers: dict[str, Pil2InnerProver] | None = None,
) -> tuple[np.ndarray, list[InstanceResult], list[np.ndarray]]:
    """Prove `sources` (``(family, witness_source, verkey)`` triples, any
    order) as one block. Returns the derived global challenge, the
    per-instance results, and the global-constraint values (all-zero for a
    sound block).

    `on_stage(instance, stage_name, result)` observes each stage as it
    lands — the byte-gates ride there during the capture-fed transition.

    `provers` is an optional caller-owned prover cache. Without it, each
    family's prover is dropped after its last prove (20 keys' compiled
    executables are device-module memory that cannot all stay loaded);
    passing a dict keeps every prover warm across calls — the caller owns
    that residency, e.g. a warm-timed benchmark pass."""
    if not sources:
        # Phase 2 cannot derive a seed from zero contributions
        # (`aggregate_contributions([])` has no lattice shape to fold), so
        # refuse here instead of crashing a phase later.
        raise ValueError("prove_block needs at least one witness source")
    lattice = global_info["latticeSize"]
    family_hash = global_info["hash"]
    release_provers = provers is None
    if provers is None:
        provers = {}

    def prover_for(fam: str, src: WitnessSource) -> Pil2InnerProver:
        if fam not in provers:
            provers[fam] = Pil2InnerProver(src.pil2_key, emit_wire=emit_dir is not None)
        return provers[fam]

    if emit_dir is not None:
        emit_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: stage-1 commits -> contributions. The commitment is dropped
    # immediately; only the root feeds the hash. The source's caches go
    # with it — the 38 traces alone are larger than host RAM, so nothing
    # per-instance may survive its own iteration (the lookahead's staged
    # trace belongs to the next one). The claim (scalars only) is taken
    # before release so its sections load once.
    stager = TraceStager()
    contribs = []
    publics = proofvalues_words = None
    staged = stager.stage(sources[0][1].trace_words())
    for i, (fam, src, vk) in enumerate(sources):
        prover = prover_for(fam, src)
        claim = src.pil2_claim()
        commitment = prover.opening.commit(InnerWitness(staged))
        # The commit consumed this source's staged upload, so its host
        # sections go now — before the lookahead materializes the next
        # trace, or three trace-sized host buffers would coexist. Safe
        # even while this source's upload is still in flight:
        # `trace_words`'s contract keeps the buffer alive until the
        # runtime's copy lands. `claim` and `si` survive release.
        src.release()
        # The commit is dispatched, the root not yet read: stage the next
        # instance now so its host read and upload ride behind this
        # instance's kernels — the in-process form of #144's double-buffer
        # lever, placed at the one point phase 1 would otherwise idle.
        staged = (
            stager.stage(sources[i + 1][1].trace_words())
            if i + 1 < len(sources)
            else None
        )
        root1 = np.asarray(commitment.root).astype(np.uint64)
        del commitment, prover
        # Same family-boundary release as phase 3: keeping every family's
        # prover (and its uploaded key sections) resident through phases
        # 1-2 is exactly the residency this module exists to bound.
        if release_provers and (i + 1 == len(sources) or sources[i + 1][0] != fam):
            dropped = provers.pop(fam, None)
            if dropped is not None:
                release_device_sections(dropped.key)
            release_shape_caches()
        gc.collect()
        av1 = (
            stage1_values(claim.airvalues, src.si["airValuesMap"])
            if src.si.get("airValuesMap")
            else np.zeros(0, dtype=np.uint64)
        )
        contribs.append(
            instance_contribution(
                vk, root1, av1, hash_family=family_hash, lattice_size=lattice
            )
        )
        if publics is None:
            publics = claim.publics
            proofvalues_words = claim.proofvalues

    # Phase 2: the seed.
    seed = global_challenge(
        publics,
        stage1_values(proofvalues_words, global_info["proofValuesMap"]),
        aggregate_contributions(contribs),
        hash_family=family_hash,
    )

    # Phase 3: every composed prove, seeded with OUR challenge. Stage
    # results are observed as they land (`on_stage`) and then dropped —
    # retaining 38 instances' stage buffers is exactly the residency the
    # recommit design exists to avoid. Phase 4 keeps only the airgroup
    # values and the first instance's stage-2 challenges.
    results = []
    stage2_challenges = None
    for i, (fam, src, _) in enumerate(sources):
        prover = prover_for(fam, src)
        claim = replace(src.pil2_claim(), global_challenge=seed)
        transcript = Transcript(
            transcript_width(src.si["starkStruct"]), src.hash_family
        )
        logup_claim = quotient_claim = opening_proof = None
        # No cross-instance pipelining here, deliberately: `prove_stages`
        # is a lazy generator whose consumption IS the dispatch, so a
        # host read inserted mid-loop stalls the device instead of hiding
        # behind it. The staged (pinned) upload still replaces the
        # pageable one; the host-read cost itself leaves with the rw
        # in-memory source (#115).
        for name, result in prover.prove_stages(
            claim, InnerWitness(stager.stage(src.trace_words())), transcript
        ):
            if name == "logup_witness":
                logup_claim = result.reduced_claim
            elif name == "quotient":
                quotient_claim = result.reduced_claim
            elif name == "opening":
                opening_proof = result.reduction_proof
            if on_stage is not None:
                on_stage(src.instance, name, result)
        assert logup_claim is not None, f"{src.instance}: no logup stage"
        if stage2_challenges is None:
            ids2 = stage_challenge_ids(src.si["challengesMap"], 2)
            stage2_challenges = {
                g: logup_claim.challenges[i] for g, i in enumerate(ids2)
            }
        wire_path = (
            emit_wire_proof(src, emit_dir, quotient_claim, opening_proof)
            if emit_dir is not None
            else None
        )
        results.append(
            InstanceResult(
                instance=src.instance,
                airgroupvalues=_airgroupvalue_words(logup_claim),
                wire_proof=wire_path,
            )
        )
        del logup_claim, quotient_claim, opening_proof, prover
        src.release()
        # A family's compiled executables are device-module memory; 20
        # families' worth cannot stay loaded at once. The manifest arrives
        # family-grouped, so the prover is dropped after its family's last
        # prove (an ungrouped manifest stays correct — `prover_for` just
        # recompiles). A caller-owned `provers` cache opts out.
        if release_provers and (i + 1 == len(sources) or sources[i + 1][0] != fam):
            dropped = provers.pop(fam, None)
            if dropped is not None:
                # The key outlives the prover (its source's `pil2_key`
                # survives release), so the uploaded sections must be
                # dropped explicitly.
                release_device_sections(dropped.key)
            # Same reason, one level up: the coset and LEv constants are
            # interned by SHAPE, not by key, so no key-scoped release reaches
            # them. Dropping them per family costs a millisecond-scale rebuild
            # on the next family that shares the shape.
            release_shape_caches()
        gc.collect()

    # Phase 4: the block binding over OUR airgroup values.
    aggregated = aggregate_airgroupvalues(
        [r.airgroupvalues for r in results], global_info["aggTypes"][0]
    )
    constraint_values = check_global_constraints(
        global_constraints,
        publics=publics,
        proofvalues=proofvalues_words,
        proof_values_map=global_info["proofValuesMap"],
        challenges=stage2_challenges,
        airgroupvalues=aggregated,
    )
    return seed, results, constraint_values
