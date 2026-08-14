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
- The witness and scalar sections still arrive from captures; the seed and
  the block binding no longer do. #115's intake swaps the witness source.
"""

from __future__ import annotations

import gc
import pathlib
from dataclasses import dataclass, replace

import frx.numpy as fnp
import numpy as np
from zorch.utils.field import split_coeffs

from zisk_zorch.harness.capture import Capture, limbs
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
    release_device_sections,
    stage_challenge_ids,
    transcript_width,
)
from zisk_zorch.harness.pil2_prover import Pil2InnerProver
from zisk_zorch.harness.proof_serializer import serialize_proof
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
    return np.asarray(split_coeffs(value)).reshape(-1).astype(np.uint64)


def _dumped_airgroupvalues(claim) -> np.ndarray:
    """A claim's airgroup values in their dumped packing: contiguous limb
    triples in id order (the byte-gates compare exactly this form)."""
    agv = claim.airgroupvalues
    if not agv:
        return np.zeros(0, dtype=np.uint64)
    return np.concatenate([limbs(agv[i]) for i in sorted(agv)]).astype(np.uint64)


def emit_wire_proof(
    cap: Capture, out_dir: pathlib.Path, claim, opening
) -> pathlib.Path:
    """Serialize one instance's own-prove wire proof — `claim` is the
    quotient-bound claim (the three roots and the values), `opening` the
    `Pil2OpeningProof` carrying `WireOpenings`."""
    proof = serialize_proof(
        cap.si,
        airgroup_values=_dumped_airgroupvalues(claim),
        air_values=np.asarray(claim.pil2.airvalues, dtype=np.uint64),
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
    path = out_dir / f"{cap.instance}_proof.npy"
    np.save(path, proof)
    return path


def prove_block(
    captures: list[tuple[str, Capture, np.ndarray]],
    *,
    global_info: dict,
    global_constraints: list[dict],
    on_stage=None,
    emit_dir: pathlib.Path | None = None,
    provers: dict[str, Pil2InnerProver] | None = None,
) -> tuple[np.ndarray, list[InstanceResult], list[np.ndarray]]:
    """Prove `captures` (``(family, capture, verkey)`` triples, any order)
    as one block. Returns the derived global challenge, the per-instance
    results, and the global-constraint values (all-zero for a sound block).

    `on_stage(instance, stage_name, result)` observes each stage as it
    lands — the byte-gates ride there during the capture-fed transition.

    `provers` is an optional caller-owned prover cache. Without it, each
    family's prover is dropped after its last prove (20 keys' compiled
    executables are device-module memory that cannot all stay loaded);
    passing a dict keeps every prover warm across calls — the caller owns
    that residency, e.g. a warm-timed benchmark pass."""
    lattice = global_info["latticeSize"]
    family_hash = global_info["hash"]
    release_provers = provers is None
    if provers is None:
        provers = {}

    def prover_for(fam: str, cap: Capture) -> Pil2InnerProver:
        if fam not in provers:
            provers[fam] = Pil2InnerProver(cap.pil2_key, emit_wire=emit_dir is not None)
        return provers[fam]

    if emit_dir is not None:
        emit_dir.mkdir(parents=True, exist_ok=True)

    # Phase 1: stage-1 commits -> contributions. The commitment is dropped
    # immediately; only the root feeds the hash. The capture's caches go
    # with it — the 38 traces alone are larger than host RAM, so nothing
    # per-instance may survive its iteration.
    contribs = []
    publics = proofvalues = None
    for i, (fam, cap, vk) in enumerate(captures):
        prover = prover_for(fam, cap)
        trace_dev = fnp.asarray(cap.trace)
        commitment = prover.opening.commit(InnerWitness(trace_dev))
        root1 = np.asarray(commitment.root).astype(np.uint64)
        del commitment, trace_dev, prover
        cap.release()
        # Same family-boundary release as phase 3: keeping every family's
        # prover (and its uploaded key sections) resident through phases
        # 1-2 is exactly the residency this module exists to bound.
        if release_provers and (i + 1 == len(captures) or captures[i + 1][0] != fam):
            dropped = provers.pop(fam, None)
            if dropped is not None:
                release_device_sections(dropped.key)
        gc.collect()
        av1 = (
            stage1_values(cap.u64("airvalues"), cap.si["airValuesMap"])
            if cap.si.get("airValuesMap")
            else np.zeros(0, dtype=np.uint64)
        )
        contribs.append(
            instance_contribution(
                vk, root1, av1, hash_family=family_hash, lattice_size=lattice
            )
        )
        if publics is None:
            publics = cap.u64("publics")
            proofvalues = stage1_values(
                cap.u64("proofvalues"), global_info["proofValuesMap"]
            )

    # Phase 2: the seed.
    seed = global_challenge(
        publics,
        proofvalues,
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
    for i, (fam, cap, _) in enumerate(captures):
        prover = prover_for(fam, cap)
        claim = replace(cap.pil2_claim(), global_challenge=seed)
        transcript = Transcript(
            transcript_width(cap.si["starkStruct"]), cap.hash_family
        )
        logup_claim = quotient_claim = opening_proof = None
        for name, result in prover.prove_stages(
            claim, InnerWitness(fnp.asarray(cap.trace)), transcript
        ):
            if name == "logup_witness":
                logup_claim = result.reduced_claim
            elif name == "quotient":
                quotient_claim = result.reduced_claim
            elif name == "opening":
                opening_proof = result.reduction_proof
            if on_stage is not None:
                on_stage(cap.instance, name, result)
        assert logup_claim is not None, f"{cap.instance}: no logup stage"
        if stage2_challenges is None:
            ids2 = stage_challenge_ids(cap.si["challengesMap"], 2)
            stage2_challenges = {
                g: logup_claim.challenges[i] for g, i in enumerate(ids2)
            }
        wire_path = (
            emit_wire_proof(cap, emit_dir, quotient_claim, opening_proof)
            if emit_dir is not None
            else None
        )
        results.append(
            InstanceResult(
                instance=cap.instance,
                airgroupvalues=_airgroupvalue_words(logup_claim),
                wire_proof=wire_path,
            )
        )
        del logup_claim, quotient_claim, opening_proof, prover
        cap.release()
        # A family's compiled executables are device-module memory; 20
        # families' worth cannot stay loaded at once. The manifest arrives
        # family-grouped, so the prover is dropped after its family's last
        # prove (an ungrouped manifest stays correct — `prover_for` just
        # recompiles). A caller-owned `provers` cache opts out.
        if release_provers and (i + 1 == len(captures) or captures[i + 1][0] != fam):
            dropped = provers.pop(fam, None)
            if dropped is not None:
                # The key outlives the prover (its capture's `pil2_key`
                # survives release), so the uploaded sections must be
                # dropped explicitly.
                release_device_sections(dropped.key)
        gc.collect()

    # Phase 4: the block binding over OUR airgroup values.
    aggregated = aggregate_airgroupvalues(
        [r.airgroupvalues for r in results], global_info["aggTypes"][0]
    )
    first = captures[0][1]
    constraint_values = check_global_constraints(
        global_constraints,
        publics=publics,
        proofvalues=first.u64("proofvalues"),
        proof_values_map=global_info["proofValuesMap"],
        challenges=stage2_challenges,
        airgroupvalues=aggregated,
    )
    return seed, results, constraint_values
