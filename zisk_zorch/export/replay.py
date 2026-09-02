"""genProof's schedule over an exported artifact — the Rust bridge's
driver, written once in Python against the same artifacts.

The artifacts hold every device stage; what remains between them is the
host protocol: the Fiat-Shamir transcript, the challenge bookkeeping, the
query draw, and the wire layout. This module is that host half over
`runtime.Artifact`, in exactly the order `gen_proof.hpp` runs it
(non-recursive: seeded by the contributions-phase global challenge, root1
never absorbed), producing the flat `proof2pointer` buffer the bridge
writes into pil2's proof buffer.

Its job is to be checkable: `stages_test` proves one instance twice — the
Python prover (`Pil2InnerProver` + `emit_wire_proof`) and this replay —
and compares the flat proofs bit-for-bit. `bridge/src/driver.rs` mirrors
THIS file step for step, so the artifacts' correctness is settled here and
the Rust only has to reproduce the host half.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from zisk_zorch.export.runtime import Artifact, words
from zisk_zorch.fri.queries import query_positions_for
from zisk_zorch.harness.pil2 import absorb_stage2_airvalues, absorb_words, to_field
from zisk_zorch.harness.proof_serializer import (
    DIGEST,
    TreeOpening,
    _n_siblings,
    serialize_proof,
)
from zisk_zorch.transcript.transcript import Transcript, absorb_section


@dataclass(frozen=True)
class Instance:
    """One `StepsParams` worth of host inputs, as canonical u64 words."""

    trace: np.ndarray  # (n, cm1 width)
    publics: np.ndarray  # (nPublics,)
    airvalues: np.ndarray  # dumped packing: stage-1 one word, else three
    proofvalues: np.ndarray  # dumped packing
    global_challenge: np.ndarray  # (3,)


@dataclass(frozen=True)
class KeySections:
    """The proving key's base-domain fixed sections."""

    const_base: np.ndarray  # (n, nConstants)
    custom_base: dict[int, np.ndarray]  # commitId -> (n, width)


def _stage_ids(schedule: dict, stage: int) -> list[int]:
    return [c["id"] for c in schedule["challenges"] if c["stage"] == stage]


def _named_id(schedule: dict, name: str) -> int:
    return next(c["id"] for c in schedule["challenges"] if c["name"] == name)


def _squeeze(transcript: Transcript, challenges: np.ndarray, ids: list[int]) -> None:
    for i in ids:
        challenges[i] = words(transcript.get_field())


def _opening(flat, last_level, width: int, n_bits: int, arity: int, llv: int):
    flat = words(flat)
    paths = flat[:, width:].reshape(flat.shape[0], -1, (arity - 1) * DIGEST)
    return TreeOpening(
        rows=flat[:, :width],
        paths=paths[:, : _n_siblings(n_bits, arity, llv)],
        last_level=words(last_level),
    )


def _layers(out: dict, old_prefix: str, new_prefix: str) -> dict:
    """A commit program's digest layers, renamed to the opening program's
    input names (the setup programs name theirs after themselves)."""
    return {
        new_prefix + k[len(old_prefix) :]: v
        for k, v in out.items()
        if k.startswith(old_prefix)
    }


def prove(art: Artifact, key: KeySections, inst: Instance) -> tuple[np.ndarray, dict]:
    """Prove `inst` through the artifacts. Returns the flat wire proof and
    the host-side products the bridge hands back to pil2 (the airgroup
    values, the stage-2 air values)."""
    sc = art.manifest
    nbe, arity, llv = sc["n_bits_ext"], sc["arity"], sc["last_level_verification"]
    steps, hashed = sc["steps"], sc["hash_commits"]
    custom_ids = [c["id"] for c in sc["custom_commits"]]
    custom_widths = {c["id"]: c["width"] for c in sc["custom_commits"]}

    # The bridge runs the fixed sections once per key; the replay pays them
    # per prove so one call is the whole schedule.
    consts = art.run("constants")
    const = art.run("const_setup", const_base=key.const_base)
    customs = {
        ci: art.run(f"custom_setup_{ci}", **{f"custom_base_{ci}": key.custom_base[ci]})
        for ci in custom_ids
    }
    custom_base = {f"custom_base_{ci}": key.custom_base[ci] for ci in custom_ids}
    custom_ext = {f"custom_ext_{ci}": customs[ci][f"custom_ext_{ci}"] for ci in custom_ids}

    scalars = {
        "publics": inst.publics,
        "airvalues": inst.airvalues,
        "proofvalues": inst.proofvalues,
    }
    trace = art.place(inst.trace, art.inputs("commit1")[0])
    if sc["witness_calc"]:
        trace = art.run(
            "witness_calc", trace=trace, const_base=key.const_base, **custom_base, **scalars
        )["trace"]

    c1 = art.run("commit1", trace=trace)
    # Non-recursive schedule: the seed already binds root1 through the
    # contributions phase, so root1 itself is never absorbed.
    transcript = Transcript(hash_family=sc["hash_family"])
    absorb_words(transcript, inst.global_challenge)

    challenges = np.zeros((len(sc["challenges"]), 3), dtype=np.uint64)
    _squeeze(transcript, challenges, _stage_ids(sc, 2))
    lg = art.run(
        "logup",
        trace=trace,
        const_base=key.const_base,
        **custom_base,
        **scalars,
        challenges=challenges,
    )
    # pil2 overwrites the instance's air-value section with the im_airval
    # results; every later reader sees those.
    airvalues = words(lg["airvalues"])
    scalars["airvalues"] = airvalues
    c2 = art.run("commit2", cm2=lg["cm2"])
    absorb_words(transcript, words(c2["root2"]))
    absorb_stage2_airvalues(transcript, airvalues, sc["airvalues"])
    airgroupvalues = np.zeros((len(sc["airgroupvalues"]), 3), dtype=np.uint64)
    if "airgroupvalue" in lg:
        airgroupvalues[sc["airgroupvalue_index"]] = words(lg["airgroupvalue"])

    _squeeze(transcript, challenges, _stage_ids(sc, sc["n_stages"] + 1))
    sections = {
        "cm1_ext": c1["cm1_ext"],
        "cm2_ext": c2["cm2_ext"],
        "const_ext": const["const_ext"],
        **custom_ext,
    }
    q_args = dict(
        sections,
        **scalars,
        challenges=challenges,
        airgroupvalues=airgroupvalues,
        zi=consts["zi"],
    )
    chunks = sc["quotient_chunks"]
    if len(chunks) == 1:
        qs = [art.run("quotient", **q_args)["q"]]
    else:
        qs, start = [], 0
        for size in chunks:
            rows = np.arange(start, start + size, dtype=np.int32)
            qs.append(art.run(f"quotient_{size}", **q_args, rows=rows)["q"])
            start += size
    qc = art.run("quotient_commit", **{f"q_{k}": q for k, q in enumerate(qs)})
    absorb_words(transcript, words(qc["rootq"]))

    _squeeze(transcript, challenges, _stage_ids(sc, sc["n_stages"] + 2))
    xi = challenges[_named_id(sc, "std_xi")]
    sections["qsec"] = qc["qsec"]
    lev = art.run("lev", xi=xi)["lev"]
    evals = words(art.run("evals", **sections, lev=lev)["evals"])
    absorb_section(transcript, to_field(evals.reshape(-1)), hashed=hashed)
    _squeeze(transcript, challenges, _stage_ids(sc, sc["n_stages"] + 3))
    fri_pol = art.run(
        "deep",
        **sections,
        evals=evals,
        domain=consts["domain"],
        xi=xi,
        vf1=challenges[_named_id(sc, "std_vf1")],
        vf2=challenges[_named_id(sc, "std_vf2")],
    )["fri_pol"]

    codeword, fri_layers, fri_roots = fri_pol, [], []
    for i in range(len(steps) - 1):
        layer = art.run(f"fri_commit_{i}", codeword=codeword)
        fri_layers.append(layer)
        fri_roots.append(words(layer[f"fri_root_{i}"]))
        absorb_words(transcript, fri_roots[-1])
        beta = words(transcript.get_field())
        codeword = art.run(f"fri_fold_{i}", codeword=codeword, beta=beta)["codeword"]
    final_pol = words(art.run("fri_final", codeword=codeword)["final_pol"])
    absorb_section(transcript, to_field(final_pol.reshape(-1)), hashed=hashed)
    challenge = transcript.get_field()
    nonce = int(np.asarray(art.run("grind", challenge=words(challenge))["nonce"]))
    positions = query_positions_for(
        challenge,
        transcript.width,
        nonce,
        n_queries=sc["n_queries"],
        n_bits_ext=nbe,
        hash_family=sc["hash_family"],
    )

    def open_tree(name: str, width: int, n_bits: int, pos: np.ndarray, **buffers):
        out = art.run(f"open_{name}", **buffers, positions=pos)
        return _opening(
            out[f"{name}_openings"], out[f"{name}_last_level"], width, n_bits, arity, llv
        )

    const_opening = open_tree(
        "const",
        sc["n_constants"],
        nbe,
        positions,
        const_ext=const["const_ext"],
        **_layers(const, "const_setup_layers_", "const_layers_"),
    )
    custom_openings = [
        open_tree(
            f"custom_{ci}",
            custom_widths[ci],
            nbe,
            positions,
            **{f"custom_ext_{ci}": customs[ci][f"custom_ext_{ci}"]},
            **_layers(customs[ci], f"custom_setup_{ci}_layers_", f"custom_layers_{ci}_"),
        )
        for ci in custom_ids
    ]
    stage_openings = [
        open_tree(
            "cm1", sc["widths"]["cm1"], nbe, positions,
            cm1_ext=c1["cm1_ext"], **_layers(c1, "cm1_layers_", "cm1_layers_"),
        ),
        open_tree(
            "cm2", sc["widths"]["cm2"], nbe, positions,
            cm2_ext=c2["cm2_ext"], **_layers(c2, "cm2_layers_", "cm2_layers_"),
        ),
        open_tree(
            "qsec", sc["widths"]["qsec"], nbe, positions,
            qsec=qc["qsec"], **_layers(qc, "qsec_layers_", "qsec_layers_"),
        ),
    ]
    fri_openings = []
    for i, layer in enumerate(fri_layers):
        leaf_bits = steps[i + 1]
        n_x = 1 << (steps[i] - leaf_bits)
        fri_openings.append(
            open_tree(
                f"fri_{i}",
                n_x * 3,
                leaf_bits,
                positions % (1 << leaf_bits),
                **{f"fri_leaves_{i}": layer[f"fri_leaves_{i}"]},
                **_layers(layer, f"fri_layers_{i}_", f"fri_layers_{i}_"),
            )
        )

    proof = serialize_proof(
        {"airgroupValuesMap": sc["airgroupvalues"], "airValuesMap": sc["airvalues"]},
        airgroup_values=airgroupvalues.reshape(-1),
        air_values=airvalues,
        roots=[words(c1["root1"]), words(c2["root2"]), words(qc["rootq"])],
        evals=evals,
        const_opening=const_opening,
        custom_openings=custom_openings,
        stage_openings=stage_openings,
        fri_roots=fri_roots,
        fri_openings=fri_openings,
        final_pol=final_pol,
        nonce=nonce,
    )
    return proof, {"airgroupvalues": airgroupvalues, "airvalues": airvalues}
