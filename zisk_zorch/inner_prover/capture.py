"""Loader for a pil2-proofman ``PIL2_DUMP_DIR`` capture bundle.

A bundle is one real ``genProof``'s per-stage sections (``tools/pil2-dump``
regenerates it) plus the AIR's proving-key artifacts, in one directory. This
module owns everything about that directory — file naming, ``.npy``
decoding, gl64 limb layout, and pil2's value-packing conventions — so the
gates in ``verify_inner_proof`` read protocol objects, never files. The
inner-prover analog of sp1-zorch's ``shard_prover/fixture_loader.py``.
"""

from __future__ import annotations

import json
import pathlib
from functools import cached_property

import frx.numpy as fnp
import numpy as np
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zk_dtypes import pfinfo
from zorch.utils.field import join_coeffs, split_coeffs

from zisk_zorch.quotient.zerofier import inv_zerofier

P = int(pfinfo(F).modulus)


def cubic(words: np.ndarray):
    """Contiguous gl64 limb triples -> a 1-D cubic device array."""
    return join_coeffs(fnp.array(words.astype(F).reshape(-1, 3)), F3)


def limbs(x) -> np.ndarray:
    """A field array flattened to its dumped form: contiguous u64 limbs."""
    return np.asarray(split_coeffs(x)).reshape(-1)


class Capture:
    """One dumped AIR instance: starkinfo/expressionsinfo shape metadata plus
    lazy, cached readers for the per-stage ``<instance>_<name>.npy`` sections."""

    def __init__(
        self, dump: pathlib.Path, instance: str, starkinfo: pathlib.Path
    ) -> None:
        self.dump, self.instance, self.starkinfo_path = dump, instance, starkinfo
        self.si = json.loads(starkinfo.read_text())
        assert self.si["nStages"] == 2, "buffer keys assume a two-stage AIR"
        assert len(self.si["boundaries"]) == 1, "only the everyRow zerofier is wired"
        ss = self.si["starkStruct"]
        self.nb, self.nbe = ss["nBits"], ss["nBitsExt"]
        self.n, self.ne = 1 << self.nb, 1 << self.nbe
        self.stride = 1 << (self.nbe - self.nb)
        self.arity = ss["merkleTreeArity"]
        self.steps = [s["nBits"] for s in ss["steps"]]
        self.cmp_map = self.si["cmPolsMap"]
        self.ev_map = self.si["evMap"]
        self.openings = self.si["openingPoints"]
        self.n_cols = self.si["mapSectionsN"]["cm1"]
        self.cm2_cols = self.si["mapSectionsN"].get("cm2", 0)
        self.n_customs = len(self.si.get("customCommits", []))

    # -- dumped sections ----------------------------------------------------

    def path(self, name: str) -> pathlib.Path:
        return self.dump / f"{self.instance}_{name}.npy"

    def u64(self, name: str) -> np.ndarray:
        arr = np.load(self.path(name))
        assert arr.dtype == np.uint64, f"{name}: expected u64 dump, got {arr.dtype}"
        return arr

    @cached_property
    def expressionsinfo(self) -> dict:
        return json.loads(
            (
                self.starkinfo_path.parent
                / self.starkinfo_path.name.replace("starkinfo", "expressionsinfo")
            ).read_text()
        )

    @cached_property
    def cexp_code(self) -> list:
        """The composite constraint expression's SSA block."""
        return next(
            e
            for e in self.expressionsinfo["expressionsCode"]
            if e["expId"] == self.si["cExpId"]
        )["code"]

    @cached_property
    def im_col_exps(self) -> tuple | None:
        """The stage-2 LogUp chain's three SSA blocks — the ``im_col`` hint's
        numerator and denominator plus the committed ``ImPol`` — or ``None``
        for an AIR whose stage-2 carries no LogUp intermediate (a range
        check, say): no hint, no imPol column, nothing to chain."""
        ei = self.expressionsinfo
        exps = {e["expId"]: e["code"] for e in ei["expressionsCode"]}
        hints = {h["name"]: h for h in ei["hintsInfo"]}
        im_pol = next(
            (p["expId"] for p in self.cmp_map if p["stage"] == 2 and p.get("imPol")),
            None,
        )
        if "im_col" not in hints or im_pol is None:
            return None

        def hint_exp(field):
            v = next(f for f in hints["im_col"]["fields"] if f["name"] == field)
            return exps[v["values"][0]["id"]]

        return hint_exp("numerator"), hint_exp("denominator"), exps[im_pol]

    @cached_property
    def const_base(self) -> np.ndarray:
        """The proving key's base-domain constant columns (pil2's own binary
        layout — the one non-npy read in the bundle)."""
        return np.fromfile(
            self.starkinfo_path.parent
            / self.starkinfo_path.name.replace(".starkinfo.json", ".const"),
            dtype=np.uint64,
        ).reshape(self.n, self.si["nConstants"])

    @cached_property
    def trace(self):
        """The settled stage-1 witness: hint-computed columns fill
        asynchronously during STEP_1, so the pre-commit dump can be
        incomplete — prefer the post-commit section."""
        name = "trace_post" if self.path("trace_post").exists() else "trace"
        words = self.u64(name)
        assert words.size == self.n * self.n_cols, "trace size mismatch"
        return fnp.array(words.astype(F).reshape(self.n, self.n_cols))

    @cached_property
    def bufs(self) -> dict:
        """Committed extended-domain sections, keyed like ``cmPolsMap`` stages
        (quotient rides as stage ``nStages + 1``)."""
        si, ne = self.si, self.ne
        bufs = {
            ("cm", 1): self.u64("cm1_ext").astype(F).reshape(ne, self.n_cols),
            ("const", 0): self.u64("const_ext").astype(F).reshape(ne, si["nConstants"]),
            ("cm", 3): self.u64("quotient_cm")
            .astype(F)
            .reshape(ne, si["mapSectionsN"]["cm" + str(si["nStages"] + 1)]),
        }
        if self.cm2_cols:
            bufs[("cm", 2)] = self.u64("cm2_ext").astype(F).reshape(ne, self.cm2_cols)
        for ci in range(self.n_customs):
            width = si["mapSectionsN"][si["customCommits"][ci]["name"] + "0"]
            bufs[("custom", ci)] = (
                self.u64(f"custom{ci}_ext").astype(F).reshape(ne, width)
            )
        return bufs

    # -- protocol objects the gates consume ----------------------------------

    def committed_column(self, entry: dict):
        """An evMap entry's committed column over the extended domain, cubic
        entries joined from their three contiguous gl64 lanes."""
        if entry["type"] == "cm":
            pm = self.cmp_map[entry["id"]]
            buf = self.bufs[("cm", pm["stage"])]
            if pm["dim"] == 1:
                return fnp.array(buf[:, pm["stagePos"]])
            lanes = buf[:, pm["stagePos"] : pm["stagePos"] + 3]
            return join_coeffs(fnp.array(np.ascontiguousarray(lanes)), F3)
        if entry["type"] == "const":
            return fnp.array(self.bufs[("const", 0)][:, entry["id"]])
        return fnp.array(self.bufs[("custom", entry["commitId"])][:, entry["id"]])

    @cached_property
    def opened_columns(self) -> list:
        return [self.committed_column(e) for e in self.ev_map]

    @cached_property
    def challenges(self):
        """The prove's transcript challenges, one cubic scalar per index."""
        return cubic(self.u64("challenges"))

    def challenge(self, name: str):
        idx = next(
            i for i, c in enumerate(self.si["challengesMap"]) if c.get("name") == name
        )
        return self.challenges[idx].reshape(())

    # -- the SSA interpreter's operand environment ----------------------------

    def _cubic_scalar(self, words):
        return cubic(np.asarray(words, dtype=np.uint64))[0]

    def _values_env(self, name: str, vmap: list) -> dict:
        """pil2 packs stage-1 values as one word, stage>=2 as three."""
        words = self.u64(name)
        out, off = {}, 0
        for i, v in enumerate(vmap):
            if v["stage"] == 1:
                out[i] = self._cubic_scalar([words[off], 0, 0])
                off += 1
            else:
                out[i] = self._cubic_scalar(words[off : off + 3])
                off += 3
        return out

    @cached_property
    def cexp_env(self) -> dict:
        """Every operand class the composite cExp reads, over the extended
        domain: committed/constant/custom columns plus the scalar values."""
        si = self.si
        publics = self.u64("publics")
        return {
            "cm": {
                i: (
                    self.committed_column({"type": "cm", "id": i})
                    if pm["dim"] == 3
                    else fnp.array(self.bufs[("cm", pm["stage"])][:, pm["stagePos"]])
                )
                for i, pm in enumerate(self.cmp_map)
            },
            "const": {
                i: fnp.array(self.bufs[("const", 0)][:, i])
                for i in range(si["nConstants"])
            },
            "custom": {
                (ci, j): fnp.array(self.bufs[("custom", ci)][:, j])
                for ci in range(self.n_customs)
                for j in range(self.bufs[("custom", ci)].shape[1])
            },
            "challenges": dict(enumerate(self.challenges)),
            "publics": {
                i: self._cubic_scalar([publics[i], 0, 0]) for i in range(len(publics))
            },
            "airvalues": self._values_env("airvalues", si["airValuesMap"]),
            "airgroupvalues": self._values_env(
                "airgroupvalues", si["airgroupValuesMap"]
            ),
            "proofvalues": self._values_env("proofvalues", si["proofValuesMap"]),
            "zi": {0: inv_zerofier(self.nb, self.nbe - self.nb)},
        }

    @cached_property
    def base_env(self) -> dict:
        """``cexp_env`` re-rooted on the base domain: the settled stage-1
        trace and the key's base constants replace the extended sections —
        what the stage-2 hint expressions evaluate over."""
        trace_np = np.asarray(self.trace).astype(np.uint64)
        return dict(
            self.cexp_env,
            cm={i: fnp.array(trace_np[:, i].astype(F)) for i in range(self.n_cols)},
            const={
                i: fnp.array(self.const_base[:, i].astype(F))
                for i in range(self.si["nConstants"])
            },
            zi={},
        )
