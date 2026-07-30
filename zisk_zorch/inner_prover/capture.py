"""Loader for a pil2-proofman ``PIL2_DUMP_DIR`` capture bundle.

A bundle is one real ``genProof``'s per-stage sections (``tools/pil2-dump``
regenerates it) plus the AIR's proving-key artifacts, in one directory.
File naming, ``.npy`` decoding, gl64 limb layout, and pil2's value-packing
conventions all live here, so the gates read protocol objects, never files.
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

from zisk_zorch.quotient.cexp_ref import _run_block
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
    def std_plan(self) -> list | None:
        """The stage-2 STD columns this AIR commits, in dependency order
        im → gsum → imPol (each read by the next), as ``(name, cm_id,
        compute)`` triples where ``compute(env)`` expects every earlier
        column already bound in ``env["cm"]``. ``None`` when the AIR
        commits no STD columns — each is optional per AIR (SpecifiedRanges
        has a gsum and an ImPol but no im column).

        The running sum scans the ``gsum_col`` hint's
        ``numerator_air/denominator_air`` — the im columns plus any direct
        bus terms that never materialize an im column (std_sum.pil's
        recurrence; #109) — not the im column alone. A hint field is an
        expression, a committed-column ref (FibonacciSquare's
        ``numerator_air`` is the im column itself), or a literal; dividing
        by the literal-1 denominator would add a dead cubic reciprocal to
        the fused graph and is skipped."""
        ei = self.expressionsinfo
        exps = {e["expId"]: e["code"] for e in ei["expressionsCode"]}
        hints = {h["name"]: h for h in ei["hintsInfo"]}
        im_pol_cid = next(
            (
                i
                for i, p in enumerate(self.cmp_map)
                if p["stage"] == 2 and p.get("imPol")
            ),
            None,
        )

        def field(hint, name):
            f = next(f for f in hints[hint]["fields"] if f["name"] == name)
            return f["values"][0]

        def operand(v, env):
            if v["op"] == "tmp":
                return _run_block(exps[v["id"]], env, 1)
            if v["op"] == "cm":
                return env["cm"][v["id"]]
            if v["op"] == "number":
                return self._cubic_scalar([int(v["value"]), 0, 0])
            raise NotImplementedError(f"hint field operand {v['op']}")

        plan = []
        if "im_col" in hints:
            num_v, den_v = field("im_col", "numerator"), field("im_col", "denominator")
            plan.append(
                (
                    "im_single",
                    field("im_col", "reference")["id"],
                    lambda env: operand(num_v, env) / operand(den_v, env),
                )
            )
        if "gsum_col" in hints:
            gnum_v = field("gsum_col", "numerator_air")
            gden_v = field("gsum_col", "denominator_air")
            trivial_den = gden_v["op"] == "number" and int(gden_v["value"]) == 1

            def gsum(env):
                term = operand(gnum_v, env)
                if not trivial_den:
                    term = term / operand(gden_v, env)
                return fnp.cumsum(term)

            plan.append(("gsum", field("gsum_col", "reference")["id"], gsum))
        if im_pol_cid is not None:
            code = exps[self.cmp_map[im_pol_cid]["expId"]]
            plan.append(("ImPol", im_pol_cid, lambda env: _run_block(code, env, 1)))
        return plan or None

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
