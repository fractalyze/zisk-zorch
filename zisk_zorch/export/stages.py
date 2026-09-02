"""The pil2 schedule as per-AIR device programs — what `gen_proof` dispatches.

The C entry point (`bridge/`) replaces pil2-stark's `gen_proof` and runs
the schedule host-side with pil2's own `TranscriptGL`, so what has to cross
to the device is exactly the heavy, transcript-free work between two
challenges: a commit, the stage-2 witness, the quotient, the openings, one
FRI round. Each such step is one StableHLO program here, lowered from the
very function the pil2-mode roles (`harness/pil2_prover.py`) run, so the
binding and the Python prover cannot compute different things.

Every program takes positional arrays and returns a tuple of arrays; the
exporter records both sides by name in the manifest, and the bridge binds
buffers by name — adding an output cannot silently shift a decode. Values
that cross to the host (roots, evals, the final polynomial, air values)
leave as base limbs in pil2's packing; device-only intermediates (extended
sections, codewords) keep their field dtype.

Scalar sections enter as the same flat buffers pil2 hands `gen_proof`
(`StepsParams`): publics as one word each, air/proof values in dumped
packing (stage-1 one word, later stages three), challenges and airgroup
values as ``(n, 3)`` limb rows — `traced_scalar_env` packs them into the SSA
interpreter's environment inside the trace.

Schedule source: ``gen_proof.hpp`` on the pinned fork
(https://github.com/fractalyze/pil2-proofman/blob/11999a69/pil2-stark/src/starkpil/gen_proof.hpp).
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array, lax
from zk_dtypes import goldilocks as F
from zk_dtypes import goldilocksx3 as F3
from zorch.pcs.fold import to_base_field
from zorch.poly.univariate import powers
from zorch.utils.field import join_coeffs, split_coeffs

from zisk_zorch.commit.openings import batched_group_proof
from zisk_zorch.commit.trace_commit import extend, merkle_tree
from zisk_zorch.evals.lev import build_lev_constants, compute_lev
from zisk_zorch.fri.queries import _grind_search_jit
from zisk_zorch.fri.seam import Pil2FriCode
from zisk_zorch.harness.pil2 import Pil2Key, hint_value, value_offsets
from zisk_zorch.harness.pil2_prover import Pil2InnerProver
from zisk_zorch.quotient.zerofier import _SHIFT, _root
from zisk_zorch.transcript.transcript import DIGEST


@dataclass(frozen=True)
class Spec:
    """One buffer at a program boundary: its element type and shape."""

    name: str
    dtype: str
    dims: tuple[int, ...]

    @staticmethod
    def of(name: str, aval) -> Spec:
        return Spec(name, str(aval.dtype), tuple(int(d) for d in aval.shape))


@dataclass(frozen=True)
class Program:
    """One device program: a positional function and its named inputs; the
    outputs are named by `output_names` in the order `fn` returns them."""

    name: str
    fn: Callable[..., tuple]
    inputs: list[Spec]
    output_names: list[str]

    def avals(self) -> list[frx.ShapeDtypeStruct]:
        return [frx.ShapeDtypeStruct(s.dims, _DTYPES[s.dtype]) for s in self.inputs]

    def outputs(self) -> list[Spec]:
        """The output specs, from tracing `fn` on the input avals — the
        manifest reports the shapes the executable produces. Traced under
        x64, as the export lowers (see `raw_boundary`)."""
        with frx.enable_x64():
            outs = frx.eval_shape(self.fn, *self.avals())
        if len(outs) != len(self.output_names):
            raise AssertionError(
                f"{self.name}: {len(self.output_names)} output names for "
                f"{len(outs)} outputs"
            )
        return [Spec.of(n, a) for n, a in zip(self.output_names, outs)]


_DTYPES = {
    "goldilocks": F,
    "goldilocksx3": F3,
    "uint64": np.uint64,
    "uint32": np.uint32,
    "int32": np.int32,
}


def raw_boundary(program: Program) -> Program:
    """`program` with every field-typed input and output carried as plain
    ``uint64`` words — the form the artifacts are exported in.

    The PJRT C API's host-buffer entry only converts XLA's standard element
    types, so a field-typed parameter cannot be fed from C or Rust (the
    plugin aborts on the type tag). Inside the trace the words are
    reinterpreted with a bitcast, which XLA compiles to nothing: a base
    element is one word, a cubic element its three limbs on a trailing
    axis. The manifest therefore describes field data as ``uint64`` with
    that extra axis, and the bridge moves words, never field types.

    JAX keeps a ``uint64`` aval 64 bits wide only under x64, so the export
    traces and lowers inside `frx.enable_x64()` — scoped, not global: the
    prover code these programs are built from runs with x64 off, and its
    integer arithmetic (index promotion in particular) changes under it."""
    specs = program.inputs

    def unwrap(x, spec: Spec):
        if spec.dtype == "goldilocks":
            return lax.bitcast_convert_type(x, F)
        if spec.dtype == "goldilocksx3":
            return _cubic(lax.bitcast_convert_type(x, F))
        return x

    def wrap(y):
        if y.dtype == F3:
            return lax.bitcast_convert_type(_limbs(y), fnp.uint64)
        if y.dtype == F:
            return lax.bitcast_convert_type(y, fnp.uint64)
        return y

    def fn(*raw):
        outs = program.fn(*(unwrap(x, s) for x, s in zip(raw, specs)))
        return tuple(wrap(y) for y in outs)

    def raw_spec(s: Spec) -> Spec:
        if s.dtype == "goldilocks":
            return Spec(s.name, "uint64", s.dims)
        if s.dtype == "goldilocksx3":
            return Spec(s.name, "uint64", (*s.dims, 3))
        return s

    return Program(program.name, fn, [raw_spec(s) for s in specs], program.output_names)


def _limbs(x: Array) -> Array:
    """A cubic array as its `(..., 3)` base limbs — the host-visible form."""
    return split_coeffs(x).reshape(*x.shape, 3)


def _cubic(limbs: Array) -> Array:
    """`(..., 3)` base limbs back to the cubic array."""
    return join_coeffs(limbs, F3).reshape(limbs.shape[:-1])


def _values_env_traced(words: Array, vmap: list) -> dict[int, Array]:
    """`pil2.values_env` inside a trace: dumped packing -> cubic scalars."""
    out = {}
    for i, off, w in value_offsets(vmap):
        limbs = (
            words[off : off + 3]
            if w == 3
            else fnp.concatenate([words[off : off + 1], fnp.zeros((2,), F)])
        )
        out[i] = _cubic(limbs)
    return out


def traced_scalar_env(
    si: dict,
    publics: Array,
    airvalues: Array,
    proofvalues: Array,
    challenges: Array,
    airgroupvalues: Array,
) -> dict:
    """`pil2.scalar_env` from the flat `StepsParams` buffers, inside the
    trace: publics ``(nPublics,)`` words, air/proof values in dumped
    packing, challenges and airgroup values as ``(n, 3)`` limb rows (every
    ``challengesMap`` / ``airgroupValuesMap`` entry, unsqueezed ones zero)."""
    zeros = fnp.zeros_like(publics)
    packed = join_coeffs(fnp.stack([publics, zeros, zeros], axis=1), F3).reshape(
        publics.shape[0]
    )
    return {
        "challenges": {i: _cubic(challenges[i]) for i in range(challenges.shape[0])},
        "publics": packed,
        "airvalues": _values_env_traced(airvalues, si.get("airValuesMap") or []),
        "airgroupvalues": {
            i: _cubic(airgroupvalues[i]) for i in range(airgroupvalues.shape[0])
        },
        "proofvalues": _values_env_traced(proofvalues, si.get("proofValuesMap") or []),
    }


def _last_level_traced(digest_layers: list[Array], arity: int, llv: int) -> Array:
    """`proof_serializer._last_level` inside a trace: the first level with at
    most ``arity^llv`` nodes, zero-padded to exactly that many."""
    if llv == 0:
        return fnp.zeros((0,), F)
    cap = arity**llv
    n = int(digest_layers[0].shape[0])
    level = 0
    while n > cap:
        n = (n + arity - 1) // arity
        level += 1
    nodes = digest_layers[level][:n]
    return fnp.zeros((cap, DIGEST), F).at[: nodes.shape[0]].set(nodes).reshape(-1)


def _layer_specs(tree, prefix: str, n_rows: int, width: int) -> list[Spec]:
    """The digest layers `tree.commit` returns for an `(n_rows, width)`
    matrix, by tracing the commit: the k-ary tree zero-pads each level to a
    multiple of the arity, so the shapes are the tree's to say, not a
    formula's."""
    layers = frx.eval_shape(
        lambda m: tree.commit(m)[1], frx.ShapeDtypeStruct((n_rows, width), F)
    )
    return [Spec.of(f"{prefix}_{j}", a) for j, a in enumerate(layers)]


class AirPrograms:
    """Every device program of one AIR's schedule, built over the pil2-mode
    roles of a `Pil2InnerProver` so each program IS the role's traced
    function. `programs()` lists them in schedule order; the exporter
    lowers each and writes the manifest."""

    def __init__(self, key: Pil2Key, prover: Pil2InnerProver) -> None:
        si, ss = key.starkinfo, key.starkinfo["starkStruct"]
        self.key = key
        self.prover = prover
        self.si = si
        self.nb, self.nbe = ss["nBits"], ss["nBitsExt"]
        self.n, self.ne = 1 << self.nb, 1 << self.nbe
        self.arity = ss["merkleTreeArity"]
        self.family = key.hash_family
        self.steps = [s["nBits"] for s in ss["steps"]]
        self.llv = ss.get("lastLevelVerification", 0)
        self.n_queries = ss["nQueries"]
        self.pow_bits = ss["powBits"]
        self.n_stages = si["nStages"]
        self.w1 = si["mapSectionsN"]["cm1"]
        self.w2 = si["mapSectionsN"]["cm2"]
        self.wq = si["mapSectionsN"][f"cm{self.n_stages + 1}"]
        self.n_const = si["nConstants"]
        self.n_publics = si["nPublics"]
        self.n_ch = len(si["challengesMap"])
        self.n_agv = len(si.get("airgroupValuesMap") or [])
        self.avs = sum(w for _, _, w in value_offsets(si.get("airValuesMap") or []))
        self.pvs = sum(w for _, _, w in value_offsets(si.get("proofValuesMap") or []))
        self.custom_ids = sorted(key.custom_ext)
        self.custom_widths = {ci: int(key.custom_ext[ci].shape[1]) for ci in self.custom_ids}
        self.n_ev = len(si["evMap"])
        self.tree = merkle_tree(self.arity, self.family)
        self.code = Pil2FriCode(tuple(self.steps))
        q = prover.quotient
        per = -(self.ne // -q.q_chunks)
        self.chunk_sizes = [
            min((k + 1) * per, self.ne) - k * per
            for k in range(q.q_chunks)
            if k * per < self.ne
        ]
        run = prover.logup._run_hint
        self.airgroupvalue_index = (
            hint_value(run, "result")["id"] if prover.logup._run_is_sum else None
        )

    # -- the scalar-section inputs every hint/expression stage shares --------

    def _scalar_specs(self, *, challenges: bool, airgroupvalues: bool) -> list[Spec]:
        specs = [
            Spec("publics", "goldilocks", (self.n_publics,)),
            Spec("airvalues", "goldilocks", (self.avs,)),
            Spec("proofvalues", "goldilocks", (self.pvs,)),
        ]
        if challenges:
            specs.append(Spec("challenges", "goldilocks", (self.n_ch, 3)))
        if airgroupvalues:
            specs.append(Spec("airgroupvalues", "goldilocks", (self.n_agv, 3)))
        return specs

    def _scalars(self, publics, airvalues, proofvalues, challenges=None, agv=None):
        if challenges is None:
            challenges = fnp.zeros((self.n_ch, 3), F)
        if agv is None:
            agv = fnp.zeros((self.n_agv, 3), F)
        return traced_scalar_env(self.si, publics, airvalues, proofvalues, challenges, agv)

    def _custom_specs(self, domain: str) -> list[Spec]:
        n = self.n if domain == "base" else self.ne
        return [
            Spec(f"custom_{domain}_{ci}", "goldilocks", (n, self.custom_widths[ci]))
            for ci in self.custom_ids
        ]

    def _customs(self, args: list) -> dict[int, Array]:
        return dict(zip(self.custom_ids, args))

    def _section_specs(self) -> list[Spec]:
        """The committed + fixed extended sections the openings read."""
        return [
            Spec("cm1_ext", "goldilocks", (self.ne, self.w1)),
            Spec("cm2_ext", "goldilocks", (self.ne, self.w2)),
            Spec("qsec", "goldilocks", (self.ne, self.wq)),
            Spec("const_ext", "goldilocks", (self.ne, self.n_const)),
            *self._custom_specs("ext"),
        ]

    def _bufs(self, cm1, cm2, qsec, const, customs: list) -> dict:
        bufs = {
            ("cm", 1): cm1,
            ("cm", 2): cm2,
            ("cm", self.n_stages + 1): qsec,
            ("const", 0): const,
        }
        for ci, buf in zip(self.custom_ids, customs):
            bufs[("custom", ci)] = buf
        return bufs

    # -- programs ---------------------------------------------------------------

    def _commit_program(self, name: str, spec: Spec, root: str, ext: str) -> Program:
        """Extend-and-merkelize of one base-domain section — the stage
        commits and the key-side constant/custom setups alike."""
        blowup = 1 << (self.nbe - self.nb)

        def fn(matrix):
            extended = extend(matrix, blowup)
            r, layers = self.tree.commit(extended)
            return (r, extended, *layers)

        layers = _layer_specs(self.tree, f"{name}_layers", self.ne, spec.dims[1])
        return Program(name, fn, [spec], [root, ext, *(s.name for s in layers)])

    def constants(self) -> Program:
        """The key-static coset points and inverse zerofier, computed in the
        trace (an interned constant would lower as an in-graph literal,
        #67's crash trigger)."""
        nb, bb = self.nb, self.nbe - self.nb

        one = fnp.ones((), F)

        def fn():
            # `zerofier._coset_points` / `inv_zerofier`, op for op: the
            # interned host copies cannot be closed over (they would lower as
            # in-graph literals), so the same arithmetic runs in the trace.
            domain = _SHIFT * powers(_root(nb + bb), self.ne)
            sn = fnp.power(_SHIFT, self.n)
            period = one / (sn * powers(_root(bb), 1 << bb) - one)
            return (fnp.tile(period, self.ne >> bb), domain)

        return Program("constants", fn, [], ["zi", "domain"])

    def const_setup(self) -> Program:
        return self._commit_program(
            "const_setup",
            Spec("const_base", "goldilocks", (self.n, self.n_const)),
            "const_root",
            "const_ext",
        )

    def custom_setup(self, ci: int) -> Program:
        return self._commit_program(
            f"custom_setup_{ci}",
            Spec(f"custom_base_{ci}", "goldilocks", (self.n, self.custom_widths[ci])),
            f"custom_root_{ci}",
            f"custom_ext_{ci}",
        )

    def witness_calc(self) -> Program | None:
        role = self.prover.witness
        if not role.active:
            return None
        nc = len(self.custom_ids)

        def fn(trace, const_base, *rest):
            customs, (publics, airvalues, proofvalues) = rest[:nc], rest[nc:]
            return (
                role.columns(
                    trace,
                    const_base,
                    self._customs(list(customs)),
                    self._scalars(publics, airvalues, proofvalues),
                ),
            )

        return Program(
            "witness_calc",
            fn,
            [
                Spec("trace", "goldilocks", (self.n, self.w1)),
                Spec("const_base", "goldilocks", (self.n, self.n_const)),
                *self._custom_specs("base"),
                *self._scalar_specs(challenges=False, airgroupvalues=False),
            ],
            ["trace"],
        )

    def commit1(self) -> Program:
        role = self.prover.opening

        def fn(trace):
            root, layers, extended = role.commit_components(trace)
            return (root, extended, *layers)

        layers = _layer_specs(self.tree, "cm1_layers", self.ne, self.w1)
        return Program(
            "commit1",
            fn,
            [Spec("trace", "goldilocks", (self.n, self.w1))],
            ["root1", "cm1_ext", *(s.name for s in layers)],
        )

    def logup(self) -> Program:
        role = self.prover.logup
        nc = len(self.custom_ids)

        def fn(trace, const_base, *rest):
            customs, (publics, airvalues, proofvalues, challenges) = rest[:nc], rest[nc:]
            matrix, result, airvalues_out = role.columns(
                trace,
                const_base,
                self._customs(list(customs)),
                self._scalars(publics, airvalues, proofvalues, challenges),
            )
            outs = [matrix]
            if result is not None:
                outs.append(_limbs(result.reshape(())))
            outs.append(airvalues_out)
            return tuple(outs)

        names = ["cm2"]
        if self.airgroupvalue_index is not None:
            names.append("airgroupvalue")
        names.append("airvalues")
        return Program(
            "logup",
            fn,
            [
                Spec("trace", "goldilocks", (self.n, self.w1)),
                Spec("const_base", "goldilocks", (self.n, self.n_const)),
                *self._custom_specs("base"),
                *self._scalar_specs(challenges=True, airgroupvalues=False),
            ],
            names,
        )

    def commit2(self) -> Program:
        role = self.prover.logup

        def fn(matrix):
            root, layers, extended = role.commit_components(matrix)
            return (root, extended, *layers)

        layers = _layer_specs(self.tree, "cm2_layers", self.ne, self.w2)
        return Program(
            "commit2",
            fn,
            [Spec("cm2", "goldilocks", (self.n, self.w2))],
            ["root2", "cm2_ext", *(s.name for s in layers)],
        )

    def _quotient_inputs(self) -> list[Spec]:
        return [
            Spec("cm1_ext", "goldilocks", (self.ne, self.w1)),
            Spec("cm2_ext", "goldilocks", (self.ne, self.w2)),
            Spec("const_ext", "goldilocks", (self.ne, self.n_const)),
            *self._custom_specs("ext"),
            *self._scalar_specs(challenges=True, airgroupvalues=True),
            Spec("zi", "goldilocks", (self.ne,)),
        ]

    def quotient(self, rows: int | None) -> Program:
        """The cExp over every row (`rows` None) or over one `rows`-long
        window handed in as an index vector — the chunked form the wide
        AIRs need (`pil2_prover._row_chunks`)."""
        role = self.prover.quotient
        nc = len(self.custom_ids)

        def fn(cm1, cm2, const, *rest):
            customs = list(rest[:nc])
            publics, airvalues, proofvalues, challenges, agv, zi = rest[nc : nc + 6]
            window = rest[nc + 6] if rows is not None else None
            scalars = self._scalars(publics, airvalues, proofvalues, challenges, agv)
            return (role.quotient(cm1, cm2, const, self._customs(customs), scalars, zi, rows=window),)

        inputs = self._quotient_inputs()
        name = "quotient"
        if rows is not None:
            inputs.append(Spec("rows", "int32", (rows,)))
            name = f"quotient_{rows}"
        return Program(name, fn, inputs, ["q"])

    def quotient_commit(self) -> Program:
        role = self.prover.quotient

        def fn(*chunks):
            q = chunks[0] if len(chunks) == 1 else fnp.concatenate(chunks)
            matrix, root, layers = role.qsec_commit(q)
            return (matrix, root, *layers)

        layers = _layer_specs(self.tree, "qsec_layers", self.ne, self.wq)
        return Program(
            "quotient_commit",
            fn,
            [Spec(f"q_{k}", "goldilocksx3", (size,)) for k, size in enumerate(self.chunk_sizes)],
            ["qsec", "rootq", *(s.name for s in layers)],
        )

    def lev(self) -> Program:
        pts = self.si["openingPoints"]

        def fn(xi):
            consts = build_lev_constants(tuple(pts), self.nb)
            return (compute_lev(_cubic(xi), pts, self.nb, consts=consts),)

        return Program("lev", fn, [Spec("xi", "goldilocks", (3,))], ["lev"])

    def evals(self) -> Program:
        role = self.prover.opening
        nc = len(self.custom_ids)

        def fn(cm1, cm2, qsec, const, *rest):
            customs, (lev,) = rest[:nc], rest[nc:]
            return (_limbs(role.evals_fn(self._bufs(cm1, cm2, qsec, const, list(customs)), lev)),)

        return Program(
            "evals",
            fn,
            [*self._section_specs(), Spec("lev", "goldilocksx3", (self.n, len(self.si["openingPoints"])))],
            ["evals"],
        )

    def deep(self) -> Program:
        role = self.prover.opening
        nc = len(self.custom_ids)

        def fn(cm1, cm2, qsec, const, *rest):
            customs = list(rest[:nc])
            evals, domain, xi, vf1, vf2 = rest[nc:]
            return (
                role.deep_fn(
                    self._bufs(cm1, cm2, qsec, const, customs),
                    _cubic(evals),
                    domain,
                    _cubic(xi),
                    _cubic(vf1),
                    _cubic(vf2),
                ),
            )

        return Program(
            "deep",
            fn,
            [
                *self._section_specs(),
                Spec("evals", "goldilocks", (self.n_ev, 3)),
                Spec("domain", "goldilocks", (self.ne,)),
                Spec("xi", "goldilocks", (3,)),
                Spec("vf1", "goldilocks", (3,)),
                Spec("vf2", "goldilocks", (3,)),
            ],
            ["fri_pol"],
        )

    def fri_commit(self, i: int) -> Program:
        """Round `i`'s commit: layer `i`'s codeword grouped into the next
        fold's cosets and merkelized (`PreFoldKGroupCommitRound`, pre-fold)."""
        code, tree = self.code, self.tree

        def fn(codeword):
            leaves = to_base_field(code.group_leaves(codeword))
            root, layers = tree.commit(leaves)
            return (leaves, root, *layers)

        n_x = 1 << (self.steps[i] - self.steps[i + 1])
        layers = _layer_specs(self.tree, f"fri_layers_{i}", 1 << self.steps[i + 1], n_x * 3)
        return Program(
            f"fri_commit_{i}",
            fn,
            [Spec("codeword", "goldilocksx3", (1 << self.steps[i],))],
            [f"fri_leaves_{i}", f"fri_root_{i}", *(s.name for s in layers)],
        )

    def fri_fold(self, i: int) -> Program:
        code = self.code

        def fn(codeword, beta):
            return (code.fold_group(codeword, _cubic(beta)),)

        return Program(
            f"fri_fold_{i}",
            fn,
            [
                Spec("codeword", "goldilocksx3", (1 << self.steps[i],)),
                Spec("beta", "goldilocks", (3,)),
            ],
            ["codeword"],
        )

    def fri_final(self) -> Program:
        def fn(codeword):
            return (_limbs(codeword),)

        return Program(
            "fri_final",
            fn,
            [Spec("codeword", "goldilocksx3", (1 << self.steps[-1],))],
            ["final_pol"],
        )

    def grind(self) -> Program:
        search = _grind_search_jit(self.family, self.pow_bits)

        def fn(challenge):
            return (search(challenge),)

        return Program("grind", fn, [Spec("challenge", "goldilocks", (3,))], ["nonce"])

    def opening(self, name: str, matrix: Spec, layers_prefix: str) -> Program:
        """The batched group proofs of one tree at the query positions plus
        its last-verified level — `proof_serializer.open_tree`'s device
        half, and the level `serialize_proof` pads."""
        tree, arity, llv = self.tree, self.arity, self.llv
        layer_specs = _layer_specs(tree, layers_prefix, matrix.dims[0], matrix.dims[1])
        depth = len(layer_specs)

        def fn(m, *rest):
            layers, idx = list(rest[:depth]), rest[depth]
            # Signed for the path walk: under x64 a uint64 index plus the
            # int64 sibling offsets promotes to float64, which cannot index.
            flat = batched_group_proof(tree, m, layers, idx.astype(fnp.int64))
            return (flat, _last_level_traced(layers, arity, llv))

        return Program(
            f"open_{name}",
            fn,
            [matrix, *layer_specs, Spec("positions", "uint64", (self.n_queries,))],
            [f"{name}_openings", f"{name}_last_level"],
        )

    def programs(self) -> list[Program]:
        out = [self.constants(), self.const_setup()]
        out += [self.custom_setup(ci) for ci in self.custom_ids]
        wc = self.witness_calc()
        if wc is not None:
            out.append(wc)
        out += [self.commit1(), self.logup(), self.commit2()]
        if len(self.chunk_sizes) == 1:
            out.append(self.quotient(None))
        else:
            for size in sorted(set(self.chunk_sizes)):
                out.append(self.quotient(size))
        out += [self.quotient_commit(), self.lev(), self.evals(), self.deep()]
        for i in range(len(self.steps) - 1):
            out += [self.fri_commit(i), self.fri_fold(i)]
        out += [self.fri_final(), self.grind()]
        ext = lambda name, width: Spec(name, "goldilocks", (self.ne, width))  # noqa: E731
        out.append(self.opening("const", ext("const_ext", self.n_const), "const_layers"))
        for ci in self.custom_ids:
            out.append(
                self.opening(
                    f"custom_{ci}", ext(f"custom_ext_{ci}", self.custom_widths[ci]), f"custom_layers_{ci}"
                )
            )
        out.append(self.opening("cm1", ext("cm1_ext", self.w1), "cm1_layers"))
        out.append(self.opening("cm2", ext("cm2_ext", self.w2), "cm2_layers"))
        out.append(self.opening("qsec", ext("qsec", self.wq), "qsec_layers"))
        for i in range(len(self.steps) - 1):
            n_x = 1 << (self.steps[i] - self.steps[i + 1])
            out.append(
                self.opening(
                    f"fri_{i}",
                    Spec(f"fri_leaves_{i}", "goldilocks", (1 << self.steps[i + 1], n_x * 3)),
                    f"fri_layers_{i}",
                )
            )
        return out

    def schedule(self) -> dict:
        """The AIR facts the bridge binds the programs with — the manifest's
        header. Everything here is also in the key's starkinfo; recording it
        beside the programs lets the driver cross-check the key it was handed
        against the artifacts it loaded."""
        si = self.si
        return {
            "n_bits": self.nb,
            "n_bits_ext": self.nbe,
            "hash_family": self.family,
            "arity": self.arity,
            "steps": self.steps,
            "n_queries": self.n_queries,
            "pow_bits": self.pow_bits,
            "hash_commits": bool(si["starkStruct"].get("hashCommits", False)),
            "last_level_verification": self.llv,
            "n_stages": self.n_stages,
            "n_publics": self.n_publics,
            "n_constants": self.n_const,
            "widths": {"cm1": self.w1, "cm2": self.w2, "qsec": self.wq},
            "custom_commits": [
                {"id": ci, "width": self.custom_widths[ci]} for ci in self.custom_ids
            ],
            "ev_map_size": self.n_ev,
            "opening_points": si["openingPoints"],
            "challenges": [
                {"id": i, "name": c.get("name"), "stage": c["stage"]}
                for i, c in enumerate(si["challengesMap"])
            ],
            "airvalues": [{"stage": v["stage"]} for v in si.get("airValuesMap") or []],
            "airgroupvalues": [{"stage": v["stage"]} for v in si.get("airgroupValuesMap") or []],
            "proofvalues": [{"stage": v["stage"]} for v in si.get("proofValuesMap") or []],
            "airgroupvalue_index": self.airgroupvalue_index,
            "quotient_chunks": self.chunk_sizes,
            "witness_calc": self.prover.witness.active,
        }

