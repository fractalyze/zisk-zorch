"""Host-side (numpy) Poseidon1-Goldilocks — the family-1 grinding engine.

pil2's family-1 grinding is `PoseidonGoldilocks<8>::grinding`: state
``[challenge(3), nonce, 0, 0, 0, 0]`` through the width-8 permutation, accept
when ``out[0] < 2^(64 - pow_bits)``, smallest nonce wins. The width-8
permutation cannot run on the frx device stack (it hangs at execution,
fractalyze/zorch#565), and grinding is host-paced in pil2 itself (an OMP
scan), so this engine evaluates the permutation in vectorized numpy — exact
u64 Goldilocks arithmetic, no device round-trips — over ascending nonce
chunks. Tables come from ``goldilocks_constants.json`` (the fixture-gen dump
of pil2's own constants; row-major ``out[i] = sum_j state[j] * M[j*W + i]``).
"""

from __future__ import annotations

import functools
import json
import pathlib

import numpy as np

_P = np.uint64(0xFFFFFFFF00000001)
_EPS = np.uint64(0xFFFFFFFF)  # 2^64 mod P
_MASK32 = np.uint64(0xFFFFFFFF)
_CONSTANTS = pathlib.Path(__file__).parent / "goldilocks_constants.json"
# Chunk of candidate nonces per vectorized scan pass — large enough that the
# numpy op overhead amortizes, small enough that a pow_bits=16 search
# (expected ~2^16 tries) doesn't overshoot by much.
_CHUNK = 8192


def _add(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a + b) mod P for canonical u64 operands, vectorized."""
    s = a + b
    # a + b < 2^65; a u64 wrap dropped 2^64 ≡ 2^32 - 1 (mod P).
    s = s + np.where(s < a, _EPS, np.uint64(0))
    return np.where(s >= _P, s - _P, s)


def _mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(a * b) mod P for canonical u64 operands, vectorized.

    32-bit split multiply to recover the 128-bit product, then the
    Goldilocks fold: with t = hi·2^64 + lo,
    t ≡ lo - (hi >> 32) + (hi & 2^32-1)·(2^32 - 1)  (mod P).
    """
    al, ah = a & _MASK32, a >> np.uint64(32)
    bl, bh = b & _MASK32, b >> np.uint64(32)
    ll = al * bl
    lh = al * bh
    hl = ah * bl
    hh = ah * bh
    mid = lh + hl
    mid_carry = np.where(mid < lh, np.uint64(1), np.uint64(0))
    lo = ll + (mid << np.uint64(32))
    lo_carry = np.where(lo < ll, np.uint64(1), np.uint64(0))
    hi = hh + (mid >> np.uint64(32)) + (mid_carry << np.uint64(32)) + lo_carry

    hi_hi = hi >> np.uint64(32)
    hi_lo = hi & _MASK32
    t = lo - hi_hi
    t = t - np.where(lo < hi_hi, _EPS, np.uint64(0))
    add = hi_lo * _EPS  # < 2^64, exact in u64
    t2 = t + add
    t2 = t2 + np.where(t2 < t, _EPS, np.uint64(0))
    return np.where(t2 >= _P, t2 - _P, t2)


def _pow7(x: np.ndarray) -> np.ndarray:
    x2 = _mul(x, x)
    x3 = _mul(x2, x)
    return _mul(_mul(x3, x3), x)


class _Tables:
    def __init__(self, entry: dict) -> None:
        self.width = entry["width"]
        self.half_f = entry["half_full_rounds"]
        self.partials = entry["n_partial_rounds"]
        u64 = lambda v: np.array([int(x) for x in v], dtype=np.uint64)
        self.c = u64(entry["c"])
        self.m = u64(entry["m"]).reshape(self.width, self.width)
        self.p = u64(entry["p"]).reshape(self.width, self.width)
        self.s = u64(entry["s"]).reshape(self.partials, 2 * self.width - 1)


@functools.cache
def _tables(width: int) -> _Tables:
    data = json.loads(_CONSTANTS.read_text())
    return _Tables(next(w for w in data["widths"] if w["width"] == width))


def _mvp(states: np.ndarray, mat: np.ndarray) -> np.ndarray:
    """out[:, i] = sum_j states[:, j] * mat[j, i] — pil2's transposed apply."""
    w = mat.shape[0]
    out = np.zeros_like(states)
    for i in range(w):
        acc = _mul(states[:, 0], mat[0, i])
        for j in range(1, w):
            acc = _add(acc, _mul(states[:, j], mat[j, i]))
        out[:, i] = acc
    return out


def permute_host(states: np.ndarray, width: int) -> np.ndarray:
    """`PoseidonGoldilocks<width>::permute_seq` over a (n, width) batch."""
    tbl = _tables(width)
    w, half_f = tbl.width, tbl.half_f
    if states.ndim != 2 or states.shape[1] != w:
        raise ValueError(f"states must be (n, {w}), got {states.shape}")
    s = states.astype(np.uint64).copy()
    with np.errstate(over="ignore"):
        s = _add(s, tbl.c[:w][None, :])
        for r in range(half_f - 1):
            s = _mvp(_add(_pow7(s), tbl.c[(r + 1) * w : (r + 2) * w][None, :]), tbl.m)
        s = _mvp(_add(_pow7(s), tbl.c[half_f * w : (half_f + 1) * w][None, :]), tbl.p)

        for r in range(tbl.partials):
            s0 = _add(_pow7(s[:, 0]), tbl.c[(half_f + 1) * w + r])
            s[:, 0] = s0
            dot = _mul(s[:, 0], tbl.s[r, 0])
            for j in range(1, w):
                dot = _add(dot, _mul(s[:, j], tbl.s[r, j]))
            for j in range(1, w):
                s[:, j] = _add(s[:, j], _mul(s0, tbl.s[r, w - 1 + j]))
            s[:, 0] = dot

        base = (half_f + 1) * w + tbl.partials
        for r in range(half_f - 1):
            s = _mvp(
                _add(_pow7(s), tbl.c[base + r * w : base + (r + 1) * w][None, :]), tbl.m
            )
        return _mvp(_pow7(s), tbl.m)


# Grinding runs the CAPACITY-width state: [challenge(3), nonce, 0-pad] at
# width 8 regardless of the key's transcript/merkle widths
# (starks_api_internal.hpp routes HashFamily::Poseidon1 to
# PoseidonGoldilocks<8>::grinding).
_GRIND_WIDTH = 8


def grind_family1(challenge: np.ndarray, pow_bits: int) -> int:
    """The family-1 grinding nonce: smallest n with
    ``permute([c0, c1, c2, n, 0...])[0] < 2^(64 - pow_bits)``."""
    chal = np.asarray(challenge, dtype=np.uint64).reshape(3)
    level = np.uint64(1) << np.uint64(64 - pow_bits)
    base = 0
    while base < 1 << 40:
        states = np.zeros((_CHUNK, _GRIND_WIDTH), dtype=np.uint64)
        states[:, :3] = chal[None, :]
        states[:, 3] = np.arange(base, base + _CHUNK, dtype=np.uint64)
        out0 = permute_host(states, _GRIND_WIDTH)[:, 0]
        hits = np.nonzero(out0 < level)[0]
        if hits.size:
            return base + int(hits[0])
        base += _CHUNK
    raise RuntimeError(f"grinding: no nonce found for pow_bits={pow_bits}")


def grind_family1_is_valid(challenge: np.ndarray, nonce: int, pow_bits: int) -> bool:
    """The verifier's O(1) grind check for a family-1 proof."""
    chal = np.asarray(challenge, dtype=np.uint64).reshape(3)
    state = np.zeros((1, _GRIND_WIDTH), dtype=np.uint64)
    state[0, :3] = chal
    state[0, 3] = np.uint64(nonce)
    out0 = permute_host(state, _GRIND_WIDTH)[0, 0]
    return bool(out0 < (np.uint64(1) << np.uint64(64 - pow_bits)))
