"""pil2-stark's Fiat-Shamir transcript — a pending/out buffer over Poseidon2.

NOT zorch's DuplexTranscript: pil2 accumulates absorbed elements in a
`width - 4` pending buffer, and on each flush permutes `pending || state[:4]`
(the running state's first 4 lanes ride in the LAST 4 input slots), keeping
the full output as both the next state and the squeeze buffer. Squeezes read
the out buffer front-to-back and any absorb invalidates it. Challenges for the
cubic extension are 3 consecutive squeezes; query indices pack 63 bits per
squeezed element, LSB first. Reference:
https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/fields/src/transcript.rs

Host-driven and eager by design — proof orchestration glue, not a traced round
body (the device-side transcript story is zorch's DuplexTranscript; pil2
byte-match requires pil2's exact buffer discipline).
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
import zk_dtypes
from frx import Array, lax
from zk_dtypes import goldilocks as F
from zorch.hash.poseidon2.poseidon2 import Poseidon2

from zisk_zorch.poseidon2.goldilocks import goldilocks_perm

# Challenges live in the cubic extension — 3 Goldilocks limbs per challenge.
CHALLENGE_LIMBS = 3
# A transcript digest is the flushed state's first four elements — pil2's
# `HASH_SIZE`, independent of the transcript width.
DIGEST = 4
# get_permutations packs 63 usable bits per squeezed element (canonical u64
# < 2^64 - 2^32 + 1, so bit 63 is biased; pil2 simply never uses it).
_BITS_PER_ELEMENT = 63


def _canonical(values: Array) -> np.ndarray:
    """Field elements -> canonical u64 on host (plain-storage read).

    Default JAX has no u64 (x64 disabled truncates the conversion), so read the
    plain goldilocks storage (limb0 is the canonical value), bitcast to u32
    halves, and recombine on host."""
    std = values.astype(zk_dtypes.goldilocks)
    halves = np.asarray(lax.bitcast_convert_type(std, fnp.uint32)).astype(np.uint64)
    return halves[..., 0] | (halves[..., 1] << np.uint64(32))


class Transcript:
    """pil2 transcript over the width-`4*transcript_arity` Poseidon2."""

    def __init__(self, width: int = 12) -> None:
        self._perm: Poseidon2 = goldilocks_perm(width)
        self.width = width
        self._state = fnp.zeros((width,), F)
        self._out = fnp.zeros((width,), F)
        self._pending: list[Array] = []
        self._out_cursor = 0

    def _update_state(self) -> None:
        pad = [fnp.zeros((), F)] * (self.width - 4 - len(self._pending))
        inputs = fnp.concatenate([fnp.stack(self._pending + pad), self._state[:4]])
        self._state = self._perm.permute(inputs)
        self._out = self._state
        self._out_cursor = self.width
        self._pending = []

    def put(self, values: Array) -> None:
        """Absorb a 1-D batch of field elements."""
        for i in range(values.shape[0]):
            self._pending.append(values[i])
            self._out_cursor = 0
            if len(self._pending) == self.width - 4:
                self._update_state()

    def get_state(self) -> Array:
        """Flush any pending absorbs and return the full state (width,)."""
        if self._pending:
            self._update_state()
        return self._state

    def get_fields1(self) -> Array:
        """Squeeze one field element."""
        if self._out_cursor == 0:
            self._update_state()
        value = self._out[(self.width - self._out_cursor) % self.width]
        self._out_cursor -= 1
        return value

    def get_field(self) -> Array:
        """Squeeze one cubic-extension challenge as its 3 Goldilocks limbs."""
        return fnp.stack([self.get_fields1() for _ in range(CHALLENGE_LIMBS)])

    def get_permutations(self, n: int, n_bits: int) -> np.ndarray:
        """Squeeze `n` query indices of `n_bits` bits — 63 bits per element,
        consumed LSB-first across element boundaries."""
        total_bits = n * n_bits
        n_fields = (total_bits - 1) // _BITS_PER_ELEMENT + 1
        fields = _canonical(fnp.stack([self.get_fields1() for _ in range(n_fields)]))
        out = np.zeros(n, dtype=np.uint64)
        cur_field, cur_bit = 0, 0
        for i in range(n):
            acc = np.uint64(0)
            for j in range(n_bits):
                bit = (int(fields[cur_field]) >> cur_bit) & 1
                acc |= np.uint64(bit << j)
                cur_bit += 1
                if cur_bit == _BITS_PER_ELEMENT:
                    cur_bit = 0
                    cur_field += 1
            out[i] = acc
        return out


def transcript_hash(values: Array, width: int = 12) -> Array:
    """pil2 ``calculateHash``: a fresh transcript absorbs the buffer and its
    flushed state's first four elements are the digest."""
    t = Transcript(width)
    t.put(values)
    return t.get_state()[:DIGEST]


def _transcript_flatten(t: Transcript) -> tuple[tuple, tuple]:
    # Leaves: the duplex state and any pending absorbs (traced through a jit
    # zone); aux: the width and squeeze cursor, which evolve host-side on the
    # fixed absorb/squeeze schedule, so they are static per call site — two
    # zones whose schedules leave different pending/cursor shapes compile
    # separately, as they must.
    return (t._state, t._out, tuple(t._pending)), (t.width, t._out_cursor)


def _transcript_unflatten(aux: tuple, children: tuple) -> Transcript:
    width, cursor = aux
    state, out, pending = children
    t = object.__new__(Transcript)
    t._perm = goldilocks_perm(width)  # cached per width — no rebuild cost
    t.width = width
    t._state = state
    t._out = out
    t._pending = list(pending)
    t._out_cursor = cursor
    return t


# A pytree, so a whole absorb/squeeze leg can cross a `frx.jit` boundary as a
# value: the zone returns the advanced transcript and the caller threads it
# forward. Mutation inside the zone acts on the unflattened copy, which is why
# zones must RETURN their transcript — the caller's object is not updated in
# place across the boundary.
frx.tree_util.register_pytree_node(
    Transcript, _transcript_flatten, _transcript_unflatten
)
