"""FRI query-position sampling + grinding — pil2-stark's PoW-seeded query draw.

After the fold loop, pil2's query phase (`gen_proof.hpp`) absorbs the final
polynomial into the running transcript and squeezes a cubic challenge, then runs
proof-of-work: search a `nonce` such that the width-4 Poseidon2 permutation of
`challenge ++ nonce` has `powBits` leading zero bits. It then seeds a FRESH
transcript with `challenge ++ nonce` and reads the query positions off it via
`getPermutations` (`nQueries` indices of `nBitsExt` bits each), and transmits
`nonce` in the proof so the verifier can re-check the grind.

The grinding search (`Poseidon2GoldilocksGrinding`) is not exported by the
v1.0.0-alpha `fields` crate, so it is ported here from pil2-stark C++ and goldened
standalone (`grinding.json`). The search is deterministic — the smallest valid
nonce — so the prover commits a canonical nonce and the verifier validates it
with an O(1) grind check rather than re-running the search.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/starkpil/gen_proof.hpp#L236-L283
https://github.com/0xPolygonHermez/pil2-proofman/blob/v1.0.0-alpha/pil2-stark/src/goldilocks/src/poseidon2_goldilocks.cpp#L177-L232
"""

from __future__ import annotations

import frx
import frx.numpy as fnp
import numpy as np
from frx import Array
from zk_dtypes import goldilocks as F

from zorch.utils.field import to_limb_rows
from zisk_zorch.poseidon2.goldilocks import goldilocks_perm
from zisk_zorch.transcript.transcript import Transcript, _canonical

# Grinding is the width-4 Poseidon2 compression (`Poseidon2GoldilocksGrinding =
# Poseidon2Goldilocks<4>`), independent of the transcript width.
_GRIND_WIDTH = 4
_GRIND_PERM = goldilocks_perm(_GRIND_WIDTH)
# Batched search stride: scan candidate nonces a chunk at a time so the per-call
# permutation runs once over the whole chunk, mirroring pil2's chunked OMP scan.
_GRIND_CHUNK = 256
_GOLDILOCKS_ORDER = 0xFFFF_FFFF_0000_0001


def _grind_level(pow_bits: int) -> int:
    """The grind threshold: an image passes iff it is `< 2^(64 - pow_bits)`,
    i.e. its top `pow_bits` bits are zero."""
    return 1 << (64 - pow_bits)


def _grind_images(challenge: Array, nonces: np.ndarray) -> np.ndarray:
    """Canonical u64 of the first lane of the width-4 Poseidon2 permutation of
    `challenge ++ nonce` for each nonce — pil2's grinding image (`state[0]`).

    `nonce` rides the last input slot as `Goldilocks::fromU64(nonce)`; the value
    cast to `F` reduces mod p, so callers must pass canonical nonces."""
    nonce_fe = fnp.asarray(nonces, dtype=F)  # (C,) canonical -> plain field
    chal = fnp.broadcast_to(challenge, (nonce_fe.shape[0], _GRIND_WIDTH - 1))
    states = fnp.concatenate([chal, nonce_fe[:, None]], axis=1)  # (C, 4)
    out0 = frx.vmap(_GRIND_PERM.permute)(states)[:, 0]
    return _canonical(out0)  # decode to canonical u64 (transcript's path)


def _grind(challenge: Array, pow_bits: int) -> int:
    """The smallest nonce whose grinding image has `pow_bits` leading zero bits,
    i.e. image `< 2^(64 - pow_bits)`. Ascending scan matching pil2's
    `Poseidon2GoldilocksGrinding::grinding` (its OMP chunking only parallelizes
    the scan; any valid nonce verifies, so the smallest is the deterministic
    choice the verifier reproduces via its O(1) check)."""
    level = _grind_level(pow_bits)
    base = 0
    while base < _GOLDILOCKS_ORDER:
        nonces = np.arange(base, base + _GRIND_CHUNK, dtype=np.uint64)
        hits = np.nonzero(_grind_images(challenge, nonces) < np.uint64(level))[0]
        if hits.size:
            return base + int(hits[0])
        base += _GRIND_CHUNK
    raise RuntimeError(f"grinding: no nonce below the field order for pow_bits={pow_bits}")


def grind_is_valid(challenge: Array, nonce: int, pow_bits: int) -> bool:
    """Whether `nonce` satisfies the grind: it is a canonical Goldilocks element
    and its image has `pow_bits` leading zero bits. The verifier's O(1) check on
    the proof-supplied nonce (pil2 `stark_verify.hpp` L195-L200)."""
    if not 0 <= nonce < _GOLDILOCKS_ORDER:
        return False
    image = int(_grind_images(challenge, np.array([nonce], dtype=np.uint64))[0])
    return image < _grind_level(pow_bits)


def grinding_seed_challenge(transcript: Transcript, final_pol: Array) -> Array:
    """Absorb `final_pol` into the running transcript and squeeze the cubic
    grinding-seed challenge (3 base limbs) — pil2's pre-grinding discipline.
    Mutates `transcript`."""
    transcript.put(to_limb_rows(final_pol).reshape(-1))  # addTranscriptGL(friPol, len*3)
    return transcript.get_field()  # cubic grinding seed (3 base limbs)


def query_positions_for(
    challenge: Array, width: int, nonce: int, *, n_queries: int, n_bits_ext: int
) -> np.ndarray:
    """Seed a fresh transcript with `challenge ++ nonce` and read the query
    positions — pil2's `getPermutations` off the grinding-seeded transcript."""
    permutation = Transcript(width)
    permutation.put(challenge)  # addTranscriptGL(challenge, FIELD_EXTENSION)
    nonce_fe = fnp.array(np.array([nonce], dtype=np.uint64), dtype=F)
    permutation.put(nonce_fe)  # addTranscriptGL((Goldilocks::Element *)&nonce, 1)
    return permutation.get_permutations(n_queries, n_bits_ext)


def sample_query_positions(
    transcript: Transcript,
    final_pol: Array,
    *,
    pow_bits: int,
    n_queries: int,
    n_bits_ext: int,
) -> tuple[np.ndarray, int]:
    """Derive the `n_queries` FRI query positions (each `n_bits_ext` bits wide)
    from the post-fold `transcript`, self-generating the grinding nonce.

    Mutates `transcript` exactly as pil2's query phase does — absorbs `final_pol`
    and squeezes the grinding-seed challenge — then searches for the PoW `nonce`
    (`pow_bits` leading zeros), seeds a fresh transcript with `challenge ++ nonce`
    and reads the positions. Returns `(positions, nonce)`; the prover transmits
    `nonce` in the proof so the verifier can re-check the grind."""
    # Validate before touching the transcript — a rejected call must not leave it
    # half-absorbed.
    if final_pol.ndim != 1:
        raise ValueError(f"final_pol must be 1-D, got shape {final_pol.shape}")
    if n_queries <= 0:
        raise ValueError(f"n_queries must be positive, got {n_queries}")
    if not 0 < n_bits_ext <= 32:
        raise ValueError(f"n_bits_ext must be in (0, 32], got {n_bits_ext}")
    if not 0 < pow_bits < 64:
        raise ValueError(f"pow_bits must be in (0, 64), got {pow_bits}")

    challenge = grinding_seed_challenge(transcript, final_pol)
    nonce = _grind(challenge, pow_bits)
    positions = query_positions_for(
        challenge, transcript.width, nonce, n_queries=n_queries, n_bits_ext=n_bits_ext
    )
    return positions, nonce
