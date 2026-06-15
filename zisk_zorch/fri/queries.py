"""FRI query-position sampling — pil2-stark's grinding-seeded query derivation.

After the fold loop, pil2's query phase (`gen_proof.hpp`) absorbs the final
polynomial into the running transcript, squeezes a cubic challenge, runs
proof-of-work to find a `nonce`, then seeds a FRESH transcript with
`challenge ++ nonce` and reads the query positions off it via `getPermutations`
(`nQueries` indices of `nBitsExt` bits each).

The PoW search itself (`Poseidon2GoldilocksGrinding`) is not exported by the
v0.18.0 `fields` crate, so it has no golden reference: `nonce` is supplied by the
caller and only the goldenable derivation — finalPol absorb -> challenge ->
reseed -> getPermutations — lives here. The nonce is absorbed as a canonical
Goldilocks element, matching the `fields` crate's `Transcript::put`.

https://github.com/0xPolygonHermez/pil2-proofman/blob/v0.18.0/pil2-stark/src/starkpil/gen_proof.hpp#L235-L282
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax import Array
from zk_dtypes import goldilocks_mont as F

from zisk_zorch.fri.seam import _cubic_to_base
from zisk_zorch.transcript.transcript import Transcript


def sample_query_positions(
    transcript: Transcript,
    final_pol: Array,
    nonce: int,
    *,
    n_queries: int,
    n_bits_ext: int,
) -> np.ndarray:
    """Derive the `n_queries` FRI query positions (each `n_bits_ext` bits wide)
    from the post-fold `transcript`.

    Mutates `transcript` exactly as pil2's query phase does — absorbs `final_pol`
    and squeezes the grinding-seed challenge — then seeds a fresh transcript with
    `challenge ++ nonce` and reads the positions. `nonce` is the PoW witness; the
    search that produces it is deferred (see module docstring), so it is taken as
    an input here."""
    # Validate before touching the transcript — a rejected call must not leave it
    # half-absorbed. A non-canonical nonce is the load-bearing check: `astype(F)`
    # reduces mod p, so nonce and nonce+p would silently draw the same queries.
    if final_pol.ndim != 1:
        raise ValueError(f"final_pol must be 1-D, got shape {final_pol.shape}")
    if n_queries <= 0:
        raise ValueError(f"n_queries must be positive, got {n_queries}")
    if not 0 < n_bits_ext <= 32:
        raise ValueError(f"n_bits_ext must be in (0, 32], got {n_bits_ext}")
    if not 0 <= nonce < 0xFFFF_FFFF_0000_0001:
        raise ValueError(f"nonce must be a canonical Goldilocks element, got {nonce}")

    transcript.put(_cubic_to_base(final_pol))  # addTranscriptGL(friPol, len*3)
    challenge = transcript.get_field()  # cubic grinding seed (3 base limbs)

    permutation = Transcript(transcript.width)
    permutation.put(challenge)  # addTranscriptGL(challenge, FIELD_EXTENSION)
    nonce_fe = jnp.array(np.array([nonce], dtype=np.uint64), dtype=F)
    permutation.put(nonce_fe)  # addTranscriptGL((Goldilocks::Element *)&nonce, 1)
    return permutation.get_permutations(n_queries, n_bits_ext)
