"""Round-trip and boundary pinning for the `proof.bin` varint envelope."""

from __future__ import annotations

import numpy as np
from absl.testing import absltest

from zisk_zorch.harness.proof_bin import decode, encode


class ProofBinTest(absltest.TestCase):
    def test_round_trip_across_varint_boundaries(self) -> None:
        # Every encoding class: single byte (<=250), u16, u32, u64 — and
        # the boundary values on each side.
        words = np.array(
            [0, 1, 250, 251, 65535, 65536, (1 << 32) - 1, 1 << 32,
             (1 << 64) - (1 << 32), 12345678901234567890],
            dtype=np.uint64,
        )
        trailer = b"\x04" + b"\xfd" + (7).to_bytes(8, "little") + b"\x00"
        buf = encode(words, trailer)
        got_words, got_trailer = decode(buf)
        self.assertTrue(np.array_equal(got_words, words))
        self.assertEqual(got_trailer, trailer)

    def test_known_byte_encodings(self) -> None:
        # Lead 0x00, length 1 as its own byte, then 300 as 0xfb + u16 LE.
        buf = encode(np.array([300], dtype=np.uint64), b"")
        self.assertEqual(buf, b"\x00\x01\xfb" + (300).to_bytes(2, "little"))

    def test_rejects_bad_lead(self) -> None:
        with self.assertRaises(ValueError):
            decode(b"\x01\x00")


if __name__ == "__main__":
    absltest.main()
