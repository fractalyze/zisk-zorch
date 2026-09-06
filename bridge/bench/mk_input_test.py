"""Pins the ZiskStdin framing `mk_input.py` writes: the fibonacci example's
committed sample (`examples/fibonacci/guest/samples/example-input.bin` at
v1.0.0-alpha, a `u8` 10) is the reference for the frame layout, and the
bincode varint widths are the ones `ziskos::io::read::<u32>()` decodes."""

from absl.testing import absltest, parameterized

from bridge.bench import mk_input


class MkInputTest(parameterized.TestCase):

    def test_frame_matches_the_committed_fibonacci_sample(self):
        # 16 bytes: len 1, the byte 0x0a, seven bytes of pad.
        self.assertEqual(
            mk_input.frame(bytes([10])),
            bytes.fromhex("0100000000000000" "0a00000000000000"),
        )

    @parameterized.parameters(
        (0, b"\x00"),
        (250, b"\xfa"),
        (251, b"\xfb\xfb\x00"),
        (1000, b"\xfb\xe8\x03"),
        (14000, b"\xfb\xb0\x36"),
        (65535, b"\xfb\xff\xff"),
        (65536, b"\xfc\x00\x00\x01\x00"),
    )
    def test_varint_widths(self, n, expected):
        self.assertEqual(mk_input.varint_u32(n), expected)

    @parameterized.parameters(1, 3, 5, 8, 9)
    def test_frame_pads_to_eight(self, data_len):
        f = mk_input.frame(bytes(data_len))
        self.assertEqual(len(f) % 8, 0)
        self.assertEqual(len(f), 8 + data_len + (8 - (8 + data_len) % 8) % 8)

    def test_rejects_non_u32(self):
        with self.assertRaises(ValueError):
            mk_input.varint_u32(1 << 32)


if __name__ == "__main__":
    absltest.main()
