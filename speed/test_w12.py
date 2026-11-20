"""Model-free, bitwise tests of the offline W12 format (CPU)."""

import unittest

import torch

from speed.w12 import decode_record, encode, validate_record


class W12CodecTest(unittest.TestCase):
    def test_all_bf16_bit_patterns(self):
        # Mix all exponents within each output row, including exceptional values
        # and both signed zeros. Also cross a 128-channel tile boundary.
        bits = torch.arange(65536, dtype=torch.int32).reshape(256, 256).T.contiguous().short()
        record = encode(bits.view(torch.bfloat16))
        validate_record(record)
        self.assertGreater(record["values"].numel(), 0)
        self.assertTrue(torch.equal(bits, decode_record(record).view(torch.int16)))

    def test_partial_tile_and_corrupt_escape_column(self):
        bits = torch.full((129, 64), 0x3F80, dtype=torch.int16)
        bits[-1, -1] = -32768  # negative zero stored as a full BF16 escape
        record = encode(bits.view(torch.bfloat16))
        self.assertEqual(len(record["values"]), 1)
        self.assertTrue(torch.equal(bits, decode_record(record).view(torch.int16)))
        record["columns"][0] = 64
        with self.assertRaises(ValueError):
            validate_record(record)


if __name__ == "__main__":
    unittest.main()
