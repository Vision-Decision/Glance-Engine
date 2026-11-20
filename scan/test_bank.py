"""Model-free storage tests, including exceptional floating-point bit patterns."""

import unittest
from types import SimpleNamespace

import torch

from scan.bank import FORMATS, StoredBank
from scan.bank_codec import decode, encode
from scan.host_bank import _copy_ranges, _layout


class HostCopyTest(unittest.TestCase):
    def test_grouped_copy_keeps_variable_tables_at_the_captured_offsets(self):
        def layout(sizes):
            return _layout((i, torch.Size([n]), torch.uint8) for i, n in enumerate(sizes))

        src = layout([8, 16, 3, 18, 0, 4])
        dst = layout([8, 16, 32, 18, 96, 4])
        plan = _copy_ranges(src, dst)
        self.assertEqual(len(plan), 3)
        source = torch.arange(src[-1][3] + src[-1][4], dtype=torch.uint8)
        target = torch.zeros(dst[-1][3] + dst[-1][4], dtype=torch.uint8)
        for start, dest, size in plan:
            target[dest : dest + size].copy_(source[start : start + size])
        for (_, _, _, start, size), (_, _, _, dest, _) in zip(src, dst):
            self.assertTrue(torch.equal(source[start : start + size], target[dest : dest + size]))


class StatePlanesTest(unittest.TestCase):
    def test_round_trip(self):
        rng = torch.Generator().manual_seed(8)
        bits = torch.randint(-(2**31), 2**31 - 1, (3, 4, 128, 128), dtype=torch.int32, generator=rng)
        bits[1] = torch.randn((4, 128, 128), generator=rng).view(torch.int32)
        bits[0, 0].zero_()
        bits[0, 1, :, ::2] = -(2**31)
        bits[0, 1, :, 1::2] = 0x7FC01234
        self.assertTrue(torch.equal(decode(encode(bits.view(torch.float32))).view(torch.int32), bits))


class BankIndexTest(unittest.TestCase):
    def test_ragged_storage_and_repeated_selection(self):
        rng = torch.Generator().manual_seed(19)
        rand = lambda *shape: torch.randn(shape, generator=rng)
        mask = torch.tensor([[0, 1, 1, 1], [0, 0, 1, 1], [1, 1, 1, 1]], dtype=torch.bool)
        k, v = rand(3, 2, 4, 8), rand(3, 2, 4, 8)
        for row, start in enumerate([1, 2, 0]):
            k[row, :, start] = 1
            v[row, :, start] = -1
        vis = rand(3, 4, 16) * mask[..., None]
        bank = SimpleNamespace(
            types=["linear_attention", "full_attention"],
            N=3,
            has_clips=False,
            p_valid=mask,
            vis_valid=mask,
            vis=vis,
            centers=rand(3, 4, 2) * mask[..., None],
            ck=rand(3, 4, 8) * mask[..., None],
            q_start=torch.arange(3),
            grid_hw=torch.ones(3, 2),
            cache=[(rand(3, 2, 4, 128), rand(3, 8, 3)), (k, v)],
        )
        ids = torch.tensor([2, 0, 2, 1])
        for fmt in FORMATS[1:]:
            source = StoredBank.pack(bank, fmt)
            self.assertEqual(source.shared, 1)
            sub = source.select(ids)
            for i in range(2):
                for expected, actual in zip(source.layer(i), sub.layer(i)):
                    self.assertTrue(torch.equal(expected[ids], actual), fmt)
            for name in ("vis", "ck", "centers"):
                self.assertTrue(torch.equal(getattr(source, name)[ids], getattr(sub, name)))
            typed = StoredBank.pack(bank, fmt, "typed")
            self.assertNotIn("vis", typed.tensors)
            self.assertNotIn("ck", typed.tensors)


if __name__ == "__main__":
    unittest.main()
