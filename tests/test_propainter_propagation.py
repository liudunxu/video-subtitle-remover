"""验证像素传播拒绝不可靠填入，并保留特征传播的原行为。"""
import unittest
from unittest.mock import patch

import torch

from backend.inpaint.video.model import propainter


class ProPainterPropagationTest(unittest.TestCase):
    def setUp(self):
        self.module = propainter.BidirectionalPropagation(3, learnable=False)

    def sample(self, target):
        frames = torch.ones(1, 2, 3, 8, 8)
        masks = torch.zeros(1, 2, 1, 8, 8)
        masks[:, target, :, 2:6, 2:6] = 1
        frames = frames * (1 - masks)
        flows = torch.zeros(1, 1, 2, 8, 8)
        return frames, masks, flows

    def test_consistent_flows_fill_from_either_direction(self):
        for target in (0, 1):
            with self.subTest(target=target), torch.no_grad():
                frames, masks, flows = self.sample(target)
                _, _, output, remaining = self.module(
                    frames, flows, flows, masks, 'nearest')
                self.assertTrue(torch.equal(output, torch.ones_like(frames)))
                self.assertEqual(remaining.count_nonzero().item(), 0)

    def test_inconsistent_flows_leave_holes_for_network(self):
        for target in (0, 1):
            with self.subTest(target=target), torch.no_grad():
                frames, masks, flows = self.sample(target)
                # 往返误差平方为 0.25：旧容差接受，新容差应拒绝。
                reverse = flows.clone()
                reverse[:, :, 0] = 0.5
                original_masks = masks.clone()
                _, _, output, remaining = self.module(
                    frames, flows, reverse, masks, 'nearest')
                self.assertTrue(torch.equal(output, frames))
                self.assertTrue(torch.equal(remaining, original_masks))
                self.assertTrue(torch.equal(masks, original_masks))

                original_check = propainter.fbConsistencyCheck
                with patch.object(propainter, 'fbConsistencyCheck',
                                  side_effect=lambda fw, bw, **kw: original_check(fw, bw)):
                    _, _, old_output, old_remaining = self.module(
                        frames, flows, reverse, masks, 'nearest')
                self.assertFalse(torch.equal(old_output, output))
                self.assertLess(old_remaining.sum().item(), remaining.sum().item())

    def test_visible_pixels_and_empty_masks_are_preserved(self):
        with torch.no_grad():
            frames, masks, flows = self.sample(0)
            _, _, output, _ = self.module(frames, flows, flows, masks, 'nearest')
            self.assertTrue(torch.equal(output * (1 - masks), frames * (1 - masks)))
            empty = torch.zeros_like(masks)
            _, _, output, remaining = self.module(frames, flows, flows, empty, 'nearest')
            self.assertTrue(torch.equal(output, frames))
            self.assertTrue(torch.equal(remaining, empty))

    def test_learned_features_match_original_consistency_check(self):
        module = propainter.BidirectionalPropagation(128, learnable=True).eval()
        generator = torch.Generator().manual_seed(7)
        features = torch.randn(1, 2, 128, 8, 8, generator=generator)
        flows = torch.zeros(1, 1, 2, 8, 8)
        reverse = flows.clone()
        reverse[:, :, 0] = 0.5
        masks = torch.zeros(1, 2, 2, 8, 8)
        original_check = propainter.fbConsistencyCheck
        with torch.no_grad():
            actual = module(features, flows, reverse, masks)
            with patch.object(propainter, 'fbConsistencyCheck',
                              side_effect=lambda fw, bw, **kw: original_check(fw, bw)):
                expected = module(features, flows, reverse, masks)
        for result, baseline in zip(actual[:3], expected[:3]):
            self.assertTrue(torch.equal(result, baseline))


if __name__ == '__main__':
    unittest.main()
