"""淡出补齐回归：当前笔画确认、范围约束及两模型的正式接入。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from backend.inpaint import glyph_pipeline as p
from backend.tools.glyph_fade import supplement_fading_glyphs


class FadeTest(unittest.TestCase):
    area = (180, 400, 30, 570)
    band = (80, 500)
    box = [70, 480, 156, 219]

    def case(self):
        background = np.full((1000, 600, 3), 60, np.uint8)
        reference = background.copy()
        cv2.putText(reference, 'HELLO WORLD', (80, 290), cv2.FONT_HERSHEY_SIMPLEX, 1.7,
                    (255, 255, 255), 3, cv2.LINE_AA)
        frames = [reference.copy() for _ in range(4)]
        frames += [(background*.7+reference*.3).astype(np.uint8), background.copy()]
        boxes = [[self.box.copy()] for _ in frames]
        masks = np.stack([p.glyph_mask(f[80:500], b) for f, b in zip(frames, boxes)])
        return frames, masks, boxes

    def run_fade(self, frames, masks, boxes, shots=None, area=None, excluded=()):
        return supplement_fading_glyphs('clip', masks, boxes, shots or [(0, len(boxes))],
            area or self.area, self.band, 1000, 25, excluded,
            lambda path: (f for f in frames), Mock())

    def test_fade_recovered_but_false_ocr_on_blank_stays_empty(self):
        frames, masks, boxes = self.case()
        old, original_boxes = masks.copy(), copy.deepcopy(boxes)
        audit = self.run_fade(frames, masks, boxes)
        self.assertEqual(audit['updated_frames'], [4])
        self.assertEqual(audit['rejected_frames'], [5])
        self.assertEqual(audit['ocr_calls'], 0)
        self.assertGreater(np.count_nonzero(masks[4]), .9*np.count_nonzero(old[0]))
        self.assertFalse(np.any(masks < old))
        np.testing.assert_array_equal(masks[[0, 1, 2, 3, 5]], old[[0, 1, 2, 3, 5]])
        self.assertEqual(boxes, original_boxes)

    def test_changed_moved_blurred_text_and_highlight_are_rejected(self):
        for kind in ('changed', 'moved', 'blurred', 'highlight', 'bright', 'faint'):
            with self.subTest(kind=kind):
                frames, masks, boxes = self.case()
                reference = frames[0]
                current = np.full_like(reference, 60)
                if kind == 'changed':
                    cv2.putText(current, 'OTHER WORDS', (80, 290), cv2.FONT_HERSHEY_SIMPLEX,
                                1.7, (120, 120, 120), 3, cv2.LINE_AA)
                elif kind == 'moved':
                    current = np.roll(frames[4], 16, axis=1)
                elif kind == 'blurred':
                    current = cv2.blur(frames[4], (25, 1))
                elif kind == 'highlight':
                    current[235:295, 80:470] = 150
                elif kind == 'bright':
                    current = reference.copy()
                else:
                    current = (reference*.03+current*.97).astype(np.uint8)
                frames[4] = current
                old = masks.copy()
                audit = self.run_fade(frames, masks, boxes)
                self.assertFalse(audit['updated_frames'])
                np.testing.assert_array_equal(masks, old)

    def test_missing_box_no_anchor_cut_expiry_and_local_paths_do_not_propagate(self):
        for kind in ('missing', 'cut', 'excluded_target', 'excluded_anchors', 'expiry', 'few_anchors'):
            with self.subTest(kind=kind):
                frames, masks, boxes = self.case()
                shots, excluded = [(0, 6)], []
                if kind == 'missing':
                    boxes[4] = boxes[5] = []
                elif kind == 'cut':
                    shots = [(0, 4), (4, 6)]
                elif kind == 'excluded_target':
                    excluded = [4, 5]
                elif kind == 'excluded_anchors':
                    excluded = [0, 1, 2, 3]
                elif kind == 'few_anchors':
                    boxes[0] = boxes[1] = []
                else:
                    boxes = boxes[:4]+[[]]*12+boxes[4:]
                    masks = np.concatenate([masks[:4], np.zeros((12, *masks.shape[1:]), np.uint8), masks[4:]])
                    shots = [(0, len(boxes))]
                read = Mock()
                audit = supplement_fading_glyphs('clip', masks, boxes, shots, self.area,
                    self.band, 1000, 25, excluded, read, Mock())
                self.assertFalse(audit['frames'])
                read.assert_not_called()

    def test_added_pixels_stay_inside_manual_area(self):
        frames, masks, boxes = self.case()
        area = (236, 299, 70, 480)
        old = masks.copy()
        self.run_fade(frames, masks, boxes, area=area)
        outside = np.ones(masks.shape[1:], bool)
        outside[156:219, 70:480] = False
        np.testing.assert_array_equal(masks[:, outside], old[:, outside])
        self.assertTrue(masks[4].any())

    def test_large_background_row_pale_anchor_and_single_symbol_are_rejected(self):
        frames, masks, boxes = self.case()
        for test_boxes in ([[[70, 480, 120, 320]]]*6, [[[70, 90, 156, 219]]]*6):
            read = Mock()
            result = supplement_fading_glyphs('clip', masks, test_boxes, [(0, 6)], self.area,
                self.band, 1000, 25, (), read, Mock())
            self.assertFalse(result['frames'])
            read.assert_not_called()
        for i in range(4):
            frames[i] = (frames[i]*.6).astype(np.uint8)
        old = masks.copy()
        self.assertFalse(self.run_fade(frames, masks, boxes)['updated_frames'])
        np.testing.assert_array_equal(masks, old)

    def test_recovered_pixels_do_not_renew_anchors(self):
        frames, masks, boxes = self.case()
        frames = frames[:4]+[frames[4]]*15
        masks = np.concatenate([masks[:4], np.repeat(masks[4:5], 15, axis=0)])
        boxes = [[self.box.copy()] for _ in frames]
        audit = self.run_fade(frames, masks, boxes)
        self.assertTrue(audit['updated_frames'])
        self.assertLess(max(audit['updated_frames']), 10)
        self.assertFalse(masks[-1].any())

    def test_truncated_decode_fails_and_generator_is_closed(self):
        frames, masks, boxes = self.case()
        closed = []
        def read(path):
            try:
                yield from frames[:4]
            finally:
                closed.append(True)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete decode'):
            supplement_fading_glyphs('clip', masks, boxes, [(0, 6)], self.area, self.band,
                1000, 25, (), read, Mock())
        self.assertEqual(closed, [True])

    def test_low_fps_does_not_extend_history_to_several_seconds(self):
        _, masks, boxes = self.case()
        read = Mock()
        audit = supplement_fading_glyphs('clip', masks, boxes, [(0, 6)], self.area,
            self.band, 1000, 1, (), read, Mock())
        self.assertFalse(audit['frames'])
        read.assert_not_called()

    def test_both_engines_receive_recovered_masks_before_title_guard(self):
        hashes = []
        for mode in ('sttn', 'propainter'):
            frames, masks, boxes = self.case()
            capture = Mock()
            capture.get.side_effect = [600, 1000, 25, 6]
            def scan(video, det, area, band, step, raw, progress, expected, local):
                raw.write_bytes(masks.tobytes())
                local.update(frames=[], trigger_frames=[])
                return boxes, [(0, 6)], 6
            def guard(video, actual, *args):
                self.assertTrue(actual[4].any())
                self.assertFalse(actual[5].any())
                return {'ocr_calls': 0, 'protected_frames': []}
            def infer(video, actual, *args):
                hashes.append(p._mask_hash(actual))
                return [], 'cpu'
            with tempfile.TemporaryDirectory() as tmp, patch.object(p.cv2, 'VideoCapture', return_value=capture), \
                    patch.object(p, '_scan', side_effect=scan), \
                    patch.object(p, '_read_frames', side_effect=lambda path: (f for f in frames)), \
                    patch.object(p, 'protect_background_titles', side_effect=guard), \
                    patch.object(p, '_infer', side_effect=infer), patch.object(p, '_render', return_value=5):
                result = p.run_glyph_removal(Path(tmp)/'clip.mp4', self.area,
                    dict(mode=mode, sttn_band_margin=100, ocr_sample_step=1), Mock(), Mock(), Mock())
                audit = json.loads(result.with_suffix('.json').read_text())
            self.assertEqual(audit['fade_recovery']['updated_frames'], [4])
            self.assertEqual(audit['profile'], 'glyph_v5')
        self.assertEqual(len(set(hashes)), 1)


if __name__ == '__main__':
    unittest.main()
