"""转场拖影需由两侧稳定字幕、背景位移和当前帧笔画共同确认。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from backend.tools.glyph_masks import glyph_mask
from backend.tools.glyph_transition import supplement_transition_glyphs
from backend.inpaint import glyph_pipeline as pipeline


class TransitionTest(unittest.TestCase):
    width, height = 600, 1000
    band = (500, 950)
    area = (680, 870, 40, 560)
    shots = [(0, 7), (7, 8), (8, 9), (9, 14)]
    lines = [[120, 480, 195, 240], [160, 450, 275, 320]]

    def scene(self, rng, words):
        background = cv2.resize(rng.integers(30, 210, (25, 15, 3), dtype=np.uint8),
                                (self.width, self.height), interpolation=cv2.INTER_LINEAR)
        text = background.copy()
        cv2.putText(text, words[0], (130, 730), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(text, words[1], (170, 810), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                    (255, 255, 255), 2, cv2.LINE_AA)
        return background, text

    def case(self, blank=False, direction=-1):
        rng = np.random.default_rng(42)
        bg_a, first = self.scene(rng, ('FIRST CAPTION', 'OLD WORDS'))
        bg_b, last = self.scene(rng, ('SECOND LINE', 'NEW WORDS'))
        frames = [first.copy() for _ in range(4)]
        for distance in (55, 55, 150, 310, 460, 520, 520):
            a, b = (bg_a, bg_b) if blank else (first, last)
            if direction < 0:
                aa = cv2.warpAffine(a, np.float32([[1, 0, -distance], [0, 1, 0]]),
                                    (self.width, self.height))
                bb = cv2.warpAffine(b, np.float32([[1, 0, self.width-distance], [0, 1, 0]]),
                                    (self.width, self.height))
                aa[:, self.width-distance:] = bb[:, self.width-distance:]
            else:
                aa = cv2.warpAffine(a, np.float32([[1, 0, distance], [0, 1, 0]]),
                                    (self.width, self.height))
                bb = cv2.warpAffine(b, np.float32([[1, 0, distance-self.width], [0, 1, 0]]),
                                    (self.width, self.height))
                aa[:, :distance] = bb[:, :distance]
            frames.append(cv2.blur(aa, (25, 1)))
        frames += [last.copy() for _ in range(3)]
        boxes = [copy.deepcopy(self.lines) for _ in range(4)]+[[] for _ in range(7)]+\
                [copy.deepcopy(self.lines) for _ in range(3)]
        masks = np.stack([glyph_mask(frame[self.band[0]:self.band[1]], b)
                          for frame, b in zip(frames, boxes)])
        return frames, boxes, masks

    def run_case(self, frames, boxes, masks, shots=None, area=None, excluded=(), read=None):
        return supplement_transition_glyphs('clip', masks, boxes, shots or self.shots,
            area or self.area, self.band, self.height, 30, excluded,
            read or (lambda _: (frame for frame in frames)), Mock())

    def test_moving_text_recovered_in_both_directions_and_clipped(self):
        for direction in (-1, 1):
            with self.subTest(direction=direction):
                frames, boxes, masks = self.case(direction=direction)
                before, original_boxes = masks.copy(), copy.deepcopy(boxes)
                audit = self.run_case(frames, boxes, masks)
                self.assertEqual(audit['updated_frames'], list(range(4, 11)))
                self.assertFalse(np.any(masks < before))
                np.testing.assert_array_equal(masks[:4], before[:4])
                np.testing.assert_array_equal(masks[11:], before[11:])
                outside = np.ones(masks.shape[1:], bool)
                outside[180:370, 40:560] = False
                np.testing.assert_array_equal(masks[:, outside], before[:, outside])
                self.assertEqual(boxes, original_boxes)

    def test_moving_background_without_text_is_not_erased(self):
        frames, boxes, masks = self.case(blank=True)
        before = masks.copy()
        audit = self.run_case(frames, boxes, masks)
        self.assertTrue(audit['events'])
        self.assertFalse(audit['updated_frames'])
        np.testing.assert_array_equal(masks, before)

    def test_one_sided_evidence_is_not_enough(self):
        frames, boxes, masks = self.case()
        background_frames, _, _ = self.case(blank=True)
        for i, distance in enumerate((55, 55, 150, 310, 460, 520, 520), start=4):
            boundary = self.width-distance
            frames[i][:, boundary:] = background_frames[i][:, boundary:]
        before = masks.copy()
        audit = self.run_case(frames, boxes, masks)
        self.assertTrue(audit['events'])
        self.assertFalse(audit['updated_frames'])
        np.testing.assert_array_equal(masks, before)

    def test_no_cut_no_stable_anchor_and_excluded_frames_do_not_decode(self):
        for kind in ('no_cut', 'no_anchor', 'excluded', 'short_gap'):
            with self.subTest(kind=kind):
                frames, boxes, masks = self.case()
                shots, excluded = self.shots, ()
                if kind == 'no_cut':
                    shots = [(0, len(boxes))]
                elif kind == 'no_anchor':
                    boxes[0] = boxes[1] = []
                elif kind == 'excluded':
                    excluded = (7,)
                else:
                    boxes[4:11] = [copy.deepcopy(self.lines) for _ in range(7)]
                    masks[4:11] = masks[3]
                reader = Mock()
                audit = self.run_case(frames, boxes, masks, shots=shots, excluded=excluded, read=reader)
                self.assertFalse(audit['events'])
                reader.assert_not_called()

    def test_truncated_decode_fails_and_closes_generator(self):
        frames, boxes, masks = self.case()
        closed = []
        def read(_):
            try:
                yield from frames[:10]
            finally:
                closed.append(True)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete decode'):
            self.run_case(frames, boxes, masks, read=read)
        self.assertEqual(closed, [True])

    def test_both_models_receive_transition_masks_before_title_guard(self):
        hashes = []
        for mode in ('sttn', 'propainter'):
            frames, old_boxes, _ = self.case()
            boxes = [[[x0, x1, y0-80, y1-80] for x0, x1, y0, y1 in row]
                     for row in old_boxes]
            raw_masks = np.stack([glyph_mask(frame[580:970], row)
                                  for frame, row in zip(frames, boxes)])
            capture = Mock()
            capture.get.side_effect = [600, 1000, 30, len(frames)]

            def scan(video, detector, area, band, step, raw, progress, expected, local):
                self.assertEqual(band, (580, 970))
                raw.write_bytes(raw_masks.tobytes())
                local.update(frames=[], trigger_frames=[])
                return boxes, self.shots, len(frames)

            def guard(video, masks, *args):
                self.assertTrue(all(masks[i].any() for i in range(4, 11)))
                return {'ocr_calls': 0, 'protected_frames': []}

            def infer(video, masks, *args):
                hashes.append(pipeline._mask_hash(masks))
                return [], 'cpu'

            with tempfile.TemporaryDirectory() as tmp, \
                    patch.object(pipeline.cv2, 'VideoCapture', return_value=capture), \
                    patch.object(pipeline, '_scan', side_effect=scan), \
                    patch.object(pipeline, '_read_frames', side_effect=lambda path: (f for f in frames)), \
                    patch.object(pipeline, '_retry_short_lines', return_value={'frames': [], 'regions': {}}), \
                    patch.object(pipeline, '_retry_narrow_lines', return_value={'frames': [], 'regions': {}, 'ocr_calls': 0}), \
                    patch.object(pipeline, '_retry_shrunk_lines', return_value={'frames': [], 'updated_frames': []}), \
                    patch.object(pipeline, '_supplement_outline_rows', return_value={'ocr_calls': 0}), \
                    patch.object(pipeline, '_supplement_bilingual_rows', return_value={'ocr_calls': 0}), \
                    patch.object(pipeline, 'protect_background_titles', side_effect=guard), \
                    patch.object(pipeline, '_infer', side_effect=infer), \
                    patch.object(pipeline, '_render', return_value=14):
                output = pipeline.run_glyph_removal(Path(tmp)/'clip.mp4', self.area,
                    dict(mode=mode, sttn_band_margin=100, ocr_sample_step=1), Mock(), Mock(), Mock())
                audit = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(audit['transition_recovery']['updated_frames'], list(range(4, 11)))
        self.assertEqual(len(set(hashes)), 1)


if __name__ == '__main__':
    unittest.main()
