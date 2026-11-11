"""双语缺行补检回归：只补当前有字的行，不删除或传播已有掩膜。"""
import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from backend.inpaint import glyph_pipeline as p
from backend.tools.glyph_context import missing_bilingual_rows
from backend.tools.subtitle_detect import SubtitleDetect


class BilingualTest(unittest.TestCase):
    area = (200, 600, 40, 560)
    band = (100, 700)
    local = (100, 500, 40, 560)
    chinese = [90, 210, 145, 215]
    english = [80, 400, 345, 415]

    def case(self):
        boxes = [[self.chinese, self.english] for _ in range(6)]
        boxes[3] = boxes[4] = [self.english]
        frames = []
        masks = np.zeros((6, 600, 600), np.uint8)
        for i in range(6):
            frame = np.full((1000, 600, 3), 60, np.uint8)
            if i != 4:
                cv2.putText(frame, 'HI', (100, 300), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            cv2.putText(frame, 'GO', (90, 500), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3)
            masks[i] = p.glyph_mask(frame[100:700], boxes[i])
            masks[i, 0, 0] = 255
            frames.append(frame)
        detector = Mock(bbox_min_inside_ratio=.995)
        detector._box_inside_ratio.side_effect = SubtitleDetect._box_inside_ratio
        detector.detect_subtitle.side_effect = [[self.chinese, self.english], [self.english]]
        return frames, masks, boxes, detector

    def supplement(self, frames, masks, boxes, detector, excluded=()):
        with patch.object(p, '_read_frames', side_effect=lambda path: (f for f in frames)):
            return p._supplement_bilingual_rows('clip', masks, boxes, [(0, 6)], detector,
                self.area, self.band, 1000, 25, set(excluded), Mock())

    def test_real_current_glyph_only_and_no_box_or_existing_mask_changes(self):
        frames, masks, boxes, detector = self.case()
        old, old_boxes = masks.copy(), copy.deepcopy(boxes)
        audit = self.supplement(frames, masks, boxes, detector)
        self.assertEqual(audit['frames'], [3, 4])
        self.assertEqual(audit['updated_frames'], [3])
        self.assertEqual(audit['empty_frames'], [4])
        self.assertEqual(audit['ocr_calls'], 2)
        self.assertGreater(audit['added_pixels'][3], 0)
        self.assertFalse(np.any(masks < old))
        self.assertEqual(boxes, old_boxes)
        np.testing.assert_array_equal(masks[[0, 1, 2, 4, 5]], old[[0, 1, 2, 4, 5]])
        np.testing.assert_array_equal(masks[:, 300:], old[:, 300:])
        outside = np.ones((600, 600), bool)
        outside[100:500, 40:560] = False
        np.testing.assert_array_equal(masks[:, outside], old[:, outside])

    def test_failed_empty_or_wrong_row_detection_never_removes_masks(self):
        for found in ([], [self.english], [[0, 600, 0, 600]], [[0, 20, 145, 215]]):
            frames, masks, boxes, detector = self.case()
            detector.detect_subtitle.side_effect = None
            detector.detect_subtitle.return_value = found
            old = masks.copy()
            audit = self.supplement(frames, masks, boxes, detector)
            self.assertFalse(audit['updated_frames'])
            np.testing.assert_array_equal(masks, old)

    def test_geometry_ocr_without_current_white_pixels_adds_nothing(self):
        frames, masks, boxes, detector = self.case()
        frames[3][:] = 60
        old = masks.copy()
        self.supplement(frames, masks, boxes, detector)
        np.testing.assert_array_equal(masks, old)

    def test_no_cross_shot_excluded_anchor_or_distant_anchor(self):
        _, _, boxes, _ = self.case()
        self.assertEqual(missing_bilingual_rows(boxes, [(0, 3), (3, 6)], self.local, 1000, 25), {})
        self.assertEqual(missing_bilingual_rows(boxes, [(0, 6)], self.local, 1000, 25, {0, 1}), {})
        self.assertEqual(missing_bilingual_rows(boxes, [(0, 6)], self.local, 1000, 25, {3, 4}), {})
        long = boxes[:3]+[[self.english]]*30
        self.assertNotIn(32, missing_bilingual_rows(long, [(0, len(long))], self.local, 1000, 25))

    def test_single_jitter_multiple_rows_and_narrow_area_do_not_trigger(self):
        _, _, boxes, _ = self.case()
        cases = [([[self.english]]*6, self.local), (boxes, (130, 230, 40, 560)),
                 ([[self.chinese, [90, 210, 180, 250]]]*3+[[self.chinese]]*3, self.local),
                 ([lines+[[90, 210, 245, 295]] for lines in boxes], self.local)]
        for lines, area in cases:
            self.assertEqual(missing_bilingual_rows(lines, [(0, 6)], area, 1000, 25), {})

    def test_no_decode_when_no_targets_and_detected_blank_frames_are_not_guessed(self):
        frames, masks, boxes, detector = self.case()
        boxes[3] = boxes[4] = []
        with patch.object(p, '_read_frames') as read:
            audit = p._supplement_bilingual_rows('clip', masks, boxes, [(0, 6)], detector,
                self.area, self.band, 1000, 25, set(), Mock())
        self.assertEqual(audit['frames'], [])
        read.assert_not_called()
        detector.detect_subtitle.assert_not_called()

    def test_ocr_error_and_truncated_decode_fail_instead_of_silent_success(self):
        frames, masks, boxes, detector = self.case()
        detector.detect_subtitle.side_effect = RuntimeError('OCR failed')
        with self.assertRaises(RuntimeError):
            self.supplement(frames, masks, boxes, detector)
        frames, masks, boxes, detector = self.case()
        with self.assertRaisesRegex(RuntimeError, 'Incomplete decode'):
            self.supplement(frames[:3], masks, boxes, detector)

    def test_shared_pipeline_uses_retry_detector_before_title_guard(self):
        hashes = []
        for mode, reuse_primary in [('sttn', False), ('propainter', False), ('sttn', True)]:
            frames, masks, boxes, retry = self.case()
            capture = Mock()
            capture.get.side_effect = [600, 1000, 25, 6]
            primary = retry if reuse_primary else Mock(bbox_min_inside_ratio=.995)
            def scan(video, det, area, band, step, raw, progress, expected, local):
                raw.write_bytes(masks.tobytes())
                local.update(frames=[], trigger_frames=[])
                return boxes, [(0, 6)], 6
            guard = p.protect_background_titles
            def check_guard(video, actual, *args):
                self.assertTrue(actual[3, 145:215, 90:210].any())
                self.assertFalse(actual[4, 145:215, 90:210].any())
                return guard(video, actual, *args)
            with tempfile.TemporaryDirectory() as tmp, patch.object(p.cv2, 'VideoCapture', return_value=capture), \
                    patch.object(p, '_scan', side_effect=scan), \
                    patch.object(p, '_read_frames', side_effect=lambda path: (f for f in frames)), \
                    patch.object(p, 'protect_background_titles', side_effect=check_guard), \
                    patch.object(p, '_infer', return_value=([], 'cpu')), patch.object(p, '_render', return_value=6):
                result = p.run_glyph_removal(Path(tmp)/'clip.mp4', self.area,
                    dict(mode=mode, sttn_band_margin=100, ocr_sample_step=1), primary, Mock(), Mock(),
                    None if reuse_primary else retry)
                audit = json.loads(result.with_suffix('.json').read_text())
            self.assertEqual(audit['bilingual_ocr']['updated_frames'], [3])
            self.assertEqual(audit['bilingual_ocr']['empty_frames'], [4])
            self.assertEqual(audit['background_title_guard']['protected_frames'], [])
            self.assertEqual(audit['ocr_frames'], 8)
            if not reuse_primary:
                primary.detect_subtitle.assert_not_called()
            hashes.append(audit['masks_sha256'])
        self.assertEqual(len(set(hashes)), 1)


if __name__ == '__main__':
    unittest.main()
