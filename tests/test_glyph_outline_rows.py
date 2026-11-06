import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from backend.inpaint import glyph_pipeline as pipeline
from backend.tools.glyph_context import outlined_text_regions, outlined_local_mask


class OutlineRowTest(unittest.TestCase):
    area = (180, 330, 40, 440)
    band = (80, 430)
    local_area = (100, 250, 40, 440)

    def frame(self, dx=0, dy=0, count=4):
        frame = np.full((600, 480, 3), 150, np.uint8)
        for x in [110, 145, 180, 215][:count]:
            cv2.rectangle(frame, (x+dx-4, 232+dy), (x+dx+14, 266+dy), (0, 0, 0), -1)
            cv2.rectangle(frame, (x+dx, 236+dy), (x+dx+10, 262+dy), (255, 255, 255), -1)
        cv2.circle(frame, (252+dx, 258+dy), 6, (0, 0, 0), -1)
        cv2.circle(frame, (252+dx, 258+dy), 2, (255, 255, 255), -1)
        return frame

    def detector(self):
        # Actual local OCR is mocked; candidates/masks still use real pixels.
        detector = Mock()
        detector.detect_subtitle.return_value = [[68, 220, 10, 42]]
        return detector

    def supplement(self, frames, masks, detector, eligible):
        with patch.object(pipeline, '_read_frames', side_effect=lambda p: (f for f in frames)):
            return pipeline._supplement_outline_rows('clip', masks, detector, self.area,
                                                       self.band, eligible, Mock())

    def test_current_rows_follow_text_and_preserve_manual_bounds(self):
        first = None
        for dx, dy in [(0, 0), (17, 23)]:
            frame = self.frame(dx, dy)
            # Large, unoutlined background lettering crosses both selection edges.
            frame[160:350, 350:362] = 255
            regions = outlined_text_regions(frame, self.area)
            self.assertEqual(len(regions), 1)
            first = first or regions[0]
            self.assertEqual(regions[0][:2], [v+dy for v in first[:2]])
            self.assertEqual(regions[0][2:], list(self.area[2:]))
            self.assertTrue(self.area[0] <= regions[0][0] < regions[0][1] <= self.area[1])
            self.assertLess(regions[0][1]-regions[0][0], self.area[1]-self.area[0])

    def test_blank_background_pale_unoutlined_short_and_clipped_text_abstain(self):
        blank = np.full_like(self.frame(), 150)
        background = blank.copy()
        for x in (100, 180, 260):
            background[160:350, x:x+15] = 255
        pale, unoutlined = self.frame(), self.frame()
        pale[np.all(pale == 255, axis=2)] = 200
        unoutlined[np.all(unoutlined == 0, axis=2)] = 150
        for frame in [blank, background, pale, unoutlined, self.frame(count=2)]:
            self.assertEqual(outlined_text_regions(frame, self.area), [])
        self.assertEqual(outlined_text_regions(self.frame(), (240, 330, 40, 440)), [])

    def test_recovers_empty_and_partial_masks_without_anchors_or_mask_loss(self):
        text = self.frame()
        blank = np.full_like(text, 150)
        frames = [text, text, text, blank, text]
        masks = np.zeros((5, 350, 480), np.uint8)
        masks[:, 0, 0] = 255  # A preexisting pixel must survive, even outside area.
        masks[1, 156:183, 110:121] = 255
        full = outlined_local_mask(text[80:430], [[108, 260, 154, 186]], self.local_area)
        masks[2] |= full
        before = masks.copy()
        detector = self.detector()
        audit = self.supplement(frames, masks, detector, {0, 1, 2, 3})
        self.assertEqual(audit['checked_frames'], [0, 1, 2, 3])
        self.assertEqual(audit['frames'], [0, 1, 2])
        self.assertEqual(audit['confirmed_frames'], [0, 1, 2])
        self.assertEqual(audit['updated_frames'], [0, 1])
        self.assertEqual(audit['ocr_calls'], 3)
        self.assertFalse(np.any((before > 0) & (masks == 0)))
        for i in (0, 1):
            self.assertTrue(np.all(masks[i, 156:183, 110:121]))
            self.assertEqual(masks[i, 178, 252], 255)  # Detached period.
            self.assertFalse(masks[i, 156:183, 131:135].any())  # Keep inter-letter background.
        outside = np.ones((350, 480), bool)
        outside[100:250, 40:440] = False
        self.assertFalse(np.any((masks > before)[:, outside]))
        np.testing.assert_array_equal(masks[2:], before[2:])
        self.assertEqual(audit['region_coordinate_space'], 'ocr_band_yxyx')
        self.assertEqual(audit['regions'][0][0][2:], [40, 440])

    def test_current_ocr_confirmation_is_required_and_failure_keeps_existing_mask(self):
        for found in [[], [[0, 400, 0, 51]], [[250, 350, 10, 42]]]:
            detector = self.detector()
            detector.detect_subtitle.return_value = found
            masks = np.zeros((1, 350, 480), np.uint8)
            masks[0, 156:183, 110:121] = 255
            before = masks.copy()
            audit = self.supplement([self.frame()], masks, detector, {0})
            self.assertEqual(audit['frames'], [0])
            self.assertEqual(audit['empty_frames'], [0])
            self.assertEqual(audit['updated_frames'], [])
            np.testing.assert_array_equal(masks, before)

    def test_no_eligible_frames_skip_decode_and_truncated_decode_fails(self):
        with patch.object(pipeline, '_read_frames') as read:
            audit = pipeline._supplement_outline_rows('clip', None, Mock(), self.area,
                                                      self.band, set(), Mock())
        read.assert_not_called()
        self.assertEqual(audit['ocr_calls'], 0)
        with self.assertRaisesRegex(RuntimeError, 'Incomplete decode'):
            self.supplement([self.frame()], np.zeros((2, 350, 480), np.uint8), self.detector(), {1})

    def test_shared_pipeline_uses_initial_local_frames_and_resets_after_cut(self):
        text = self.frame()
        frames = [text, np.full_like(text, 150), np.full_like(text, 255)]
        capture = Mock()
        capture.get.side_effect = lambda key: {
            cv2.CAP_PROP_FRAME_WIDTH: 480, cv2.CAP_PROP_FRAME_HEIGHT: 600,
            cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: len(frames),
        }[key]
        hashes = []
        for mode in ('sttn', 'propainter'):
            detector = Mock(bbox_min_inside_ratio=1)
            detector._box_inside_ratio.return_value = 0
            def detect(crop):
                if crop.shape[:2] == (350, 480):
                    return [] if np.all(crop == 255) else [[0, 480, 0, 350]]
                if crop.shape[:2] == (51, 400):
                    return [[68, 220, 10, 42]]
                return []  # Original area-cropped OCR misses; no anchor can be built.
            detector.detect_subtitle.side_effect = detect
            model = Mock()
            model.inpaint.side_effect = lambda frames, masks, **kwargs: frames
            original_infer = pipeline._infer
            def infer(video, masks, *args, **kwargs):
                self.assertTrue(masks[0].any())
                self.assertFalse(masks[1:].any())
                # A recovered OCR-empty frame cannot be used as a clean reference.
                self.assertNotIn(0, args[-1])
                return original_infer(video, masks, *args, **kwargs)
            with tempfile.TemporaryDirectory() as folder, patch.object(
                    pipeline.cv2, 'VideoCapture', return_value=capture), patch.object(
                    pipeline, '_read_frames', side_effect=lambda p: (f for f in frames)), patch.object(
                    pipeline, '_make_model', return_value=(model, 'cpu')), patch.object(
                    pipeline, '_infer', side_effect=infer), patch.object(pipeline, '_render', return_value=1):
                output = pipeline.run_glyph_removal(Path(folder)/'clip.mp4', self.area,
                    dict(mode=mode, ocr_sample_step=1, sttn_band_margin=100), detector, Mock(), Mock())
                audit = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(audit['narrow_ocr']['groups'], [])
            self.assertEqual(audit['outline_ocr']['checked_frames'], [0, 1])
            self.assertEqual(audit['outline_ocr']['updated_frames'], [0])
            self.assertEqual(audit['outline_ocr']['ocr_calls'], 1)
            self.assertEqual(audit['ocr_frames'], 5)
            self.assertTrue(model.inpaint.called)
            hashes.append(audit['masks_sha256'])
        self.assertEqual(hashes[0], hashes[1])


if __name__ == '__main__':
    unittest.main()
