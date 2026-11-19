"""静止片头保护：字幕优先、逐帧确认与异常回退。"""
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from backend.tools.glyph_opening_title import protect_opening_title, _uncovered_outlined_stroke
from backend.tools.glyph_masks import glyph_mask
from backend.inpaint import glyph_pipeline as pipeline


class OpeningTitleTest(unittest.TestCase):
    band = (0, 1000)
    area = (100, 800, 0, 600)
    titles = [[30, 570, 180, 270], [30, 570, 330, 460]]
    subtitle = [145, 370, 245, 285]

    def fixture(self, text=None, count=5):
        frame = np.zeros((1000, 600, 3), np.uint8)
        for x0, x1, y0, y1 in self.titles:
            cv2.rectangle(frame, (x0, y0), (x1-1, y1-1), (0, 160, 230), -1)
        if text:
            cv2.putText(frame, text, (150, 275), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, text, (150, 275), cv2.FONT_HERSHEY_SIMPLEX, .9, (255, 255, 255), 2, cv2.LINE_AA)
        masks = np.zeros((count, 1000, 600), np.uint8)
        for x0, x1, y0, y1 in self.titles:
            masks[:, y0:y1, x0:x1] = 255
        if text:
            masks[:, 245:285, 145:370] = 255
        return [frame.copy() for _ in range(count)], masks

    def run_guard(self, frames, masks, found, primary=None, shots=None, excluded=(), area=None):
        detector = Mock()
        detector.detect_subtitle.return_value = found
        audit = protect_opening_title('video', masks, [primary or self.titles]*len(masks),
            shots or [(0, len(masks))], area or self.area, self.band, 1000, 25,
            detector, set(excluded), lambda _: (f for f in frames), Mock())
        return audit, detector

    def test_static_title_without_subtitle_is_preserved(self):
        frames, masks = self.fixture()
        audit, det = self.run_guard(frames, masks, self.titles)
        self.assertEqual(audit['protected_frames'], list(range(5)))
        self.assertEqual(audit['subtitle_frames'], [])
        self.assertEqual(det.detect_subtitle.call_count, 5)
        self.assertFalse(masks.any())

    def test_overlapping_subtitle_retains_actual_glyphs(self):
        frames, masks = self.fixture('HELLO')
        truth = np.zeros((1000, 600), np.uint8)
        cv2.putText(truth, 'HELLO', (150, 275), cv2.FONT_HERSHEY_SIMPLEX, .9, 255, 5, cv2.LINE_AA)
        audit, _ = self.run_guard(frames, masks, self.titles+[self.subtitle], self.titles+[self.subtitle])
        self.assertEqual(audit['protected_frames'], list(range(5)))
        self.assertEqual(audit['subtitle_frames'], list(range(5)))
        self.assertGreater(np.count_nonzero(masks[0]), 0)
        self.assertEqual(np.count_nonzero((truth > 0) & (masks[0] == 0)), 0)
        self.assertFalse(masks[0, 200:240].any())

    def test_undetected_short_caption_or_unknown_color_falls_back(self):
        for text, found in [('I', self.titles), ('HELLO', self.titles+[[145, 370, 245, 305]])]:
            frames, masks = self.fixture(text)
            if text == 'HELLO':
                # 不把未满足白色字幕判断的检测框当成背景片名删除。
                frames = [np.where(f == 255, 180, f).astype('uint8') for f in frames]
            before = masks.copy()
            audit, _ = self.run_guard(frames, masks, found)
            self.assertEqual(audit['protected_frames'], [], text)
            np.testing.assert_array_equal(masks, before)

    def test_uncovered_colored_or_dim_outlined_text_vetoes_white_mask(self):
        for color in [(0, 0, 255), (0, 220, 255), (170, 170, 170)]:
            frame = np.zeros((1000, 600, 3), np.uint8)
            cv2.putText(frame, 'HELLO', (150, 275), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(frame, 'HELLO', (150, 275), cv2.FONT_HERSHEY_SIMPLEX, .9, color, 2, cv2.LINE_AA)
            mask = glyph_mask(frame, [self.subtitle])
            self.assertTrue(_uncovered_outlined_stroke(frame, [self.subtitle], mask, self.area, 1000), color)

    def test_title_highlights_do_not_hide_colored_subtitle(self):
        for color in [(0, 0, 255), (0, 220, 255), (170, 170, 170)]:
            frames, masks = self.fixture()
            for frame in frames:
                cv2.rectangle(frame, (320, 245), (348, 274), (0, 0, 0), -1)
                cv2.rectangle(frame, (323, 248), (345, 271), (255, 255, 255), -1)
                cv2.putText(frame, 'HELLO', (150, 275), cv2.FONT_HERSHEY_SIMPLEX, .9, (0, 0, 0), 5, cv2.LINE_AA)
                cv2.putText(frame, 'HELLO', (150, 275), cv2.FONT_HERSHEY_SIMPLEX, .9, color, 2, cv2.LINE_AA)
            before = masks.copy()
            lines = self.titles+[self.subtitle]
            audit, _ = self.run_guard(frames, masks, lines, lines)
            self.assertFalse(audit['protected_frames'], color)
            np.testing.assert_array_equal(masks, before)

    def test_excluded_or_unstable_frames_do_not_form_confirmation_run(self):
        frames, masks = self.fixture()
        before = masks.copy()
        audit, _ = self.run_guard(frames, masks, self.titles, excluded={2})
        self.assertEqual(audit['protected_frames'], [])
        np.testing.assert_array_equal(masks, before)
        frames, masks = self.fixture()
        frames[2][100:800] = 0
        before = masks.copy()
        audit, _ = self.run_guard(frames, masks, self.titles)
        self.assertEqual(audit['protected_frames'], [])
        np.testing.assert_array_equal(masks, before)

    def test_narrow_selection_no_title_and_other_shots_do_not_trigger(self):
        frames, masks = self.fixture()
        for area, shots, primary in [((280, 320, 145, 370), None, None),
                                     (None, [(0, 2), (2, 5)], None),
                                     (None, None, [[100, 200, 280, 320]])]:
            current = masks.copy()
            audit, det = self.run_guard(frames, current, self.titles, primary, shots, area=area)
            det.detect_subtitle.assert_not_called()
            self.assertEqual(audit['protected_frames'], [])
            np.testing.assert_array_equal(current, masks)

    def test_truncated_decode_does_not_modify_masks(self):
        frames, masks = self.fixture()
        before = masks.copy()
        with self.assertRaises(RuntimeError):
            self.run_guard(frames[:-1], masks, self.titles)
        np.testing.assert_array_equal(masks, before)

    def test_ocr_failure_does_not_partially_modify_masks(self):
        frames, masks = self.fixture()
        before = masks.copy()
        detector = Mock()
        detector.detect_subtitle.side_effect = [self.titles, self.titles, RuntimeError('OCR failed')]
        with self.assertRaises(RuntimeError):
            protect_opening_title('video', masks, [self.titles]*5, [(0, 5)],
                self.area, self.band, 1000, 25, detector, set(),
                lambda _: (f for f in frames), Mock())
        np.testing.assert_array_equal(masks, before)


    def test_pipeline_passes_protected_masks_to_both_models(self):
        frames, raw = self.fixture(count=5)
        for _ in range(3):
            f = np.zeros((1000, 600, 3), np.uint8)
            f[600:640, 150:400] = 255
            frames.append(f)
        final_raw = np.zeros((8, 900, 600), np.uint8)
        final_raw[:5] = raw[:, :900]
        final_raw[5:, 600:640, 150:400] = 255
        boxes = [self.titles]*5+[[[150, 400, 600, 640]]]*3
        hashes = []
        for mode in ('sttn', 'propainter'):
            capture = Mock()
            capture.get.side_effect = [600, 1000, 25, 8]
            retry = Mock()
            retry.detect_subtitle.return_value = self.titles
            def scan(video, detector, area, band, step, output, progress, expected, local):
                output.write_bytes(final_raw.tobytes())
                local.update(frames=[], trigger_frames=[])
                return boxes, [(0, 5), (5, 8)], 8
            def infer(video, actual, jobs, *args):
                self.assertFalse(actual[:5].any())
                self.assertTrue(actual[5:].any())
                return [], 'cpu'
            replacements = {
                '_scan': scan,
                '_read_frames': lambda _: (f for f in frames),
                '_retry_short_lines': Mock(return_value=dict(updated_frames=[])),
                '_retry_narrow_lines': Mock(return_value=dict(frames=[], regions={}, ocr_calls=0)),
                '_recover_local_gaps': Mock(return_value=[]),
                '_retry_shrunk_lines': Mock(return_value=dict(frames=[], updated_frames=[])),
                'recover_vertical_border': lambda frame, boxes, stable: np.zeros(stable.shape, np.uint8),
                '_supplement_outline_rows': Mock(return_value=dict(updated_frames=[], ocr_calls=0)),
                '_supplement_bilingual_rows': Mock(return_value=dict(updated_frames=[], ocr_calls=0)),
                'supplement_fading_glyphs': Mock(return_value=dict(updated_frames=[])),
                'supplement_transition_glyphs': Mock(return_value=dict(updated_frames=[])),
                'protect_background_titles': Mock(return_value=dict(ocr_calls=0)),
                '_infer': infer,
                '_render': Mock(return_value=3),
            }
            with tempfile.TemporaryDirectory() as tmp, ExitStack() as stack:
                stack.enter_context(patch.object(pipeline.cv2, 'VideoCapture', return_value=capture))
                for name, replacement in replacements.items():
                    stack.enter_context(patch.object(pipeline, name, replacement))
                result = pipeline.run_glyph_removal(Path(tmp)/'clip.mp4', self.area,
                    dict(mode=mode, sttn_band_margin=100, ocr_sample_step=1), Mock(), Mock(), Mock(), retry)
                audit = json.loads(result.with_suffix('.json').read_text())
            self.assertEqual(audit['opening_title_guard']['protected_frames'], list(range(5)))
            hashes.append(audit['masks_sha256'])
        self.assertEqual(hashes[0], hashes[1])


if __name__ == '__main__':
    unittest.main()
