"""覆盖双行学习、重叠字幕否决、连续帧确认及两种模型共用的接入位置。"""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np

from backend.tools.glyph_title_guard import protect_background_titles
from backend.inpaint import glyph_pipeline as pipeline


class TitleGuardTest(unittest.TestCase):
    area = (200, 600, 0, 600)
    band = (100, 700)
    small = [[150, 400, 150, 190], [150, 400, 350, 390]]
    large = [[50, 550, 100, 220], [50, 550, 260, 390]]

    def setup_case(self, title_count=4):
        boxes = [self.small]*4+[self.large]*title_count
        frames, masks = [], np.zeros((len(boxes), 600, 600), np.uint8)
        for i, lines in enumerate(boxes):
            frame = np.zeros((1000, 600, 3), np.uint8)
            for x0, x1, y0, y1 in lines:
                frame[100+y0:100+y1, x0:x1] = 255
                masks[i, y0:y1, x0:x1] = 255
            frames.append(frame)
        detector = Mock()
        detector.detect_subtitle.return_value = self.large
        retry = Mock()
        retry.detect_subtitle.return_value = []
        return frames, masks, boxes, detector, retry

    def run_guard(self, frames, masks, boxes, detector, retry, shots=None, excluded=(), area=None):
        return protect_background_titles('video', masks, boxes, shots or [(0, 4), (4, len(boxes))],
            area or self.area, self.band, 1000, detector, retry, set(excluded),
            lambda p: (f for f in frames), Mock())

    def test_stable_titles_keep_earlier_subtitle_masks_and_record_audit(self):
        frames, masks, boxes, det, retry = self.setup_case()
        old = masks.copy()
        audit = self.run_guard(frames, masks, boxes, det, retry)
        self.assertEqual(audit['protected_frames'], [4, 5, 6, 7])
        self.assertEqual(audit['ocr_calls'], 12)
        np.testing.assert_array_equal(masks[:4], old[:4])
        self.assertFalse(masks[4:].any())
        self.assertEqual(audit['regions'][4], [[134, 206, 0, 600], [334, 406, 0, 600]])

    def test_actual_local_subtitle_vetoes_whole_frame_protection(self):
        frames, masks, boxes, det, retry = self.setup_case()
        # 标题和字幕合并成大框时，宽带检测无法否决；局部检测必须阻止清除。
        retry.detect_subtitle.return_value = [[100, 300, 12, 52]]
        before = masks.copy()
        audit = self.run_guard(frames, masks, boxes, det, retry)
        self.assertEqual(audit['veto_frames'], [4, 5, 6, 7])
        np.testing.assert_array_equal(masks, before)

    def test_moved_normal_text_also_vetoes_protection(self):
        frames, masks, boxes, det, retry = self.setup_case()
        for frame in frames[4:]:
            frame[330:370, 100:300] = 255
        det.detect_subtitle.return_value = self.large+[[100, 300, 230, 270]]
        before = masks.copy()
        audit = self.run_guard(frames, masks, boxes, det, retry)
        self.assertEqual(audit['candidate_frames'], [])
        retry.detect_subtitle.assert_not_called()
        np.testing.assert_array_equal(masks, before)

    def test_no_protection_for_single_line_jitter_pale_or_missing_history(self):
        for variant in ('single', 'jitter', 'pale', 'no_history'):
            frames, masks, boxes, det, retry = self.setup_case()
            for i in range(4):
                if variant == 'single':
                    boxes[i] = self.small[:1]
                elif variant == 'jitter':
                    boxes[i] = [self.small[0], [150, 400, 180, 220]]
                elif variant == 'pale':
                    frames[i][frames[i] == 255] = 180
                else:
                    boxes[i] = []
            old = masks.copy()
            audit = self.run_guard(frames, masks, boxes, det, retry)
            self.assertEqual(audit['protected_frames'], [], variant)
            np.testing.assert_array_equal(masks, old)

    def test_no_propagation_across_cut_or_unconfirmed_frame(self):
        for shots, excluded in [([(0, 4), (4, 6), (6, 8)], ()),
                                ([(0, 4), (4, 8)], (5,))]:
            frames, masks, boxes, det, retry = self.setup_case()
            old = masks.copy()
            audit = self.run_guard(frames, masks, boxes, det, retry, shots, excluded)
            self.assertFalse(audit['protected_frames'])
            np.testing.assert_array_equal(masks, old)

    def test_no_extra_decode_for_narrow_area_or_no_large_boxes(self):
        for area, boxes in [((200, 300, 0, 600), [self.large]*5), (self.area, [self.small]*5)]:
            read = Mock()
            audit = protect_background_titles('clip', np.zeros((5, 600, 600), np.uint8), boxes,
                [(0, 5)], area, self.band, 1000, Mock(), Mock(), set(), read, Mock())
            read.assert_not_called()
            self.assertFalse(audit['protected_frames'])

    def test_three_supported_rows_invalidate_earlier_two_row_profile(self):
        frames, masks, boxes, det, retry = self.setup_case()
        extra = [150, 400, 250, 290]
        for i in range(1, 4):
            boxes[i] = self.small+[extra]
            frames[i][350:390, 150:400] = 255
            masks[i, 250:290, 150:400] = 255
        before = masks.copy()
        audit = self.run_guard(frames, masks, boxes, det, retry)
        self.assertEqual(audit['protected_frames'], [])
        np.testing.assert_array_equal(masks, before)

    def test_failed_ocr_and_truncated_decode_leave_masks_untouched(self):
        for failure in ('ocr', 'decode'):
            frames, masks, boxes, det, retry = self.setup_case()
            if failure == 'ocr':
                retry.detect_subtitle.side_effect = RuntimeError('OCR failed')
            else:
                frames = frames[:-1]
            old = masks.copy()
            with self.assertRaises(RuntimeError):
                self.run_guard(frames, masks, boxes, det, retry)
            np.testing.assert_array_equal(masks, old)

    def test_both_models_receive_final_protected_masks(self):
        hashes = []
        for mode in ('sttn', 'propainter'):
            frames, masks, boxes, det, retry = self.setup_case()
            capture = Mock()
            capture.get.side_effect = [600, 1000, 25, len(frames)]
            def scan(video, detector, area, band, step, raw, progress, expected, local):
                raw.write_bytes(masks.tobytes())
                local.update(frames=[], trigger_frames=[])
                return boxes, [(0, 4), (4, 8)], 8
            def infer(video, actual, jobs, *args):
                self.assertTrue(actual[:4].any())
                self.assertFalse(actual[4:].any())
                self.assertTrue(all(j['stop'] <= 4 for j in jobs))
                return [], 'cpu'
            with tempfile.TemporaryDirectory() as tmp, patch.object(pipeline.cv2, 'VideoCapture', return_value=capture), \
                    patch.object(pipeline, '_scan', side_effect=scan), \
                    patch.object(pipeline, '_read_frames', side_effect=lambda p: (f for f in frames)), \
                    patch.object(pipeline, '_infer', side_effect=infer), \
                    patch.object(pipeline, '_render', return_value=4):
                result = pipeline.run_glyph_removal(Path(tmp)/'clip.mp4', self.area,
                    dict(mode=mode, sttn_band_margin=100, ocr_sample_step=1), det, Mock(), Mock(), retry)
                audit = json.loads(result.with_suffix('.json').read_text())
            self.assertEqual(audit['background_title_guard']['protected_frames'], [4, 5, 6, 7])
            hashes.append(audit['masks_sha256'])
        self.assertEqual(hashes[0], hashes[1])


if __name__ == '__main__':
    unittest.main()
