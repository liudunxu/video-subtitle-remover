import tempfile
import unittest
import json
from pathlib import Path
from unittest.mock import Mock, patch
from copy import deepcopy

import cv2
import numpy as np

from backend.tools.glyph_context import (
    needs_local_ocr, local_ocr_boxes, outlined_local_mask, select_reference,
    short_text_regions, short_text_mask, photometric_boundaries,
    narrow_text_regions,
)
from backend.inpaint import glyph_pipeline as pipeline


class LocalOCRTest(unittest.TestCase):
    def test_trigger_requires_tall_overlapping_box(self):
        area = (90, 150, 40, 400)
        self.assertTrue(needs_local_ocr([[30, 410, 40, 200]], area))
        for coords in ([], [[50, 300, 100, 140]], [[0, 20, 40, 200]], [[50, 300, 145, 160]]):
            self.assertFalse(needs_local_ocr(coords, area))

    def test_local_crop_offsets_and_background_rejection(self):
        frame = np.zeros((240, 432, 3), np.uint8)
        detector = Mock()
        detector.detect_subtitle.return_value = [[10, 150, 10, 50], [0, 200, 0, 60], [10, 50, 5, 55]]
        boxes = local_ocr_boxes(frame, detector, (90, 150, 40, 400))
        self.assertEqual(boxes, [[50, 190, 100, 140]])
        self.assertEqual(detector.detect_subtitle.call_args.args[0].shape, (60, 360, 3))

    def test_local_mask_preserves_unoutlined_bright_background_and_bounds(self):
        frame = np.full((100, 220, 3), 150, np.uint8)
        frame[33:55, 95:101] = 255  # Unoutlined background stroke.
        cv2.putText(frame, 'Hi.', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 7)
        cv2.putText(frame, 'Hi.', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        mask = outlined_local_mask(frame, [[20, 70, 35, 64]], (20, 80, 10, 120))
        self.assertTrue(mask[35:60, 25:60].any())
        self.assertFalse(mask[33:55, 95:101].any())
        self.assertFalse(mask[:20].any())
        self.assertFalse(mask[80:].any())
        self.assertFalse(mask[:, 120:].any())

    def test_recovery_stays_framewise_until_cut_then_resets(self):
        frames = [np.zeros((240, 432, 3), np.uint8) for _ in range(3)]
        frames.append(np.full_like(frames[0], 200))
        detector = Mock(bbox_min_inside_ratio=1)
        detector._box_inside_ratio.return_value = 1
        def detect(crop):
            if crop.shape[0] == 60:
                return [[60, 200, 10, 50]]
            return [[50, 300, 100, 140]] if crop[0, 0, 0] else [[30, 410, 40, 200]]
        detector.detect_subtitle.side_effect = detect
        audit = {}
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, '_read_frames', return_value=iter(frames)):
            boxes, shots, count = pipeline._scan('clip', detector, (90, 150, 40, 400), (0, 240),
                                                 2, Path(folder)/'masks', Mock(), 4, audit)
        self.assertEqual(audit, dict(frames=[0, 1, 2], trigger_frames=[0]))
        self.assertEqual(count, 5)
        self.assertEqual(shots, [(0, 3), (3, 4)])
        self.assertEqual(boxes[1], [[100, 240, 100, 140]])
        self.assertEqual(boxes[3], [[50, 300, 100, 140]])

    def test_shrink_retry_does_not_overwrite_constrained_mask(self):
        detector = Mock()
        full, short = [10, 110, 20, 50], [10, 75, 20, 50]
        with patch.object(pipeline, '_read_frames') as read:
            result = pipeline._retry_shrunk_lines('clip', [[full], [short], [full]], [(0, 3)],
                None, detector, (0, 80, 0, 160), (0, 80), Mock(), {1})
        read.assert_not_called()
        self.assertEqual(result['frames'], [])

    def test_local_gap_requires_current_strokes_and_both_neighbors_same_shot(self):
        frame = np.full((100, 220, 3), 150, np.uint8)
        cv2.putText(frame, 'Hello', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 7)
        cv2.putText(frame, 'Hello', (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        area = (20, 80, 0, 220)
        line = [20, 110, 35, 64]
        mask = outlined_local_mask(frame, [line], area)
        for middle, shots, expected in [(frame, [(0, 3)], [1]),
                                        (np.full_like(frame, 150), [(0, 3)], []),
                                        (frame, [(0, 1), (1, 3)], [])]:
            raw = np.stack([mask, np.zeros_like(mask), mask])
            boxes = [[line], [], [line]]
            with patch.object(pipeline, '_read_frames', return_value=(f.copy() for f in [frame, middle, frame])):
                recovered = pipeline._recover_local_gaps('clip', boxes, shots, raw, area, (0, 100), {0, 1, 2})
            self.assertEqual(recovered, expected)
            self.assertEqual(bool(raw[1].any()), bool(expected))

    def test_empty_wide_ocr_probes_narrow_selection_and_recovers_first_frame(self):
        frame = np.full((1000, 220, 3), 150, np.uint8)
        detector = Mock(bbox_min_inside_ratio=1)
        detector.detect_subtitle.side_effect = [[], [[20, 110, 15, 44]], [[20, 110, 15, 44]]]
        audit = {}
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, '_read_frames', return_value=iter([frame, frame])):
            boxes, _, count = pipeline._scan('clip', detector, (420, 480, 0, 220), (320, 580),
                                             2, Path(folder)/'masks', Mock(), 2, audit)
        self.assertEqual(count, 3)
        self.assertEqual(audit['frames'], [0, 1])
        self.assertEqual(audit['trigger_frames'], [0])
        self.assertEqual(boxes, [[[20, 110, 115, 144]]]*2)


class NarrowOCRTest(unittest.TestCase):
    area = (400, 650, 0, 600)
    band = (300, 750)
    line = [116, 233, 166, 206]
    tall = [50, 550, 80, 330]

    def frame(self, text='Hello', offset=0):
        frame = np.full((1200, 600, 3), 150, np.uint8)
        if text:
            cv2.putText(frame, text, (120, 500+offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 7)
            cv2.putText(frame, text, (120, 500+offset), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 2)
        return frame

    def retry(self, frames, boxes, detector, shots=None, fps=30, area=None):
        raw = np.zeros((len(boxes), 450, 600), np.uint8)
        # Distinct preexisting masks make accidental replacement observable.
        raw[:, 0, 0] = 255
        with patch.object(pipeline, '_read_frames', side_effect=lambda p: (f for f in frames)):
            audit = pipeline._retry_narrow_lines('clip', boxes, shots or [(0, len(boxes))], raw,
                detector, area or self.area, self.band, 1200, fps, Mock())
        return raw, audit

    def test_automatic_row_tracks_translation_and_current_caption_width(self):
        first = None
        for offset in [0, 43]:
            line = [*self.line[:2], self.line[2]+offset, self.line[3]+offset]
            boxes = [[line], [line], [self.tall], []]
            frames = [self.frame(offset=offset), self.frame(offset=offset),
                      self.frame('Longer', offset), self.frame('')]
            detector = Mock()
            detector.detect_subtitle.side_effect = [[[115, 268, 10, 47]], []]
            raw, audit = self.retry(frames, boxes, detector)
            self.assertEqual(audit['frames'], [2, 3])
            self.assertEqual(audit['confirmed_frames'], [2])
            self.assertEqual(audit['empty_frames'], [3])
            self.assertEqual(audit['ocr_calls'], 2)
            self.assertGreater(boxes[2][0][1], line[1])
            self.assertTrue(raw[2].any())
            self.assertFalse(raw[3].any())
            np.testing.assert_array_equal(raw[:2, 0, 0], [255, 255])
            self.assertEqual(boxes[3], [])
            region = audit['groups'][0]['region']
            first = first or region
            self.assertEqual(region[:2], [v+offset for v in first[:2]])
            self.assertEqual(detector.detect_subtitle.call_args_list[0].args[0].shape, (64, 600, 3))

    def test_no_anchor_single_anchor_and_scene_boundary_do_not_retry(self):
        for boxes, shots in [([[], [], []], [(0, 3)]),
                              ([[self.line], [], []], [(0, 3)]),
                              ([[self.line], [self.line], []], [(0, 2), (2, 3)])]:
            with patch.object(pipeline, '_read_frames') as read:
                result = pipeline._retry_narrow_lines('clip', boxes, shots, None, Mock(),
                    self.area, self.band, 1200, 30, Mock())
            read.assert_not_called()
            self.assertEqual(result['frames'], [])

    def test_anchors_require_current_outlined_pixels(self):
        pale, unoutlined = self.frame(), self.frame()
        pale[np.all(pale == 255, axis=2)] = 200
        unoutlined[np.all(unoutlined == 0, axis=2)] = 150
        for frame in [self.frame(''), pale, unoutlined]:
            detector = Mock()
            raw, audit = self.retry([frame]*3, [[self.line], [self.line], []], detector)
            self.assertFalse(audit['groups'])
            detector.detect_subtitle.assert_not_called()
            np.testing.assert_array_equal(raw[:, 0, 0], [255]*3)

    def test_blank_pixels_cannot_confirm_a_false_positive_ocr_box(self):
        detector = Mock()
        detector.detect_subtitle.return_value = [[115, 268, 10, 47]]
        boxes = [[self.line], [self.line], [self.tall]]
        raw, audit = self.retry([self.frame(), self.frame(), self.frame('')], boxes, detector)
        self.assertEqual(audit['empty_frames'], [2])
        self.assertEqual(boxes[2], [])
        self.assertFalse(raw[2].any())

    def test_expiry_is_renewed_only_by_current_confirmations(self):
        frames = [self.frame(), self.frame(), self.frame('Longer')] + [self.frame('')]*6
        detector = Mock()
        detector.detect_subtitle.side_effect = [[[115, 268, 10, 47]], [], [], []]
        raw, audit = self.retry(frames, [[self.line], [self.line]]+[[] for _ in range(7)], detector, fps=3)
        self.assertEqual(audit['frames'], [2, 3, 4, 5])
        self.assertEqual(audit['confirmed_frames'], [2])
        self.assertEqual(audit['empty_frames'], [3, 4, 5])
        self.assertFalse(raw[3:6].any())
        np.testing.assert_array_equal(raw[6:, 0, 0], [255]*3)  # Original path after expiry.

    def test_retry_does_not_cross_cut_or_replace_short_and_multiline_frames(self):
        short, other = [120, 140, 166, 206], [100, 300, 260, 300]
        boxes = [[self.line], [self.line], [short], [self.line, other], [], []]
        original = deepcopy(boxes)
        detector = Mock()
        detector.detect_subtitle.return_value = []
        raw, audit = self.retry([self.frame()]*6, boxes, detector, [(0, 5), (5, 6)])
        self.assertEqual(audit['frames'], [4])
        self.assertEqual(boxes[:4], original[:4])
        np.testing.assert_array_equal(raw[:4, 0, 0], [255]*4)
        self.assertEqual(raw[5, 0, 0], 255)

    def test_unrelated_tall_detection_keeps_normal_caption_on_original_path(self):
        # The side box overlaps vertically and lies inside the old merge gap.
        # It must not turn a valid horizontal caption into an oversized line.
        side = [40, 50, 140, 330]
        mixed = pipeline.merge_line_boxes([side, self.line])
        detector = Mock()
        detector.detect_subtitle.return_value = []
        boxes = [[self.line], [self.line], mixed, []]
        raw, audit = self.retry([self.frame()]*3+[self.frame('')], boxes, detector)
        self.assertEqual(audit['frames'], [3])
        self.assertIn(self.line, boxes[2])
        self.assertEqual(raw[2, 0, 0], 255)  # Retry must not clear the existing mask.
        self.assertFalse(raw[3].any())

    def test_multiple_supported_rows_abstain_and_manual_area_is_respected(self):
        other = [100, 300, 260, 300]
        self.assertEqual(narrow_text_regions({0: [self.line, other], 1: [self.line, other]},
                         [(0, 3)], (100, 350, 0, 600)), [])
        detector = Mock()
        detector.detect_subtitle.return_value = []
        _, audit = self.retry([self.frame()]*3, [[self.line], [self.line], []], detector,
                              area=(470, 550, 140, 450))
        self.assertFalse(audit['groups'])  # The line is outside the manual selection.
        region = narrow_text_regions({0: [self.line], 1: [self.line]},
                                     [(0, 3)], (160, 212, 100, 450))[0]['region']
        self.assertEqual(region, [160, 212, 100, 450])

    def test_incomplete_anchor_or_retry_decode_fails(self):
        boxes = [[self.line], [self.line], []]
        for lengths in [(1,), (2, 2)]:
            streams = [(self.frame() for _ in range(n)) for n in lengths]
            with patch.object(pipeline, '_read_frames', side_effect=streams), self.assertRaisesRegex(RuntimeError, 'Incomplete decode'):
                pipeline._retry_narrow_lines('clip', boxes, [(0, 3)], np.zeros((3, 450, 600), np.uint8),
                    Mock(), self.area, self.band, 1200, 30, Mock())

    def test_shared_pipeline_delivers_same_corrected_masks_to_both_models(self):
        frames = [self.frame(), self.frame(), self.frame('Longer'), self.frame('')]
        capture = Mock()
        capture.get.side_effect = lambda key: {
            cv2.CAP_PROP_FRAME_WIDTH: 600, cv2.CAP_PROP_FRAME_HEIGHT: 1200,
            cv2.CAP_PROP_FPS: 30, cv2.CAP_PROP_FRAME_COUNT: 4,
        }[key]
        hashes = []
        for mode in ['sttn', 'propainter']:
            detector = Mock(bbox_min_inside_ratio=1)
            detector._box_inside_ratio.return_value = 1
            detector.detect_subtitle.side_effect = [
                [self.line], [self.line], [[50, 550, 110, 340]], [], [[115, 268, 10, 47]], []]
            model = Mock()
            model.inpaint.side_effect = lambda frames, masks, **kwargs: frames
            original_infer = pipeline._infer
            checked = []
            def infer(video, masks, *args, **kwargs):
                self.assertTrue(masks[2].any())
                self.assertFalse(masks[2, :154].any())
                self.assertFalse(masks[2, 218:].any())
                self.assertFalse(masks[3].any())
                checked.append(True)
                return original_infer(video, masks, *args, **kwargs)
            with tempfile.TemporaryDirectory() as folder, patch.object(pipeline.cv2, 'VideoCapture', return_value=capture), patch.object(
                    pipeline, '_read_frames', side_effect=lambda p: (f for f in frames)), patch.object(
                    pipeline, '_make_model', return_value=(model, 'cpu')), patch.object(
                    pipeline, '_infer', side_effect=infer), patch.object(pipeline, '_render', return_value=3):
                output = pipeline.run_glyph_removal(Path(folder)/'clip.mp4', self.area,
                    dict(mode=mode, ocr_sample_step=1, sttn_band_margin=100), detector, Mock(), Mock())
                audit = json.loads(output.with_suffix('.json').read_text())
            self.assertEqual(checked, [True])
            self.assertTrue(model.inpaint.called)
            self.assertEqual(audit['local_ocr']['frames'], [2, 3])
            self.assertEqual(audit['narrow_ocr']['confirmed_frames'], [2])
            self.assertEqual(audit['narrow_ocr']['empty_frames'], [3])
            self.assertEqual(audit['ocr_frames'], 6)
            hashes.append(audit['masks_sha256'])
        self.assertEqual(hashes[0], hashes[1])


class ShortTextTest(unittest.TestCase):
    def test_regions_require_repeated_nearby_anchors_and_respect_area(self):
        line = [90, 110, 100, 130]
        boxes = [[line], [], [line]]
        regions = short_text_regions(boxes, [(0, 3)], (80, 145, 80, 125), 1000, 25)
        self.assertEqual(regions[1], [80, 145, 80, 125])
        for candidates, shots in [([[line], [], []], [(0, 3)]),
                                   (boxes, [(0, 2), (2, 3)]),
                                   ([[], [], []], [(0, 3)]),
                                   ([[line], [], [[190, 210, 100, 130]]], [(0, 3)])]:
            self.assertFalse(short_text_regions(candidates, shots, (0, 200, 0, 300), 1000, 25))
        distant = [[line], [line]] + [[] for _ in range(30)]
        self.assertNotIn(20, short_text_regions(distant, [(0, 32)], (0, 200, 0, 300), 1000, 25))

    def test_normal_captions_and_overlap_recovery_are_excluded(self):
        line = [90, 110, 100, 130]
        boxes = [[line], [[30, 200, 100, 130]], [line], []]
        regions = short_text_regions(boxes, [(0, 4)], (0, 200, 0, 300), 1000, 25, {3})
        self.assertEqual(set(regions), {0, 2})
        self.assertFalse(short_text_regions(boxes, [(0, 4)], (0, 200, 0, 300), 1000, 25, {0, 2}))

    def short_frame(self):
        frame = np.full((160, 180, 3), 150, np.uint8)
        # Synthetic short body and detached upper mark, each with a dark outline.
        for x0, y0, x1, y1 in [(72, 72, 87, 98), (75, 51, 83, 55)]:
            cv2.rectangle(frame, (x0-4, y0-4), (x1+4, y1+4), (0, 0, 0), -1)
            cv2.rectangle(frame, (x0, y0), (x1, y1), (255, 255, 255), -1)
        return frame

    def test_current_ocr_body_and_upper_mark_are_covered_within_region(self):
        detector = Mock()
        detector.detect_subtitle.return_value = [[30, 50, 40, 75]]
        area = (30, 130, 40, 130)
        boxes, mask = short_text_mask(self.short_frame(), detector, area)
        self.assertEqual(boxes, [[70, 90, 70, 105]])
        self.assertTrue(np.all(mask[72:99, 72:88]))
        self.assertTrue(np.all(mask[51:56, 75:84]))
        self.assertFalse(mask[:30].any())
        self.assertFalse(mask[130:].any())
        self.assertFalse(mask[:, :40].any())
        self.assertFalse(mask[:, 130:].any())

    def test_blank_pale_unoutlined_or_detached_mark_alone_cannot_confirm_text(self):
        detector = Mock()
        detector.detect_subtitle.return_value = [[30, 50, 40, 75]]
        pale = self.short_frame()
        pale[np.all(pale == 255, axis=2)] = 215
        unoutlined = self.short_frame()
        unoutlined[np.all(unoutlined == 0, axis=2)] = 150
        detached = self.short_frame()
        detached[65:110] = 150
        for frame in [np.full_like(pale, 150), pale, unoutlined, detached]:
            boxes, mask = short_text_mask(frame, detector, (30, 130, 40, 130))
            self.assertEqual(boxes, [])
            self.assertFalse(mask.any())
        for coords in [[], [[0, 50, 40, 75]], [[5, 85, 40, 75]]]:
            detector.detect_subtitle.return_value = coords
            self.assertFalse(short_text_mask(self.short_frame(), detector, (30, 130, 40, 130))[1].any())

    def test_enlarged_retry_maps_detections_back_to_original_pixels(self):
        detector = Mock()
        detector.detect_subtitle.side_effect = [[], [[60, 100, 80, 150]]]
        boxes, mask = short_text_mask(self.short_frame(), detector, (30, 130, 40, 130))
        self.assertEqual(boxes, [[70, 90, 70, 105]])
        self.assertTrue(np.all(mask[72:99, 72:88]))
        self.assertTrue(np.all(mask[51:56, 75:84]))
        self.assertEqual([c.args[0].shape[:2] for c in detector.detect_subtitle.call_args_list],
                         [(100, 90), (200, 180)])

    def test_retry_requires_current_detection_and_keeps_existing_mask(self):
        line = [70, 90, 70, 105]
        for present in [True, False]:
            boxes = [[line], [], [line]]
            raw = np.zeros((3, 160, 180), np.uint8)
            raw[0, 10, 10] = 255
            detector = Mock()
            # Each automatically proposed region is [17, 140, 21, 139].
            detector.detect_subtitle.return_value = [[49, 69, 53, 88]] if present else []
            with patch.object(pipeline, '_read_frames', return_value=(self.short_frame() for _ in range(3))):
                audit = pipeline._retry_short_lines('clip', boxes, [(0, 3)], raw, detector,
                    (0, 160, 0, 180), (0, 160), 1000, 25, set(), Mock())
            self.assertEqual(audit['updated_frames'], [0, 1, 2] if present else [])
            self.assertEqual(raw[0, 10, 10], 255)
            self.assertEqual(bool(raw[1].any()), present)
            self.assertEqual(bool(boxes[1]), present)

    def test_retry_skips_decode_without_targets_and_fails_on_truncation(self):
        detector = Mock()
        with patch.object(pipeline, '_read_frames') as read:
            audit = pipeline._retry_short_lines('clip', [[], []], [(0, 2)], None, detector,
                (0, 160, 0, 180), (0, 160), 1000, 25, set(), Mock())
        read.assert_not_called()
        self.assertEqual(audit['frames'], [])
        line = [70, 90, 70, 105]
        with patch.object(pipeline, '_read_frames', return_value=(f for f in [])), self.assertRaisesRegex(RuntimeError, 'Incomplete decode'):
            pipeline._retry_short_lines('clip', [[line], [line]], [(0, 2)], None, detector,
                (0, 160, 0, 180), (0, 160), 1000, 25, set(), Mock())

    def test_shared_pipeline_recovers_short_gap_before_narrow_retry(self):
        frame = np.full((1000, 220, 3), 150, np.uint8)
        frame[:160, :180] = self.short_frame()
        frames = [frame.copy() for _ in range(5)] + [np.full_like(frame, 150)]
        short, long = [70, 90, 70, 105], [30, 145, 70, 105]
        for mode in ['sttn', 'propainter']:
            with self.subTest(mode=mode):
                main = iter([[long], [long], [short], [], [short], []])
                detector = Mock(bbox_min_inside_ratio=1)
                detector._box_inside_ratio.return_value = 1
                def detect(crop):
                    if crop.shape[:2] == (300, 220):
                        return next(main)
                    if crop.shape[:2] == (123, 118) and crop.max() == 255:
                        return [[49, 69, 53, 88]]
                    return []  # Narrow full-width OCR still misses this word.
                detector.detect_subtitle.side_effect = detect
                capture = Mock()
                capture.get.side_effect = lambda key: {
                    cv2.CAP_PROP_FRAME_WIDTH: 220, cv2.CAP_PROP_FRAME_HEIGHT: 1000,
                    cv2.CAP_PROP_FPS: 25, cv2.CAP_PROP_FRAME_COUNT: len(frames),
                }[key]
                def infer(video, masks, *args, **kwargs):
                    self.assertTrue(np.all(masks[3, 72:99, 72:88]))
                    self.assertTrue(np.all(masks[3, 51:56, 75:84]))
                    self.assertFalse(masks[5].any())  # Real blank remains blank.
                    return [], 'cpu'
                with tempfile.TemporaryDirectory() as folder, patch.object(
                        pipeline.cv2, 'VideoCapture', return_value=capture), patch.object(
                        pipeline, '_read_frames', side_effect=lambda p: (f for f in frames)), patch.object(
                        pipeline, '_infer', side_effect=infer), patch.object(pipeline, '_render', return_value=5):
                    output = pipeline.run_glyph_removal(Path(folder)/'clip.mp4', (0, 200, 0, 220),
                        dict(mode=mode, ocr_sample_step=1, sttn_band_margin=100), detector, Mock(), Mock())
                    audit = json.loads(output.with_suffix('.json').read_text())
                self.assertIn(3, audit['short_ocr']['updated_frames'])
                self.assertNotIn(3, audit['narrow_ocr']['frames'])


class PhotometricContextTest(unittest.TestCase):
    def test_gradual_flash_and_fade_split_but_small_exposure_change_does_not(self):
        frames = [np.full((40, 80, 3), value, np.uint8) for value in [70, 73, 85, 97, 85]]
        masks = [np.zeros((40, 80), np.uint8) for _ in frames]
        self.assertEqual(photometric_boundaries(frames, masks), [0, 2, 3, 4, 5])

    def test_local_motion_masked_subtitles_and_insufficient_context_do_not_split(self):
        clean = np.full((40, 80, 3), 70, np.uint8)
        changed = clean.copy()
        changed[:20] = 200  # Large local movement, but no coherent global shift.
        empty = np.zeros((40, 80), np.uint8)
        self.assertEqual(photometric_boundaries([clean, changed], [empty, empty]), [0, 2])
        changed[:32] = 200
        mask = empty.copy()
        mask[:32] = 255
        self.assertEqual(photometric_boundaries([clean, changed], [empty, mask]), [0, 2])
        mask[:] = 255
        self.assertEqual(photometric_boundaries([clean, clean+20], [mask, mask]), [0, 2])

    def test_propainter_isolates_context_duplicates_singletons_and_passes_empty_segments(self):
        frames = [np.full((240, 432, 3), value, np.uint8) for value in [70, 70, 90, 110, 110]]
        masks = np.zeros((5, 240, 432), np.uint8)
        masks[:3, 80:100, 80:100] = 255
        jobs = [dict(id=0, start=0, stop=5, crop_y=[0, 240])]
        for mode in ['propainter', 'sttn']:
            model = Mock()
            model.inpaint.side_effect = lambda frames, masks: [f.copy() for f in frames]
            with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, '_make_model', return_value=(model, 'cpu')), patch.object(
                    pipeline, '_read_frames', return_value=(f.copy() for f in frames)):
                records, _ = pipeline._infer('clip', masks, jobs, 0, {'mode': mode}, Path(folder), Mock())
                predicted = np.load(Path(folder)/'job00000.npy')
            np.testing.assert_array_equal(predicted, frames)
            contexts = [[int(f[0, 0, 0]) for f in call.args[0]] for call in model.inpaint.call_args_list]
            self.assertEqual(contexts, [[70, 70], [90, 90]] if mode == 'propainter' else [[70, 70, 90, 110, 110]])
            self.assertEqual(records[0]['photometric_cuts'], [2, 3] if mode == 'propainter' else [])

    def test_incomplete_propainter_segment_fails_instead_of_losing_frames(self):
        frame = np.full((240, 432, 3), 70, np.uint8)
        masks = np.zeros((1, 240, 432), np.uint8)
        masks[:, 80:100, 80:100] = 255
        model = Mock()
        model.inpaint.return_value = [frame]  # Singleton is sent as a pair.
        with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, '_make_model', return_value=(model, 'cpu')), patch.object(
                pipeline, '_read_frames', return_value=(f for f in [frame])), self.assertRaisesRegex(RuntimeError, 'incomplete segment'):
            pipeline._infer('clip', masks, [dict(id=0, start=0, stop=1, crop_y=[0, 240])], 0,
                            {'mode': 'propainter'}, Path(folder), Mock())


class ReferenceSelectionTest(unittest.TestCase):
    def setUp(self):
        self.clean = np.full((240, 432, 3), 70, np.uint8)
        for x in range(20, 430, 35):
            cv2.line(self.clean, (x, 20), (x+20, 220), (190, 190, 190), 2)
        self.text = self.clean.copy()
        cv2.putText(self.text, 'Hello', (100, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 7)
        cv2.putText(self.text, 'Hello', (100, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        self.mask = np.zeros((240, 432), np.uint8)
        cv2.putText(self.mask, 'Hello', (100, 135), cv2.FONT_HERSHEY_SIMPLEX, 1, 255, 9)
        self.empty = np.zeros_like(self.mask)

    def test_selects_clean_and_rejects_false_empty_ocr(self):
        index, rows = select_reference([self.text, self.text, self.clean],
                                      [self.mask, self.empty, self.empty], {1, 2})
        self.assertEqual(index, 2)
        self.assertEqual(rows[0]['reason'], 'subtitle_pattern_remains')

    def test_abstains_for_movement_title_disappearance_and_no_reference(self):
        for candidate in [np.roll(self.clean, 15, axis=1), np.full_like(self.clean, 70), self.text]:
            index, _ = select_reference([self.text, candidate], [self.mask, self.empty], {1})
            self.assertIsNone(index)

    def test_unverified_empty_mask_and_changing_text_are_not_references(self):
        index, _ = select_reference([self.text, self.clean], [self.mask, self.empty], set())
        self.assertIsNone(index)
        changed = np.roll(self.mask, 70, axis=1)
        index, rows = select_reference([self.text, self.text, self.clean], [self.mask, changed, self.empty], {2})
        self.assertIsNone(index)
        self.assertEqual(rows[0]['reason'], 'changing_subtitle_layout')

    def test_inference_routes_reference_only_to_sttn_local_recovery_windows(self):
        masks = np.array([self.mask, self.empty])
        jobs = [dict(id=0, start=0, stop=2, crop_y=[0, 240])]
        for mode, local_frames, expected in [('sttn', {0, 1}, True), ('sttn', set(), False), ('propainter', {0, 1}, False)]:
            with self.subTest(mode=mode, local_frames=local_frames):
                model = Mock()
                model.inpaint.side_effect = lambda frames, masks, **kwargs: frames
                with tempfile.TemporaryDirectory() as folder, patch.object(pipeline, '_make_model', return_value=(model, 'cpu')), patch.object(
                    pipeline, '_read_frames', return_value=(f.copy() for f in [self.text, self.clean])):
                    records, _ = pipeline._infer('clip', masks, jobs, 0, {'mode': mode}, Path(folder), Mock(), local_frames, {1})
                self.assertEqual(model.inpaint.call_args.kwargs, {'extra_reference_indices': [1]} if expected else {})
                self.assertEqual(records[0]['clean_reference_frame'], 1 if expected else None)

    def test_sttn_extra_references_are_deduplicated_and_do_not_leak_to_next_call(self):
        import torch
        from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
        model = object.__new__(STTNDetInpaint)
        model.device = torch.device('cpu')
        model.neighbor_stride, model.ref_length = 1, 3
        model.model_input_width, model.model_input_height = 32, 16
        model.model = Mock()
        model.model.encoder.side_effect = lambda x: torch.arange(len(x), dtype=torch.float32).reshape(-1, 1, 1, 1)
        observed = []
        def infer(feats, masks):
            observed.append(feats[:, 0, 0, 0].tolist())
            return feats
        model.model.infer.side_effect = infer
        model.model.decoder.side_effect = lambda feats: torch.zeros((len(feats), 3, 16, 32))
        frames = [np.full((16, 32, 3), 70, np.uint8) for _ in range(4)]
        masks = [np.zeros((16, 32), np.uint8) for _ in frames]
        model.inpaint([f.copy() for f in frames], [m.copy() for m in masks], extra_reference_indices=[2, 2])
        self.assertEqual(observed[0], [0, 1, 2, 3])
        self.assertTrue(all(len(ids) == len(set(ids)) for ids in observed))
        observed.clear()
        model.inpaint([f.copy() for f in frames], [m.copy() for m in masks])
        self.assertEqual(observed[0], [0, 1, 3])
        with self.assertRaises(ValueError):
            model.inpaint(frames, masks, extra_reference_indices=[4])


if __name__ == '__main__':
    unittest.main()
