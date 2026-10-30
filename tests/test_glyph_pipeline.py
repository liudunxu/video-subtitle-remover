import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

import api
from backend.tools.glyph_masks import (
    glyph_mask, recover_vertical_border, stabilize_masks, build_windows,
    overlap_weight, blend_prediction, merge_line_boxes, resize_model_mask, find_shrunk_lines,
)
from backend.inpaint import glyph_pipeline as pipeline


class GlyphOptionsTest(unittest.TestCase):
    def test_opt_in_and_both_engines(self):
        self.assertEqual(api._normalize_options({})['mask_strategy'], 'rectangle')
        for mode in ('sttn', 'propainter'):
            self.assertEqual(api._normalize_options({'mask_strategy': 'glyph', 'mode': mode})['mask_strategy'], 'glyph')

    def test_reject_incompatible_options(self):
        for field, value in [('mode', 'lama'), ('post_lama_refine', True),
                             ('auto_residual_cleanup', True), ('post_verify_blur', True),
                             ('sttn_skip_detection', True), ('empty_detection_full_inpaint', True),
                             ('sttn_input_width', 864), ('ocr_sample_step', 3),
                             ('subtitle_area_deviation_pixel', 12), ('propainter_max_load_num', 70)]:
            with self.subTest(field=field), self.assertRaises(api.RequestError):
                api._normalize_options({'options': {'mask_strategy': 'glyph', field: value}})
        with self.assertRaises(api.RequestError):
            api._normalize_options({'mask_strategy': 'typo'})

    def test_dispatch_does_not_enter_legacy_remover(self):
        options = api._normalize_options({'mask_strategy': 'glyph'})
        with patch.object(api, '_run_glyph_remover', return_value=Path('result.mp4')) as run:
            self.assertEqual(api._run_subtitle_remover(Path('in.mp4'), None, options), (Path('result.mp4'), False))
            run.assert_called_once()

    def test_requested_ocr_size_reaches_detector(self):
        options = api._normalize_options({'options': {'mask_strategy': 'glyph', 'glyph_ocr_max_side': 960}})
        with patch('backend.tools.subtitle_detect.SubtitleDetect') as detector, patch.object(
                pipeline, 'run_glyph_removal', side_effect=pipeline.NoGlyphsError('empty')):
            with self.assertRaises(api.RequestError):
                api._run_glyph_remover(Path('in.mp4'), None, options)
        self.assertEqual(detector.call_args.kwargs['det_limit_side_len'], 960)
        detector.assert_called_once()  # Already-high-res OCR needs no retry detector.
        for supplied, clamped in ((639, 640), (1281, 1280)):
            self.assertEqual(api._normalize_options({'glyph_ocr_max_side': supplied})['glyph_ocr_max_side'], clamped)

    def test_default_api_supplies_lazy_high_resolution_retry_detector(self):
        options = api._normalize_options({'mask_strategy': 'glyph'})
        primary, retry = Mock(), Mock()
        with patch('backend.tools.subtitle_detect.SubtitleDetect', side_effect=[primary, retry]) as detector, patch.object(
                pipeline, 'run_glyph_removal', side_effect=pipeline.NoGlyphsError('empty')) as run:
            with self.assertRaises(api.RequestError):
                api._run_glyph_remover(Path('in.mp4'), None, options)
        self.assertEqual([call.kwargs['det_limit_side_len'] for call in detector.call_args_list], [640, 960])
        self.assertIs(run.call_args.kwargs['retry_detector'], retry)
        retry.detect_subtitle.assert_not_called()

    def test_empty_glyphs_are_http_422(self):
        options = api._normalize_options({'mask_strategy': 'glyph'})
        with patch('backend.tools.subtitle_detect.SubtitleDetect'), patch.object(
                pipeline, 'run_glyph_removal', side_effect=pipeline.NoGlyphsError('empty')):
            with self.assertRaises(api.RequestError) as exc:
                api._run_glyph_remover(Path('in.mp4'), None, options)
        self.assertEqual(exc.exception.status, 422)


class GlyphMaskTest(unittest.TestCase):
    def test_white_letters_keep_spaces_and_reject_white_background(self):
        frame = np.full((100, 220, 3), 70, np.uint8)
        cv2.putText(frame, 'I I', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        mask = glyph_mask(frame, [[15, 100, 25, 70]])
        self.assertGreater(np.count_nonzero(mask), 0)
        self.assertEqual(mask[40:65, 51:54].sum(), 0)
        self.assertEqual(glyph_mask(np.full_like(frame, 255), [[15, 100, 25, 70]]).sum(), 0)

    def test_border_recovery_covers_clipped_descender(self):
        frame = np.full((120, 220, 3), 70, np.uint8)
        cv2.putText(frame, 'gy', (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)
        boxes = [[25, 100, 30, 68]]
        raw = glyph_mask(frame, boxes)
        revised = raw | recover_vertical_border(frame, boxes, raw)
        white = np.all(frame[60:80, 30:60] == 255, axis=2)
        self.assertTrue(np.all(revised[60:80, 30:60][white] > 0))

    def test_disconnected_accent_outside_ocr_box_is_covered(self):
        frame = np.full((100, 160, 3), 70, np.uint8)
        cv2.putText(frame, 'HI', (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.circle(frame, (30, 73), 1, (255, 255, 255), -1)
        mask = glyph_mask(frame, [[15, 85, 25, 67]])
        self.assertEqual(mask[73, 30], 255)
        self.assertEqual(mask[90, 30], 0)

    def test_letter_connected_to_bright_background_gets_coverage_guard(self):
        frame = np.full((100, 160, 3), 70, np.uint8)
        cv2.putText(frame, 'HI', (30, 65), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
        cv2.rectangle(frame, (34, 0), (38, 50), (255, 255, 255), -1)
        mask = glyph_mask(frame, [[25, 95, 25, 70]])
        white = np.all(frame[30:65, 30:90] == 255, axis=2)
        self.assertTrue(np.all(mask[30:65, 30:90][white] == 255))
        self.assertFalse(mask[:, 130:].any())

    def test_clipped_white_symbols_recovered_above_and_below_line(self):
        frame = np.full((150, 170, 3), 70, np.uint8)
        frame[55:85, 38:45] = 255
        frame[35:42, 39:44] = 255  # Crosses old top search boundary (38).
        frame[98:105, 39:44] = 255  # Crosses old bottom boundary (101).
        mask = glyph_mask(frame, [[20, 130, 50, 90]])
        self.assertTrue(np.all(mask[35:42, 39:44] == 255))
        self.assertTrue(np.all(mask[98:105, 39:44] == 255))
        self.assertFalse(mask[:, 75:85].any())  # Preserve background between glyphs.

    def test_clipped_symbol_rejects_pale_fabric_and_unaligned_highlights(self):
        for value, x in [(215, 39), (255, 100)]:
            with self.subTest(value=value, x=x):
                frame = np.full((150, 170, 3), 70, np.uint8)
                frame[55:85, 38:45] = 255
                frame[98:105, x:x+5] = value
                mask = glyph_mask(frame, [[20, 130, 50, 90]])
                self.assertFalse(mask[98:105, x:x+5].any())

    def test_symbol_recovery_keeps_empty_and_frame_boundary_safe(self):
        frame = np.full((100, 170, 3), 70, np.uint8)
        frame[55:85, 38:45] = 255
        frame[98:100, 39:44] = 255  # Continues through the physical frame edge.
        self.assertFalse(glyph_mask(frame, []).any())
        mask = glyph_mask(frame, [[20, 130, 50, 90]])
        self.assertFalse(mask[98:100, 39:44].any())

    def test_thin_mask_core_never_blends_original_text_back(self):
        source = np.full((40, 60, 3), 255, np.uint8)
        mask = np.zeros((40, 60), np.uint8)
        mask[20, 30] = 255
        result = blend_prediction(source, np.zeros_like(source), mask)
        np.testing.assert_array_equal(result[20, 30], [0, 0, 0])
        np.testing.assert_array_equal(result[0, 0], source[0, 0])

    def test_model_resize_preserves_subpixel_accents_and_empty_frames(self):
        mask = np.zeros((240, 1080), np.uint8)
        mask[100, 501] = 255
        resized = resize_model_mask(mask)
        self.assertTrue(resized[99:102, 199:203].any())
        self.assertFalse(resize_model_mask(np.zeros_like(mask)).any())

    def test_stability_keeps_empty_and_cut_separate(self):
        raw = np.zeros((6, 30, 40), np.uint8)
        raw[0:2, 10:20, 10:20] = 255
        raw[1, 10:20, 20] = 255
        raw[3:6, 10:20, 10:20] = 255
        raw[3, 10:20, 9] = 255
        result = np.zeros_like(raw)
        stabilize_masks(raw, [(0, 1), (1, 3), (3, 6)], result)
        np.testing.assert_array_equal(result[0], raw[0])
        self.assertFalse(result[2].any())
        self.assertTrue(np.all(result[4, 10:20, 9] == 255))
        self.assertFalse(result[1, :, 9].any())

    def test_single_character_boxes_merge_before_glyph_filter(self):
        self.assertEqual(merge_line_boxes([[10, 20, 10, 40], [30, 40, 12, 42]]), [[10, 40, 10, 42]])

    def test_windows_coverage_overlap_and_cut(self):
        masks = np.zeros((220, 300, 20), np.uint8)
        masks[10:190, 100:140, 5:10] = 255
        masks[219, 100:140, 5:10] = 255
        shots = [(0, 170), (170, 219), (219, 220)]
        jobs = build_windows(masks, shots, 40)
        coverage = np.zeros(220, int)
        for job in jobs:
            self.assertLessEqual(job['stop']-job['start'], 80)
            a, b = shots[job['shot']]
            self.assertTrue(a <= job['start'] < job['stop'] <= b)
            coverage[job['start']:job['stop']] += 1
        self.assertTrue(np.all(coverage[np.any(masks, axis=(1, 2))] > 0))
        self.assertLessEqual(coverage.max(), 2)
        first, second = jobs[:2]
        self.assertEqual(first['stop']-second['start'], 20)
        for i in range(second['start'], first['stop']):
            self.assertAlmostEqual(overlap_weight(i, first)+overlap_weight(i, second), 1)
        self.assertEqual(jobs[-1]['stop']-jobs[-1]['start'], 1)

    def test_blend_keeps_outside_and_empty_exact(self):
        rng = np.random.default_rng(7)
        source = rng.integers(0, 256, (100, 120, 3), dtype=np.uint8)
        mask = np.zeros(source.shape[:2], np.uint8)
        prediction = np.full((20, 30, 3), (10, 120, 230), np.uint8)
        np.testing.assert_array_equal(blend_prediction(source, prediction, mask), source)
        mask[40:60, 50:70] = 255
        result = blend_prediction(source, prediction, mask)
        np.testing.assert_array_equal(result[:25], source[:25])
        np.testing.assert_array_equal(result[50, 60], prediction[0, 0])


class ScanAndInferenceTest(unittest.TestCase):
    def test_shrink_retry_requires_both_sides_and_same_shot(self):
        full = [10, 110, 20, 50]
        short = [10, 75, 20, 50]
        boxes = [[full], [short], [short], [full]]
        self.assertEqual(set(find_shrunk_lines(boxes, [(0, 4)])), {1, 2})
        self.assertFalse(find_shrunk_lines(boxes, [(0, 2), (2, 4)]))
        self.assertFalse(find_shrunk_lines([[full], [short], [short], [short]], [(0, 4)]))
        self.assertFalse(find_shrunk_lines([[full], [], [full]], [(0, 3)]))
        self.assertFalse(find_shrunk_lines([[full], [[10, 103, 20, 50]], [full]], [(0, 3)]))

    def test_retry_uses_current_frame_detections_not_neighbor_boxes(self):
        full = [10, 110, 20, 50]
        short = [10, 75, 20, 50]
        frame = np.full((80, 160, 3), 70, np.uint8)
        cv2.putText(frame, 'HI', (20, 45), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)
        for detected, changed in [([full], True), ([short], False), ([], False), ([[120, 155, 20, 50]], False)]:
            with self.subTest(detected=detected):
                boxes = [[full], [short], [full]]
                raw = np.zeros((3, 80, 160), np.uint8)
                detector = Mock(bbox_min_inside_ratio=1)
                detector._box_inside_ratio.return_value = 1
                detector.detect_subtitle.return_value = detected
                with patch.object(pipeline, '_read_frames', return_value=(frame.copy() for _ in range(3))):
                    audit = pipeline._retry_shrunk_lines('clip', boxes, [(0, 3)], raw, detector,
                                                       (0, 80, 0, 160), (0, 80), Mock())
                self.assertEqual(audit['frames'], [1])
                self.assertEqual(audit['updated_frames'], [1] if changed else [])
                self.assertEqual(boxes[1], [full] if changed else [short])
                self.assertEqual(bool(raw.any()), changed)
                detector.detect_subtitle.assert_called_once()

    def test_retry_preserves_area_acceptance_and_skips_unneeded_ocr(self):
        full, short = [10, 110, 20, 50], [10, 75, 20, 50]
        detector = Mock(bbox_min_inside_ratio=1)
        detector._box_inside_ratio.return_value = .5
        detector.detect_subtitle.return_value = [full]
        raw = np.zeros((3, 80, 160), np.uint8)
        frames = (np.zeros((80, 160, 3), np.uint8) for _ in range(3))
        with patch.object(pipeline, '_read_frames', return_value=frames):
            result = pipeline._retry_shrunk_lines('clip', [[full], [short], [full]], [(0, 3)], raw,
                                                  detector, (0, 80, 0, 90), (0, 80), Mock())
        self.assertFalse(result['updated_frames'])
        with patch.object(pipeline, '_read_frames') as read:
            result = pipeline._retry_shrunk_lines('clip', [[full]]*3, [(0, 3)], raw,
                                                  detector, (0, 80, 0, 160), (0, 80), Mock())
            read.assert_not_called()
        self.assertFalse(result['frames'])

    def test_inference_decodes_sequentially_with_exact_overlap(self):
        # Some MP4 seek calls land one frame early. Inference must share OCR's
        # sequential numbering, including skipped leading frames and overlap.
        frames = [np.full((240, 432, 3), i, np.uint8) for i in range(6)]
        capture = Mock()
        capture.read.side_effect = [(True, f) for f in frames] + [(False, None)]
        model = Mock()
        model.inpaint.side_effect = lambda batch, masks: batch
        masks = np.full((6, 240, 432), 255, np.uint8)
        jobs = [dict(id=0, start=1, stop=4, crop_y=[0, 240]),
                dict(id=1, start=3, stop=6, crop_y=[0, 240])]
        with tempfile.TemporaryDirectory() as temp, patch.object(
                pipeline, '_make_model', return_value=(model, 'cpu')), patch.object(
                pipeline.cv2, 'VideoCapture', return_value=capture):
            pipeline._infer('offset.mp4', masks, jobs, 0, {'mode': 'sttn'}, Path(temp), Mock())
        capture.set.assert_not_called()
        batches = [[int(f[0, 0, 0]) for f in call.args[0]] for call in model.inpaint.call_args_list]
        self.assertEqual(batches, [[1, 2, 3], [3, 4, 5]])
        self.assertEqual(capture.read.call_count, 6)
        capture.release.assert_called_once()

    def test_scan_supplements_boundaries_and_never_bridges_cuts(self):
        frames = [np.full((80, 120, 3), i, np.uint8) for i in range(7)]
        frames[3][:] = 180  # One-frame shot; both sides must be inspected.
        detector = Mock()
        detector.bbox_min_inside_ratio = 1
        detector._box_inside_ratio.return_value = 1
        seen = []
        def detect(frame):
            value = int(frame[0, 0, 0]); seen.append(value)
            return [] if value in (4, 6) else [[10, 70, 20, 50]]
        detector.detect_subtitle.side_effect = detect
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline, '_read_frames', return_value=iter(frames)):
            boxes, shots, calls = pipeline._scan('unused', detector, (0, 80, 0, 120),
                                                (0, 80), 2, Path(temp)/'raw', Mock(), len(frames))
            self.assertEqual((Path(temp)/'raw').stat().st_size, 7*80*120)
        self.assertEqual(len(boxes), 7)
        self.assertEqual(shots, [(0, 3), (3, 4), (4, 7)])
        self.assertIn(180, seen)
        self.assertFalse(boxes[4])
        self.assertEqual(calls, len(seen))

    def test_scan_odd_final_frame_is_detected(self):
        frames = [np.zeros((60, 100, 3), np.uint8) for _ in range(4)]
        detector = Mock(bbox_min_inside_ratio=1)
        detector._box_inside_ratio.return_value = 1
        detector.detect_subtitle.return_value = []
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline, '_read_frames', return_value=iter(frames)):
            boxes, _, count = pipeline._scan('unused', detector, (0, 60, 0, 100), (0, 60), 2,
                                             Path(temp)/'raw', Mock(), 4)
        self.assertEqual(len(boxes), 4)
        self.assertEqual(count, 3)  # 0, 2 and final frame 3.

    def test_propainter_single_frame_duplicates_only_same_shot(self):
        source = np.full((240, 432, 3), (1, 20, 90), np.uint8)
        model = Mock()
        model.inpaint.side_effect = lambda frames, masks: frames
        capture = Mock()
        capture.read.return_value = (True, source)
        masks = np.zeros((1, 240, 432), np.uint8)
        masks[:, 80:120, 100:200] = 255
        jobs = build_windows(masks, [(0, 1)], 0)
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline, '_make_model', return_value=(model, 'cpu')), patch.object(pipeline.cv2, 'VideoCapture', return_value=capture):
            pipeline._infer('unused', masks, jobs, 0, {'mode': 'propainter'}, Path(temp), Mock())
            result = np.load(Path(temp)/'job00000.npy')
        frames, model_masks = model.inpaint.call_args.args
        self.assertEqual(len(frames), 2)
        np.testing.assert_array_equal(frames[0], frames[1])
        np.testing.assert_array_equal(model_masks[0], model_masks[1])
        self.assertEqual(result.shape, (1, 240, 432, 3))

    def test_mask_reader_rejects_misalignment_and_keeps_empty_frames(self):
        from backend.inpaint.propainter_inpaint import read_mask
        masks = [np.array([[0, 127], [128, 255]], np.uint8), np.zeros((2, 2), np.uint8)]
        flow, fill = read_mask(masks, 2, (2, 2), 0, 0)
        np.testing.assert_array_equal(np.array(fill[0]), [[0, 0], [255, 255]])
        self.assertFalse(np.array(flow[1]).any())
        with self.assertRaises(ValueError):
            read_mask(masks, 3, (2, 2))
        with self.assertRaises(ValueError):
            read_mask(masks, 2, (3, 2))


class StrictWriterTest(unittest.TestCase):
    def test_nonzero_encoder_exit_is_failure(self):
        from backend.tools.video_io import FFmpegVideoWriter
        writer = object.__new__(FFmpegVideoWriter)
        writer.strict = True
        writer._process = Mock(returncode=1)
        with self.assertRaises(RuntimeError):
            writer.release()

    def test_broken_pipe_is_failure(self):
        from backend.tools.video_io import FFmpegVideoWriter
        writer = object.__new__(FFmpegVideoWriter)
        writer.strict = True
        writer._process = Mock()
        writer._process.stdin.write.side_effect = BrokenPipeError
        with self.assertRaises(RuntimeError):
            writer.write(np.zeros((2, 2, 3), np.uint8))

    def test_mux_offsets_relative_to_silent_video_origin(self):
        from backend.tools.ffmpeg_cli import FFmpegCLI
        cli = Mock(ffmpeg_path='ffmpeg', ffprobe_path='ffprobe')
        with tempfile.TemporaryDirectory() as temp:
            target = Path(temp)/'output.mp4'
            target.touch()
            with patch.object(FFmpegCLI, 'instance', return_value=cli), patch(
                'backend.tools.ffmpeg_cli.probe_audio_codec', return_value='aac'
            ), patch.object(api.subprocess, 'check_output', side_effect=[
                json.dumps({'streams': [{'start_time': '0.239'}]}).encode(),
                json.dumps({'streams': [{'start_time': '0.125'}]}).encode(), b'',
            ]) as run:
                api._mux_audio_onto_video('silent.mp4', 'source.mp4', target, preserve_timestamps=True)
            command = run.call_args.args[0]
            self.assertAlmostEqual(float(command[command.index('-itsoffset')+1]), .114)
            self.assertIn('-copyts', command)
            self.assertNotIn('-shortest', command)


class CompositionAndCleanupTest(unittest.TestCase):
    def test_render_preserves_empty_frames_and_does_not_mix_shots(self):
        frames = [np.full((100, 120, 3), 60+i, np.uint8) for i in range(4)]
        originals = [frame.copy() for frame in frames]
        masks = np.zeros((4, 100, 120), np.uint8)
        masks[1:3, 40:60, 50:70] = 255
        jobs = build_windows(masks, [(0, 2), (2, 4)], 0)
        saved = []
        writer = Mock()
        writer.write.side_effect = lambda frame: saved.append(frame.copy())
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline, '_read_frames', return_value=iter(frames)), patch('backend.tools.video_io.FFmpegVideoWriter', return_value=writer):
            folder = Path(temp)
            for job in jobs:
                np.save(folder/f'job{job["id"]:05}.npy', np.full(
                    (job['stop']-job['start'], 240, 432, 3), 100+job['id']*80, np.uint8))
            active = pipeline._render('unused', masks, jobs, 0, folder, folder/'out.mp4', 24, (120, 100), Mock())
        self.assertEqual(active, 2)
        np.testing.assert_array_equal(saved[0], originals[0])
        np.testing.assert_array_equal(saved[3], originals[3])
        np.testing.assert_array_equal(saved[1][50, 60], [100]*3)
        np.testing.assert_array_equal(saved[2][50, 60], [180]*3)
        for frame, original in zip(saved, originals):
            np.testing.assert_array_equal(frame[:25], original[:25])

    def test_model_failure_cleans_temp_files(self):
        frame = np.full((100, 120, 3), 70, np.uint8)
        capture = Mock()
        capture.get.side_effect = lambda key: {
            cv2.CAP_PROP_FRAME_WIDTH: 120, cv2.CAP_PROP_FRAME_HEIGHT: 100,
            cv2.CAP_PROP_FPS: 24, cv2.CAP_PROP_FRAME_COUNT: 1,
        }[key]
        def scan(video, detector, area, band, step, path, progress, count):
            mask = np.zeros((100, 120), np.uint8)
            mask[40:60, 50:70] = 255
            path.write_bytes(mask.tobytes())
            return [[]], [(0, 1)], 1
        mux = Mock()
        with tempfile.TemporaryDirectory() as temp, patch.object(pipeline.cv2, 'VideoCapture', return_value=capture), patch.object(pipeline, '_scan', side_effect=scan), patch.object(pipeline, '_read_frames', return_value=iter([frame])), patch.object(pipeline, '_infer', side_effect=RuntimeError('model failed')):
            with self.assertRaisesRegex(RuntimeError, 'model failed'):
                pipeline.run_glyph_removal(Path(temp)/'in.mp4', None,
                    api._normalize_options({'mask_strategy': 'glyph'}), Mock(), Mock(), mux)
            self.assertEqual(list(Path(temp).iterdir()), [])
        mux.assert_not_called()


if __name__ == '__main__':
    unittest.main()
