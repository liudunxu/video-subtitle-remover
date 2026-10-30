import tempfile
import unittest
from pathlib import Path
from http import HTTPStatus
from unittest.mock import Mock, patch

import api


class RemovalOptionsTest(unittest.TestCase):
    def test_default_request_uses_quality_parameters(self):
        options = api._normalize_options({})
        expected = {
            "mode": "sttn",
            "auto_detect_area": True,
            "sttn_skip_detection": False,
            "subtitle_area_deviation_pixel": 12,
            "subtitle_area_deviation_pixel_y": 16,
            "sttn_band_margin": 80,
            "post_lama_refine": False,
            "auto_residual_cleanup": False,
            "post_verify_blur": False,
            "empty_detection_full_inpaint": False,
        }
        self.assertEqual({key: options[key] for key in expected}, expected)

    def test_callers_can_restore_previous_defaults(self):
        previous = {
            "subtitle_area_deviation_pixel": 44,
            "subtitle_area_deviation_pixel_y": 44,
            "sttn_band_margin": 0,
            "post_lama_refine": True,
            "empty_detection_full_inpaint": True,
        }
        options = api._normalize_options({"options": previous})
        self.assertEqual({key: options[key] for key in previous}, previous)

    def test_refinement_mode_still_opts_in(self):
        options = api._normalize_options({"options": {"mode": "sttn_then_lama"}})
        self.assertEqual(options["mode"], "sttn")
        self.assertTrue(options["post_lama_refine"])

    def test_multiline_explicit_opt_out_is_preserved(self):
        options = api._normalize_options({"options": {
            "ocr_preset": "multiline",
            "post_lama_refine": False,
            "auto_residual_cleanup": False,
        }})
        self.assertTrue(options["_explicit_post_lama_refine"])
        self.assertTrue(options["_explicit_auto_residual_cleanup"])
        self.assertFalse(options["post_lama_refine"])
        self.assertFalse(options["auto_residual_cleanup"])

    def test_capabilities_matches_refinement_default(self):
        handler = object.__new__(api.RemoverAPIHandler)
        handler.headers = {}
        handler.path = "/api/capabilities"
        handler._send_json = Mock()
        handler.do_GET()
        status, payload = handler._send_json.call_args.args
        self.assertEqual(status, HTTPStatus.OK)
        self.assertEqual(
            payload["features"]["post_lama_refine_default"],
            api._normalize_options({})["post_lama_refine"],
        )


class SubAreaMarginTest(unittest.TestCase):
    def test_default_margin_is_zero(self):
        self.assertEqual(api._normalize_options({})["sub_area_margin"], 0)

    def test_top_level_key_is_whitelisted(self):
        self.assertEqual(api._normalize_options({"sub_area_margin": 24})["sub_area_margin"], 24)

    def test_margin_expands_all_sides(self):
        self.assertEqual(
            api._expand_sub_area((100, 200, 50, 300), 1920, 1080, 16),
            [84, 216, 34, 316],
        )

    def test_expansion_is_clamped_to_frame(self):
        self.assertEqual(
            api._expand_sub_area((4, 1076, 2, 1918), 1920, 1080, 40),
            [0, 1080, 0, 1920],
        )

    def test_zero_margin_still_clamps_out_of_bounds_area(self):
        self.assertEqual(
            api._expand_sub_area((100, 5000, 50, 300), 1920, 1080, 0),
            [100, 1080, 50, 300],
        )

    def test_area_outside_frame_is_rejected(self):
        with self.assertRaises(api.RequestError) as ctx:
            api._expand_sub_area((2000, 3000, 50, 300), 1920, 1080, 0)
        self.assertEqual(ctx.exception.status, HTTPStatus.BAD_REQUEST)


class RemovalAreaRoutingTest(unittest.TestCase):
    def test_glyph_auto_area_is_full_width_but_explicit_and_legacy_stay_unchanged(self):
        detected = [831, 1871, 108, 1066]
        explicit = [1400, 1600, 100, 980]
        cases = [
            ('sttn', 'glyph', None, True, [831, 1871, 0, 1080]),
            ('propainter', 'glyph', None, True, [831, 1871, 0, 1080]),
            ('sttn', 'rectangle', None, True, detected),
            ('sttn', 'glyph', explicit, True, explicit),
            ('propainter', 'glyph', explicit, True, explicit),
            ('sttn', 'glyph', None, False, None),
        ]
        for mode, strategy, supplied, auto, expected in cases:
            with self.subTest(mode=mode, strategy=strategy, supplied=supplied, auto=auto):
                with tempfile.TemporaryDirectory() as temp:
                    source = Path(temp)/'source.mp4'
                    source.touch()
                    payload = {'_uploaded_video_path': str(source), 'options': {
                        'mode': mode, 'mask_strategy': strategy, 'auto_detect_area': auto,
                    }}
                    if supplied is not None:
                        payload['sub_area'] = supplied
                    handler = object.__new__(api.RemoverAPIHandler)
                    handler.headers = {}
                    handler.path = '/api/remove-subtitle'
                    handler._send_json = Mock()
                    # Exercise the real request routing up to the model boundary.
                    with patch.object(api, 'WORK_DIR', Path(temp)), patch.object(
                        api, '_read_payload', return_value=payload
                    ), patch.object(api, '_sweep_stale_job_dirs'), patch.object(
                        api, '_set_job_progress'
                    ), patch.object(api, '_probe_video_stream_geometry', return_value={
                        'width': 1080, 'height': 1920,
                    }), patch.object(api, '_run_subtitle_area_detection', return_value={
                        'sub_area': detected, 'video_size': {'width': 1080, 'height': 1920},
                    }) as detect, patch.object(api, '_run_subtitle_remover', side_effect=
                        api.RequestError(HTTPStatus.UNPROCESSABLE_ENTITY, 'model boundary')
                    ) as run:
                        handler.do_POST()
                    run.assert_called_once()
                    area = run.call_args.args[1]
                    self.assertEqual(list(area) if area is not None else None, expected)
                    self.assertEqual(detect.call_count, int(supplied is None and auto))
                    self.assertEqual(handler._send_json.call_args.args,
                                     (HTTPStatus.UNPROCESSABLE_ENTITY, {'error': 'model boundary'}))


if __name__ == "__main__":
    unittest.main()
