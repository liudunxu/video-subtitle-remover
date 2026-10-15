import unittest
from http import HTTPStatus
from unittest.mock import Mock

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


if __name__ == "__main__":
    unittest.main()
