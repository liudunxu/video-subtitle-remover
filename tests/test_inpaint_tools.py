"""Tests for backend.tools.inpaint_tools.create_mask.

Run with:
    python3 -m unittest tests.test_inpaint_tools -v
"""
import sys
import unittest
from types import ModuleType
from unittest.mock import MagicMock

import numpy as np


def _install_qfluentwidgets_stub():
    """Install a minimal qfluentwidgets stub so backend.config can be imported.

    The real qfluentwidgets library is a PyQt5/PySide6 GUI dep. In headless
    test envs, we inject a stub that exposes only the names backend.config
    touches: qconfig (with load/set), ConfigItem, OptionsConfigItem,
    RangeConfigItem, QConfig, OptionsValidator, BoolValidator, EnumSerializer,
    RangeValidator, ConfigValidator.
    """
    if 'qfluentwidgets' in sys.modules:
        return

    qfw = ModuleType('qfluentwidgets')

    class _ConfigItem:
        def __init__(self, group, name, default, validator=None, restart=False):
            self.group = group
            self.name = name
            self.default = default
            self.validator = validator
            self.value = default

        def __repr__(self):
            return f"<ConfigItem {self.group}/{self.name} value={self.value!r}>"

    class _QConfig:
        @classmethod
        def set(cls, item, value):
            item.value = value

    class _QConfigSingleton:
        def __init__(self):
            self._store = {}

        def load(self, path, config):
            # No-op: stub doesn't persist anything.
            pass

        def set(self, item, value):
            item.value = value

    qfw.qconfig = _QConfigSingleton()
    qfw.ConfigItem = _ConfigItem
    qfw.OptionsConfigItem = _ConfigItem
    qfw.RangeConfigItem = _ConfigItem
    qfw.QConfig = _QConfig
    qfw.OptionsValidator = MagicMock
    qfw.BoolValidator = MagicMock
    qfw.EnumSerializer = MagicMock
    qfw.RangeValidator = MagicMock
    qfw.ConfigValidator = MagicMock

    sys.modules['qfluentwidgets'] = qfw


# Install stub BEFORE importing backend.config.
_install_qfluentwidgets_stub()

from backend.config import config  # noqa: E402
from backend.tools.inpaint_tools import create_mask  # noqa: E402


class CreateMaskTest(unittest.TestCase):
    def test_basic_xy_split_and_morphology(self):
        """dev_x=10 / dev_y=22: rectangle (90,28)-(210,102) filled, then dilate (1,3) extends ±1 row in Y."""
        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 22)

        # bbox: xmin=100, xmax=200, ymin=50, ymax=80
        coords = [(100, 200, 50, 80)]
        # generous frame so we don't clip
        mask = create_mask((200, 300, 3), coords)

        # Phase 1: rectangle (90, 28) - (210, 102) inclusive on both endpoints.
        # Phase 2: dilate with (1, 3) kernel extends 1 row above and 1 below.
        # Final Y range: 27..103 (1 extra above and below).
        # Final X range: 90..210 (unchanged — kernel width is 1).
        expected_y_start, expected_y_end = 27, 103
        expected_x_start, expected_x_end = 90, 210

        for y in range(expected_y_start, expected_y_end + 1):
            row = mask[y, expected_x_start:expected_x_end + 1, 0]
            self.assertTrue(
                np.all(row == 255),
                f"row {y} should be fully 255 in X range, got {row}",
            )

        # Just outside the morph-extended Y range:
        # row 26 should NOT be 255 in the X range
        self.assertFalse(np.any(mask[26, expected_x_start:expected_x_end + 1, 0] == 255))
        # row 104 should NOT be 255 in the X range
        self.assertFalse(np.any(mask[104, expected_x_start:expected_x_end + 1, 0] == 255))
        # col 89 should NOT be 255 in the Y range
        self.assertFalse(np.any(mask[expected_y_start:expected_y_end + 1, 89, 0] == 255))
        # col 211 should NOT be 255 in the Y range
        self.assertFalse(np.any(mask[expected_y_start:expected_y_end + 1, 211, 0] == 255))

    def test_empty_coords_returns_zero_mask_without_morphology(self):
        """Empty coords_list returns all-zero mask. cv2.dilate must NOT be called."""
        from unittest.mock import patch

        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 22)

        with patch("backend.tools.inpaint_tools.cv2.dilate") as mock_dilate:
            mask = create_mask((50, 80, 3), [])

        self.assertEqual(mask.shape, (50, 80, 3))
        self.assertTrue(np.all(mask == 0))
        mock_dilate.assert_not_called()

    def test_boundary_clipping(self):
        """bbox that would expand past frame edges gets clipped to [0, h-1] / [0, w-1]."""
        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 22)

        # bbox right at top-left corner
        coords = [(0, 50, 0, 30)]
        mask = create_mask((100, 200, 3), coords)

        # After dev_y=22: y1 would be -22, clipped to 0; y2 = 30+22=52.
        # After dilate (1,3): y range stays 0..52 (already at top, can't extend above 0).
        # X: x1=0 (clipped), x2=50+10=60. Dilate doesn't extend X.
        # So filled region: rows 0..52, cols 0..60.
        self.assertTrue(np.all(mask[0:53, 0:61, 0] == 255))
        # col 61 should be all 0
        self.assertFalse(np.any(mask[0:53, 61, 0] == 255))
        # row 53 should be all 0 in cols 0..60
        self.assertFalse(np.any(mask[53, 0:61, 0] == 255))

    def test_y_zero_skips_morphology(self):
        """dev_y=0 skips cv2.dilate entirely."""
        from unittest.mock import patch

        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 0)

        coords = [(100, 200, 50, 80)]
        with patch("backend.tools.inpaint_tools.cv2.dilate") as mock_dilate:
            mask = create_mask((200, 300, 3), coords)

        mock_dilate.assert_not_called()
        # X range: 90..210. Y range: 50..80 (no expansion, no morphology).
        self.assertTrue(np.all(mask[50:81, 90:211, 0] == 255))
        # Y=49 and Y=81 should be 0
        self.assertFalse(np.any(mask[49, 90:211, 0] == 255))
        self.assertFalse(np.any(mask[81, 90:211, 0] == 255))

    def test_morphology_does_not_affect_x(self):
        """Kernel (1, 3) is 1 col wide; X is not affected by dilate."""
        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 22)

        coords = [(100, 200, 50, 80)]
        mask = create_mask((200, 300, 3), coords)

        # X: dev_x=10 → x1=90, x2=210. With (1,3) kernel, X range unchanged.
        # col 89 must be 0 across the Y range
        self.assertFalse(np.any(mask[:, 89, 0] == 255))
        # col 211 must be 0 across the Y range
        self.assertFalse(np.any(mask[:, 211, 0] == 255))
        # but col 90 and col 210 are filled in the Y range
        self.assertTrue(np.any(mask[:, 90, 0] == 255))
        self.assertTrue(np.any(mask[:, 210, 0] == 255))
