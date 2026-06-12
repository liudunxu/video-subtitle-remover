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
        """dev_x=10 / dev_y=22: rectangle (90,28)-(210,102) filled, then
        vertical dilate (1, 5) iter=2 extends ±4 rows in Y and horizontal
        dilate (3, 1) iter=1 extends ±1 col in X.
        """
        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 22)

        # bbox: xmin=100, xmax=200, ymin=50, ymax=80
        coords = [(100, 200, 50, 80)]
        # generous frame so we don't clip
        mask = create_mask((200, 300, 3), coords)

        # Phase 1: rectangle (90, 28) - (210, 102) inclusive on both endpoints.
        # Phase 2 vertical: kernel (1, 1+2*max(1, 22//8)) = (1, 5), iter=2 →
        # Y range extends 4 rows above and below.
        # Phase 2 horizontal: kernel (1+2*max(1, 10//8), 1) = (3, 1), iter=1 →
        # X range extends 1 col each side.
        # Final Y range: 24..106 (4 extra above and below).
        # Final X range: 89..211 (1 extra each side).
        expected_y_start, expected_y_end = 24, 106
        expected_x_start, expected_x_end = 89, 211

        for y in range(expected_y_start, expected_y_end + 1):
            row = mask[y, expected_x_start:expected_x_end + 1, 0]
            self.assertTrue(
                np.all(row == 255),
                f"row {y} should be fully 255 in X range, got {row}",
            )

        # Just outside the morph-extended range:
        # row 23 should NOT be 255 in the X range
        self.assertFalse(np.any(mask[23, expected_x_start:expected_x_end + 1, 0] == 255))
        # row 107 should NOT be 255 in the X range
        self.assertFalse(np.any(mask[107, expected_x_start:expected_x_end + 1, 0] == 255))
        # col 88 should NOT be 255 in the Y range (1 col before the morph extent)
        self.assertFalse(np.any(mask[expected_y_start:expected_y_end + 1, 88, 0] == 255))
        # col 212 should NOT be 255 in the Y range
        self.assertFalse(np.any(mask[expected_y_start:expected_y_end + 1, 212, 0] == 255))

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
        # After dilate (1,5) iter=2: top can't extend above 0 (clipped), but
        # bottom extends 4 rows down to 56.
        # X: x1=0 (clipped), x2=50+10=60. After horizontal dilate (3,1)
        # iter=1: x2 extends to 61.
        # So filled region: rows 0..56, cols 0..61.
        self.assertTrue(np.all(mask[0:57, 0:62, 0] == 255))
        # col 62 should be all 0 across the filled Y range
        self.assertFalse(np.any(mask[0:57, 62, 0] == 255))
        # row 57 should be all 0 in cols 0..61
        self.assertFalse(np.any(mask[57, 0:62, 0] == 255))

    def test_y_zero_skips_vertical_morphology(self):
        """dev_y=0 skips the vertical dilate; horizontal dilate still runs
        (gated independently on dev_x).
        """
        from unittest.mock import patch
        import cv2 as real_cv2

        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 0)

        coords = [(100, 200, 50, 80)]
        # side_effect=real_cv2.dilate so the real morphology still runs
        # while mock_dilate records call counts. A naive patch would
        # replace dilate with a MagicMock and break the mask shape
        # assertions below.
        with patch("backend.tools.inpaint_tools.cv2.dilate", side_effect=real_cv2.dilate) as mock_dilate:
            mask = create_mask((200, 300, 3), coords)

        # dilate IS called (for the horizontal pass) but the vertical
        # kernel never gets used. We don't assert kernel shape here — the
        # call-count check below plus the test_x_zero_skips_horizontal
        # test cover both halves of the per-axis gating.
        self.assertGreaterEqual(mock_dilate.call_count, 1)
        # X range after dev_x=10 + horizontal dilate (3,1) iter=1: 89..211.
        # Y range with dev_y=0 + no vertical dilate: 50..80.
        self.assertTrue(np.all(mask[50:81, 89:212, 0] == 255))
        # Y=49 and Y=81 should be 0
        self.assertFalse(np.any(mask[49, 89:212, 0] == 255))
        self.assertFalse(np.any(mask[81, 89:212, 0] == 255))

    def test_x_zero_skips_horizontal_morphology(self):
        """dev_x=0 skips the horizontal dilate; vertical dilate still runs."""
        from unittest.mock import patch
        import cv2 as real_cv2

        config.set(config.subtitleAreaDeviationPixelX, 0)
        config.set(config.subtitleAreaDeviationPixelY, 22)

        coords = [(100, 200, 50, 80)]
        with patch("backend.tools.inpaint_tools.cv2.dilate", side_effect=real_cv2.dilate) as mock_dilate:
            mask = create_mask((200, 300, 3), coords)

        # dilate IS called (for the vertical pass) but the horizontal
        # kernel never gets used.
        self.assertGreaterEqual(mock_dilate.call_count, 1)
        # X range with dev_x=0 + no horizontal dilate: 100..200.
        # Y range after dev_y=22 + vertical dilate (1,5) iter=2: 24..106.
        self.assertTrue(np.all(mask[24:107, 100:201, 0] == 255))
        # col 99 and col 201 should be 0 (X dilate didn't run)
        self.assertFalse(np.any(mask[24:107, 99, 0] == 255))
        self.assertFalse(np.any(mask[24:107, 201, 0] == 255))

    def test_both_axes_zero_no_dilate(self):
        """dev_x=0 AND dev_y=0 → no dilate is called (per-axis gating
        both kick in).
        """
        from unittest.mock import patch

        config.set(config.subtitleAreaDeviationPixelX, 0)
        config.set(config.subtitleAreaDeviationPixelY, 0)

        coords = [(100, 200, 50, 80)]
        with patch("backend.tools.inpaint_tools.cv2.dilate") as mock_dilate:
            mask = create_mask((200, 300, 3), coords)

        mock_dilate.assert_not_called()
        # Tight rectangle, no morphology: 100..200 × 50..80.
        self.assertTrue(np.all(mask[50:81, 100:201, 0] == 255))

    def test_horizontal_morphology_does_not_affect_y(self):
        """Horizontal kernel is 1 row tall; Y is not affected by it."""
        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 22)

        coords = [(100, 200, 50, 80)]
        mask = create_mask((200, 300, 3), coords)

        # Row 23 (1 row above the vertical morph extent) must be 0 across X
        self.assertFalse(np.any(mask[23, :, 0] == 255))
        # Row 107 (1 row below) must be 0 across X
        self.assertFalse(np.any(mask[107, :, 0] == 255))
