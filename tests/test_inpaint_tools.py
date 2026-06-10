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
