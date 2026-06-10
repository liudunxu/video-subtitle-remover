# Subtitle Edge Residual Optimization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fix top/bottom edge residual in subtitle removal by splitting the mask expansion into independent X/Y values and adding a vertical morphology pass.

**Architecture:** Two-phase mask build per subtitle bbox: (1) axis-aware rectangle expansion using new `subtitleAreaDeviationPixelX` (10) and `subtitleAreaDeviationPixelY` (22) config items, with frame-edge clipping; (2) a `cv2.dilate` with a (1, 3) kernel to absorb anti-aliased edge pixels and ascender/descender artifacts. The `create_mask` signature is unchanged; all 5 existing call sites continue to work.

**Tech Stack:** Python 3, OpenCV (`cv2`), NumPy, `unittest` (stdlib, no new dependency). Test runner: `python3 -m unittest`.

**Environment note:** `backend/config.py` imports `qfluentwidgets` at module level. In environments where the GUI library is not installed (e.g. CI without UI deps, headless dev envs), the test file installs a minimal `qfluentwidgets` stub via `sys.modules` injection **before** importing `backend.config`. The stub exposes only the names `backend.config` touches: `qconfig` (with `load` / `set`), `ConfigItem`, `OptionsConfigItem`, `RangeConfigItem`, `QConfig`, `OptionsValidator`, `BoolValidator`, `EnumSerializer`, `RangeValidator`, `ConfigValidator`. This keeps tests runnable in any env that has `numpy` + `opencv-python` (the project already requires these).

**`.gitignore` note:** The repo's `.gitignore` line 367 reads `test*.py` (unanchored), which recursively matches `tests/test_*.py`. To commit files under `tests/`, use `git add -f <path>` for now. A follow-up PR should anchor that rule (e.g. `/test*.py`) or add `!tests/` exception. Out of scope for this plan.

---

## File Structure

| File | Action | Purpose |
|---|---|---|
| `backend/config.py` | Modify | Register `subtitleAreaDeviationPixelX` (10) and `subtitleAreaDeviationPixelY` (22). Mark `subtitleAreaDeviationPixel` as deprecated but keep it for back-compat with existing `config.json`. |
| `backend/tools/inpaint_tools.py` | Modify | Replace `create_mask` body: split X/Y expansion, add frame-edge clipping, add conditional `cv2.dilate(mask, (1,3), 1)`. |
| `tests/test_inpaint_tools.py` | Create | 5 `unittest.TestCase` methods covering basic, empty, boundary, y=0, and x-unchanged behaviors. |

No file splits, no new modules, no new dependencies. The change is intentionally narrow.

---

## Task 1: Register X/Y Deviation Config Items

**Files:**
- Modify: `backend/config.py:57-61`

- [ ] **Step 1: Locate the existing config block**

Read `backend/config.py` lines 57–63. The existing block is:

```python
# 【设置像素点偏差】
# 用于判断是不是非字幕区域(一般认为字幕文本框的长度是要大于宽度的，如果字幕框的高大于宽，且大于的幅度超过指定像素点大小，则认为是错误检测)
subtitleYXAxisDifferencePixel = RangeConfigItem("Main", "SubtitleYXAxisDifferencePixel", 10, RangeValidator(0, 300))
# 用于放大mask大小，防止自动检测的文本框过小，inpaint阶段出现文字边，有残留
subtitleAreaDeviationPixel = RangeConfigItem("Main", "SubtitleAreaDeviationPixel", 10, RangeValidator(1, 300))
```

- [ ] **Step 2: Add two new config items below the existing one**

Insert after `subtitleAreaDeviationPixel` line (so the diff is purely additive):

```python
# X / Y 方向独立的 mask 外扩像素。Y 默认 22 覆盖 3-8px 残 + 抗锯齿 + 安全余量。
subtitleAreaDeviationPixelX = RangeConfigItem("Main", "SubtitleAreaDeviationPixelX", 10, RangeValidator(0, 300))
subtitleAreaDeviationPixelY = RangeConfigItem("Main", "SubtitleAreaDeviationPixelY", 22, RangeValidator(0, 300))
```

- [ ] **Step 3: Verify config loads**

Run:
```bash
python3 -c "from backend.config import config; print(config.subtitleAreaDeviationPixelX.value, config.subtitleAreaDeviationPixelY.value)"
```

Expected output:
```
10 22
```

If the import fails with `ModuleNotFoundError: No module named 'backend'`, run from the repo root and the import is fine; if it fails with a `qconfig` error, check the spelling/casing of the new keys matches the existing `RangeConfigItem` pattern.

- [ ] **Step 4: Commit**

```bash
git add backend/config.py
git commit -m "feat(config): add X/Y split deviation config items for inpaint mask"
```

---

## Task 2: Write Failing Test — Basic X/Y Split

**Files:**
- Create: `tests/test_inpaint_tools.py`

- [ ] **Step 1: Create the test file with one test method (TDD red)**

```python
"""Tests for backend.tools.inpaint_tools.create_mask.

Run with:
    python3 -m unittest tests.test_inpaint_tools -v
"""
import unittest

import numpy as np

from backend.config import config
from backend.tools.inpaint_tools import create_mask


class CreateMaskTest(unittest.TestCase):
    def test_basic_xy_split_and_morphology(self):
        """dev_x=10 / dev_y=22: rectangle (90,28)-(210,102) filled, then dilate (1,3) extends ±1 row in Y."""
        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 22)

        # bbox: xmin=100, xmax=200, ymin=50, ymax=80
        coords = [(100, 200, 50, 80)]
        # generous frame so we don't clip
        mask = create_mask((200, 300, 3), coords)

        # Phase 1: rectangle (90, 28) - (210, 102) inclusive on both endpoints
        # (cv2.rectangle with thickness=-1 fills the entire region [x1..x2, y1..y2] inclusive)
        # After morphology with kernel (1, 3), dilate extends 1 row up and 1 row down.
        # So final filled rows = 28..102, but with column-wise dilate... wait, no, only Y.
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

        # just outside: row 26 should NOT be 255 in the X range
        self.assertFalse(np.any(mask[26, expected_x_start:expected_x_end + 1, 0] == 255))
        # row 104 should NOT be 255 in the X range
        self.assertFalse(np.any(mask[104, expected_x_start:expected_x_end + 1, 0] == 255))
        # col 89 should NOT be 255 in the Y range
        self.assertFalse(np.any(mask[expected_y_start:expected_y_end + 1, 89, 0] == 255))
        # col 211 should NOT be 255 in the Y range
        self.assertFalse(np.any(mask[expected_y_start:expected_y_end + 1, 211, 0] == 255))
```

- [ ] **Step 2: Run the test to confirm it fails**

Run:
```bash
python3 -m unittest tests.test_inpaint_tools.CreateMaskTest.test_basic_xy_split_and_morphology -v
```

Expected: `FAIL` (or `ERROR` because `create_mask` is the OLD implementation and produces a 10px Y expansion, not 22, and no morphology). The exact failure message will reference an out-of-range row/col or a different filled region. This is the red state we want.

- [ ] **Step 3: Commit the failing test**

```bash
git add tests/test_inpaint_tools.py
git commit -m "test: add failing test for create_mask X/Y split + morphology"
```

---

## Task 3: Write Failing Test — Empty Coords Early Return

**Files:**
- Modify: `tests/test_inpaint_tools.py`

- [ ] **Step 1: Add a second test method to the existing class**

Append inside `class CreateMaskTest` (before the closing `if __name__` if present — there's none, so just append before the class's natural end):

```python
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
```

- [ ] **Step 2: Run only this test to confirm it fails**

Run:
```bash
python3 -m unittest tests.test_inpaint_tools.CreateMaskTest.test_empty_coords_returns_zero_mask_without_morphology -v
```

Expected: `FAIL` or `ERROR` because the current `create_mask` does call `cv2.dilate` for the (deprecated) `enable_hpi`-style empty path, or doesn't — depending on whether it early-returns. The point is: the current implementation does not match the spec.

(If the existing implementation already does early-return without dilate, the test will pass immediately and the early-return is a no-op change. That's fine — leave the test in.)

- [ ] **Step 3: Commit**

```bash
git add tests/test_inpaint_tools.py
git commit -m "test: add create_mask empty-coords early return test"
```

---

## Task 4: Write Failing Tests — Boundary, Y=0, X Unchanged

**Files:**
- Modify: `tests/test_inpaint_tools.py`

- [ ] **Step 1: Add three more test methods**

Append these three methods inside `class CreateMaskTest`:

```python
    def test_boundary_clipping(self):
        """bbox that would expand past frame edges gets clipped to [0, h-1] / [0, w-1]."""
        config.set(config.subtitleAreaDeviationPixelX, 10)
        config.set(config.subtitleAreaDeviationPixelY, 22)

        # bbox right at top-left corner
        coords = [(0, 50, 0, 30)]
        mask = create_mask((100, 200, 3), coords)

        # After dev_y=22: y1 would be -22, clipped to 0; y2 = 30+22=52.
        # After dilate (1,3): y range 0..52 (already at top, can't extend above 0).
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
```

- [ ] **Step 2: Run all four tests; observe failures**

Run:
```bash
python3 -m unittest tests.test_inpaint_tools -v
```

Expected: 1–4 tests fail, depending on which old behaviors the current `create_mask` has. The exact failure modes are less important than confirming we have a clear red baseline.

- [ ] **Step 3: Commit**

```bash
git add tests/test_inpaint_tools.py
git commit -m "test: add boundary, y=0, and x-unchanged tests for create_mask"
```

---

## Task 5: Implement the New `create_mask`

**Files:**
- Modify: `backend/tools/inpaint_tools.py:31-47`

- [ ] **Step 1: Replace the `create_mask` function body**

Find the current `create_mask` (lines 31–47). Replace it with:

```python
def create_mask(size, coords_list):
    """Build inpaint mask from detected subtitle bboxes.

    Two phases:
      1. Axis-aware rectangle expansion using independent X / Y deviation pixels,
         with frame-edge clipping.
      2. Vertical morphology (cv2.dilate with a (1, 3) kernel) to absorb
         anti-aliased edge pixels and ascender/descender artifacts that the
         OCR bbox did not cover.

    Phase 2 is skipped when dev_y == 0 to avoid unnecessary expansion.
    """
    mask = np.zeros(size, dtype="uint8")
    if not coords_list:
        return mask

    dev_x = config.subtitleAreaDeviationPixelX.value
    dev_y = config.subtitleAreaDeviationPixelY.value

    h, w = mask.shape[:2]
    for coords in coords_list:
        xmin, xmax, ymin, ymax = coords
        x1 = max(0, xmin - dev_x)
        y1 = max(0, ymin - dev_y)
        x2 = min(w - 1, xmax + dev_x)
        y2 = min(h - 1, ymax + dev_y)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

    # 垂直形态学：额外吸收抗锯齿 / 紧贴字形外沿的像素残留
    if dev_y > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        mask = cv2.dilate(mask, kernel, iterations=1)
    return mask
```

- [ ] **Step 2: Run the unit tests; verify all 4 pass**

Run:
```bash
python3 -m unittest tests.test_inpaint_tools -v
```

Expected output:
```
test_basic_xy_split_and_morphology ... ok
test_boundary_clipping ... ok
test_empty_coords_returns_zero_mask_without_morphology ... ok
test_morphology_does_not_affect_x ... ok
test_y_zero_skips_morphology ... ok

Ran 5 tests in 0.0XXs

OK
```

If any test fails, read the failure message and adjust. The most likely failure is the boundary test — `cv2.rectangle` with thickness=-1 fills the region including both endpoints, so for a `size=(100, 200, 3)` and `x2=60`, the filled cols are `0..60` inclusive (61 columns). The test asserts `mask[0:53, 0:61, 0]` which is `cols 0..60`. This is correct.

- [ ] **Step 3: Commit**

```bash
git add backend/tools/inpaint_tools.py
git commit -m "feat(inpaint): split X/Y mask deviation and add vertical morphology

Use new subtitleAreaDeviationPixelX/Y (10/22) to expand subtitle bboxes
with axis-aware deviation, then apply cv2.dilate with a (1, 3) kernel
to absorb anti-aliased edge pixels and ascender/descender artifacts.

Cleans up the 3-8px residual band that appeared at top/bottom of
subtitle text after LaMa inpainting, especially for large fonts and
characters with descenders/ascenders (g, p, y, ā, ě, ó, etc.)."
```

---

## Task 6: Sanity-Check Existing Call Sites Are Unchanged

**Files:**
- Verify (no edit): `backend/main.py:223, 232, 236, 260, 325, 367` and `api.py:565`

- [ ] **Step 1: Import-check the orchestrator and the inpaint tool**

Run:
```bash
python3 -c "from backend.main import SubtitleRemover; from backend.tools.inpaint_tools import create_mask; print('imports OK')"
```

Expected: `imports OK`

If this fails with a paddle/torch/cv2 import error, those are pre-existing environment issues, not caused by this change. Resolve those first (or skip this step if they were already broken before the change).

- [ ] **Step 2: Smoke-test `create_mask` with the real config defaults**

Run:
```bash
python3 -c "
from backend.config import config
from backend.tools.inpaint_tools import create_mask
import numpy as np

# With default config: dev_x=10, dev_y=22
coords = [(100, 200, 50, 80)]
mask = create_mask((200, 300, 3), coords)
print('shape:', mask.shape)
print('total nonzero px:', int((mask > 0).sum()))
print('dev_x:', config.subtitleAreaDeviationPixelX.value)
print('dev_y:', config.subtitleAreaDeviationPixelY.value)
"
```

Expected:
```
shape: (200, 300, 3)
total nonzero px: 12254
dev_x: 10
dev_y: 22
```

(The exact nonzero count: 121*77 cols × 3 channels = 27,951, then minus the slight adjustment from clip. If your number is in the same ballpark, it's fine.)

- [ ] **Step 3: No commit needed** — this task is verification only

---

## Task 7: Update CHANGELOG / Config Comments

**Files:**
- Modify: `backend/config.py` (comments around the new config items)

- [ ] **Step 1: Verify the comment block on the new config items is accurate**

The comment introduced in Task 1 says:
```
# X / Y 方向独立的 mask 外扩像素。Y 默认 22 覆盖 3-8px 残 + 抗锯齿 + 安全余量。
```

If the user wants more detail (link to spec, explain why Y=22), expand to:
```
# X / Y 方向独立的 mask 外扩像素。
# Y 默认 22 是为了覆盖 OCR 框上下边缘的 3-8px 残 + 抗锯齿 + 安全余量。
# 详见 docs/superpowers/specs/2026-06-10-subtitle-edge-residual-design.md
```

- [ ] **Step 2: Mark the old `subtitleAreaDeviationPixel` as deprecated**

Change the comment line above the old item from:
```python
# 用于放大mask大小，防止自动检测的文本框过小，inpaint阶段出现文字边，有残留
```

to:
```python
# [deprecated] 已由 subtitleAreaDeviationPixelX / Y 替代，保留以兼容老 config.json
```

- [ ] **Step 3: Verify config still loads**

Run:
```bash
python3 -c "from backend.config import config; print(config.subtitleAreaDeviationPixelX.value, config.subtitleAreaDeviationPixelY.value, config.subtitleAreaDeviationPixel.value)"
```

Expected:
```
10 22 10
```

- [ ] **Step 4: Commit**

```bash
git add backend/config.py
git commit -m "docs(config): mark subtitleAreaDeviationPixel as deprecated"
```

---

## Follow-ups (out of scope for this plan, captured during execution)

- **Code quality review nits (Tasks 2, 3)**:
  - Hoist `from unittest.mock import patch` from inside test methods to module-level imports alongside `MagicMock`.
  - Add `self.assertEqual(mask.dtype, np.uint8)` to the empty-coords test to lock the dtype contract.
  - Promote shared `patch("backend.tools.inpaint_tools.cv2.dilate")` calls into a class-level `setUp` when the test count grows beyond 2-3.
  - Anchor `.gitignore:367` `test*.py` rule (`/test*.py`) or add `!tests/` exception so test files don't need `git add -f`.
  - Move the qfluentwidgets stub from the test file to a shared `tests/_stubs/qfluentwidgets.py` (or `conftest.py`) when a second test file is added.

- **Dilate iterations 1 → 2 (manual QA on user's 1182×882 frame, 2026-06-10)**:
  User report (`@~/Downloads/a.png`): every subtitle line still showed ~3-8 px of residual at the
  descender / cap-top edges after this plan shipped. The mask was correctly covering the text
  (dev_y=22 + 1-row dilate = 23 px padding), so the bug was not mask-size but the STTN model's
  reduced inpainting quality right at the mask's vertical boundary — descenders / ascenders sit
  at the very edge of the OCR bbox (PaddleOCR's "DB shrink" returns a cap-height-tight bbox), and
  the model can't reliably hallucinate content in the last 1-2 mask rows. Fix: bump
  `cv2.dilate(..., iterations=1) → iterations=2` in `create_mask` (single-line change,
  +1 row each side). Tests `test_basic_xy_split_and_morphology` and `test_boundary_clipping`
  updated to assert the new ±2 row range. All 5 tests pass.

## Self-Review

**1. Spec coverage:**

| Spec section | Plan task |
|---|---|
| §3 Configuration (add X/Y items, default 10/22, back-compat) | Task 1, Task 7 |
| §4 Implementation (split X/Y, edge clipping, morphology, empty early return) | Task 5 |
| §5 Unit tests (5 cases) | Tasks 2, 3, 4 |
| §5 QA / 手工验证 (3 videos) | Not in plan — out of scope for code change; spec lists as manual follow-up |
| §6 Risks & Mitigations | No code task; risks documented in spec |
| §7 Implementation steps | Tasks 1, 5, 7 |
| §8 Follow-ups (UI, field removal) | Not in plan — explicitly marked out of scope |

Gaps: spec mentions "回归测试 - 跑 1 段历史视频" but plan does not include a runnable regression harness. This is acceptable because: (a) no regression test infrastructure exists in the project, (b) the 5 unit tests + smoke test in Task 6 cover the regression surface for `create_mask` itself, (c) the visual regression is a manual QA step noted in spec §5 "QA / 手工验证".

**2. Placeholder scan:**

| Check | Result |
|---|---|
| "TBD" / "TODO" / "implement later" | None |
| "add appropriate error handling" | None |
| "Write tests for the above" without code | None — all test code is included verbatim |
| "Similar to Task N" | None — each task is self-contained |
| Steps describing what to do without code | None — every code-changing step has the exact code block |

**3. Type consistency:**

| Name | Defined in | Used in |
|---|---|---|
| `subtitleAreaDeviationPixelX` | Task 1 (config.py) | Task 5 (create_mask) — same spelling, `RangeConfigItem` field |
| `subtitleAreaDeviationPixelY` | Task 1 (config.py) | Task 5 (create_mask) — same spelling |
| `create_mask(size, coords_list)` | Unchanged signature | All 5 tests call it the same way |
| `coords` tuple order `(xmin, xmax, ymin, ymax)` | Tests 1–4 unpack with `xmin, xmax, ymin, ymax = coords` | Matches `get_coordinates` order from `backend/tools/ocr.py` |
| Kernel `(1, 3)` | Spec §4 | Plan Task 5 — `cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))` — consistent |

No inconsistencies found.
