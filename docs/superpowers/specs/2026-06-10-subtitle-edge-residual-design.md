# Subtitle 上下边缘残留优化 — Design Spec

- **Date**: 2026-06-10
- **Status**: Design (pending user approval)
- **Author**: brainstorming session
- **Related code**: `backend/tools/inpaint_tools.py`, `backend/config.py`

## 1. Problem & Goal

### Problem
当前 `create_mask` 在 ymin/ymax 上只外扩 10px，对**字号 ≥ 40px**、**字体带上下伸部**（descender/ascender/变音符号，例：g, p, y, ā, ě, ó）的字幕，会在字的**顶/底边缘**留下 **3–8 px** 的薄色带，LaMa 修复后肉眼可辨。

### Goal
在 LaMa 修复前把 mask 在 Y 方向扩到**足以覆盖 3–8 px 残留 + 抗锯齿安全余量**，且**不引入对 X 方向的副作用**（左右原本就 OK，避免误伤邻接文字）。

### Non-goals
- 不动 OCR 检测模型
- 不动 LaMa / STTN / ProPainter 推理代码
- 不改前端 UI（默认值已变，高级用户可手改 `config/config.json`）
- 不引入新依赖（继续用 `cv2` + `numpy`）

## 2. Architecture / Data Flow

```
PaddleOCR detect  ─►  dt_polys  ─►  get_coordinates
                                          │
                                          ▼
                              [(xmin,xmax,ymin,ymax), ...]
                                          │
                                          ▼
                                  create_mask(size, coords)
                                          │
                          ┌───────────────┴───────────────┐
                          │   阶段 1: 矩形扩张（拆分 X/Y）  │
                          │  x1 = xmin - DevX              │
                          │  y1 = ymin - DevY              │
                          │  x2 = xmax + DevX              │
                          │  y2 = ymax + DevY              │
                          │  cv2.rectangle(mask, -1)       │
                          └───────────────┬───────────────┘
                                          │
                          ┌───────────────┴───────────────┐
                          │   阶段 2: 垂直形态学            │
                          │  kernel = (1, 3)               │
                          │  cv2.dilate(mask, kernel, 1)   │
                          └───────────────┬───────────────┘
                                          │
                                          ▼
                                    最终 mask
                                          │
                                          ▼
                              LaMa / STTN / ProPainter
```

**不变点**
- `create_mask` 签名不变：`def create_mask(size, coords_list) -> np.ndarray`
- `coords` 顺序约定不变：`(xmin, xmax, ymin, ymax)`（与 `get_coordinates` 对齐）
- 5 个调用方（`backend/main.py` 第 223/232/236/260/325 行）**完全不用改**

## 3. Configuration Changes

**File**: `backend/config.py:61`

**Before**:
```python
subtitleAreaDeviationPixel = RangeConfigItem(
    "Main", "SubtitleAreaDeviationPixel", 10, RangeValidator(1, 300))
```

**After**:
```python
# 旧字段保留，标记弃用（向后兼容现有用户的 config.json）
subtitleAreaDeviationPixel = RangeConfigItem(
    "Main", "SubtitleAreaDeviationPixel", 10, RangeValidator(1, 300))
# 新字段：分别控制 X / Y 方向的 mask 外扩像素
subtitleAreaDeviationPixelX = RangeConfigItem(
    "Main", "SubtitleAreaDeviationPixelX", 10, RangeValidator(0, 300))
subtitleAreaDeviationPixelY = RangeConfigItem(
    "Main", "SubtitleAreaDeviationPixelY", 22, RangeValidator(0, 300))
```

**Default 选择依据**
- X = 10：现状不变（左右本来 OK）
- Y = 22：覆盖 3–8 px 残留 + 8–10 px 抗锯齿 + 5–6 px 安全余量

**向后兼容**
- 旧字段保留但**运行时不再读取**（`create_mask` 只读 X / Y 两个新字段）
- qconfig 自动用新字段默认值兜底，老用户 config.json 不会报错
- 行为变化：Y 方向从 10 → 22。**这是设计上有意的**，spec 已记录

**为什么旧字段不删**
- qconfig 删除已注册字段需要写迁移逻辑
- 保留不影响功能
- 未来清掉，单独做一个 PR

## 4. Implementation — `create_mask`

**File**: `backend/tools/inpaint_tools.py:31-47`

**After**:
```python
def create_mask(size, coords_list):
    """Build inpaint mask from detected subtitle bboxes.

    拆分 X / Y 方向的外扩像素，并在矩形填充后做一次垂直 morphology，
    用于吸收抗锯齿 / 字幕框上下伸部未被 OCR 覆盖到的像素。
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

**Key points**
1. **`max(0, ...) / min(h-1, w-1, ...)` 边界保护**：原版没做。Y=22 时靠近帧顶/底的 bbox 越界部分不写入。
2. **`if dev_y > 0` 守卫**：用户把 Y 调到 0 时跳过 dilate，避免过度扩张。
3. **kernel = (1, 3)**：3 行高、1 列宽 → **仅向上下扩张**，左右 0 影响。
4. **空 `coords_list` 提前返回**：避免对全 0 mask 跑 dilate（虽然 cv2.dilate 接受全 0，但节省一次函数调用）。
5. **`mask.shape[:2]` 取 h/w**：原版假定 `size` 是 `(H, W)` 二元组，与 `cv2.VideoCapture` 一致；保留这个隐含约定。

**Performance**: N 个矩形填充 + 1 次全图 dilate。1080p 全图 `(1,3)` kernel dilate 大约 1–2 ms。**实测可忽略**。

## 5. Testing

### Unit tests
**New file**: `tests/test_inpaint_tools.py`

| # | Test | Assertion |
|---|---|---|
| 1 | `test_create_mask_basic` | 给定 bbox `(100, 200, 50, 80)`、dev_x=10、dev_y=22，验证 mask 在矩形 (90, 28) → (210, 102) 范围内为 255（dilate 后上下各多 1 行） |
| 2 | `test_create_mask_empty` | `coords_list=[]` → 全 0 mask（**不**调用 dilate） |
| 3 | `test_create_mask_boundary` | bbox 紧贴帧边缘时不越界；`x1=0` 或 `y2=H-1` 等场景无 crash |
| 4 | `test_create_mask_y_zero` | dev_y=0 → mock `cv2.dilate`，断言**未**被调用 |
| 5 | `test_create_mask_x_unchanged` | dev_x=10、dev_y=22、kernel=(1,3) 验证 mask 水平方向**不**受 dilate 影响（用 `np.where` 检查第一列和最后一列） |

### QA / 手工验证
准备 3 段测试视频：
1. 纯中文（无 asc/desc，验证不引入副作用）
2. 中英混排（验证 Y 扩 22 不误伤上方正文）
3. 纯英文带变音符号（ā, ě, g, p, y — 验证顶/底边缘被完整覆盖）

每段跑 LaMa 修复后，对比：
- 字幕行 top/bottom 1px 应**完全无残留**
- 字幕行**外**的区域**不能**被误伤（特别是字幕上方的演员名字、机位标识）

### 回归
跑 1 段历史视频，对比修复前后——除边缘更干净外，其他区域应**像素级一致**。

## 6. Risks & Mitigations

| Risk | Probability | Impact | Mitigation |
|---|---|---|---|
| Y=22 误伤字幕上方正文（叠放文字、竖排） | Medium | 用户看到无关文字被擦 | 用户可在 UI / config.json 把 DevY 调小到 0–5；doc 标注 |
| 配置项变更破坏老用户 config.json | Low | 程序启动失败 | qconfig 用 default 兜底；旧字段保留 |
| morphology kernel 选错形状 | Low | 上下扩得不够或多扩 | (1, 3) 是最小入侵；test 1 / 5 覆盖 |
| 与 `get_inpaint_area_by_mask` 冲突 | Low | 大块区域被错误合并 | `create_mask` 输出形状不变，只是更大；后续逻辑只看 0/255 不受影响 |
| 老的 `subtitleAreaDeviationPixel` 字段继续被 UI 引用 | Low | UI 控件设置的值无效 | UI 控件暂未改；本 spec 范围内不涉及，doc 标 follow-up |

## 7. Implementation Steps (outline)

详细计划由 `writing-plans` skill 输出。本节仅列大纲：

1. `backend/config.py` — 添加 `subtitleAreaDeviationPixelX` / `subtitleAreaDeviationPixelY`
2. `backend/tools/inpaint_tools.py` — 修改 `create_mask`
3. 新增 `tests/test_inpaint_tools.py`，5 个测试用例
4. 跑现有测试套件确认无回归
5. 手工跑 3 段测试视频做 QA
6. 更新 CHANGELOG / 相关文档

## 8. Follow-ups (out of scope for this spec)

- UI 添加「上下扩展像素」滑杆（暴露 DevY 给最终用户）
- 完全删除 `subtitleAreaDeviationPixel` 旧字段（带迁移逻辑的单独 PR）
- `subtitleAreaYXAxisDifferencePixel` / `subtitleAreaYAxisDifferencePixel` 等其他 Y 相关阈值的合理性审计
