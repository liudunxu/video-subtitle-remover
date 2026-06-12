# 字幕擦除边缘残留优化 — 一周策略回顾与经验总结

- **Date**: 2026-06-12
- **Scope**: 2026-06-05 ~ 2026-06-12 的字幕擦除边缘残留优化记录
- **Audience**: 接手字幕擦除方向的工程师 / 维护者
- **Status**: Recap — Round 1（核心优化）已合并；Round 2（旋钮清理 + 覆盖度补强）见 §3.9 / §6 / §7

---

## 1. 一句话总结

**字幕上下边缘残留的本质是「OCR 给的 bbox 比字形紧」+「STTN 边界处 inpaint 质量差」的叠加。**
这两层问题被分别用「mask 形态学外扩」和「残字兜底两阶段修复」两套独立策略解决。
X 轴 / Y 轴必须分开调；Y 方向调多调少还要随 dev_y 自适应。

---

## 2. 问题溯源

| 现象 | 用户报告 | 根因（事后定位） |
|---|---|---|
| LaMa 修复后字幕顶/底有 3–8px 薄色带 | 字号 ≥ 40px、中英混排、变音符号场景 | PaddleOCR 的 DB shrink 输出 cap-height 紧框，`create_mask` 在 ymin/ymax 各只外扩 10px，覆盖不到字形外伸部 + 抗锯齿 |
| STTN inpaint 完成后字形侧边仍有薄边 | 1080p 视频、3 行字幕 | STTN 在 mask 边界 ±1–2 px 内 inpaint 质量急剧下降，对抗锯齿像素补不出来 |
| 字幕下方有 1–3px 的「阴影带」| descender / 变音符号场景 | 残字兜底阶段的对称 dilate 触及不到 glyph body 上面那条 1–3px 的暗带 |
| 多行字幕越往下行擦得越差 | 3 行 / 双行字幕 | `subtitleAreaDeviationPixelX` 一个值被同时映射给 X/Y，多行场景 Y 扩得不够 |

---

## 3. 已采纳的优化策略（按时间线 + 主题分组）

### 3.1 方向拆分：Mask 偏差 X / Y 分离

**提交链**：
- `9df4cd1` — `feat(inpaint): split X/Y mask deviation and add vertical morphology`
- `31d726d` — `feat(api,inpaint): add subtitle_area_deviation_pixel_y and scale morphology by dev_y`
- `9c99446` — `feat(api): sync axis-specific deviation fields & feather blur ROI edges`

**做了什么**：
- `backend/config.py` 新增 `subtitleAreaDeviationPixelX` (默认 10) / `subtitleAreaDeviationPixelY` (默认 22→44)
- `backend/tools/inpaint_tools.py:create_mask` 用 X/Y 独立值做矩形外扩
- 旧字段 `subtitleAreaDeviationPixel` 保留但标记 deprecated（兼容老 config.json）

**为什么 X 不用加**：
- 字幕横向是连续的字符，左右邻近字符也属于「需要擦除」目标
- 横向多扩 1px 不会引入新问题；纵向多扩 1px 可能误伤上方演员名 / 机位标识

**Y 默认值 22 → 44 的提升**：
- 22 在多行字幕下仍偏紧；44 在 descender / asc 上 8px 残 + AA + 安全余量都覆盖到
- 配合后面 3.2 的「morphology 跟 dev_y 自适应」放大后才稳

### 3.2 垂直形态学：dev_y 自适应

**提交链**：
- `9df4cd1` — 引入 `cv2.dilate(mask, (1, 3), iter=1)`
- `cf2b786` — `iter=1 → iter=2`（覆盖 STTN 边界 ±1–2 px 质量塌陷）
- `a1fb18f` — `kernel (1,3) → (1,5)`（进一步加 2 行）
- `31d726d` — `kernel/iter 随 dev_y 自适应`

**最终形态**（`inpaint_tools.py:70-74`）：
```python
if dev_y > 0:
    morph_h = min(1 + 2 * (dev_y // 10), 11)
    morph_iters = 2 if dev_y <= 30 else 3
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, morph_h))
    mask = cv2.dilate(mask, kernel, iterations=morph_iters)
```

**经验**：形态学参数不能写死，要随 Y 偏差线性放大。
- 3 行字幕 → dev_y 自然会大 → morphology 也更宽，刚好补偿
- 单行字幕 → dev_y 小 → morphology 收窄，不影响字幕上方区域

### 3.3 OCR bbox 垂直外扩（默认关闭）

**提交链**：
- `0662557` — `Expand OCR subtitle bbox y padding`（引入 `ocr_bbox_y_pad` 字段，默认 16）
- `db4ddc1` — `Disable OCR bbox y padding by default`（默认 16 → 0）

**做了什么**：
- 新增 `_padded_ocr_coords` / `_build_padded_subtitle_detector`，对 PaddleOCR 返回的 bbox 在 ymin/ymax 方向各加 N 像素
- 三处使用：`_run_subtitle_remover`、`_run_blur_cover`、`_run_subtitle_area_detection`
- `_normalize_options` / `_normalize_detect_options` 同步暴露 API 字段

**回退理由**（默认 16 → 0）：
- 加了 y_pad 后 mask 边界扩大 → STTN inpaint 区域扩大 → 部分视频上方正文被误伤
- 已经在 `create_mask` 阶段把 dev_y 拉到了 44，OCR 端再扩反而过度
- 默认值保守，0 由用户按需开启

**保留价值**：作为 API 调参旋钮保留，0–120 范围仍然合法。文字密集且字幕与上方文字间距小的视频可以试 6–12。

### 3.4 残字兜底：上下边缘非对称外扩

**提交链**：
- `80d22f9` — `feat(api): add residual_bottom_extra_px to mirror top edge handling`

**做了什么**（`api.py:1635-1644, 1719-1730`）：
- 已有 `residual_top_extra_px`（默认 6）— STTN 后对 residual mask 向上偏移 6px
- 新增 `residual_bottom_extra_px`（默认 6）— 向下偏移
- 用 `(1, 1+2*extra)` 的垂直 kernel 做 dilate，再 numpy 切片平移
- 范围 0..12，对齐已有 `residual_top_extra_px` 合约

**为什么需要这一步**：
- STTN 的 text-trace mask 上下对称，但字形的上沿 / 下沿残影不对称
- 单独靠 3.2 的形态学扩 mask 不够 — STTN 出来的图里残留的暗像素要靠二次 inpaint 补
- top_extra 先于 bottom_extra 做，是先「向上推 mask」后「向下推 mask」

### 3.5 残字兜底：垂直 close 步骤

**提交链**：与 `80d22f9` 同批

**做了什么**（`api.py:1650-1697`）：
- 新增 `residual_vertical_close_px`（默认 3）
- 对 residual mask 跑 `cv2.MORPH_CLOSE(vertical_close_kernel)`
- 消除「顶部阴影带与 glyph body 之间的 1–3 px 缝隙」

**为什么不能省**：
- top_extra 是「先 dilate 再向上平移」— 如果 dilate 后的 mask 与残影 mask 中间有 1–3 px 断带，shift 拿不到那个残影
- close 用 `(1, 1+2*vertical_close_px)` 形态学连接两者，让后续 shift 能「拖上去」

**和 3.4 的协同顺序**（`_run_residual_cleanup` 主循环）：
```
residual = _residual_text_mask(crop, options)        # 找残字
if vertical_close_px > 0:
    residual = cv2.morphologyEx(residual, MORPH_CLOSE, vertical_close_kernel)
# 主体循环：dilate → top_extra shift → bottom_extra shift → inpaint → 重检
```

### 3.6 Post-verify 模糊：ROI 边缘羽化

**提交链**：`9c99446` 同 commit

**做了什么**（`api.py` 的 post-verify blur 阶段）：
- 原本：硬性把模糊后的 bbox 区域 paste 回原图（能看到矩形 patch 边）
- 改为：用 np.linspace 生成的 ramp 在 bbox 边缘做 alpha 混合
- 四边各自独立做羽化：上下 ramp `(H,1)`，左右 ramp `(1,W)`

**经验**：**当 inpaint 区域和原图非 inpaint 区域有色差时，羽化边界是必修项**。硬 paste 在背景颜色变化、字幕底部阴影场景下肉眼可辨。

### 3.7 配置系统同步：API → backend

**提交链**：
- `9c99446` — `_apply_config_options` 把 `subtitleAreaDeviationPixel` 同时设给 X/Y 字段
- `31d726d` — 修正：Y 字段读 `subtitleAreaDeviationPixelY` 选项（不是 X 的复用）

**踩坑**：
- backend 的 `create_mask` 只读新 X/Y 字段，不读老字段
- 如果只设老的 `subtitleAreaDeviationPixel`，backend 用的是默认值，API 调用方完全感受不到自己在控制 mask
- 用户在前端把偏差调到 60，期望 mask 真的变宽；不同步就是「设置无效」

**铁律**：API 层的 option → backend 的 config 字段必须显式映射，写注释说明哪两个字段联动。

### 3.8 其他配套改动（同期）

| Commit | 作用 |
|---|---|
| `267bd01` | PaddleOCR GPU 检测改用 `paddle.device.cuda.device_count()`（`paddle.device.is_available()` 在 2.6.x 被移除） |
| `c564194` | 进度日志 throttle 10% → 2%，长任务可见性更好 |
| `79fb897` | PaddleOCR 在装了 paddlepaddle-gpu 时自动切 GPU |
| `bdb4014` | 新增 AGENTS.md / CLAUDE.md（karpathy 行为准则） |

### 3.9 Round 2：旋钮清理 + 覆盖度补强（基于 §6 评估执行）

在 §6「哪些可以去掉」的基础上执行的两组清理。每一项都跟「默认值选择」「旋钮暴露面」直接相关。

**3.9.1 合并 `residual_top_extra_px` + `residual_bottom_extra_px` → `residual_v_extra_px`**

- 提交范围：`_normalize_options` + `_run_residual_cleanup`
- 替换两个独立旋钮为一个 `v_extra_px`（默认 6，范围 0..12）
- 上下对称：单个 `cv2.dilate` + 上下同时 shift 一份代码
- 净收益：旋钮 -1；`_run_residual_cleanup` 主体代码 ~30 行 → ~10 行
- 不变点：默认行为（默认 6 等价于旧 top=6 / bottom=6）保持兼容

**3.9.2 `_residual_text_mask` 加彩色字幕检测**

- 新增第 3 路信号 `colorful_mask`：`S >= 60 AND V ∈ [80, 230]`
- 覆盖蓝（H≈100）、红（H≈0/180）、绿（H≈60）、紫（H≈130）、青 等彩色字幕
- 之前完全漏检（白 / 黄信号对蓝/红/绿命中率为 0）
- 暗像素（D 信号）的「需要 text-color 邻居」门控同步改为接受 colorful 邻居

**3.9.3 4 个未暴露的 residual 旋钮改模块常量**

| 旧 `options.get()` 旋钮 | 改成 | 理由 |
|---|---|---|
| `residual_dark_v_min` (40) | 常量 | 调过 30 就会破坏「dark glyph 范围」语义 |
| `residual_dark_v_max` (140) | 常量 | 同上 |
| `residual_dark_s_max` (90) | 常量 | 同上 |
| `residual_max_blob_ratio` (0.45) | 常量 | 调小会丢半行字幕；调大误伤 clothing |

剩余 user-facing 旋钮（`residual_dark_nbhd_radius` / `residual_dark_vertical_strip_px`）覆盖几何门控参数；HSV 窗口和 blob 阈值作为 class-conditional constants 封死。

**3.9.4 `create_mask` 形态学公式去 cap + 加 horizontal morph**

- 旧：`morph_h = min(1 + 2*(dev_y//10), 11)`（dev_y=80 跟 dev_y=50 扩一样宽）
- 新：`morph_h = 1 + 2*max(1, dev_y//8)`（连续函数，dev_y=100 → 25 行 kernel）
- 旧：只有垂直方向 dilate，X 方向不处理
- 新：加 `MORPH_RECT (morph_w, 1)` 水平 dilate iter=1，对称覆盖 CJK 字符水平 stroke 末端
- per-axis gating：dev_x=0 不跑 X dilate，dev_y=0 不跑 Y dilate
- 行为兼容：dev_y=22 时 kernel=(1,5)/iter=2，跟旧版完全一致

**3.9.5 `ocr_bbox_y_pad` doc 化为 deprecated**

- 保留 `_normalize_options` / `_normalize_detect_options` 的字段处理（API 兼容）
- 在两处 `_bounded_int` 调用上方加 `[deprecated]` 注释
- 指向 `subtitle_area_deviation_pixel_y` 替代
- 代码不删（破坏 API），但客户端读源码会看到迁移建议

**3.9.6 清理 `sttn_lama` / `sttn_lama_refine` alias**

- 旧：`if mode in {"sttn_lama", "sttn_lama_refine", "sttn_then_lama"}: ...`
- 新：`if mode == "sttn_then_lama": ...`（仅保留 canonical name）
- 行为：旧 alias 现在会触发 `RequestError: mode must be sttn, lama, propainter, lama_area, or blur_cover` — 客户端会得到清晰错误信息而不是默默走非预期路径

---

## 4. 每一轮 QA 都学到了什么

### Round 1：`(1, 3) / iter=1` → 不够
- 测试帧：1080p 单行字幕
- 现象：descender 下方还有薄带
- 结论：单次 dilate 只扩 ±1px，STTN 边界处模型能力差

### Round 2：`iter=1 → iter=2` → 部分改善
- 测试帧：同上
- 现象：顶/底色带变窄但仍可见
- 结论：iter 加了只多 1px，不足以 cover 8px 残影

### Round 3：`kernel (1,3) → (1,5)` → 基本 OK
- 测试帧：同上，3 行字幕
- 现象：3 行场景下「I」「m」这种 descender 残影明显减少
- 结论：宽 kernel 一次性扩 ±2px 比 iter=2 更有效

### Round 4：dev_y 22 → 44 → 在 3 行字幕上稳了
- 现象：单行 OK 但 3 行底行仍有残
- 结论：dev_y 本身要放大，不是只有 morphology

### Round 5：OCR bbox y_pad 默认 16 → 用户报告误伤 → 默认改回 0
- 现象：文字密集视频上方正文被擦
- 结论：dev_y 已经够大，OCR 端再扩属于「双倍补偿」反而坏事

**总结**：每一轮 QA 都暴露一个**没考虑到的边界**；修复方案都是「放大单点参数」，**但放大到一定程度就开始误伤**。最终解法是**多个独立旋钮让用户按场景调**，默认值取最保守。

---

## 5. 经验沉淀（给后续维护者）

### 5.1 字幕擦除优化的「三层」心智模型

```
第 1 层：OCR bbox 紧 → mask 不够大
       解：create_mask 阶段 Y 方向外扩 + 垂直形态学（X 不动）
       
第 2 层：STTN inpaint 在 mask 边界质量塌陷
       解：morphology 宽度自适应 dev_y；iter + kernel 双调
       
第 3 层：inpaint 后仍有残影
       解：_run_residual_cleanup 二次修补；top/bottom_extra + vertical_close
```

**任何一层调过头都会误伤**。修第 1 层是预防；修第 2 层是「即使 mask 准了模型也补不全」；修第 3 层是「实在补不全的最后兜底」。

### 5.2 默认值选择的「保守 → 激进 → 看用户反馈」节奏

- 一开始：默认值要保守（不影响干净视频），让用户主动开高级开关
- 收到「修了还有残」反馈后：放开 1–2 个最有效的旋钮
- 看到「误伤」反馈：回退激进值，让用户自己调

这次的 OCR bbox y_pad（16 → 0）就是「激进 → 回退」的标准案例。

### 5.3 自适应参数优于写死

- 形态学 kernel 高度 / iter 次数都跟 dev_y 联动
- 单行 / 多行字幕用同一份配置，不需要为多行单独建 preset
- **写死 `kernel = (1, 5)` 总会在某类字幕上失败**

### 5.4 边缘处理：clip + 羽化 + 形态学

- `max(0, ...)` / `min(h-1, w-1, ...)` 防止越界（Round 1 必加）
- mask 边界用 GaussianBlur 做羽化，避免硬边
- 模糊覆盖的 ROI 边缘用 np.linspace ramp 做 alpha 混合（3.6）

### 5.5 配置项的命名要一致

| 字段 | 命名风格 | 例子 |
|---|---|---|
| backend qfluentwidgets | 驼峰 + 像素描述 | `subtitleAreaDeviationPixelY` |
| API options | 蛇形 + 像素描述 | `subtitle_area_deviation_pixel_y` |
| 残字兜底 | `residual_<stage>_<param>` | `residual_top_extra_px` |

**每个旋钮必须 API / backend 双侧都有**。如果只在一侧，配置项就是「摆设」（3.7 踩坑）。

### 5.6 TDD 在 cv2 图像处理上依然有效

- 测试不依赖 GPU / 大模型，纯 numpy + cv2 跑得快
- mock `cv2.dilate` 验证「dev_y=0 时不调用」
- 边界 / 空输入 / X 不变 / Y=0 四种场景覆盖一个 mask 函数的全部行为分支
- 看 spec：`docs/superpowers/plans/2026-06-10-subtitle-edge-residual.md` 5 个测试 case 的设计可以套到任何 `create_*` 函数上

### 5.7 文档先行的好处

- 这一周有 3 篇关键文档：design spec、implementation plan、recap
- 每次 QA 发现新问题，先把 follow-up 写进 plan（不是直接改代码）
- 维护者接手时不用从 commit message 倒推设计意图

---

## 6. 当前 API / backend 默认值速查

| 旋钮 | 默认 | 范围 | 适用 |
|---|---|---|---|
| `subtitle_area_deviation_pixel` (X) | 44 | 0..180 | mask 横向外扩 |
| `subtitle_area_deviation_pixel_y` | 44 | 0..300 | mask 纵向外扩（多行字幕建议 ≥ 30） |
| `ocr_bbox_y_pad` | **0（deprecated）** | 0..120 | 旧 API 字段，用 `subtitle_area_deviation_pixel_y` 替代 |
| `residual_v_extra_px` | 6 | 0..12 | 残字二次修补上下对称外扩（旧 `top_extra_px` / `bottom_extra_px` 合并） |
| `residual_vertical_close_px` | 3 | 0..8 | 残字 mask 垂直 close |
| `auto_residual_cleanup` | False | bool | 是否跑残字兜底（默认关，脏视频手动开） |
| `post_refine_feather` | 3 | 0..12 | post-verify 模糊羽化宽度 |

**调参建议**（按场景）：
- 单行 / 中文干净视频：默认值即可
- 3 行字幕 / 多 descender：把 `subtitle_area_deviation_pixel_y` 提到 60–80
- 字幕上方有演员名 / 机位标识：把 `ocr_bbox_y_pad` 保持 0，dev_y 不要超过 60
- STTN 跑完还有暗色残影：开 `auto_residual_cleanup=true`
- 蓝/红/绿等彩色字幕：默认就能识别（Round 2 加的 colorful 信号）

---

## 7. 后续待办

### 7.1 Round 2 已完成（旋钮清理 + 覆盖度补强）

- [x] 合并 `residual_top/bottom_extra_px` → `residual_v_extra_px`
- [x] `_residual_text_mask` 加彩色字幕检测
- [x] 4 个未暴露的 residual 旋钮改模块常量
- [x] `create_mask` 形态学公式去 cap + 加 horizontal morph
- [x] `ocr_bbox_y_pad` doc 化为 deprecated
- [x] 清理 `sttn_lama` / `sttn_lama_refine` alias
- [x] 更新 `tests/test_inpaint_tools.py` 反映 per-axis gating（5 个 → 7 个测试）

### 7.2 还未做（按优先级）

- [ ] **跨场景回归测试套件**（P1）— 3-5 段真实视频 + PSNR/SSIM 对比 baseline。改 default 值有量化信号。
- [ ] UI 添加「上下扩展像素」滑杆（暴露 DevY 给最终用户）
- [ ] 完全删除 `subtitleAreaDeviationPixel` 旧字段（带迁移逻辑的单独 PR）
- [ ] `subtitleAreaYXAxisDifferencePixel` / `subtitleAreaYAxisDifferencePixel` 等其他 Y 相关阈值的合理性审计
- [ ] `.gitignore` 第 367 行 `test*.py` 加锚（`/test*.py`）或加 `!tests/` 例外，避免 `git add -f`
- [ ] qfluentwidgets stub 抽到 `tests/_stubs/` 共享
- [ ] Post-verify blur 在 4K 视频上的耗时 profile（高斯模糊 51×51 在 4K 上偏慢）
- [ ] `_residual_text_mask` 的 `max_blob_ratio=0.45` 在小 frame 上偏紧（30×200 frame 上 2800px band 被滤掉）— 评估改成 min(60, 0.45 * area) 还是别的
- [ ] dry-run / preview 模式（`--max-frames N`）让调参便宜
