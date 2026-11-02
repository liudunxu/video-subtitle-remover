简体中文 | [English](README_en.md)

<div align="center">
  <img src="design/icon_1024.PNG" alt="VSR Logo" width="128" height="128">
</div>

## 项目简介

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.11+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)

Video-subtitle-remover (VSR) 是一款基于 AI 技术，将视频中的硬字幕去除的软件。

本仓库面向 **HTTP API / GPU 服务化部署**：桌面 GUI 已移除（配置系统同步去除了 PySide6/qfluentwidgets 依赖），保留 **API 服务** 与 **命令行 CLI** 两种用法。需要桌面图形界面请使用上游项目 [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover)。

主要功能：

- **无损分辨率**去除视频中的硬字幕，生成去除字幕后的文件
- 通过 AI 算法模型（STTN / LaMa / ProPainter / OpenCV）对字幕区域进行填充（非相邻像素填充与马赛克）
- PaddleOCR 自动检测字幕位置，也支持传入自定义字幕区域
- HTTP API：字幕擦除、字幕区域自动检测、进度轮询、结果下载
- 命令行 CLI：`python backend/main.py -i input.mp4 -o output.mp4`

![demo.png](https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.png)

## 快速开始

### 1. 安装依赖（uv，推荐）

```shell
git clone https://github.com/YaoFANGUK/video-subtitle-remover.git
cd video-subtitle-remover

# NVIDIA GPU (CUDA 12.6)
uv sync --extra cuda12

# CPU only
uv sync --extra cpu

# macOS (Apple Silicon)
uv sync --extra macos

# AMD/Intel GPU (Windows DirectML)
uv sync --extra directml
```

传统 pip 方式：`pip install -r requirements.txt`，再按平台安装 paddlepaddle / torch（见 pyproject.toml 中各 extra 的版本钉板）。

### 2. 启动 API 服务

```shell
uv run python api.py            # 默认 127.0.0.1:6006
HOST=0.0.0.0 PORT=6006 uv run python api.py
```

### 3. 命令行 CLI

```shell
python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# 指定字幕区域（可多次指定）与算法
python backend/main.py -i test/test.mp4 -o out.mp4 -c 900 1000 100 1800 --inpaint-mode sttn-det
```

```
options:
  -h, --help            show this help message and exit
  --input INPUT, -i INPUT
                        Input video file path
  --output OUTPUT, -o OUTPUT
                        Output video file path (optional)
  --subtitle-area-coords YMIN YMAX XMIN XMAX, -c YMIN YMAX XMIN XMAX
                        Subtitle area coordinates (ymin ymax xmin xmax). Can be specified multiple times for multiple areas.
  --inpaint-mode {sttn-auto,sttn-det,lama,propainter,opencv}
                        Inpaint mode, default is sttn-auto
```

## HTTP API

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查（别名 `/api/health`），返回 `{"ok": true, "service", "api_schema_version", ...}` |
| `/api/capabilities` | GET | 能力清单：modes / ocr_presets / 特性开关 / 别名使用量 |
| `/api/status` | GET | 服务状态：版本 / 运行时长 / 工作目录 / 磁盘余量 / 活跃任务数 |
| `/api/remove-subtitle` | POST | 提交字幕擦除任务（别名 `/api/remove`、`/remove`，别名响应带 `Deprecation` 头） |
| `/api/detect-subtitle-area` | POST | 自动检测字幕区域（别名 `/api/detect`、`/detect-subtitle-area`） |
| `/api/progress/{job_id}` | GET | 轮询任务进度（phase / percent / stage） |
| `/download/{job_id}/{filename}` | GET | 下载处理结果 |

响应约定（与 OmniVoice 对齐）：所有 JSON 响应携带 `{"ok": true|false, ...}` 信封；健康、能力、状态、擦除和检测响应还包含 `api_schema_version: "vsr-removal-v1"`。请求头带 `X-Request-ID` 时会在响应头中原样返回（不带则服务端生成）；错误响应包含 `{"ok": false, "error": "..."}`（不回传 traceback），进度和错误响应不保证携带 schema 版本。

以下示例的服务地址为 `http://127.0.0.1:6006`；访问 Vast.ai 时替换为实例当前的公网 IP 和映射端口，可用 `python scripts/vast_ai_manager.py status` 查询。`GET /health` 中的 `runtime_commit_id` / `runtime_source` 可用于核对线上代码版本。

### 擦除：POST /api/remove-subtitle

两种视频输入方式（二选一）：

- JSON body 里的 `video_url`：HTTP(S) URL、`file://` 路径或服务端本地路径（Vast.ai 无法通过此字段读取调用方电脑上的文件）
- `multipart/form-data`：`video` 文件字段上传，参数放在 `payload` JSON 字段（100MB+ 视频推荐 multipart，不要用 base64 JSON）

**同步调用**：POST 会等待处理完成才返回下载地址，并非立即返回任务 ID 的异步提交接口；并发任务会排队。客户端和反向代理的超时需覆盖上传、排队及推理时间。需要中途查进度时，在提交前自定义唯一的 `X-Job-Id`，并从另一个连接轮询。

| 请求字段 | 类型 | 用途 |
|----------|------|------|
| `video_url` | string | JSON 请求的视频来源；multipart 上传 `video` 时省略 |
| `sub_area` | array，可选 | `[ymin, ymax, xmin, xmax]`，原始视频像素坐标，原点在左上；需满足 `0 ≤ ymin < ymax ≤ 高度`、`0 ≤ xmin < xmax ≤ 宽度` |
| `options` | object，可选 | 擦除算法、遮罩和后处理参数，见下方推荐组合 |
| `detect_options` | object，可选 | 自动识别 area 的采样、`padding_x` / `padding_y` 等参数；与 `options` 同级 |
| `progress_job_id` | string，可选 | 仅 JSON 请求可用此字段指定任务 ID；两种输入方式统一推荐使用 `X-Job-Id` 请求头 |

#### 逐帧字形掩膜与切镜分段（2026-10-29，可选）

`glyph_v5` 自动处理选区与背景文字重叠的情况，不增加开关：宽带 OCR 出现跨越选区的异常高框时，改为按 `sub_area` 实际裁剪做局部 OCR；窄选区（高度不超过画面 15%）未检出可接受框时也尝试局部补检。仅补检成功或异常框触发后，该镜头剩余部分逐帧走局部检测，切镜后重置。局部候选排除贴裁剪边缘、过高或非横向文字框，只提取有深色描边的白色组件，掩膜限制在选区内，不执行会重新带入背景文字的宽泛笔画恢复。孤立一帧漏检仅在同镜头两侧框一致、当前帧重新提取的笔画像素与两侧掩膜 IoU 均至少 0.75 时恢复；不复制邻帧掩膜。

上述局部路径中，STTN 自动在当前同镜头、最多 80 帧的处理窗口内找清晰参考帧：必须经局部 OCR 检查且无有效文字框/掩膜，背景结构相符，原字幕描边图案已消失。窗口内字幕字形明显变化、背景移动或证据不足时不添加参考帧，保持原固定间隔参考帧列表；不跨窗口额外搜索。筛选复用 432×240 输入，不缓存整片原尺寸帧。ProPainter 共用局部 OCR/掩膜，但不添加这类参考帧。旧 rectangle 路径不受影响。审计中的 `local_ocr` 记录触发、逐帧检测和像素恢复帧，`jobs[].clean_reference_frame` 记录额外参考帧（无则为 null），`reference_candidates` 给出窗口内局部索引和拒绝原因。精确的 `sub_area` 仍是重要前提；颜色字幕、持续 OCR 漏检及运动场景不保证恢复。

同一 `glyph_v5` 还包含以下自动修复，无需新增参数：

行合并会保留高度相差超过两倍的独立 OCR 框，避免背景竖框吞掉正常字幕行。短字补检先于自动窄区域补检执行，已恢复的短字保留原掩膜，避免被空的窄带补检结果覆盖；背景文字重叠路径的描边约束保持不变。

- **自动窄区域补检（2026-10-31）**：同镜头至少两帧出现高度、位置一致的正常横向字幕，且当前图像中有白色描边字形时，按字幕行高度和上下位置自动推导窄检测带，限制在 `sub_area` 内。后续空框或异常大框在该窄带内重新 OCR，再从当前帧提取掩膜；已有正常字幕（含短字）保持原流程，不复制邻帧文字。补检成功后可继续跟踪，距最近可靠检测超过一秒则停止补检，切镜重置；没有足够依据或同镜头有多个得到支持的字幕行时沿用原流程。STTN 和 ProPainter 共用此步骤；掩膜进入既有稳定、分段和推理流程，不修改模型尺寸。存在候选时最多增加两次顺序解码（只保留当前帧）及目标帧局部 OCR，实际额外耗时取决于异常帧数量。审计 `narrow_ocr` 记录参考帧、自动区域、补检/确认/空检测帧和调用次数，区域坐标为 OCR 带内 `[ymin,ymax,xmin,xmax]`；`local_ocr.frames` 包含这些补检帧。E01 的 10 秒 STTN 实验改善了拉丁字幕与背景中文合框的坏例，但重叠笔画仍有修复痕迹；没有验证 ProPainter 在该样本的画质或无可靠初始字幕、多行字幕的恢复效果。
- **短字补检与上方符号补全**：同镜头半秒内（最多前后各 15 帧）至少两帧出现位置一致的短字框时，按框尺寸推导局部搜索区，并限制在 `sub_area` 内。逐帧重做局部 OCR；没有合格短字框时，将同一小裁剪放大两倍补检一次，再映射回原图。只接受当前画面实际检测到且存在亮白内核、深色描边的短字，同时向上搜索分离音符。邻帧只提供搜索位置，不复制字形。正常长句和上述背景文字重叠路径不进入此补检；全程无短字检测依据的字幕仍无法恢复。审计 `short_ocr` 记录补检、更新帧和搜索区，搜索区坐标为 OCR 带内的 `[ymin,ymax,xmin,xmax]`。仅有候选时增加一次顺序解码及候选帧局部 OCR。
- **ProPainter 闪场隔离**：在既有模型窗口内，排除掩膜像素后，若相邻帧亮度差中位数绝对值 >6，且至少 75% 的有效像素同向变化 >3，则分开推理，避免闪白帧污染正常帧的修复。有效背景至少占输入的 20%；普通局部运动或字幕变化不应触发。单帧段复制自身满足光流输入要求，只保留一帧输出；空掩膜段直接通过，输出帧数和时间轴不变。审计 `jobs[].photometric_cuts` 记录视频内从零开始的分界帧。此项不改变 STTN 的分段。

这些约束用于减少误补和串色；复杂背景的模糊、扭曲仍可能存在，短片验证不代表所有字体、运动或曝光变化都已覆盖。

白色描边字幕可以使用 `options.mask_strategy="glyph"`，模型支持 `sttn` 和 `propainter`：

```bash
curl --fail-with-body -sS http://127.0.0.1:6006/api/remove-subtitle \
  -H 'X-Job-Id: glyph-example-1' \
  -F 'video=@input.mp4' \
  -F 'payload={"options":{"mode":"propainter","mask_strategy":"glyph"}}'
```

省略 `sub_area` 时先自动检测字幕带，再保持其上下范围、将横向接受范围扩到画面全宽，避免采样区域过窄时丢弃长句；已知字幕位置时，在 `payload` 中传入 `sub_area: [ymin,ymax,xmin,xmax]`。该区域是 OCR 接受范围，字形描边及羽化允许稍微越过检测框。没有写死样本坐标；补全候选按 OCR 行尺寸限制。显式选区仍按调用方设置使用（默认不扩边），选区过窄时默认的整框包含规则仍可能丢掉字幕行。彩色字幕、移动字幕和其它字体尚未验证；旧路径仍为默认值 `mask_strategy="rectangle"`。

`glyph_v5`（2026-10-30）使用 PaddleOCR 文字检测（最长边 640）、默认隔帧采样和边界补检、同一行检测框合并、白色字形与描边提取及原分辨率 5px 膨胀。额外恢复 OCR 框附近的独立音符/小笔画；若框内仍有未覆盖的亮色描边像素，只在这些漏笔画周围补全掩膜，不把整行及字间背景改成矩形重绘。亮色皮肤、衣服与文字粘连时优先覆盖文字边缘，局部修复面积可能略增。保留上下边界笔画恢复、镜头内 ±2 帧且掩膜 IoU≥0.75 才传播。切镜依据 96×54 BGR 缩略图相邻帧平均绝对差 >40。字幕换句时若框位置完全不变，隔帧采样仍可能漏掉极短字幕；需要更密检测可传 `ocr_sample_step=1`。

新增局部补全：扩大字形搜索窗口，只恢复跨越旧窗口上下边界、与已有字形列对齐的小组件，并要求足够的亮白内核和描边对比，以排除浅色织物纹理。该规则接入两模型共用的掩膜提取，初次 OCR 和 960 补检都会使用；审计标记 `profile=glyph_v5`、`clipped_symbol_recovery=aligned_strong_white_components`。在 My Poet 的 31/90/119 秒附近各 50 帧短片上验证，未做全片或所有字体保证。

配音前端的擦除选区可由用户手动调整，调用方传入的 `sub_area` 默认不会自动扩宽；如果长字幕超出选区，可手动放宽后重试。笔画补全需要部署擦除服务新代码且调用 `options.mask_strategy="glyph"`，省略该参数仍走旧 rectangle 路径。

`glyph_ocr_max_side` 控制字形路径的 OCR 最长边（默认 640，范围 640–1280）。亮色背景上的短字/句尾漏检时可试 960；这只提高 OCR 分辨率，模型输入仍为 432×240。960 在局部坏例中补回了 640 漏掉的越南语句尾，但尚未完成全片质量与耗时验证，因此不更改默认值。

当基础 OCR 最长边小于 960 时，自动检查同一镜头前后各 2 帧：当前行宽缩短至少 15%，且两侧较宽框一致时，才以 960 重检该帧。只接受当前帧重新检测到、与原行及邻帧范围相符的扩展框，不把邻帧框直接复制成掩膜。普通换句、空字幕帧及切镜边界不会直接补框；持续漏检、整行漏检和片段首尾不在此补检规则的覆盖范围内。基础分辨率已达到 960 时不重复补检。`processing` 返回补检及更新帧数，审计 JSON 记录对应帧号。

两模型共用 432×240 输入、最多 80 帧的分段和 20 帧重叠；每个镜头独立裁剪、处理与融合。推理与 OCR/合成都按顺序解码，缓存缩小后的重叠帧，避免 MP4 随机 seek 偏帧后将带字帧配上空掩膜。模型掩膜缩放保留任何非零覆盖，再统一膨胀 1 个模型像素，避免细笔画缩没。`sttn_band_margin`（默认 80）控制上下文，`sttn_reference_length`（默认 20）控制两模型参考帧间隔；`sttn_neighbor_stride` 仅控制 STTN。ProPainter 使用 FP32、内部膨胀为 0。合成时掩膜内部完全使用修复结果，只在外侧以 sigma=2 羽化，空掩膜帧在编码前保持原像素。保留原音频与音视频起始偏移。

Glyph 的 STTN/ProPainter 均从原视频解码帧进行 OCR 和推理，按上述模型尺寸处理；最终合成帧直接以 H.264 `CRF 18 / preset fast` 编码，回灌音轨时复制视频流。此前输出为 `CRF 12 / veryfast`；新设置缩小输出体积，只编码一次，不先压缩输入视频或增加中间有损转码。CRF 影响输出画质和体积，不改变 OCR、掩膜或模型参数，也不保证固定文件大小。

新路径不支持 `lama`/`blur_cover`、整区重绘兜底、额外修边/残字清理/模糊后处理、大尺寸模型输入以及旧矩形膨胀/装载帧数参数；不兼容组合返回 400。检测不到可擦除的白色字形返回 422。`ocr_sample_step` 仅支持 0（默认 2）、1、2。`multiline` preset 在此只调整 OCR 阈值，不隐式开启后处理。

响应的 `processing` 包含实际 profile、设备、OCR 帧数、有效掩膜帧数、分段数、掩膜哈希和 `audit_url`。审计 JSON 还包含每段源帧及模型掩膜哈希，可用于两模型公平对照。临时掩膜使用磁盘映射，结束或失败时清理；需要约 `帧数×字幕带宽×字幕带高×4` 字节的掩膜空间，另加模型预测及视频编码空间。每次模型推理最多加载 80 张缩小后的帧。CUDA 可用时使用 CUDA；其它设备暂用 CPU，避免未经验证的 MPS/DirectML 路径。当前按固定帧率重编码，适用于 CFR 视频。

#### 实测推荐参数（2026-10-14，1080p 竖屏质量/性能折中）

以减轻毛玻璃、涂抹和误伤背景为目标，1080p 竖屏优先试 **864×240 模型输入、margin 80、mask X/Y=12/16、关闭后处理**。同一 6 秒样片上，比 432×240 的针织纵向纹理略好，耗时约增加 12%，但仍有重建痕迹，并非所有视频的通用最优。将以下内容保存为 `payload.json`，可直接用于后面的调用示例：

```json
{
  "options": {
    "mode": "sttn",
    "auto_detect_area": true,
    "sttn_skip_detection": false,
    "subtitle_area_deviation_pixel": 12,
    "subtitle_area_deviation_pixel_y": 16,
    "sttn_band_margin": 80,
    "sttn_input_width": 864,
    "sttn_input_height": 240,
    "post_lama_refine": false,
    "auto_residual_cleanup": false,
    "post_verify_blur": false,
    "empty_detection_full_inpaint": false
  }
}
```

**推荐请求不等于全局默认**：除模型输入尺寸外，上述字段均为当前 API 默认值。尺寸不传（或传 0）仍使用环境变量，未配置时为 **432×240**；本轮不将单个竖屏样片的结论推广为全局默认。速度优先或其他素材可先用 432×240，与此推荐对照。

- 先增加模型输入宽度，不盲目增加 margin：864×480 在本次样片耗时约增加 78%，视觉收益有限；margin 160 也没有明确优势。高分辨率并不意味着 area、mask 和上下文余量都应同比扩大。
- 遮罩外扩由旧默认的 X/Y 各 44 px 收紧至 12/16 px，减少需要重新生成的背景面积；若有字边残留，再小幅增加相应方向的值。
- `post_lama_refine=false` 关闭 STTN 后的修边，避免把背景纹理进一步抹平；`auto_residual_cleanup` / `post_verify_blur` 也保持关闭。仅在确认有残字时单独尝试开启。
- `empty_detection_full_inpaint=false` 表示 STTN 的擦除 OCR 结果为空时返回 **HTTP 422**，不自动整区重绘。自动 area 识别失败仍会进入全帧 OCR；这两个阶段不要混淆。
- ≤480p 素材可将 `mode` 改成 `lama`，其他字段保持不变（STTN 专属字段此时不生效）。此前低清样片静帧纹理更自然，但耗时约翻倍；本次 1080p 竖屏反而出现矩形接缝和纹理断裂，不推荐直接切换，且仍需检查播放时是否闪烁。
- `ocr_preset=multiline` 在没有显式关闭时，仍可对三行及以上字幕自动启用修边/残字清理；希望保持清晰度优先时保留上面的显式 `false`。

需要恢复旧版默认行为的客户端，可显式设置 `subtitle_area_deviation_pixel=44`、`subtitle_area_deviation_pixel_y=44`、`sttn_band_margin=0`、`post_lama_refine=true`、`empty_detection_full_inpaint=true`；其中 margin 为 0 表示沿用环境变量，未设置环境变量时为 40 px。

#### 调用示例

```shell
# JSON + 远程视频 URL：复用上面的竖屏推荐参数（需要 jq）
jq --arg video_url 'https://example.com/input.mp4' \
  '. + {video_url: $video_url}' payload.json > request.json
curl --fail-with-body -sS http://127.0.0.1:6006/api/remove-subtitle \
  -H 'Content-Type: application/json' \
  -H 'X-Job-Id: my-job-0001' \
  --data-binary @request.json \
  -o result.json

# multipart 本地上传：payload.json 使用上面的推荐组合
curl --fail-with-body -sS http://127.0.0.1:6006/api/remove-subtitle \
  -H 'X-Job-Id: my-job-0002' \
  -F 'video=@test/test.mp4' \
  -F 'payload=<payload.json' \
  -o result.json

# 横屏使用服务默认参数 + 手动 area（此坐标示例适用于 1920×1080 视频）
curl --fail-with-body -sS http://127.0.0.1:6006/api/remove-subtitle \
  -H 'Content-Type: application/json' \
  -H 'X-Job-Id: my-job-0003' \
  -d '{"video_url": "https://example.com/input.mp4", "sub_area": [900, 1050, 100, 1820]}' \
  -o result.json
```

示例中的远程 URL、文件路径和任务 ID 需按实际替换，每次提交使用新的 ID。multipart 的 `payload` 是普通文本字段（`<payload.json`），不要写成文件上传 `@payload.json`，也不要自行设置缺少 boundary 的 `Content-Type`。

`sub_area` 不传时默认先自动识别字幕区域，识别失败或为空则退回全帧 OCR；响应里的 `area` 与 `auto_detected_area` 标明实际使用的区域与来源。成功响应示例（`options` 为节选，实际返回规范化后的完整参数）：

```json
{
  "ok": true,
  "url": "http://127.0.0.1:6006/download/my-job-0002/test_no_sub.mp4",
  "download_url": "http://127.0.0.1:6006/download/my-job-0002/test_no_sub.mp4",
  "job_id": "my-job-0002",
  "request_id": "2267e968a79c",
  "api_schema_version": "vsr-removal-v1",
  "filename": "test_no_sub.mp4",
  "area": [261, 480, 143, 708],
  "auto_detected_area": true,
  "refine_area": null,
  "options": {
    "mode": "sttn",
    "subtitle_area_deviation_pixel": 12,
    "subtitle_area_deviation_pixel_y": 16,
    "sttn_band_margin": 80,
    "sttn_input_width": 864,
    "sttn_input_height": 240,
    "post_lama_refine": false,
    "empty_detection_full_inpaint": false
  },
  "fallback_triggered": false,
  "elapsed_seconds": 123.4
}
```

自定义任务 ID 需匹配 `[A-Za-z0-9_-]{1,64}`，不指定则服务端生成。进度查询返回 `found`、`phase`、`percent`、`stage`；`phase=done` 表示完成，`phase=error` 表示失败。`found=false` 可能是上传/区域检测/排队期间尚未登记进度，或记录已过期，不能单凭它判定处理失败；最终以 POST 响应为准。

```shell
# 处理期间在另一个终端查询
curl --fail-with-body -sS http://127.0.0.1:6006/api/progress/my-job-0002

# POST 成功后下载结果；此示例需要 jq
curl --fail --location "$(jq -er '.download_url' result.json)" -o output.mp4
```

常见错误：400（参数、坐标或任务 ID 不合法）、411（缺少 Content-Length）、413（请求或视频超过限制）、422（STTN OCR 为空且禁止整区兜底，或 blur_cover 检测为空）、499（上传时客户端断开）、502（视频 URL 拉取失败）、500（处理失败）。调用方应检查 HTTP 状态和 `ok`，收到 422 时先检查区域是否覆盖字幕及 OCR 灵敏度，再决定是否显式启用整区重绘。

输出保证：产物 duration 与源视频一致（尾部帧缺失时冻结末帧补齐，主体视频流不重编码）；音轨按源全长保留（AAC 流拷贝，其他 codec 转 AAC 320k，不用 `-shortest` 截断）；H.264 封装 + faststart；尾部修复 pass 在零改动时直接沿用上游输出，不多产生一代重编码。

### 主要 options 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `mode` | `sttn` | `sttn`（检测驱动，真人视频推荐）/ `lama`（动画或低清素材可试）/ `propainter`（剧烈运动，耗显存）/ `lama_area`（整区 LaMa）/ `blur_cover`（高斯模糊覆盖） |
| `subtitle_area_deviation_pixel` | 12 | mask 的 X 方向外扩像素；不是 area 的 padding |
| `subtitle_area_deviation_pixel_y` | 16 | mask 的 Y 方向外扩像素；残留上下字边时可小幅调大 |
| `sttn_skip_detection` | false | 跳过 OCR 检测，整区 STTN_AUTO 重绘（最快，可能误伤无字幕帧） |
| `sttn_neighbor_stride` / `sttn_reference_length` / `sttn_max_load_num` | 5 / 20 / 120 | STTN 参考帧步长 / 数量 / 单次最大加载帧数（调大效果更好、更慢、更耗显存） |
| `sttn_input_width` / `sttn_input_height` | 0（用 env/默认 432×240） | STTN_DET 模型输入尺寸；本次 1080p 竖屏推荐显式传 864×240（纹理略好、耗时约 +12%）；864×480 收益有限、耗时约 +78%。须为 432/240 的整数倍且 ≤864×480，非法值回退默认 |
| `sttn_band_margin` | 80 | STTN_DET 裁剪带上下文余量 px；增大可提供更多周边纹理，但不保证更清晰；传 0 才沿用 `VSR_STTN_BAND_MARGIN`（未设置时 40） |
| `ocr_preset` | `default` | 检测灵敏度：`default` / `fast` / `turbo` / `fuzzy` / `ultra` / `multiline` / `aggressive`。`multiline` 检测到 3 行及以上字幕时自动对多行帧启用 STTN 修边 + 残字兜底 |
| `bbox_min_inside_ratio` | 1.0 | OCR 检测框与字幕区域的交集面积占比下限；1.0 = 完全落在区域内才保留，调小（如 0.6）可保留压在区域边缘的检测框 |
| `ocr_sample_step` | 0 | OCR 采样间隔；0 = 按帧率自适应，1 = 逐帧检测（最慢但最不易漏），最大 30 |
| `empty_detection_full_inpaint` | false | STTN OCR 检测不到字幕时：false = 返回 422；true = 整区每帧 STTN_AUTO 重绘。响应里的 `fallback_triggered` 标记是否触发了整区重绘兜底 |
| `auto_detect_area` | true | 未传 `sub_area` 时先自动识别字幕区域再擦除（识别失败/为空退回全帧；传了 `sub_area` 或设为 `false` 则不触发） |
| `sub_area_margin` | 0 | 仅对显式传入的 `sub_area` 生效：向四周扩张的像素数（clamp 到帧内，最大 400）。默认 0 = 原样使用，服务端不会自动放宽调用方的区域——STTN_DET 下 area 只是 OCR 接受窗口，悄悄放宽会把邻近的水印/台标也擦掉。`sttn_skip_detection=true` 或触发空检测兜底时这个矩形**就是** mask，贴着文字传会留下描边残字，此时建议传 16~40。越界坐标一律 clamp 到帧内；完全落在帧外的 area 返回 400 |
| `post_lama_refine` | false | STTN 后接 text-trace 修边；默认方法为 `telea_text`，并非切换主算法到 LaMa。可清除残影，也可能抹平纹理；有残字时再显式开启 |
| `auto_residual_cleanup` | false | 尾部残字扫描 + cv2.inpaint 兜底（`residual_*` 系列参数可调） |
| `post_verify_blur` | false | 尾部 OCR 复检，仍有残字的检测框做高斯模糊（`post_verify_blur_*` 系列参数可调） |
| `blur_kernel` / `blur_passes` / `blur_pad` / `blur_feather` / `blur_temporal_window` | 51 / 2 / 24 / 20 / 3 | 仅 `blur_cover` 模式：高斯核 / 叠加次数 / 扩边 px / 羽化 px / 时序平滑窗口（0 关闭） |
| `blur_cover_allow_passthrough` | false | `blur_cover` 检测失败或未检测到字幕时：false = 返回 422（让调用方感知字幕仍在）；true = 原样返回原视频（旧行为） |

preset 的 `det_db_thresh` / `det_db_box_thresh` / `det_limit_side_len` 会按安装的大版本映射成 PaddleOCR 真正接受的参数名（2.x 同名；3.x/PaddleX 为 `thresh` / `box_thresh` / `limit_side_len`），构造日志会打印实际生效值。

STTN 修复相关环境变量（docker 镜像已内置默认值）：

| 变量 | 默认 | 说明 |
|------|------|------|
| `VSR_STTN_BAND_MARGIN` | 40 | STTN_DET 的 crop 高度按 mask 字幕带自适应：最高字幕带 + 2×margin（px），空 mask 回退旧的比例逻辑 |
| `VSR_STTN_INPUT_WIDTH` / `VSR_STTN_INPUT_HEIGHT` | 432 / 240 | STTN_DET 模型输入尺寸；受注意力 patch 网格约束仅允许 432/240 的整数倍且 ≤864×480，非法值回退 432×240 |

API 现默认传入 `sttn_band_margin=80`，优先于环境变量；仅显式传 `sttn_band_margin=0` 时使用上述 margin 环境配置。CLI/模型层的默认值没有随 API 一起修改。

其他服务级环境变量：`HOST` / `PORT`、`VSR_API_WORK_DIR`（任务工作目录）、`VSR_API_MAX_UPLOAD_MB` / `VSR_API_MAX_DOWNLOAD_MB`（默认 2048）、`VSR_API_DOWNLOAD_TIMEOUT`（默认 300 秒）、`VSR_API_JOB_TTL_SECONDS`（job 目录磁盘保留时长，默认 21600 秒 / 6 小时，超时自动清理）、`VSR_REQUIRE_GPU`（docker entrypoint 启动时强制 GPU 检查）。

### 字幕区域检测：POST /api/detect-subtitle-area

输入方式与擦除端点相同，返回检测到的字幕区域（可直接回传给擦除端点）：

```shell
curl -X POST http://127.0.0.1:6006/api/detect-subtitle-area \
  -H 'Content-Type: application/json' \
  -d '{"video_url": "https://example.com/input.mp4", "detect_options": {"sample_count": 72}}'
```

响应包含 `area`（`{"x", "y", "width", "height"}` 对象）和 `sub_area`（`[ymin, ymax, xmin, xmax]` 数组）；与擦除响应的 `area` 数组格式不同，建议直接将检测响应的 `sub_area` 传给擦除接口。`raw_area` / `raw_sub_area` 为最终 padding 前的区域（已经包含 `raw_area_buffer_y`）；另有 `confidence`、`boxes`、`video_size`、`fps` 等。检测没有找到区域时仍可返回 HTTP 200，但 `area` / `sub_area` 为 `null`、`confidence=0`，调用方需检查。

常用检测参数：`sample_count`（默认请求值 72，非 HPI 快速路径可减少实际采样，显式传入可固定）、`ocr_preset`、`bbox_min_inside_ratio`、`crop_to_band`、`early_stop_hit_frames`、`multi_group_union`（true = 不做带宽过滤、≥最高分 60% 的强组取并集，适合底部横排字幕与侧边竖排文字并存的多区域文本）。响应的 `detect_options` / `frames_sampled` 可用于核对实际执行设置。

#### area 是否需要扩大 X/Y？

区分三个独立设置：

| 设置 | 参数位置 | 实际作用 |
|------|----------|----------|
| 检测区域外扩 | `detect_options.padding_x` / `padding_y` | 防止字幕检测框被 area 边缘截掉；不直接提高重建清晰度 |
| 擦除遮罩外扩 | `options.subtitle_area_deviation_pixel` / `_y` | 决定检测框周围多少原始像素需要重新生成；过大容易涂抹背景 |
| 模型周边上下文 | `options.sttn_band_margin` | 控制送入 STTN 的裁剪带余量；本次推荐 80 px，更多上下文挤入固定输入也可能加剧缩放损失 |

独立检测接口的 X/Y padding 默认均为 16 px；擦除接口自动识别 area 时，会将 X padding 至少设为 96 px。Y 方向还已有 `raw_area_buffer_y`（默认 60 px）及最终上下各 50 px 的缓冲，并会裁到画面边界，因此通常无需再盲目扩大 Y。

只有看到字幕超出 area 时再试以下请求；传了手动 `sub_area` 时不会执行自动 area 检测，这些 padding 不会帮你扩展手动坐标：

```json
{
  "video_url": "https://example.com/input.mp4",
  "detect_options": {"padding_x": 160, "padding_y": 64},
  "options": {"mode": "sttn", "auto_detect_area": true}
}
```

#### Vast.ai 实测结果与适用范围

**第一轮：基础默认值。** 2026-10-13 在 RTX 3090 Ti 上，以旧版 `de4b8f2` 服务运行 10 组串行对比：1080×1920/30fps 和 852×480/25fps 两段各 6 秒样片。下表“基础组合”为 12/16、margin 80、关修边，STTN 输入均为 **432×240**，对应 2026-10-14 起的新 API 默认值；时间为单次 API 返回的处理耗时，不是稳定吞吐基准。

| 参数组 | 1080p 竖屏 | 480p 横屏 | 观察 |
|--------|------------|-----------|------|
| 旧默认（44/44、margin 40、开修边） | 94.0s | 36.9s | 低清样片出现明显色块和大面积涂抹 |
| 基础组合（12/16、margin 80、关修边） | 74.2s | 31.4s | 低清改善明显；1080p 针织纹理仍有模糊 |
| 基础组合 + area padding 160/64 | 86.0s | 34.0s | 两段输出分别与未扩大 area 的基础组合 SHA-256 完全相同 |
| 基础组合 + margin 120 | 84.2s | 35.3s | 没有显示出足以替换 margin 80 的优势 |
| 旧默认仅关闭修边 | 未测 | 31.4s | 色块减轻，但大面积发糊仍在 |
| 基础组合改用 LaMa | 未测 | 61.6s | 静帧纹理更自然，需检查时序闪烁 |

10 组均未触发整区兜底，分辨率、帧数和音轨存在性检查通过，与输入容器时长偏差不超过 38ms。此轮未测试更大模型输入，后续竖屏对比见下表。

**第二轮：竖屏输入尺寸与 margin。** 2026-10-14 在 RTX 3090、`e0a2f36` 服务上，使用同一个 1080×1920/30fps、6 秒/180 帧样片；保持 mask 12/16、关闭后处理，自动 area 均为 `[1188,1653,241,835]`。GPU 与第一轮不同，不直接跨表比较速度。

| 配置 | 耗时 | 观察 |
|------|------|------|
| 432×240、margin 80（当前默认） | 63.2s | 字幕区仍有明显模糊带 |
| **864×240、margin 80（竖屏推荐）** | **70.6s** | 针织纵向纹理略改善，仍有网格状重建痕迹；约 +12% 耗时 |
| 864×480、margin 80 | 112.8s | 视觉收益有限，约 +78% 耗时，不作为首选 |
| 864×480、margin 160 | 102.7s | 没有明确胜过 margin 80；单次耗时有波动，不代表 margin 160 更快 |
| LaMa | 65.1s | 明显矩形接缝、纹理断裂，不推荐用于该竖屏样片 |

新增 4 组均成功、未触发兜底，保留 1080×1920、180 帧及音轨，容器时长偏差小于 50ms；服务日志确认实际使用了 864×240/864×480 模型输入。视觉判断基于 0.5、2.5、4.5 秒抽帧，没有无字幕真值，也未做正式时序闪烁量化或长视频验证。横竖屏样片内容不同，不能将差异仅归因于画面方向。收紧遮罩可能增加字边残留；这些参数不能保证彻底消除复杂纹理上的毛玻璃感。

## 部署

### Docker（GPU API 镜像）

`docker/Dockerfile.gpu` 构建面向 Vast.ai 的 GPU API 镜像（CUDA 12.6 + paddlepaddle-gpu + 启动时强制 GPU 自检），由 GitHub Actions 工作流 `build-vast-gpu` 发布到 Docker Hub：

```shell
docker build -f docker/Dockerfile.gpu -t <namespace>/video-subtitle-remover:vast-gpu .
docker run --gpus all -p 6006:6006 <namespace>/video-subtitle-remover:vast-gpu
curl http://127.0.0.1:6006/health
```

### Vast.ai 一键开/关实例

```bash
export VAST_API_KEY=<your-key>
python scripts/vast_ai_manager.py status   # 查看实例
python scripts/vast_ai_manager.py start    # 选 offer、建实例、等 /health 通过
python scripts/vast_ai_manager.py stop     # 删除全部匹配实例
python scripts/vast_ai_manager.py logs <instance_id>
```

详细运维手册见 [docs/vast_ai_gpu_service_runbook_zh.md](docs/vast_ai_gpu_service_runbook_zh.md)。

## 开发与测试

```shell
python -m pytest tests/ -v     # 单元测试（无需 GPU / Paddle 模型）
python api.py --help           # API 服务参数
```

算法/接口行为的关键约定记录在 `AGENTS.md` 第 8 节；训练自己的模型可参考 `backend/tools/train/`。

## 常见问题

1. 处理速度慢怎么办

- `mode=sttn` + `sttn_skip_detection=true`：跳过字幕检测，整区重绘（可能误伤无字幕帧或漏擦）。
- 调小 `sttn_max_load_num` / `sttn_reference_length` 降低显存占用（效果略降）。

2. 去除效果不好怎么办

- STTN：真人视频效果较好，速度快，可跳过字幕检测。
- LaMa：对图片效果最好，对动画类视频效果好，速度一般，不可跳过字幕检测。
- ProPainter：需要消耗大量显存，速度较慢，对运动非常剧烈的视频效果较好。
- 出现毛玻璃或涂抹：优先使用上方实测默认组合；低清素材可试 `mode=lama`，扩大 area 不等于提升重建清晰度。
- 仍有残字：先检查 area 和遮罩是否覆盖字边，再单独尝试 `post_lama_refine`、`auto_residual_cleanup` 或 `post_verify_blur`，这些后处理可能牺牲纹理清晰度；多行字幕可试 `ocr_preset=multiline`。
- 对模型效果不满意，可以查看 design 文件夹里面的训练方法，利用 `backend/tools/train/` 里面的代码训练后替换旧模型。

## 赞助

<img src="https://github.com/YaoFANGUK/video-subtitle-extractor/raw/main/design/sponsor.png" width="600">
