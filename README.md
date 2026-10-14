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

#### 实测推荐参数（2026-10-14 起为 API 默认值）

以减轻毛玻璃、涂抹和误伤背景为目标，以下组合在本次样片中表现较好。所有字段均可省略而使用新默认值；显式传参便于固定客户端行为。将以下内容保存为 `payload.json`，可直接用于后面的 multipart 示例：

```json
{
  "options": {
    "mode": "sttn",
    "auto_detect_area": true,
    "sttn_skip_detection": false,
    "subtitle_area_deviation_pixel": 12,
    "subtitle_area_deviation_pixel_y": 16,
    "sttn_band_margin": 80,
    "post_lama_refine": false,
    "auto_residual_cleanup": false,
    "post_verify_blur": false,
    "empty_detection_full_inpaint": false
  }
}
```

- 遮罩外扩由旧默认的 X/Y 各 44 px 收紧至 12/16 px，减少需要重新生成的背景面积；若有字边残留，再小幅增加相应方向的值。
- `post_lama_refine=false` 关闭 STTN 后的修边，避免把背景纹理进一步抹平；`auto_residual_cleanup` / `post_verify_blur` 也保持关闭。仅在确认有残字时单独尝试开启。
- `empty_detection_full_inpaint=false` 表示 STTN 的擦除 OCR 结果为空时返回 **HTTP 422**，不自动整区重绘。自动 area 识别失败仍会进入全帧 OCR；这两个阶段不要混淆。
- ≤480p 素材可将 `mode` 改成 `lama`，其他字段保持不变（STTN 专属字段此时不生效）。本次静帧纹理更自然，但耗时约翻倍，仍需检查播放时是否闪烁。
- `ocr_preset=multiline` 在没有显式关闭时，仍可对三行及以上字幕自动启用修边/残字清理；希望保持清晰度优先时保留上面的显式 `false`。

需要恢复旧版默认行为的客户端，可显式设置 `subtitle_area_deviation_pixel=44`、`subtitle_area_deviation_pixel_y=44`、`sttn_band_margin=0`、`post_lama_refine=true`、`empty_detection_full_inpaint=true`；其中 margin 为 0 表示沿用环境变量，未设置环境变量时为 40 px。

#### 调用示例

```shell
# JSON + 远程视频 URL：使用新默认参数和自动 area
curl --fail-with-body -sS http://127.0.0.1:6006/api/remove-subtitle \
  -H 'Content-Type: application/json' \
  -H 'X-Job-Id: my-job-0001' \
  -d '{"video_url": "https://example.com/input.mp4"}' \
  -o result.json

# multipart 本地上传：payload.json 使用上面的推荐组合
curl --fail-with-body -sS http://127.0.0.1:6006/api/remove-subtitle \
  -H 'X-Job-Id: my-job-0002' \
  -F 'video=@test/test.mp4' \
  -F 'payload=<payload.json' \
  -o result.json

# 已知字幕位置：手动 area（此坐标示例适用于 1920×1080 视频）
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
| `sttn_input_width` / `sttn_input_height` | 0（用 env/默认 432×240） | STTN_DET 模型输入尺寸；864×480 可减少缩放损失，但更慢更耗显存且不保证观感改善，本次对比未测试；须为 432/240 的整数倍且 ≤864×480，非法值回退默认 |
| `sttn_band_margin` | 80 | STTN_DET 裁剪带上下文余量 px；增大可提供更多周边纹理，但不保证更清晰；传 0 才沿用 `VSR_STTN_BAND_MARGIN`（未设置时 40） |
| `ocr_preset` | `default` | 检测灵敏度：`default` / `fast` / `turbo` / `fuzzy` / `ultra` / `multiline` / `aggressive`。`multiline` 检测到 3 行及以上字幕时自动对多行帧启用 STTN 修边 + 残字兜底 |
| `bbox_min_inside_ratio` | 1.0 | OCR 检测框与字幕区域的交集面积占比下限；1.0 = 完全落在区域内才保留，调小（如 0.6）可保留压在区域边缘的检测框 |
| `ocr_sample_step` | 0 | OCR 采样间隔；0 = 按帧率自适应，1 = 逐帧检测（最慢但最不易漏），最大 30 |
| `empty_detection_full_inpaint` | false | STTN OCR 检测不到字幕时：false = 返回 422；true = 整区每帧 STTN_AUTO 重绘。响应里的 `fallback_triggered` 标记是否触发了整区重绘兜底 |
| `auto_detect_area` | true | 未传 `sub_area` 时先自动识别字幕区域再擦除（识别失败/为空退回全帧；传了 `sub_area` 或设为 `false` 则不触发） |
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
| 模型周边上下文 | `options.sttn_band_margin` | 控制送入 STTN 的裁剪带余量；本次推荐 80 px |

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

2026-10-13 在现有 RTX 3090 Ti 上，以旧版 `de4b8f2` 服务运行 10 组串行对比：1080×1920/30fps 和 852×480/25fps 两段各 6 秒样片。下表“推荐组合”即上方参数，也是 2026-10-14 起的新 API 默认值；时间为单次 API 返回的处理耗时，不是稳定吞吐基准。

| 参数组 | 1080p 竖屏 | 480p 横屏 | 观察 |
|--------|------------|-----------|------|
| 旧默认（44/44、margin 40、开修边） | 94.0s | 36.9s | 低清样片出现明显色块和大面积涂抹 |
| 推荐组合（12/16、margin 80、关修边） | 74.2s | 31.4s | 低清改善明显；1080p 针织纹理仍有模糊 |
| 推荐组合 + area padding 160/64 | 86.0s | 34.0s | 两段输出分别与未扩大 area 的推荐组合 SHA-256 完全相同 |
| 推荐组合 + margin 120 | 84.2s | 35.3s | 没有显示出足以替换 margin 80 的优势 |
| 旧默认仅关闭修边 | 未测 | 31.4s | 色块减轻，但大面积发糊仍在 |
| 推荐组合改用 LaMa | 未测 | 61.6s | 静帧纹理更自然，需检查时序闪烁 |

10 组均未触发整区兜底，分辨率、帧数和音轨存在性检查通过，与输入容器时长偏差不超过 38ms。结论仅适用于本次样片：未验证长视频、864×480 模型输入及 LaMa 时序闪烁，收紧遮罩可能增加字边残留；这些参数不能保证彻底消除复杂纹理上的毛玻璃感。

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
