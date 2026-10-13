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

响应约定（与 OmniVoice 对齐）：所有 JSON 响应携带 `{"ok": true|false, ...}` 信封与 `api_schema_version`；请求头带 `X-Request-ID` 时会在响应头中原样返回（不带则服务端生成）；错误响应统一为 `{"ok": false, "error": "..."}`（4xx/422/499/5xx，不回传 traceback）；客户端上传途中断开返回 499。

### 擦除：POST /api/remove-subtitle

两种视频输入方式（二选一）：

- JSON body 里的 `video_url`：HTTP(S) URL、`file://` 路径或本地路径
- `multipart/form-data`：`video` 文件字段上传，参数放在 `payload` JSON 字段（100MB+ 视频推荐 multipart，不要用 base64 JSON）

```shell
# JSON + 远程视频 URL
curl -X POST http://127.0.0.1:6006/api/remove-subtitle \
  -H 'Content-Type: application/json' \
  -H 'X-Job-Id: my-job-0001' \
  -d '{
    "video_url": "https://example.com/input.mp4",
    "sub_area": [900, 1050, 100, 1820],
    "options": {"mode": "sttn", "ocr_preset": "default"}
  }'

# multipart 本地上传
curl -X POST http://127.0.0.1:6006/api/remove-subtitle \
  -F 'video=@test/test.mp4' \
  -F 'payload={"sub_area": [900, 1050, 100, 1820], "options": {"mode": "sttn"}}'
```

`sub_area` 为 `[ymin, ymax, xmin, xmax]`（像素，可选）。不传时默认先自动识别字幕区域（`auto_detect_area`，见下表），识别失败或为空则退回全帧处理；响应里的 `area` 与 `auto_detected_area` 会标明实际使用的区域与来源。响应：

```json
{
  "ok": true,
  "url": "http://127.0.0.1:6006/download/<job_id>/test_no_sub.mp4",
  "download_url": "...",
  "job_id": "...",
  "request_id": "...",
  "api_schema_version": "vsr-removal-v1",
  "filename": "test_no_sub.mp4",
  "fallback_triggered": false,
  "elapsed_seconds": 123.4
}
```

进度轮询：`GET /api/progress/{job_id}`（job_id 可用 `X-Job-Id` 头或 `progress_job_id` 字段指定，否则服务端生成；自定义 id 需匹配 `[A-Za-z0-9_-]{1,64}`）。

输出保证：产物 duration 与源视频一致（尾部帧缺失时冻结末帧补齐，主体视频流不重编码）；音轨按源全长保留（AAC 流拷贝，其他 codec 转 AAC 320k，不用 `-shortest` 截断）；H.264 封装 + faststart；尾部修复 pass 在零改动时直接沿用上游输出，不多产生一代重编码。

### 主要 options 参数

| 参数 | 默认 | 说明 |
|------|------|------|
| `mode` | `sttn` | `sttn`（检测驱动，真人视频推荐）/ `lama`（动画类）/ `propainter`（剧烈运动，耗显存）/ `lama_area`（整区 LaMa）/ `blur_cover`（高斯模糊覆盖） |
| `subtitle_area_deviation_pixel` | 44 | mask 的 X 方向外扩像素 |
| `subtitle_area_deviation_pixel_y` | 44 | mask 的 Y 方向外扩像素（多行字幕调大） |
| `sttn_skip_detection` | false | 跳过 OCR 检测，整区 STTN_AUTO 重绘（最快，可能误伤无字幕帧） |
| `sttn_neighbor_stride` / `sttn_reference_length` / `sttn_max_load_num` | 5 / 20 / 120 | STTN 参考帧步长 / 数量 / 单次最大加载帧数（调大效果更好、更慢、更耗显存） |
| `ocr_preset` | `default` | 检测灵敏度：`default` / `fast` / `turbo` / `fuzzy` / `ultra` / `multiline` / `aggressive`。`multiline` 检测到 3 行及以上字幕时自动对多行帧启用 STTN 修边 + 残字兜底 |
| `bbox_min_inside_ratio` | 1.0 | OCR 检测框与字幕区域的交集面积占比下限；1.0 = 完全落在区域内才保留，调小（如 0.6）可保留压在区域边缘的检测框 |
| `ocr_sample_step` | 0 | OCR 采样间隔；0 = 按帧率自适应，1 = 逐帧检测（最慢但最不易漏），最大 30 |
| `empty_detection_full_inpaint` | true | OCR 检测不到字幕时：true = 整区每帧 STTN_AUTO 重绘；false = 返回 422。响应里的 `fallback_triggered` 标记是否触发了整区重绘兜底 |
| `auto_detect_area` | true | 未传 `sub_area` 时先自动识别字幕区域再擦除（识别失败/为空退回全帧；传了 `sub_area` 或设为 `false` 则不触发） |
| `post_lama_refine` | true | STTN 后接一遍 text-trace 修边，清除 STTN 残影（diff-gated，只处理 STTN 改动过的像素；显式传 `false` 关闭；`post_refine_*` 系列参数可调） |
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

其他服务级环境变量：`HOST` / `PORT`、`VSR_API_WORK_DIR`（任务工作目录）、`VSR_API_MAX_UPLOAD_MB` / `VSR_API_MAX_DOWNLOAD_MB`（默认 2048）、`VSR_API_DOWNLOAD_TIMEOUT`（默认 300 秒）、`VSR_API_JOB_TTL_SECONDS`（job 目录磁盘保留时长，默认 21600 秒 / 6 小时，超时自动清理）、`VSR_REQUIRE_GPU`（docker entrypoint 启动时强制 GPU 检查）。

### 字幕区域检测：POST /api/detect-subtitle-area

输入方式与擦除端点相同，返回检测到的字幕区域（可直接回传给擦除端点）：

```shell
curl -X POST http://127.0.0.1:6006/api/detect-subtitle-area \
  -H 'Content-Type: application/json' \
  -d '{"video_url": "https://example.com/input.mp4", "detect_options": {"sample_count": 72}}'
```

响应包含 `area` / `sub_area`（padded）、`raw_area` / `raw_sub_area`（未 padding）、`confidence`、`boxes`、`video_size`、`fps` 等。常用检测参数：`sample_count`（采样帧数，默认 72）、`ocr_preset`、`bbox_min_inside_ratio`、`crop_to_band`、`early_stop_hit_frames`。

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
- 仍有残字：开启 `post_lama_refine`、`auto_residual_cleanup` 或 `post_verify_blur` 尾部修复链；多行字幕试 `ocr_preset=multiline`。
- 对模型效果不满意，可以查看 design 文件夹里面的训练方法，利用 `backend/tools/train/` 里面的代码训练后替换旧模型。

## 赞助

<img src="https://github.com/YaoFANGUK/video-subtitle-extractor/raw/main/design/sponsor.png" width="600">
