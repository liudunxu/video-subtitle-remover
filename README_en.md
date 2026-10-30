[简体中文](README.md) | English

<div align="center">
  <img src="design/icon_1024.PNG" alt="VSR Logo" width="128" height="128">
</div>

## Project Introduction

![License](https://img.shields.io/badge/License-Apache%202-red.svg)
![python version](https://img.shields.io/badge/Python-3.11+-blue.svg)
![support os](https://img.shields.io/badge/OS-Windows/macOS/Linux-green.svg)

Video-subtitle-remover (VSR) is an AI-based tool that removes hardcoded subtitles from videos.

This repository is focused on **HTTP API / GPU service deployment**: the desktop GUI has been removed (along with the PySide6/qfluentwidgets dependency in the config system), keeping two ways to use it — an **API service** and a **command-line CLI**. For the desktop GUI, use the upstream project [YaoFANGUK/video-subtitle-remover](https://github.com/YaoFANGUK/video-subtitle-remover).

Main features:

- **Lossless resolution** removal of hardcoded subtitles from videos
- Fills the subtitle area with AI inpainting models (STTN / LaMa / ProPainter / OpenCV) — not adjacent-pixel smearing or mosaics
- Automatic subtitle detection with PaddleOCR; custom subtitle areas can also be passed in
- HTTP API: subtitle removal, automatic subtitle-area detection, progress polling, result download
- CLI: `python backend/main.py -i input.mp4 -o output.mp4`

![demo.png](https://github.com/YaoFANGUK/video-subtitle-remover/raw/main/design/demo.png)

## Quick Start

### 1. Install dependencies (uv, recommended)

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

With plain pip: `pip install -r requirements.txt`, then install paddlepaddle / torch for your platform (versions pinned in the extras in pyproject.toml).

### 2. Start the API service

```shell
uv run python api.py            # defaults to 127.0.0.1:6006
HOST=0.0.0.0 PORT=6006 uv run python api.py
```

### 3. Command-line CLI

```shell
python backend/main.py -i test/test.mp4 -o test/test_no_sub.mp4

# Custom subtitle area (repeatable) and inpaint mode
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

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check (alias `/api/health`), returns `{"ok": true, "service", "api_schema_version", ...}` |
| `/api/capabilities` | GET | Capability list: modes / ocr_presets / feature flags / deprecated-alias usage |
| `/api/status` | GET | Service status: version / uptime / work dir / free disk / active jobs |
| `/api/remove-subtitle` | POST | Submit a subtitle removal job (aliases `/api/remove`, `/remove` — alias responses carry a `Deprecation` header) |
| `/api/detect-subtitle-area` | POST | Auto-detect the subtitle area (aliases `/api/detect`, `/detect-subtitle-area`) |
| `/api/progress/{job_id}` | GET | Poll job progress (phase / percent / stage) |
| `/download/{job_id}/{filename}` | GET | Download the processed result |

Response conventions (aligned with OmniVoice): every JSON response carries the `{"ok": true|false, ...}` envelope and `api_schema_version`; an `X-Request-ID` request header is echoed back as a response header (generated when absent); errors are uniformly `{"ok": false, "error": "..."}` (4xx/422/499/5xx, no traceback in the body); a client disconnect mid-upload returns 499.

### Removal: POST /api/remove-subtitle

Opt-in white outlined subtitle profile (2026-10-29): send
`"options":{"mode":"propainter","mask_strategy":"glyph"}` (also supports `sttn`).
Omitting `sub_area` keeps automatic area detection; an explicit area is an OCR
acceptance window, not a rectangular repaint mask. The default strategy remains
`rectangle`. Colored/moving captions and other fonts are not validated.

`glyph_v3` (2026-10-30) uses real PaddleOCR detection, per-frame glyph/outline
masks with 5px dilation, disconnected-accent and clipped-stroke recovery, and
scene-local temporal stabilization. If outlined bright pixels still escape a
glyph mask, local growth around those pixels repairs coverage without filling
the entire OCR line or its glyph gaps. This handles letters connected to bright
skin/clothing while limiting extra repair area. Uniform highlights alone do not
activate the guard.
`glyph_ocr_max_side` sets the OCR maximum side (default 640, range 640–1280).
Try 960 for missed short words on bright backgrounds. This changes OCR resolution
only; full-video quality/cost at 960 is not yet validated.
Below a base OCR size of 960, a line shrinking by at least 15% between agreeing
wider detections within two frames on each side triggers a 960 retry. Retries
use actual current-frame detections, never copied neighbor boxes, and stay within
the shot and accepted area. Empty frames, persistent misses and clip edges are
not recovered by this rule. Base OCR at 960 or above skips retries. The response
reports retry/update counts; the audit records frame indices.
Both models use 432×240 inputs and 80-frame windows with 20-frame overlap.
Inference decodes sequentially like OCR/render, caching resized overlap frames;
MP4 random seeks can otherwise pair a captioned frame with the next empty mask.
Mask resizing preserves any nonzero coverage and adds one model-pixel margin.
ProPainter uses FP32 and zero internal dilation. Original-resolution composition
fully replaces mask cores and feathers only outside them (sigma 2).
Scene cuts use mean absolute BGR difference >40 on 96×54 thumbnails. OCR runs
every two frames with boundary checks (`ocr_sample_step=1` enables every frame;
only 0/1/2 are supported). A brief caption between identical neighboring layouts
can still be missed with sampling. `sttn_band_margin` and
`sttn_reference_length` apply to both models; `sttn_neighbor_stride` is STTN-only.

Incompatible legacy rectangle-dilation/chunk-size options, larger model inputs,
whole-area fallback, and additional cleanup/refinement/blur return HTTP 400.
No editable white glyphs returns 422. The multiline preset adjusts detection
thresholds only. The response's `processing.audit_url` exposes actual device,
mask hash, cuts, and per-window input/mask hashes. Temporary masks use disk maps
(about frames × band width × band height × 4 bytes, plus predictions and output),
cleaned on completion/failure. CUDA is used when enabled/available; other devices
use CPU. Audio and source A/V offset are preserved; video encoding is CFR.

Two ways to supply the video (pick one):

- `video_url` in a JSON body: HTTP(S) URL, `file://` path, or local path
- `multipart/form-data`: upload via the `video` file field, parameters in the `payload` JSON field (recommended for 100MB+ videos; do not use base64 JSON)

```shell
# JSON + remote video URL
curl -X POST http://127.0.0.1:6006/api/remove-subtitle \
  -H 'Content-Type: application/json' \
  -H 'X-Job-Id: my-job-0001' \
  -d '{
    "video_url": "https://example.com/input.mp4",
    "sub_area": [900, 1050, 100, 1820],
    "options": {"mode": "sttn", "ocr_preset": "default"}
  }'

# multipart local upload
curl -X POST http://127.0.0.1:6006/api/remove-subtitle \
  -F 'video=@test/test.mp4' \
  -F 'payload={"sub_area": [900, 1050, 100, 1820], "options": {"mode": "sttn"}}'
```

`sub_area` is `[ymin, ymax, xmin, xmax]` in pixels (optional). When omitted, the server first auto-detects the subtitle area (`auto_detect_area`, see below) and falls back to full-frame processing if detection fails or finds nothing; the response `area` and `auto_detected_area` fields report what was actually used. Response:

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

Progress polling: `GET /api/progress/{job_id}` (set the job id via the `X-Job-Id` header or the `progress_job_id` field; otherwise the server generates one; custom ids must match `[A-Za-z0-9_-]{1,64}`).

Output guarantees: the result duration matches the source video (missing tail frames are padded with a frozen last frame — the main body is NOT re-encoded); the audio track is kept at full source length (AAC stream-copied, other codecs transcoded to AAC 320k, never truncated with `-shortest`); H.264 + faststart; tail repair passes that modify zero frames return the upstream output unchanged instead of adding another lossy generation.

### Main options

| Option | Default | Description |
|--------|---------|-------------|
| `mode` | `sttn` | `sttn` (detection-driven, best for live-action) / `lama` (animation) / `propainter` (heavy motion, VRAM-hungry) / `lama_area` (whole-area LaMa) / `blur_cover` (Gaussian blur cover) |
| `subtitle_area_deviation_pixel` | 12 | Mask expansion in X (px); increase slightly if glyph edges remain |
| `subtitle_area_deviation_pixel_y` | 16 | Mask expansion in Y (px); increase slightly if glyph edges remain |
| `sttn_skip_detection` | false | Skip OCR detection and repaint the whole area with STTN_AUTO (fastest; may touch subtitle-free frames) |
| `sttn_neighbor_stride` / `sttn_reference_length` / `sttn_max_load_num` | 5 / 20 / 120 | STTN reference-frame stride / count / max frames per batch (larger = better quality, slower, more VRAM) |
| `sttn_input_width` / `sttn_input_height` | 0 (use env/default 432×240) | STTN_DET model input size; try explicit 864×240 for 1080p portrait (slightly better texture, ~12% more time on the tested clip); 864×480 gave limited benefit at ~78% more time. Must be multiples of 432/240, capped at 864×480; invalid values fall back to default |
| `sttn_band_margin` | 80 | STTN_DET crop context margin (px); larger is not necessarily sharper. Pass 0 to use `VSR_STTN_BAND_MARGIN` (40 if unset) |
| `ocr_preset` | `default` | Detection sensitivity: `default` / `fast` / `turbo` / `fuzzy` / `ultra` / `multiline` / `aggressive`. `multiline` auto-enables STTN refinement + residual cleanup on frames with 3+ subtitle lines |
| `bbox_min_inside_ratio` | 1.0 | Minimum intersection-over-box ratio for keeping an OCR box inside the subtitle area; 1.0 = box must be fully inside (old behavior), lower values (e.g. 0.6) keep boxes straddling the area edge |
| `ocr_sample_step` | 0 | OCR sampling interval; 0 = fps-adaptive, 1 = every frame (slowest, fewest misses), max 30 |
| `empty_detection_full_inpaint` | false | When STTN OCR finds no subtitles: false = HTTP 422; true = whole-area STTN_AUTO repaint. The response field `fallback_triggered` reports whether the fallback fired |
| `auto_detect_area` | true | When `sub_area` is omitted, auto-detect the subtitle area before removal (falls back to full-frame on failure/empty; ignored when `sub_area` is passed or this is `false`) |
| `sub_area_margin` | 0 | Applies only to a caller-supplied `sub_area`: expand it by this many px on all sides (clamped to the frame, max 400). Default 0 = used as-is — the server never silently widens the caller's area, because in STTN_DET the area is the OCR acceptance window and widening it can erase nearby watermarks/channel logos. Under `sttn_skip_detection=true` or the empty-detection fallback the rectangle *is* the mask, so a box drawn tight to the glyphs leaves stroke fringes; pass 16~40 there. Out-of-bounds coordinates are clamped to the frame; an area entirely outside the frame returns HTTP 400 |
| `post_lama_refine` | false | Optional text-trace refinement after STTN (default method: `telea_text`); may remove residue but also flatten background texture. Enable explicitly when needed |
| `auto_residual_cleanup` | false | Tail pass that scans for residual text pixels and inpaints them (tunable via `residual_*`) |
| `post_verify_blur` | false | Tail pass that re-runs OCR on the output and Gaussian-blurs remaining text boxes (tunable via `post_verify_blur_*`) |
| `blur_kernel` / `blur_passes` / `blur_pad` / `blur_feather` / `blur_temporal_window` | 51 / 2 / 24 / 20 / 3 | `blur_cover` mode only: Gaussian kernel / stacked passes / box padding px / feather px / temporal smoothing window (0 disables) |
| `blur_cover_allow_passthrough` | false | When `blur_cover` detection fails or finds no subtitles: false = HTTP 422 (the caller should know the subtitles are still there); true = return the original video (old behavior) |

As of 2026-10-14, API defaults use smaller masks (12/16), margin 80, no refinement, and HTTP 422 on empty STTN OCR. Explicit options can restore the previous 44/44, margin 0 (environment fallback), refinement enabled, and whole-area fallback enabled. CLI/model defaults are unchanged. See the [Chinese README](README.md) for complete request examples and both GPU comparisons; these settings do not guarantee artifact-free output on every video.

**1080p portrait quality/performance recommendation:** save this as `payload.json` and upload with `-F 'video=@input.mp4' -F 'payload=<payload.json'`. This explicitly selects 864×240; omitting the size options still uses env/default **432×240**, not this recommendation.

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

On 2026-10-14, a single 6-second 1080×1920 clip on an RTX 3090 took 63.2s at 432×240/margin 80, 70.6s at 864×240/margin 80, 112.8s at 864×480/margin 80, and 102.7s at 864×480/margin 160. Wider input slightly improved knit texture, but artifacts remained; increasing margin had no clear benefit. LaMa took 65.1s and introduced visible rectangular seams, despite looking better on the earlier low-resolution clip. These are single-run timings and sampled-frame assessments, not long-video or temporal-flicker benchmarks. Do not scale area padding, mask expansion, and context margin together just because the source is higher resolution.

Preset values `det_db_thresh` / `det_db_box_thresh` / `det_limit_side_len` are mapped to the parameter names the installed PaddleOCR actually accepts (2.x uses the same names; 3.x/PaddleX uses `thresh` / `box_thresh` / `limit_side_len`); the applied values are logged at construction.

STTN-related environment variables (the docker image sets defaults):

| Variable | Default | Description |
|----------|---------|-------------|
| `VSR_STTN_BAND_MARGIN` | 40 | STTN_DET crop height adapts to the mask: tallest subtitle band + 2×margin (px); empty masks fall back to the old proportional split |
| `VSR_STTN_INPUT_WIDTH` / `VSR_STTN_INPUT_HEIGHT` | 432 / 240 | STTN_DET model input size; constrained by the attention patch grid to multiples of 432/240, capped at 864×480; invalid values fall back to 432×240 |

Other service-level variables: `HOST` / `PORT`, `VSR_API_WORK_DIR` (job work dir), `VSR_API_MAX_UPLOAD_MB` / `VSR_API_MAX_DOWNLOAD_MB` (default 2048), `VSR_API_DOWNLOAD_TIMEOUT` (default 300s), `VSR_API_JOB_TTL_SECONDS` (how long finished job dirs are kept on disk, default 21600s / 6h — older dirs are swept automatically), `VSR_REQUIRE_GPU` (docker entrypoint enforces GPU checks at startup).

### Area detection: POST /api/detect-subtitle-area

Same input styles as the removal endpoint; returns the detected subtitle area (can be fed straight back to the removal endpoint):

```shell
curl -X POST http://127.0.0.1:6006/api/detect-subtitle-area \
  -H 'Content-Type: application/json' \
  -d '{"video_url": "https://example.com/input.mp4", "detect_options": {"sample_count": 72}}'
```

The response contains `area` / `sub_area` (padded), `raw_area` / `raw_sub_area` (unpadded), `confidence`, `boxes`, `video_size`, `fps`, etc. Common detect options: `sample_count` (default 72), `ocr_preset`, `bbox_min_inside_ratio`, `crop_to_band`, `early_stop_hit_frames`, `multi_group_union` (true = no band filtering, groups scoring ≥ 60% of the top group get unioned — for videos with multiple text regions such as bottom subtitles plus side credits).

## Deployment

### Docker (GPU API image)

`docker/Dockerfile.gpu` builds the GPU API image for Vast.ai (CUDA 12.6 + paddlepaddle-gpu + mandatory GPU self-check at startup), published to Docker Hub by the `build-vast-gpu` GitHub Actions workflow:

```shell
docker build -f docker/Dockerfile.gpu -t <namespace>/video-subtitle-remover:vast-gpu .
docker run --gpus all -p 6006:6006 <namespace>/video-subtitle-remover:vast-gpu
curl http://127.0.0.1:6006/health
```

### One-command Vast.ai instances

```bash
export VAST_API_KEY=<your-key>
python scripts/vast_ai_manager.py status   # list instances
python scripts/vast_ai_manager.py start    # pick an offer, create, wait for /health
python scripts/vast_ai_manager.py stop     # delete all matching instances
python scripts/vast_ai_manager.py logs <instance_id>
```

See the full operations runbook (Chinese): [docs/vast_ai_gpu_service_runbook_zh.md](docs/vast_ai_gpu_service_runbook_zh.md).

## Development & Testing

```shell
python -m pytest tests/ -v     # unit tests (no GPU / Paddle models needed)
python api.py --help           # API service arguments
```

Key behavioral conventions for the algorithm/API layer are documented in `AGENTS.md` section 8; to train your own models see `backend/tools/train/`.

## FAQ

1. Processing is slow

- `mode=sttn` + `sttn_skip_detection=true`: skips detection and repaints the whole area (may touch subtitle-free frames or miss some subtitles).
- Lower `sttn_max_load_num` / `sttn_reference_length` to reduce VRAM usage (slightly worse quality).

2. Removal quality is poor

- STTN: best for live-action videos, fast, can skip detection.
- LaMa: best for images, good for animation, moderate speed, cannot skip detection.
- ProPainter: VRAM-hungry and slow, best for videos with very heavy motion.
- Residual text left: enable the `post_lama_refine`, `auto_residual_cleanup`, or `post_verify_blur` tail passes; try `ocr_preset=multiline` for 3+ line subtitles.
- If you are not satisfied with the models, check the training guide in the design folder and train a replacement with `backend/tools/train/`.

## Sponsor

<img src="https://github.com/YaoFANGUK/video-subtitle-extractor/raw/main/design/sponsor.png" width="600">
