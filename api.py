import argparse
import json
import mimetypes
import multiprocessing
import os
import re
import shutil
import subprocess
import tempfile
import threading
import time
import traceback
import uuid
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import quote, unquote, urlparse
from urllib.request import Request, urlopen
from urllib.error import HTTPError, URLError


# Mitigate Paddle CUDNN non-deterministic failures on repeated runs
os.environ.setdefault("FLAGS_cudnn_deterministic", "1")
os.environ.setdefault("FLAGS_cudnn_exhaustive_search", "0")

WORK_DIR = Path(
    os.environ.get(
        "VSR_API_WORK_DIR",
        str(Path(tempfile.gettempdir()) / "video_subtitle_remover_api"),
    )
).resolve()
MAX_BODY_BYTES = 1024 * 1024
MAX_DOWNLOAD_BYTES = int(os.environ.get("VSR_API_MAX_DOWNLOAD_MB", "2048")) * 1024 * 1024
DOWNLOAD_TIMEOUT_SECONDS = int(os.environ.get("VSR_API_DOWNLOAD_TIMEOUT", "300"))
PROCESS_LOCK = threading.Lock()
# Live progress registry: {job_id: {phase, percent, stage, updated_at}}
#   phase: "sttn" | "refine" | "stitch" | "finalize" | "done" | "error"
#   percent: 0-100 (within the current phase, except for the very first "sttn"
#            phase which we report on a 0-70 scale and the refine pass on
#            70-95 so the caller can render a continuous bar).
_ACTIVE_JOBS: dict = {}
_ACTIVE_JOBS_LOCK = threading.Lock()
JOB_TTL_SECONDS = 1800
# Per-job throttle for stdout PROGRESS lines: {job_id: (phase, 10%-bucket)}.
# We only print on phase change, every 10% within a phase, or on terminal
# states — keeps the api-server log readable during long removals.
_last_logged_progress: dict = {}
_last_logged_progress_LOCK = threading.Lock()


def _set_job_progress(job_id, phase, percent, stage=""):
    if not job_id:
        return
    percent = max(0.0, min(100.0, float(percent)))
    with _ACTIVE_JOBS_LOCK:
        _ACTIVE_JOBS[job_id] = {
            "phase": phase,
            "percent": percent,
            "stage": stage,
            "updated_at": time.time(),
        }


def _prune_active_jobs():
    now = time.time()
    with _ACTIVE_JOBS_LOCK:
        stale = [k for k, v in _ACTIVE_JOBS.items() if now - v.get("updated_at", 0) > JOB_TTL_SECONDS]
        for k in stale:
            _ACTIVE_JOBS.pop(k, None)


def _get_job_progress(job_id):
    with _ACTIVE_JOBS_LOCK:
        entry = _ACTIVE_JOBS.get(job_id)
    if not entry:
        return None
    return {
        "job_id": job_id,
        "phase": entry.get("phase") or "unknown",
        "percent": entry.get("percent") or 0.0,
        "stage": entry.get("stage") or "",
        "updated_at": entry.get("updated_at") or 0.0,
    }


def _clear_gpu_cache():
    """Try to release GPU memory after a failed run to reduce fragmentation."""
    try:
        import paddle
        if paddle.device.is_compiled_with_cuda():
            paddle.device.cuda.empty_cache()
            print("INFO: paddle.device.cuda.empty_cache() called")
    except Exception:
        pass
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("INFO: torch.cuda.empty_cache() called")
    except Exception:
        pass


def _is_cudnn_error(exc):
    msg = str(exc)
    return "CUDNN" in msg or "ExternalError" in msg or "cudnn" in msg.lower()


def _patch_numpy_compat():
    import numpy as np

    for name, value in {
        "int": int,
        "float": float,
        "complex": complex,
        "object": object,
        "str": str,
        "int0": np.intp,
    }.items():
        if not hasattr(np, name):
            setattr(np, name, value)


class RequestError(Exception):
    def __init__(self, status, message):
        super().__init__(message)
        self.status = status
        self.message = message


def _json_bytes(payload):
    return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8")


def _safe_filename(name, default_name="input.mp4"):
    name = unquote(name or "").strip()
    name = Path(name).name
    if not name:
        name = default_name
    name = re.sub(r"[^A-Za-z0-9._-]+", "_", name)
    name = name.strip("._")
    if not name:
        name = default_name
    suffix = Path(name).suffix
    if not suffix or len(suffix) > 10:
        name = f"{Path(name).stem or 'input'}.mp4"
    if len(name) > 160:
        suffix = Path(name).suffix
        stem_limit = max(1, 160 - len(suffix))
        name = f"{Path(name).stem[:stem_limit]}{suffix}"
    return name


def _to_int(value, field_name):
    try:
        return int(round(float(value)))
    except (TypeError, ValueError):
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{field_name} must be a number")


def _to_bool(value):
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return False


def _bounded_int(value, field_name, default, minimum, maximum):
    if value in (None, ""):
        return default
    number = _to_int(value, field_name)
    return max(minimum, min(maximum, number))


def _normalize_options(payload):
    raw_options = payload.get("options")
    if not isinstance(raw_options, dict):
        raw_options = payload.get("remover_options")
    if not isinstance(raw_options, dict):
        raw_options = {}
    merged = {**raw_options, **{key: payload[key] for key in payload.keys() if key in {
        "mode",
        "inpaint_mode",
        "subtitle_area_deviation_pixel",
        "mask_deviation",
        "sttn_skip_detection",
        "sttn_neighbor_stride",
        "sttn_reference_length",
        "sttn_max_load_num",
        "lama_super_fast",
        "propainter_max_load_num",
        "post_lama_refine",
        "post_refine_method",
        "post_refine_dilate",
        "post_refine_bright_threshold",
        "post_refine_dark_threshold",
        "post_refine_edge_threshold",
        "post_refine_diff_threshold",
        "post_refine_inpaint_radius",
        "post_refine_feather",
        "ocr_preset",
        "post_verify_blur",
        "post_verify_blur_force",
        "post_verify_blur_sample_every",
        "post_verify_blur_boundary",
        "post_verify_blur_max_ratio",
        "post_verify_blur_kernel",
        "post_verify_blur_pad",
        "auto_residual_cleanup",
        "residual_inpaint_radius",
        "residual_max_passes",
        "residual_dilate_iters",
        "residual_bright_threshold",
        "residual_early_stop_ratio",
        "residual_top_extra_px",
        "residual_bottom_extra_px",
        "residual_vertical_close_px",
        "residual_dark_vertical_strip_px",
        "residual_dark_nbhd_radius",
    }}}
    mode = str(merged.get("mode") or merged.get("inpaint_mode") or "sttn").strip().lower().replace("-", "_")
    post_lama_refine = _to_bool(merged.get("post_lama_refine", False))
    if mode in {"sttn_lama", "sttn_lama_refine", "sttn_then_lama"}:
        mode = "sttn"
        post_lama_refine = True
    aliases = {
        "sttn_area": "sttn",
        "sttn_no_detection": "sttn",
        "lama_direct": "lama_area",
        "direct_lama": "lama_area",
        "area_lama": "lama_area",
    }
    mode = aliases.get(mode, mode)
    if mode not in {"sttn", "lama", "propainter", "lama_area", "blur_cover"}:
        raise RequestError(HTTPStatus.BAD_REQUEST, "mode must be sttn, lama, propainter, lama_area, or blur_cover")
    return {
        "mode": mode,
        "subtitle_area_deviation_pixel": _bounded_int(
            merged.get("subtitle_area_deviation_pixel", merged.get("mask_deviation")),
            "subtitle_area_deviation_pixel",
            44,
            0,
            180,
        ),
        "sttn_skip_detection": _to_bool(merged.get("sttn_skip_detection", False)),
        "sttn_neighbor_stride": _bounded_int(merged.get("sttn_neighbor_stride"), "sttn_neighbor_stride", 5, 1, 30),
        "sttn_reference_length": _bounded_int(merged.get("sttn_reference_length"), "sttn_reference_length", 20, 1, 60),
        "sttn_max_load_num": _bounded_int(merged.get("sttn_max_load_num"), "sttn_max_load_num", 120, 2, 240),
        "lama_super_fast": _to_bool(merged.get("lama_super_fast", False)),
        "propainter_max_load_num": _bounded_int(merged.get("propainter_max_load_num"), "propainter_max_load_num", 70, 2, 180),
        "post_lama_refine": post_lama_refine,
        "post_refine_method": str(merged.get("post_refine_method") or "telea_text").strip().lower().replace("-", "_"),
        # Aggressive defaults so frames with clear subtitles get cleaned
        # thoroughly on the first pass. See the post_lama_refine tuning
        # notes in AGENTS.md — these values were picked so a 1080p frame
        # with white/yellow subtitles cleans down to a uniform patch.
        "post_refine_dilate": _bounded_int(merged.get("post_refine_dilate"), "post_refine_dilate", 5, 1, 24),
        "post_refine_bright_threshold": _bounded_int(merged.get("post_refine_bright_threshold"), "post_refine_bright_threshold", 145, 80, 255),
        "post_refine_dark_threshold": _bounded_int(merged.get("post_refine_dark_threshold"), "post_refine_dark_threshold", 80, 0, 160),
        "post_refine_edge_threshold": _bounded_int(merged.get("post_refine_edge_threshold"), "post_refine_edge_threshold", 40, 0, 120),
        "post_refine_diff_threshold": _bounded_int(merged.get("post_refine_diff_threshold"), "post_refine_diff_threshold", 5, 0, 80),
        "post_refine_inpaint_radius": _bounded_int(merged.get("post_refine_inpaint_radius"), "post_refine_inpaint_radius", 4, 1, 12),
        "post_refine_feather": _bounded_int(merged.get("post_refine_feather"), "post_refine_feather", 3, 0, 12),
        "ocr_preset": str(merged.get("ocr_preset") or "default").strip().lower(),
        # Auto residual cleanup tail pass — see _run_residual_cleanup.
        # 默认 False: 残字兜底是 STTN 之后的二次修补,在 STTN 已经被时域参数
        # 优化后(NEIGHBOR=8 / REF=7 / MAX_LOAD=100)效果已经不错的情况下,默认
        # 跑残字擦除会让大多数干净视频也多耗 30-60 秒。用户想要"修了还有残
        # 影"时再勾选。
        "auto_residual_cleanup": _to_bool(merged.get("auto_residual_cleanup", False)),
        "residual_inpaint_radius": _bounded_int(merged.get("residual_inpaint_radius"), "residual_inpaint_radius", 7, 2, 12),
        "residual_max_passes": _bounded_int(merged.get("residual_max_passes"), "residual_max_passes", 3, 1, 4),
        "residual_dilate_iters": _bounded_int(merged.get("residual_dilate_iters"), "residual_dilate_iters", 2, 0, 4),
        "residual_bright_threshold": _bounded_int(merged.get("residual_bright_threshold"), "residual_bright_threshold", 130, 80, 200),
        "residual_early_stop_ratio": float(_bounded_int(merged.get("residual_early_stop_ratio"), "residual_early_stop_ratio", 2, 0, 500)) / 10000.0,
        "residual_top_extra_px": _bounded_int(merged.get("residual_top_extra_px"), "residual_top_extra_px", 6, 0, 12),
        "residual_bottom_extra_px": _bounded_int(merged.get("residual_bottom_extra_px"), "residual_bottom_extra_px", 6, 0, 12),
        "residual_vertical_close_px": _bounded_int(merged.get("residual_vertical_close_px"), "residual_vertical_close_px", 3, 0, 8),
        "residual_dark_vertical_strip_px": _bounded_int(merged.get("residual_dark_vertical_strip_px"), "residual_dark_vertical_strip_px", 8, 0, 16),
        "residual_dark_nbhd_radius": _bounded_int(merged.get("residual_dark_nbhd_radius"), "residual_dark_nbhd_radius", 7, 0, 16),
        # Post-verify blur: re-runs PaddleOCR text_detector on the STTN
        # output and blurs any residual-text bboxes. On by default
        # after the 92.8% residual report — the helper itself auto-skips
        # when no residual is found, so the cost is bounded to the
        # sampling PaddleOCR pass (~20% of a full pass) plus the no-op
        # candidate blur loop when nothing's there. Users can still
        # uncheck the box on /remove-subtitle for the cleanest videos.
        "post_verify_blur": _to_bool(merged.get("post_verify_blur", False)),
        "post_verify_blur_force": _to_bool(merged.get("post_verify_blur_force", False)),
        "post_verify_blur_sample_every": _bounded_int(merged.get("post_verify_blur_sample_every"), "post_verify_blur_sample_every", 5, 1, 30),
        "post_verify_blur_boundary": _bounded_int(merged.get("post_verify_blur_boundary"), "post_verify_blur_boundary", 3, 0, 15),
        "post_verify_blur_max_ratio": float(_bounded_int(merged.get("post_verify_blur_max_ratio"), "post_verify_blur_max_ratio", 85, 0, 100)) / 100.0,
        "post_verify_blur_kernel": _bounded_int(merged.get("post_verify_blur_kernel"), "post_verify_blur_kernel", 51, 3, 99),
        "post_verify_blur_pad": _bounded_int(merged.get("post_verify_blur_pad"), "post_verify_blur_pad", 8, 0, 32),
    }


def _normalize_area_value(raw_area, field_name="area"):
    if raw_area is None:
        return None

    if isinstance(raw_area, dict):
        if {"ymin", "ymax", "xmin", "xmax"}.issubset(raw_area):
            values = (
                raw_area["ymin"],
                raw_area["ymax"],
                raw_area["xmin"],
                raw_area["xmax"],
            )
        elif {"x", "y", "width", "height"}.issubset(raw_area):
            x = _to_int(raw_area["x"], f"{field_name}.x")
            y = _to_int(raw_area["y"], f"{field_name}.y")
            width = _to_int(raw_area["width"], f"{field_name}.width")
            height = _to_int(raw_area["height"], f"{field_name}.height")
            values = (y, y + height, x, x + width)
        elif {"left", "top", "width", "height"}.issubset(raw_area):
            x = _to_int(raw_area["left"], f"{field_name}.left")
            y = _to_int(raw_area["top"], f"{field_name}.top")
            width = _to_int(raw_area["width"], f"{field_name}.width")
            height = _to_int(raw_area["height"], f"{field_name}.height")
            values = (y, y + height, x, x + width)
        else:
            raise RequestError(
                HTTPStatus.BAD_REQUEST,
                f"{field_name} must contain ymin/ymax/xmin/xmax, x/y/width/height, or left/top/width/height",
            )
    elif isinstance(raw_area, (list, tuple)) and len(raw_area) == 4:
        values = tuple(raw_area)
    else:
        raise RequestError(
            HTTPStatus.BAD_REQUEST,
            f"{field_name} must be an object or a 4-item list in backend order [ymin, ymax, xmin, xmax]",
        )

    ymin, ymax, xmin, xmax = (
        _to_int(values[0], f"{field_name}.ymin"),
        _to_int(values[1], f"{field_name}.ymax"),
        _to_int(values[2], f"{field_name}.xmin"),
        _to_int(values[3], f"{field_name}.xmax"),
    )
    if min(ymin, ymax, xmin, xmax) < 0:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{field_name} values must be non-negative")
    if ymax <= ymin or xmax <= xmin:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"{field_name} must satisfy ymax > ymin and xmax > xmin")
    return ymin, ymax, xmin, xmax


def _normalize_area(payload):
    raw_area = payload.get("sub_area")
    if raw_area is None:
        raw_area = payload.get("area")
    return _normalize_area_value(raw_area, "area")


def _normalize_refine_area(payload):
    for key in ("refine_sub_area", "refine_area", "post_lama_refine_area", "raw_area"):
        if payload.get(key) is not None:
            return _normalize_area_value(payload.get(key), key)
    return None


def _read_json(handler):
    content_length = handler.headers.get("Content-Length")
    if content_length is None:
        raise RequestError(HTTPStatus.LENGTH_REQUIRED, "Content-Length is required")
    try:
        body_size = int(content_length)
    except ValueError:
        raise RequestError(HTTPStatus.BAD_REQUEST, "Invalid Content-Length")
    if body_size > MAX_BODY_BYTES:
        raise RequestError(HTTPStatus.REQUEST_ENTITY_TOO_LARGE, "Request body is too large")

    raw_body = handler.rfile.read(body_size)
    try:
        payload = json.loads(raw_body.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise RequestError(HTTPStatus.BAD_REQUEST, f"Invalid JSON: {exc.msg}")
    if not isinstance(payload, dict):
        raise RequestError(HTTPStatus.BAD_REQUEST, "JSON body must be an object")
    return payload


def _download_http_url(video_url, target_path):
    """Download a video file from `video_url` to `target_path`. Retries on
    transient 5xx (CDN / origin hiccups) up to 3 times with short backoff.
    4xx and connection errors are not retried — they're permanent.

    Catches urllib.error.HTTPError (e.g. HTTP 530 from the file CDN) and
    converts to a clean RequestError with the upstream status code in the
    message, instead of letting it leak as a raw urllib traceback.
    """
    import time as _time
    request = Request(video_url, headers={"User-Agent": "video-subtitle-remover-api/1.0"})
    last_exc = None
    for attempt in range(3):
        try:
            response = urlopen(request, timeout=DOWNLOAD_TIMEOUT_SECONDS)
        except HTTPError as exc:
            # 4xx is permanent (not found, forbidden, etc) — don't retry.
            # 5xx is transient (CDN/origin issues) — retry with backoff.
            if 500 <= exc.code < 600 and attempt < 2:
                _time.sleep(1.0 * (attempt + 1))
                last_exc = exc
                continue
            raise RequestError(
                HTTPStatus.BAD_GATEWAY,
                f"Video URL returned HTTP {exc.code} {exc.reason} (after {attempt + 1} attempt(s))",
            )
        except (URLError, TimeoutError) as exc:
            # DNS / connection refused / read timeout — retry.
            if attempt < 2:
                _time.sleep(1.0 * (attempt + 1))
                last_exc = exc
                continue
            raise RequestError(
                HTTPStatus.BAD_GATEWAY,
                f"Video URL fetch failed: {exc.reason if hasattr(exc, 'reason') else exc} (after {attempt + 1} attempt(s))",
            )

        try:
            status = getattr(response, "status", HTTPStatus.OK)
            if status >= HTTPStatus.BAD_REQUEST:
                raise RequestError(
                    HTTPStatus.BAD_GATEWAY,
                    f"Video URL returned HTTP {status} (after {attempt + 1} attempt(s))",
                )

            content_length = response.headers.get("Content-Length")
            if content_length and MAX_DOWNLOAD_BYTES:
                try:
                    content_length_value = int(content_length)
                except ValueError:
                    content_length_value = None
                if content_length_value and content_length_value > MAX_DOWNLOAD_BYTES:
                    limit_mb = MAX_DOWNLOAD_BYTES // 1024 // 1024
                    raise RequestError(
                        HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                        f"Video exceeds {limit_mb} MB",
                    )

            bytes_written = 0
            with target_path.open("wb") as output:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    bytes_written += len(chunk)
                    if MAX_DOWNLOAD_BYTES and bytes_written > MAX_DOWNLOAD_BYTES:
                        limit_mb = MAX_DOWNLOAD_BYTES // 1024 // 1024
                        raise RequestError(
                            HTTPStatus.REQUEST_ENTITY_TOO_LARGE,
                            f"Video exceeds {limit_mb} MB",
                        )
                    output.write(chunk)
            # success
            break
        finally:
            try:
                response.close()
            except Exception:
                pass
    else:
        # All retries failed.
        if last_exc is not None:
            raise RequestError(
                HTTPStatus.BAD_GATEWAY,
                f"Video URL fetch failed after retries: {last_exc}",
            )

    if target_path.stat().st_size == 0:
        raise RequestError(HTTPStatus.BAD_GATEWAY, "Downloaded video is empty")


def _stage_video(video_url, job_dir, filename_hint):
    parsed = urlparse(video_url)
    if parsed.scheme in ("http", "https"):
        source_name = _safe_filename(Path(parsed.path).name, filename_hint)
        target_path = job_dir / source_name
        _download_http_url(video_url, target_path)
        return target_path

    if parsed.scheme == "file":
        source_path = Path(unquote(parsed.path)).expanduser().resolve()
    elif parsed.scheme == "":
        source_path = Path(video_url).expanduser().resolve()
    else:
        raise RequestError(HTTPStatus.BAD_REQUEST, "video_url must use http, https, file, or a local path")

    if not source_path.is_file():
        raise RequestError(HTTPStatus.BAD_REQUEST, f"Local video does not exist: {source_path}")

    target_path = job_dir / _safe_filename(source_path.name, filename_hint)
    if source_path.resolve() != target_path.resolve():
        shutil.copy2(source_path, target_path)
    return target_path


def _apply_config_options(config, options):
    # 适配当前项目 qfluentwidgets 配置系统（属性名为驼峰命名）
    mode_map = {
        "sttn": config.InpaintMode.STTN_DET,
        "lama": config.InpaintMode.LAMA,
        "propainter": config.InpaintMode.PROPAINTER,
    }
    if options.get("mode") == "sttn" and options.get("sttn_skip_detection", False):
        mode_map["sttn"] = config.InpaintMode.STTN_AUTO

    config_overrides = {
        "inpaintMode": mode_map.get(options["mode"], config.InpaintMode.STTN_DET),
        "subtitleAreaDeviationPixel": options["subtitle_area_deviation_pixel"],
        "sttnNeighborStride": options["sttn_neighbor_stride"],
        "sttnReferenceLength": options["sttn_reference_length"],
        "sttnMaxLoadNum": max(
            options["sttn_max_load_num"],
            options["sttn_neighbor_stride"] * options["sttn_reference_length"],
        ),
        "propainterMaxLoadNum": options["propainter_max_load_num"],
    }
    # 当前 Config 类没有的属性，作为动态属性设置
    dynamic_overrides = {
        "STTN_SKIP_DETECTION": options["sttn_skip_detection"],
        "LAMA_SUPER_FAST": options["lama_super_fast"],
    }

    old_values = {}
    for name, value in config_overrides.items():
        item = getattr(config, name, None)
        if item is not None and hasattr(item, "value"):
            old_values[name] = (item, item.value)
            config.set(item, value)

    for name, value in dynamic_overrides.items():
        old_values[name] = getattr(config, name, None)
        setattr(config, name, value)

    return old_values


def _restore_config_options(config, old_values):
    for name, entry in old_values.items():
        if isinstance(entry, tuple) and len(entry) == 2:
            item, old_value = entry
            config.set(item, old_value)
        else:
            setattr(config, name, entry)


def _run_lama_area_remover(remover, area):
    _patch_numpy_compat()
    import cv2
    import numpy as np
    from backend import config
    from backend.inpaint.lama_inpaint import LamaInpaint
    from backend.tools.inpaint_tools import create_mask

    if area is not None:
        ymin, ymax, xmin, xmax = area
    else:
        ymin, ymax, xmin, xmax = 0, remover.frame_height, 0, remover.frame_width
    xmin = max(0, min(remover.frame_width - 1, int(xmin)))
    xmax = max(xmin + 1, min(remover.frame_width, int(xmax)))
    ymin = max(0, min(remover.frame_height - 1, int(ymin)))
    ymax = max(ymin + 1, min(remover.frame_height, int(ymax)))
    mask = create_mask(remover.mask_size, [(xmin, xmax, ymin, ymax)])
    feathered_mask = cv2.GaussianBlur(mask.astype(np.float32), (0, 0), 2.0)
    feathered_3ch = np.stack([feathered_mask] * 3, axis=-1)
    lama_inpaint = LamaInpaint()
    frame_no = 0
    while remover.video_cap.isOpened():
        ret, frame = remover.video_cap.read()
        if not ret or frame is None:
            break
        frame_no += 1
        if config.LAMA_SUPER_FAST:
            inpainted = cv2.inpaint(frame, mask, 3, cv2.INPAINT_TELEA)
        else:
            inpainted = lama_inpaint(frame, mask)
        frame = (feathered_3ch * inpainted.astype(np.float32) + (1.0 - feathered_3ch) * frame.astype(np.float32)).astype(np.uint8)
        remover.video_writer.write(frame)
        if remover.frame_count:
            remover.progress_total = int(100 * frame_no / remover.frame_count)
    remover.video_cap.release()
    remover.video_writer.release()
    remover.merge_audio_to_video()
    if os.path.exists(remover.video_temp_file.name):
        try:
            os.remove(remover.video_temp_file.name)
        except OSError:
            pass


def _run_blur_cover(video_path, area, options, ocr_preset=None):
    """模糊覆盖模式：自动检测字幕位置并应用高斯模糊。

    Tunable parameters (all optional in `options`):
      - blur_kernel: Gaussian kernel size (default 51; odd integer >= 3)
      - blur_passes: stacked Gaussian passes (default 2; 5 was the old
        value but it was overkill and ~3x slower for the same visual result)
      - blur_pad: padding around detected box in pixels (default 24)
      - blur_feather: feather width in pixels (default 20)
      - blur_temporal_window: smooth box positions over this many frames
        (default 3). Set 0 to disable temporal smoothing.
    """
    _patch_numpy_compat()
    import cv2
    import numpy as np
    from backend.main import SubtitleDetect, SubtitleRemover

    blur_kernel = max(3, int(options.get("blur_kernel") or 51))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    blur_passes = max(1, int(options.get("blur_passes") or 2))
    blur_pad = max(0, int(options.get("blur_pad") or 24))
    blur_feather = max(0, int(options.get("blur_feather") or 20))
    blur_temporal_window = max(0, int(options.get("blur_temporal_window") or 3))

    ocr_preset_name = str(options.get("ocr_preset") or "default").strip().lower()
    det_db_thresh = None
    det_db_box_thresh = None
    det_limit_side_len = None
    if ocr_preset_name in ("ultra", "aggressive") and ocr_preset:
        det_db_thresh = ocr_preset.get("det_db_thresh")
        det_db_box_thresh = ocr_preset.get("det_db_box_thresh")
        det_limit_side_len = ocr_preset.get("det_limit_side_len")
    elif ocr_preset_name == "fuzzy" and ocr_preset:
        det_db_thresh = ocr_preset.get("det_db_thresh")
        det_db_box_thresh = ocr_preset.get("det_db_box_thresh")
        det_limit_side_len = ocr_preset.get("det_limit_side_len")

    print(
        f"INFO: _run_blur_cover ocr_preset={ocr_preset_name}, "
        f"kernel={blur_kernel}, passes={blur_passes}, pad={blur_pad}, "
        f"feather={blur_feather}, temporal_window={blur_temporal_window}"
    )

    # Step 1: 使用 SubtitleRemover 查找字幕帧和位置
    remover = SubtitleRemover(str(video_path), sub_area=area)
    if det_db_thresh is not None:
        remover.sub_detector = SubtitleDetect(
            str(video_path), sub_area=area,
            det_db_thresh=det_db_thresh,
            det_db_box_thresh=det_db_box_thresh,
            det_limit_side_len=det_limit_side_len,
        )

    # 查找字幕位置（检测可能因CUDNN失败，降级处理）
    try:
        sub_list = remover.sub_detector.find_subtitle_frame_no(sub_remover=remover)
    except Exception as detect_err:
        print(f"WARNING: _run_blur_cover detection failed: {detect_err}, skipping blur")
        # 检测失败时直接复制原视频
        remover.video_cap.release()
        remover.video_writer.release()
        import shutil
        shutil.copy2(video_path, remover.video_out_name)
        remover.merge_audio_to_video()
        return Path(remover.video_out_name)
    if not sub_list:
        print("WARNING: _run_blur_cover no subtitles detected, skipping blur")
        # 仍然输出原视频
        remover.video_cap.release()
        remover.video_writer.release()
        import shutil
        shutil.copy2(video_path, remover.video_out_name)
        remover.merge_audio_to_video()
        return Path(remover.video_out_name)

    # Step 1.5: Temporal smoothing of detected box positions to reduce
    # frame-to-frame flicker when the detector jitters. Boxes from
    # frames within `blur_temporal_window` of the current frame are merged
    # into one smoothed box, then expanded by the merge of all boxes'
    # extents so the blur covers the union. Empty frames inherit the
    # previous smoothed box.
    sorted_frames = sorted(sub_list.keys())
    smoothed_by_frame: dict[int, list[tuple[int, int, int, int]]] = {}
    for idx, fn in enumerate(sorted_frames):
        if blur_temporal_window <= 0:
            smoothed_by_frame[fn] = sub_list[fn]
            continue
        window_start = max(0, idx - blur_temporal_window)
        window_end = min(len(sorted_frames) - 1, idx + blur_temporal_window)
        merged_x1 = min(
            box[0] for j in range(window_start, window_end + 1)
            for box in sub_list[sorted_frames[j]]
        )
        merged_y1 = min(
            box[1] for j in range(window_start, window_end + 1)
            for box in sub_list[sorted_frames[j]]
        )
        merged_x2 = max(
            box[2] for j in range(window_start, window_end + 1)
            for box in sub_list[sorted_frames[j]]
        )
        merged_y2 = max(
            box[3] for j in range(window_start, window_end + 1)
            for box in sub_list[sorted_frames[j]]
        )
        smoothed_by_frame[fn] = [(merged_x1, merged_y1, merged_x2, merged_y2)]
    sub_list = smoothed_by_frame

    print(f"INFO: _run_blur_cover detected subtitles on {len(sorted_frames)} frames (smoothed)")

    # Step 2: 模糊处理（优化：2 次高斯 + 羽化，省掉慢的双边滤波）
    frame_no = 0
    feather_cache: dict[tuple[int, int], np.ndarray] = {}
    while remover.video_cap.isOpened():
        ret, frame = remover.video_cap.read()
        if not ret or frame is None:
            break
        frame_no += 1
        boxes = sub_list.get(frame_no)
        if boxes:
            for (xmin, ymin, xmax, ymax) in boxes:
                # 确保坐标在范围内
                x1 = max(0, int(xmin))
                x2 = min(frame.shape[1], int(xmax))
                y1 = max(0, int(ymin))
                y2 = min(frame.shape[0], int(ymax))
                if x2 > x1 and y2 > y1:
                    # 扩大模糊区域
                    x1_pad = max(0, x1 - blur_pad)
                    x2_pad = min(frame.shape[1], x2 + blur_pad)
                    y1_pad = max(0, y1 - blur_pad)
                    y2_pad = min(frame.shape[0], y2 + blur_pad)
                    roi = frame[y1_pad:y2_pad, x1_pad:x2_pad].copy()
                    # 多次高斯模糊（OpenCV 内部用 separable filter）
                    for _ in range(blur_passes):
                        roi = cv2.GaussianBlur(roi, (blur_kernel, blur_kernel), 0)
                    # 羽化边缘：渐变 mask 让融合更自然
                    h, w = roi.shape[:2]
                    cache_key = (h, w)
                    mask = feather_cache.get(cache_key)
                    if mask is None:
                        mask = np.ones((h, w), dtype=np.float32)
                        if blur_feather * 2 < h and blur_feather * 2 < w:
                            mask[:blur_feather, :] *= np.linspace(0, 1, blur_feather)[:, np.newaxis]
                            mask[-blur_feather:, :] *= np.linspace(1, 0, blur_feather)[:, np.newaxis]
                            mask[:, :blur_feather] *= np.linspace(0, 1, blur_feather)[np.newaxis, :]
                            mask[:, -blur_feather:] *= np.linspace(1, 0, blur_feather)[np.newaxis, :]
                        feather_cache[cache_key] = mask
                    roi = (
                        roi.astype(np.float32) * mask[:, :, np.newaxis]
                        + frame[y1_pad:y2_pad, x1_pad:x2_pad].astype(np.float32) * (1 - mask[:, :, np.newaxis])
                    ).astype(np.uint8)
                    frame[y1_pad:y2_pad, x1_pad:x2_pad] = roi
        remover.video_writer.write(frame)
        if remover.frame_count:
            remover.progress_total = int(100 * frame_no / remover.frame_count)

    remover.video_cap.release()
    remover.video_writer.release()
    remover.merge_audio_to_video()
    if os.path.exists(remover.video_temp_file.name):
        try:
            os.remove(remover.video_temp_file.name)
        except OSError:
            pass
    return Path(remover.video_out_name)


def _text_trace_mask(mask_frame, target_frame, area, options):
    import cv2
    import numpy as np

    if area is not None:
        ymin, ymax, xmin, xmax = area
    else:
        ymin, ymax, xmin, xmax = 0, mask_frame.shape[0], 0, mask_frame.shape[1]
    height, width = mask_frame.shape[:2]
    xmin = max(0, min(width - 1, int(xmin)))
    xmax = max(xmin + 1, min(width, int(xmax)))
    ymin = max(0, min(height - 1, int(ymin)))
    ymax = max(ymin + 1, min(height, int(ymax)))

    source_crop = mask_frame[ymin:ymax, xmin:xmax]
    target_crop = target_frame[ymin:ymax, xmin:xmax]
    if source_crop.size == 0 or target_crop.size == 0:
        return np.zeros((height, width), dtype=np.uint8)

    hsv = cv2.cvtColor(source_crop, cv2.COLOR_BGR2HSV)
    hue, saturation, value = cv2.split(hsv)
    gray = cv2.cvtColor(source_crop, cv2.COLOR_BGR2GRAY)
    bright_threshold = int(options.get("post_refine_bright_threshold") or 145)
    dark_threshold = int(options.get("post_refine_dark_threshold") or 80)
    edge_threshold = int(options.get("post_refine_edge_threshold") or 40)
    white_text = (value >= bright_threshold) & (saturation <= 130)
    yellow_text = (value >= max(120, bright_threshold - 20)) & (saturation >= 55) & (hue >= 8) & (hue <= 45)
    seed_pixels = (white_text | yellow_text).astype(np.uint8) * 255
    if edge_threshold > 0:
        edge_pixels = cv2.Canny(gray, edge_threshold, edge_threshold * 3)
        seed_pixels = cv2.bitwise_or(
            seed_pixels,
            cv2.bitwise_and(edge_pixels, cv2.dilate(seed_pixels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))),
        )
    dark_pixels = ((value <= dark_threshold) & (saturation <= 170)).astype(np.uint8) * 255
    seed_neighborhood = cv2.dilate(seed_pixels, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7)))
    outline_pixels = cv2.bitwise_and(dark_pixels, seed_neighborhood)
    text_pixels = cv2.bitwise_or(seed_pixels, outline_pixels)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(text_pixels, 8)
    filtered = np.zeros_like(text_pixels)
    crop_h, crop_w = text_pixels.shape[:2]
    max_component_area = max(40, int(crop_h * crop_w * 0.35))
    for label in range(1, component_count):
        x, y, w, h, area_pixels = stats[label]
        if area_pixels < 3 or area_pixels > max_component_area:
            continue
        if h < 2 or w < 2 or h > crop_h * 0.9:
            continue
        filtered[labels == label] = 255

    diff_threshold = int(options.get("post_refine_diff_threshold") or 5)
    if diff_threshold > 0:
        diff = cv2.absdiff(source_crop, target_crop)
        diff_gray = cv2.cvtColor(diff, cv2.COLOR_BGR2GRAY)
        diff_mask = (diff_gray >= diff_threshold).astype(np.uint8) * 255
        diff_mask = cv2.dilate(diff_mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)), iterations=1)
        filtered = cv2.bitwise_and(filtered, diff_mask)

    dilate = int(options.get("post_refine_dilate") or 5)
    kernel_size = max(3, dilate * 2 + 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    filtered = cv2.morphologyEx(filtered, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)))
    filtered = cv2.dilate(filtered, kernel, iterations=1)

    full_mask = np.zeros((height, width), dtype=np.uint8)
    full_mask[ymin:ymax, xmin:xmax] = filtered
    return full_mask


def _run_text_trace_refine(video_path, mask_source_path, area, options, progress_callback=None, start_percent=70.0, end_percent=95.0):
    """Refine STTN output frame-by-frame using a CPU text-trace mask + cv2.inpaint.

    Avoids instantiating a second SubtitleRemover (which would re-load Paddle and
    try to allocate GPU memory). Instead we open the STTN output with a plain
    cv2.VideoCapture, run the refinement loop, and write to a temp mp4 that we
    then mux with the source audio.
    """
    _patch_numpy_compat()
    import cv2
    import subprocess
    import tempfile

    def _report(percent, stage="inpainting"):
        if progress_callback is None:
            return
        try:
            scaled = start_percent + (float(percent) / 100.0) * (end_percent - start_percent)
            progress_callback("refine", scaled, stage)
        except Exception:
            pass

    src_cap = cv2.VideoCapture(str(video_path))
    if not src_cap.isOpened():
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Text-trace refinement: cannot open STTN output")
    fps = src_cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        src_cap.release()
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Text-trace refinement: invalid video dimensions")

    mask_cap = cv2.VideoCapture(str(mask_source_path))
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video.close()
    writer = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        src_cap.release()
        if mask_cap.isOpened():
            mask_cap.release()
        try:
            os.remove(temp_video.name)
        except OSError:
            pass
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Text-trace refinement: cannot open video writer")

    output_path = Path(str(video_path)).with_name(Path(str(video_path)).stem + "_refined.mp4").resolve()
    if output_path.is_file():
        try:
            output_path.unlink()
        except OSError:
            pass
    # The refine loop is CPU-bound (cv2.inpaint is a pure-Python-callable
    # OpenCV op that releases the GIL). Run it across a thread pool sized
    # to the host's CPU count, with a bounded in-flight buffer so peak
    # memory stays at chunk_size * frame_size.
    from concurrent.futures import ThreadPoolExecutor

    def _inpaint_one_frame(frame, mask_frame, area, options):
        """Per-frame refine: text-trace mask + cv2.inpaint + optional feather.
        Pure function so it's safe to run from worker threads."""
        # Fast pre-check: if the area has no white or yellow text-like
        # pixels at all, skip the expensive text-trace mask computation
        # (which runs morphology + connectedComponents). This makes
        # refine nearly free for the majority of frames that have no
        # subtitles in them.
        if area is not None:
            ymin, ymax, xmin, xmax = area
        else:
            h, w = frame.shape[:2]
            ymin, ymax, xmin, xmax = 0, h, 0, w
        ymin = max(0, ymin)
        ymax = min(frame.shape[0], ymax)
        xmin = max(0, xmin)
        xmax = min(frame.shape[1], xmax)
        if ymin >= ymax or xmin >= xmax:
            return frame
        crop = frame[ymin:ymax, xmin:xmax]
        # Skip pre-check for very small areas — the per-pixel overhead
        # doesn't pay off.
        if crop.size > 32 * 32:
            try:
                import numpy as _np
                hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
                h_ch, s_ch, v_ch = cv2.split(hsv)
                bright_thr = int(options.get("post_refine_bright_threshold") or 145)
                yellow_v = max(120, bright_thr - 20)
                has_white = _np.any((v_ch >= bright_thr) & (s_ch <= 130))
                if not has_white:
                    has_yellow = _np.any(
                        (v_ch >= yellow_v) & (s_ch >= 55) & (h_ch >= 8) & (h_ch <= 45)
                    )
                else:
                    has_yellow = False
                if not (has_white or has_yellow):
                    return frame
            except Exception:
                # Pre-check is best-effort; fall through to the full mask.
                pass
        mask = _text_trace_mask(mask_frame, frame, area, options)
        if cv2.countNonZero(mask) > 0:
            radius = int(options.get("post_refine_inpaint_radius") or 4)
            refined = cv2.inpaint(frame, mask, radius, cv2.INPAINT_TELEA)
            feather = int(options.get("post_refine_feather") or 3)
            if feather > 0:
                alpha = cv2.GaussianBlur(mask, (0, 0), feather).astype("float32") / 255.0
                alpha = cv2.merge([alpha, alpha, alpha])
                return (refined.astype("float32") * alpha + frame.astype("float32") * (1.0 - alpha)).astype("uint8")
            return refined
        return frame

    cpu_count = os.cpu_count() or 1
    # Reserve at least 1 core for the main loop + writer; cap to a reasonable
    # upper bound so a 64-core box doesn't spawn 60+ threads per request.
    max_workers = max(1, min(8, cpu_count - 1))
    print(
        f"INFO: text_trace_refine: parallel inpaint with max_workers={max_workers} (cpu_count={cpu_count})"
    )

    chunk_depth = max(max_workers * 2, 8)
    executor = ThreadPoolExecutor(
        max_workers=max_workers,
        thread_name_prefix="text-trace-inpaint",
    )
    pending = []  # list of (frame_no, future) in submit order
    frame_no = 0
    last_reported = -1

    def _drain_buffer():
        """Wait for in-order futures and write them out. Returns the
        last reported percent after this drain pass."""
        nonlocal last_reported
        for fno, fut in pending:
            refined = fut.result()
            writer.write(refined)
            if frame_count > 0:
                current_pct = int(100 * (fno) / frame_count)
                if current_pct != last_reported:
                    _report(current_pct, "inpainting")
                    last_reported = current_pct
        pending.clear()

    try:
        _report(0.0, "starting")
        while True:
            ret, frame = src_cap.read()
            if not ret:
                break
            frame_no += 1
            mask_ret, mask_frame = mask_cap.read() if mask_cap.isOpened() else (False, None)
            if not mask_ret or mask_frame is None or mask_frame.shape[:2] != frame.shape[:2]:
                mask_frame = frame
            fut = executor.submit(_inpaint_one_frame, frame, mask_frame, area, options)
            pending.append((frame_no, fut))
            if len(pending) >= chunk_depth:
                _drain_buffer()
        _drain_buffer()
    finally:
        src_cap.release()
        if mask_cap.isOpened():
            mask_cap.release()
        writer.release()
        executor.shutdown(wait=True, cancel_futures=False)

    if frame_no == 0:
        try:
            os.remove(temp_video.name)
        except OSError:
            pass
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Text-trace refinement: no frames decoded from input")

    # Mux source audio onto the refined video via ffmpeg stream copy when
    # possible (much faster than re-encoding).
    ffmpeg_path = "ffmpeg"
    try:
        from backend import config as _vsr_config  # noqa: WPS433
        ffmpeg_path = getattr(_vsr_config, "FFMPEG_PATH", "ffmpeg") or "ffmpeg"
    except Exception:
        pass
    # The temp writer uses cv2's mp4v fourcc, which ffmpeg can sometimes
    # stream-copy directly. When that fails (e.g. ffmpeg can't parse the
    # mp4v muxer cleanly), fall back to re-encoding with libx264.
    copy_cmd = [
        ffmpeg_path, "-y",
        "-i", temp_video.name,
        "-i", str(mask_source_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "copy",
        "-c:a", "copy",
        "-shortest",
        "-loglevel", "error",
        str(output_path),
    ]
    reencode_cmd = [
        ffmpeg_path, "-y",
        "-i", temp_video.name,
        "-i", str(mask_source_path),
        "-map", "0:v:0",
        "-map", "1:a:0?",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "copy",
        "-shortest",
        "-loglevel", "error",
        str(output_path),
    ]
    try:
        with open(os.devnull) as devnull:
            try:
                subprocess.check_output(copy_cmd, stdin=devnull, timeout=600)
                print("INFO: text_trace_refine: ffmpeg stream-copy mux succeeded")
            except Exception as copy_exc:
                print(
                    f"WARN: text_trace_refine: stream-copy mux failed ({copy_exc}), re-encoding with libx264"
                )
                subprocess.check_output(reencode_cmd, stdin=devnull, timeout=600)
    except Exception:
        # Fallback: just copy the temp video (no audio mux).
        try:
            shutil.copy2(temp_video.name, output_path)
        except Exception as copy_exc:
            try:
                os.remove(temp_video.name)
            except OSError:
                pass
            raise RequestError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"Text-trace refinement: failed to mux audio ({copy_exc})",
            )
    finally:
        try:
            os.remove(temp_video.name)
        except OSError:
            pass

    if not output_path.is_file():
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Text-trace refinement did not create an output file")
    return output_path


def _run_post_verify_blur(video_path, area, options, progress_callback=None, start_percent=95.0, end_percent=99.0):
    """后置 OCR 校验 + 模糊兜底:STTN 跑完后,用 PaddleOCR text_detector
    在输出帧上重新检测一遍,如果还有"残字"就在检测到的 bbox 上做
    GaussianBlur 兜底(0 误擦干净帧,只模糊有残字的帧)。

    性能:全帧 PaddleOCR 很慢(1 分钟 1080p 视频 60+ 秒),所以用**粗采样 +
    边界扩展**:
      - 每 sample_every 帧(默认 5)采 1 帧跑 PaddleOCR
      - 采样帧有文字 → 在 ±boundary 帧再跑确认范围
      - 候选帧 < threshold(默认 30% 的总帧数)才进入模糊阶段,否则跳过
        (大量残字 = STTN 整体失败,这种素材应该走其他修复路径)

    Returns: new video path with blur applied on residual-text frames.
    """
    _patch_numpy_compat()
    import cv2
    import numpy as np
    import subprocess
    import tempfile
    from backend.main import SubtitleDetect

    sample_every = max(1, int(options.get("post_verify_blur_sample_every") or 5))
    boundary = max(0, int(options.get("post_verify_blur_boundary") or 3))
    # 0.30 -> 0.70 -> 0.40 -> 0.85: user goal is "blur any residual"
    # not "give up when there's a lot". The old guard misfired on
    # text-dense videos (lectures, integral chinese subtitles) where
    # 50%+ frames legitimately have residual. 0.85 keeps an upper
    # bound only for the pathological "PaddleOCR sees text everywhere"
    # case (e.g. OCR melted down on a noisy background).
    max_ratio = float(options.get("post_verify_blur_max_ratio") or 0.85)
    force = bool(options.get("post_verify_blur_force", False))
    # 51 -> 81 -> 121: 单次高斯还能透出字幕轮廓,加大 kernel + 多次叠加
    # 才能把字形彻底糊成一团色块。3 次 121×121 高斯叠加 ≈ 一次 ~210×210
    # 高斯(σ ∝ √N),但分次计算比单次大 kernel 快得多。
    blur_kernel = max(3, int(options.get("post_verify_blur_kernel") or 121))
    if blur_kernel % 2 == 0:
        blur_kernel += 1
    # 8 -> 16: 横向 pad 也加大,避免 blur 区域的边缘还能描出字幕的左右边。
    blur_pad = max(0, int(options.get("post_verify_blur_pad") or 16))
    # Y 方向再额外加 20px——字幕上下沿(中文笔画的撇捺、英文的 g/j/p
    # 下伸部分)经常超出 OCR 检测出的 bbox,纯横向 pad 不够吃。
    blur_pad_y = blur_pad + int(options.get("post_verify_blur_pad_y_extra") or 20)
    # 多次叠加 + 一次 median pre-pass(median 能把细笔画"侵蚀"掉,然后
    # Gaussian 再把残余抹匀)。3 次 Gaussian 是经验值,再多收益递减。
    blur_passes = max(1, int(options.get("post_verify_blur_passes") or 3))
    median_kernel = max(0, int(options.get("post_verify_blur_median_kernel") or 21))
    if median_kernel > 0 and median_kernel % 2 == 0:
        median_kernel += 1

    def _report(percent, stage="verifying"):
        if progress_callback is None:
            return
        try:
            scaled = start_percent + (float(percent) / 100.0) * (end_percent - start_percent)
            progress_callback("refine", scaled, stage)
        except Exception:
            pass

    src_cap = cv2.VideoCapture(str(video_path))
    if not src_cap.isOpened():
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "verify_blur: cannot open video")
    fps = src_cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0 or frame_count <= 0:
        src_cap.release()
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "verify_blur: invalid video info")

    # Restrict detection to the user-defined area only — same as blur_cover.
    if area is not None:
        ay_min, ay_max, ax_min, ax_max = area
    else:
        ay_min, ay_max, ax_min, ax_max = 0, height, 0, width
    ay_min = max(0, ay_min); ay_max = min(height, ay_max)
    ax_min = max(0, ax_min); ax_max = min(width, ax_max)
    if ay_min >= ay_max or ax_min >= ax_max:
        src_cap.release()
        return Path(video_path)
    print(
        f"DEBUG: post_verify_blur area: x=[{ax_min},{ax_max}] y=[{ay_min},{ay_max}] "
        f"frame_size={width}x{height} crop_size={ax_max-ax_min}x{ay_max-ay_min}"
    )

    # PaddleOCR detector (lightweight, no recognizer). Cached detector on
    # SubtitleDetect; we create a fresh instance per call (no easy global
    # cache to safely reuse without leaking model memory across requests).
    #
    # NOTE: do NOT inherit the main flow's ocr_preset. The 'aggressive'
    # preset (0.005/0.02/2560) catches the residual text but also pulls
    # in background texture as false positives. The verify pass needs
    # SLIGHTLY-stricter thresholds than the detection pass — not as
    # extreme as the 0.15/0.30 we tried earlier (those missed faint
    # STTN ghost text). 0.10 / 0.20 is the sweet spot: catches greyed-
    # out partial-erase residuals while still rejecting bare halo noise.
    det_args = {
        "det_db_thresh": 0.10,
        "det_db_box_thresh": 0.20,
        "det_limit_side_len": 1280,
    }
    detector = SubtitleDetect(str(video_path), sub_area=area, **det_args)

    # Phase 1: coarse sample — find candidate frame ranges
    _report(0.0, "ocr_sampling")
    sampled_hits = []  # list of frame_no (1-based) where sampled frame had text
    frame_no = 0
    while True:
        ret, frame = src_cap.read()
        if not ret or frame is None:
            break
        frame_no += 1
        if (frame_no - 1) % sample_every != 0:
            continue
        crop = frame[ay_min:ay_max, ax_min:ax_max]
        if crop.size == 0:
            continue
        try:
            coords = detector.detect_subtitle(crop)
        except Exception:
            continue
        # `detect_subtitle` returns a list of (xmin, xmax, ymin, ymax)
        # tuples already in crop-local pixel space — no further
        # `get_coordinates` conversion needed.
        # Validity-only filter. The earlier 50%-of-crop sanity check
        # was based on the wrong assumption that get_coordinates()
        # returns un-rescaled bbox coords — PaddleOCR's text_detector
        # already maps boxes back to input-image pixel space, so the
        # coords ARE crop-local. The 50% guard then wrongly dropped
        # real single-line subtitles that happen to span > 50% of
        # crop_w/crop_h (common when the user draws a tight sub_area).
        coords = [
            (xmin, xmax, ymin, ymax)
            for (xmin, xmax, ymin, ymax) in coords
            if 0 <= xmin < xmax and 0 <= ymin < ymax
        ]
        if coords:
            sampled_hits.append(frame_no)
        if frame_no % 20 == 0:
            _report(50.0 * frame_no / frame_count, "ocr_sampling")

    # Summary diagnostic: total sampled frames, total hits, ratio
    sampled_count = (frame_count + sample_every - 1) // sample_every
    print(
        f"DEBUG: post_verify_blur Phase 1 summary: "
        f"sampled={sampled_count}, hits={len(sampled_hits)}, "
        f"hit_ratio={100.0 * len(sampled_hits) / max(1, sampled_count):.1f}%"
    )

    if not sampled_hits:
        src_cap.release()
        print("INFO: post_verify_blur: no residual text detected, skipping blur")
        return Path(video_path)

    # Phase 2: expand each hit ± boundary frames
    candidate_set = set()
    for hit in sampled_hits:
        for f in range(max(1, hit - boundary), min(frame_count, hit + boundary + 1)):
            candidate_set.add(f)
    if not force and len(candidate_set) > max_ratio * frame_count:
        # Too many candidate frames — likely a global STTN failure, not
        # edge cases. Skip and let the user re-run with stronger params.
        # 'force' bypasses this guard for users who want to see what the
        # verify pass actually produces on a globally-failing case
        # (e.g. all 188 frames get GaussianBlur'd).
        print(
            f"WARN: post_verify_blur: {len(candidate_set)}/{frame_count} candidate frames "
            f"({100.0 * len(candidate_set) / frame_count:.1f}%) exceed max_ratio={max_ratio*100:.0f}%, "
            "skipping (likely a global inpaint failure, not a residual edge case). "
            "Set post_verify_blur_force=True to bypass."
        )
        src_cap.release()
        return Path(video_path)

    # Phase 3: re-read the full video, blur only candidate frames on their
    # detected bbox regions.
    src_cap.release()
    src_cap = cv2.VideoCapture(str(video_path))
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video.close()
    writer = cv2.VideoWriter(temp_video.name, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        src_cap.release()
        try:
            os.remove(temp_video.name)
        except OSError:
            pass
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "verify_blur: cannot open writer")
    output_path = Path(str(video_path)).with_name(Path(str(video_path)).stem + "_verified_blur.mp4").resolve()
    if output_path.is_file():
        try:
            output_path.unlink()
        except OSError:
            pass

    frame_no = 0
    blurred_count = 0
    while True:
        ret, frame = src_cap.read()
        if not ret or frame is None:
            break
        frame_no += 1
        if frame_no in candidate_set:
            crop = frame[ay_min:ay_max, ax_min:ax_max]
            try:
                coords = detector.detect_subtitle(crop)
            except Exception:
                coords = []
            # Validity-only filter. See Phase 1 comment for why the
            # earlier 50%-of-crop sanity check was wrong.
            crop_h = ay_max - ay_min
            crop_w = ax_max - ax_min
            coords = [
                (xmin, xmax, ymin, ymax)
                for (xmin, xmax, ymin, ymax) in coords
                if 0 <= xmin < xmax and 0 <= ymin < ymax
            ]
            # Aggregate area floor: drop frames whose total bbox area
            # is below a "this is just noise" threshold. 2% balances:
            # - real residual subtitles easily clear it (typical
            #   600x40 = 24000px ≈ 4.7% of 1080x477 sub_area)
            # - PaddleOCR halo / single-glyph false positives are
            #   well under 2% and get correctly dropped
            # Earlier values: 5% (too strict — dropped short single-
            # line subs around 600x40 ≈ 4.6%), 0.5% (too lax — let
            # halo noise through).
            total_bbox_area = sum(
                (xmax - xmin) * (ymax - ymin)
                for (xmin, xmax, ymin, ymax) in coords
            )
            if crop_h * crop_w > 0 and total_bbox_area < 0.02 * crop_h * crop_w:
                coords = []
            for (xmin, xmax, ymin, ymax) in coords:
                # coords are in crop-local coords; convert to full-frame.
                # 同时把 blur 范围钳在 sub_area 内,避免 blur_pad 扩到
                # sub_area 之外 — 用户说"只检测标的框里的文字框",
                # blur 也要只作用在标的框里。
                fxmin = max(ax_min, ax_min + xmin - blur_pad)
                fxmax = min(ax_max, ax_min + xmax + blur_pad)
                fymin = max(ay_min, ay_min + ymin - blur_pad_y)
                fymax = min(ay_max, ay_min + ymax + blur_pad_y)
                if fxmax <= fxmin or fymax <= fymin:
                    continue
                roi = frame[fymin:fymax, fxmin:fxmax]
                # 先 median 一次把细笔画"侵蚀"掉,再 Gaussian 多次叠加抹匀
                # 残余色块。比单次大 kernel Gaussian 更能消除字形痕迹。
                if median_kernel >= 3 and min(roi.shape[:2]) >= median_kernel:
                    roi = cv2.medianBlur(roi, median_kernel)
                blurred_roi = roi
                for _ in range(blur_passes):
                    blurred_roi = cv2.GaussianBlur(blurred_roi, (blur_kernel, blur_kernel), 0)
                frame[fymin:fymax, fxmin:fxmax] = blurred_roi
            if coords:
                blurred_count += 1
        writer.write(frame)
        if frame_no % 30 == 0:
            _report(50.0 + 50.0 * frame_no / frame_count, "ocr_blurring")
    src_cap.release()
    writer.release()
    print(
        f"INFO: post_verify_blur: blurred {blurred_count} frame(s) at PaddleOCR-detected bbox regions"
    )

    # Phase 4: mux audio (same pattern as _run_text_trace_refine)
    from backend import config as _vsr_cfg
    ffmpeg_path = getattr(_vsr_cfg, "FFMPEG_PATH", "ffmpeg") or "ffmpeg"
    audio_tmp = tempfile.NamedTemporaryFile(suffix=".aac", delete=False)
    audio_tmp.close()
    has_audio = False
    try:
        subprocess.check_output(
            [ffmpeg_path, "-y", "-i", str(video_path), "-acodec", "copy", "-vn", "-loglevel", "error", audio_tmp.name],
            stdin=open(os.devnull), timeout=600,
        )
        has_audio = os.path.exists(audio_tmp.name) and os.path.getsize(audio_tmp.name) > 0
    except Exception:
        has_audio = False
    try:
        merge_cmd = [ffmpeg_path, "-y", "-i", temp_video.name]
        if has_audio:
            merge_cmd += ["-i", audio_tmp.name]
        merge_cmd += [
            "-c:v", "libx264" if _vsr_cfg.USE_H264 else "copy",
            "-c:a", "copy",
            "-loglevel", "error",
            str(output_path),
        ]
        subprocess.check_output(merge_cmd, stdin=open(os.devnull), timeout=600)
    except Exception as copy_exc:
        print(f"WARN: post_verify_blur: mux failed ({copy_exc}), copying raw")
        shutil.copy2(temp_video.name, output_path)
    finally:
        try:
            os.remove(temp_video.name)
        except OSError:
            pass
        if os.path.exists(audio_tmp.name):
            try:
                os.remove(audio_tmp.name)
            except OSError:
                pass

    if not output_path.is_file():
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "verify_blur did not create output file")
    _report(100.0, "ocr_done")
    return output_path


# ---------------------------------------------------------------------------
# Auto residual cleanup (runs after STTN / LaMa to catch leftover text)
# ---------------------------------------------------------------------------
# Goal: keep STTN as the primary engine (per user direction "尽量复用现有
# STTN, 只调参数"), but auto-append a final pass that scans the STTN output
# for white / yellow text-like pixels that survived the upstream inpaint and
# erases them in place with cv2.inpaint. Two specific badcases this catches:
#   1. "整行没擦" — STTN's internal mask missed a row of text. We re-scan
#      the whole area for residual text-like pixels and inpaint them.
#   2. "边界没擦" — STTN's text-trace mask was tighter than the actual
#      glyphs. We dilate the residual mask so the inpaint radius covers
#      the edges that survived.
#
# Critical: we ONLY touch pixels inside the user-defined `area`. Pixels
# outside `area` are bit-identical to the upstream output. Within `area`,
# only residual pixels are replaced — surrounding context inside the area
# is preserved wherever no text-like pixel was detected.
#
# This is wired into _run_subtitle_remover as a tail pass on modes that
# produce a real inpaint (sttn, lama, propainter, lama_area, blur_cover).
# Mode "blur_cover" is skipped because the whole area is already blurred.
# ---------------------------------------------------------------------------


def _residual_text_mask(crop_bgr, options):
    """Return a binary mask of residual text-like pixels inside `crop_bgr`.

    Three signals ORed together:
      A. White text   — V >= residual_bright_threshold AND S <= 130.
      B. Yellow text  — V in [110..230], S >= 60, H in [8, 45].
      C. Dark glyphs / outlines — V in [40..140] AND S <= 90, only counted
         if at least one A/B neighbor exists in a 7x7 ellipse. This catches
         the dark fill or stroke that surrounds a white/yellow subtitle
         when STTN's main pass left a hollow ghost (very common in
         real-shot drama / 1080p SD material).

    Each candidate pixel is then ANDed with a 5x5 ellipse neighborhood of
    itself to suppress stray bright texture that happens to match HSV but
    isn't text.
    """
    import cv2
    import numpy as np

    hsv = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)
    h_ch, s_ch, v_ch = cv2.split(hsv)
    bright = int(options.get("residual_bright_threshold", 130))
    white_mask = (v_ch >= bright) & (s_ch <= 130)
    yellow_v_min = max(110, bright - 40)
    yellow_v_max = 230
    yellow_mask = (
        (v_ch >= yellow_v_min) & (v_ch <= yellow_v_max)
        & (s_ch >= 60)
        & (h_ch >= 8) & (h_ch <= 45)
    )
    # Signal C: dark glyphs/outlines that survive alongside signal A/B.
    # The "needs a bright neighbor" gate stops us from erasing genuine dark
    # features in the video (e.g. clothing, shadows, hair) when there's no
    # subtitle nearby.
    dark_v_min = int(options.get("residual_dark_v_min", 40))
    dark_v_max = int(options.get("residual_dark_v_max", 140))
    dark_s_max = int(options.get("residual_dark_s_max", 90))
    bright_signal = (white_mask | yellow_mask)
    # Two neighborhoods — the original 7x7 ellipse catches dark pixels
    # *around* a glyph, and the vertical-strip dilation catches dark pixels
    # sitting directly above/below a glyph (top/bottom edge shadow or
    # stroke that the symmetric nbhd can miss when the glyph is already
    # too thin to register horizontally).
    nbhd_radius = int(options.get("residual_dark_nbhd_radius", 7))
    bright_nbhd_2d = cv2.dilate(
        bright_signal.astype(np.uint8),
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (nbhd_radius, nbhd_radius)),
    )
    vertical_strip_px = int(options.get("residual_dark_vertical_strip_px", 8))
    bright_above_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1 + vertical_strip_px))
    bright_signal_uint8 = bright_signal.astype(np.uint8)
    bright_above_dilated = cv2.dilate(bright_signal_uint8, bright_above_kernel, iterations=1)
    # Shift UP by vertical_strip_px so the "is there a bright pixel ABOVE
    # me in the same column?" check works on the dark-glyph mask below.
    bright_above = np.zeros_like(bright_above_dilated)
    bright_below = np.zeros_like(bright_above_dilated)
    if vertical_strip_px < bright_above.shape[0]:
        bright_above[: bright_above.shape[0] - vertical_strip_px] = bright_above_dilated[vertical_strip_px:]
        bright_below[vertical_strip_px:] = bright_above_dilated[: bright_above.shape[0] - vertical_strip_px]
    dark_glyph = (
        (v_ch >= dark_v_min) & (v_ch <= dark_v_max)
        & (s_ch <= dark_s_max)
    )
    # A dark pixel is residual if EITHER it sits in a 7x7 neighborhood of
    # a bright pixel (existing rule, catches outlines on the sides) OR it
    # sits directly above/below a bright pixel in the same column within
    # the configured strip. This catches upper and lower subtitle fringes.
    dark_mask = (dark_glyph & ((bright_nbhd_2d > 0) | (bright_above > 0) | (bright_below > 0))).astype(np.uint8) * 255

    raw = ((white_mask | yellow_mask).astype(np.uint8) * 255) | dark_mask
    if int(raw.sum()) == 0:
        return raw

    # Require at least one text-like neighbor (5x5 ellipse) — kills isolated
    # bright/dark texture that happens to match HSV but isn't text.
    nbhd = cv2.dilate(raw, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    raw = cv2.bitwise_and(raw, nbhd)
    if int(raw.sum()) == 0:
        return raw

    # Connected-components filter: drop tiny noise (<3 px) and huge blobs
    # (likely a bright background region, not a glyph). max_blob raised
    # to 0.45 of crop so a half-line of residual subtitle is NOT thrown out
    # as "background" — the previous 0.25 cap silently dropped exactly the
    # badcase the user is reporting.
    n_labels, labels, stats, _ = cv2.connectedComponentsWithStats(raw, 8)
    filtered = np.zeros_like(raw)
    crop_h, crop_w = raw.shape[:2]
    max_blob_ratio = float(options.get("residual_max_blob_ratio", 0.45))
    max_blob = max(60, int(crop_h * crop_w * max_blob_ratio))
    for label in range(1, n_labels):
        x, y, w, h, area = stats[label]
        if area < 3 or area > max_blob:
            continue
        if w < 2 or h < 2 or h > crop_h * 0.85:
            continue
        filtered[labels == label] = 255
    return filtered


def _run_residual_cleanup(input_video_path, output_video_path, area, options, progress_callback=None, start_percent=70.0, end_percent=95.0):
    """Multi-pass residual inpaint inside `area` only.

    For each frame: extract the area crop, scan for residual text pixels,
    build an inpaint mask (with a small elliptical dilation so we also catch
    the edges that STTN's text-trace mask sometimes misses), and run
    cv2.inpaint with INPAINT_NS (Navier-Stokes) — gives cleaner fills on
    text-like masks than TELEA, which tends to leave banding. We re-scan
    after each pass and stop early when the residual area drops below the
    configured threshold.

    The output is the same mp4 (same fps / dimensions) as the input. Audio
    is NOT touched here — caller is expected to mux the source audio on top
    of the returned file, or to replace the input file with the output and
    run a no-op audio stream-copy.
    """
    _patch_numpy_compat()
    import cv2
    import numpy as np

    src_cap = cv2.VideoCapture(str(input_video_path))
    if not src_cap.isOpened():
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Residual cleanup: cannot open input video")
    width = int(src_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(src_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    fps = float(src_cap.get(cv2.CAP_PROP_FPS) or 25.0)
    frame_count = int(src_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if width <= 0 or height <= 0:
        src_cap.release()
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Residual cleanup: invalid input video dimensions")

    if area is not None:
        ymin, ymax, xmin, xmax = area
    else:
        ymin, ymax, xmin, xmax = 0, height, 0, width
    ymin = max(0, min(height - 1, int(ymin)))
    ymax = max(ymin + 1, min(height, int(ymax)))
    xmin = max(0, min(width - 1, int(xmin)))
    xmax = max(xmin + 1, min(width, int(xmax)))
    crop_h, crop_w = ymax - ymin, xmax - xmin
    if crop_h <= 0 or crop_w <= 0:
        src_cap.release()
        raise RequestError(HTTPStatus.BAD_REQUEST, "Residual cleanup: empty `area`")

    max_passes = max(1, int(options.get("residual_max_passes", 3)))
    dilate_iters = max(0, int(options.get("residual_dilate_iters", 2)))
    inpaint_radius = int(options.get("residual_inpaint_radius", 7))
    if inpaint_radius < 2:
        inpaint_radius = 2
    early_stop_ratio = float(options.get("residual_early_stop_ratio", 0.0002))
    early_stop_pixels = max(4, int(crop_h * crop_w * early_stop_ratio))

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    # Vertical-bias kernels for edge fixes. STTN's text-trace mask is
    # symmetric, but a real subtitle's top/bottom edges have anti-aliased
    # fringe + dark stroke that can need the inpaint mask to extend
    # further vertically than left/right.
    top_extra_px = max(0, int(options.get("residual_top_extra_px", 6)))
    if top_extra_px > 0:
        top_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1 + 2 * top_extra_px))
    else:
        top_kernel = None
    bottom_extra_px = max(0, int(options.get("residual_bottom_extra_px", 6)))
    if bottom_extra_px > 0:
        bottom_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1 + 2 * bottom_extra_px))
    else:
        bottom_kernel = None
    # Vertical-close step: MORPH_CLOSE on the raw mask with a tall thin
    # kernel. Closes the 1-3 px vertical gap between the top-edge shadow
    # band and the glyph body that often survives STTN. Without this the
    # top_extra_px shift can't reach the shadow because there's no
    # connecting mask to "drag" upward.
    vertical_close_px = max(0, int(options.get("residual_vertical_close_px", 3)))
    vertical_close_kernel = None
    if vertical_close_px > 0:
        vertical_close_kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (1, 1 + 2 * vertical_close_px)
        )

    output_path = Path(output_video_path)
    if output_path.is_file():
        try:
            output_path.unlink()
        except OSError:
            pass
    writer = cv2.VideoWriter(str(output_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
    if not writer.isOpened():
        src_cap.release()
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Residual cleanup: cannot open video writer")

    def _report(pct_int, stage="cleaning"):
        if progress_callback is None:
            return
        try:
            scaled = start_percent + (pct_int / 100.0) * (end_percent - start_percent)
            progress_callback("refine", scaled, stage)
        except Exception:
            pass

    last_pct = -1
    passes_used = 0
    frames_with_residual = 0
    frame_no = 0
    try:
        while True:
            ret, frame = src_cap.read()
            if not ret or frame is None:
                break
            frame_no += 1
            crop = frame[ymin:ymax, xmin:xmax]
            if crop.size == 0:
                writer.write(frame)
            else:
                residual = _residual_text_mask(crop, options)
                # Vertical close: if the residual has a 1-3 px gap between
                # the top-edge shadow band and the glyph body, fill it so
                # the per-pass inpaint mask covers both. Optional — only
                # runs when residual_vertical_close_px > 0.
                if vertical_close_kernel is not None and cv2.countNonZero(residual) > 0:
                    residual = cv2.morphologyEx(residual, cv2.MORPH_CLOSE, vertical_close_kernel)
                if cv2.countNonZero(residual) == 0:
                    # No residual text detected — keep this frame untouched.
                    writer.write(frame)
                else:
                    frames_with_residual += 1
                    inpainted = crop
                    passes_for_this_frame = 0
                    for pass_idx in range(max_passes):
                        if cv2.countNonZero(residual) <= early_stop_pixels:
                            break
                        passes_for_this_frame = pass_idx + 1
                        if dilate_iters > 0:
                            mask_for_inpaint = cv2.dilate(residual, kernel, iterations=dilate_iters)
                        else:
                            mask_for_inpaint = residual
                        # Top-edge extension: dilate with a 1xN vertical
                        # kernel, then shift the result up by `top_extra_px`
                        # so the inpaint covers the anti-aliased fringe
                        # ABOVE each detected text pixel (the symmetric
                        # dilate alone doesn't reach high enough on the
                        # top side).
                        if top_kernel is not None and top_extra_px > 0:
                            vert = cv2.dilate(residual, top_kernel, iterations=1)
                            top_ext = np.zeros_like(vert)
                            if top_extra_px < vert.shape[0]:
                                top_ext[: vert.shape[0] - top_extra_px] = vert[top_extra_px:]
                            mask_for_inpaint = cv2.bitwise_or(mask_for_inpaint, top_ext)
                        if bottom_kernel is not None and bottom_extra_px > 0:
                            vert = cv2.dilate(residual, bottom_kernel, iterations=1)
                            bottom_ext = np.zeros_like(vert)
                            if bottom_extra_px < vert.shape[0]:
                                bottom_ext[bottom_extra_px:] = vert[: vert.shape[0] - bottom_extra_px]
                            mask_for_inpaint = cv2.bitwise_or(mask_for_inpaint, bottom_ext)
                        inpainted = cv2.inpaint(inpainted, mask_for_inpaint, inpaint_radius, cv2.INPAINT_NS)
                        residual = _residual_text_mask(inpainted, options)
                    if passes_for_this_frame > passes_used:
                        passes_used = passes_for_this_frame
                    # Write the inpainted crop back into the full frame —
                    # surrounding pixels stay bit-identical to the upstream
                    # output.
                    frame[ymin:ymax, xmin:xmax] = inpainted
                    writer.write(frame)
            if frame_count > 0:
                current_pct = int(100 * frame_no / frame_count)
                if current_pct != last_pct:
                    _report(current_pct, "residual_cleaning")
                    last_pct = current_pct
    finally:
        src_cap.release()
        writer.release()
    print(
        f"INFO: residual_cleanup: scanned {frame_no} frames, "
        f"{frames_with_residual} had residual text, "
        f"max passes per frame = {passes_used}, output = {output_path}"
    )
    if not output_path.is_file():
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Residual cleanup did not create output file")
    return output_path


def _ensure_h264(video_path, timeout=1800):
    """Re-encode `video_path` to H.264 (libx264) if its video stream is not
    already H.264. Browsers (Chrome/Safari) cannot play mpeg4 / Xvid inside
    an MP4 container, so any non-H.264 output from upstream STTN or the
    text-trace refiner would surface as 'audio only' on the result page.

    Returns the (possibly-replaced) output path. Stream-copies the audio
    track to keep transcode cost low. Adds +faststart for web streaming.
    """
    import subprocess
    import cv2

    path = Path(video_path)
    if not path.is_file():
        return path
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        cap.release()
        return path
    fourcc_int = int(cap.get(cv2.CAP_PROP_FOURCC) or 0)
    cap.release()
    if fourcc_int:
        codec_chars = "".join(
            chr((fourcc_int >> (8 * i)) & 0xFF) for i in range(4)
        ).lower()
        if codec_chars in ("avc1", "h264"):
            return path  # already H.264

    # ffmpeg binary (with vendor FFMPEG_PATH override if available).
    ffmpeg_path = "ffmpeg"
    try:
        from backend import config as _vsr_cfg
        ffmpeg_path = getattr(_vsr_cfg, "FFMPEG_PATH", "ffmpeg") or "ffmpeg"
    except Exception:
        pass

    h264_path = path.with_name(path.stem + "_h264.mp4")
    if h264_path.is_file():
        try:
            h264_path.unlink()
        except OSError:
            pass
    cmd = [
        ffmpeg_path, "-y", "-loglevel", "error",
        "-i", str(path),
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "20",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(h264_path),
    ]
    with open(os.devnull) as devnull:
        subprocess.check_output(cmd, stdin=devnull, timeout=timeout)
    if not h264_path.is_file():
        raise RequestError(
            HTTPStatus.INTERNAL_SERVER_ERROR,
            "Failed to transcode output to H.264",
        )
    # Replace the original in place so callers don't need to track a new path.
    try:
        path.unlink()
    except OSError:
        pass
    h264_path.replace(path)
    print(f"INFO: _ensure_h264: re-encoded to H.264 ({path})")
    return path



def _probe_video_info(path):
    """用 ffprobe 快速获取视频流时长和帧数标签。"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=duration,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            capture_output=True,
            text=True,
            timeout=30,
        )
        data = json.loads(result.stdout)
        stream = data.get("streams", [{}])[0]
        duration = float(stream.get("duration") or 0)
        nb_frames = stream.get("nb_frames")
        nb_frames = int(float(nb_frames)) if nb_frames else 0
        return {"duration": duration, "nb_frames": nb_frames}
    except Exception:
        return {"duration": 0, "nb_frames": 0}


_STTN_PATCHED = False


def _patch_sttn_none_safe_and_quality():
    global _STTN_PATCHED
    if _STTN_PATCHED:
        return

    try:
        from backend.inpaint.sttn_inpaint import STTNInpaint, STTNVideoInpaint
    except ImportError:
        print("WARNING: Cannot import STTNInpaint/STTNVideoInpaint, skip None-safe+quality patch")
        return

    import copy
    import cv2
    import numpy as np

    _original_inpaint_call = STTNInpaint.__call__

    def _safe_sttn_inpaint_call(self, input_frames, input_mask):
        _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
        mask = mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        split_h = int(W_ori * 3 / 16)
        inpaint_area = self.get_inpaint_area_by_mask(H_ori, split_h, mask)

        none_mask = [f is None for f in input_frames]
        valid_frames = [f for f in input_frames if f is not None]

        if not valid_frames or not inpaint_area:
            return [f if f is not None else np.zeros((H_ori, W_ori, 3), dtype=np.uint8) for f in input_frames]

        frames_hr = [f.copy() for f in valid_frames]
        frames_scaled = {}
        comps = {}

        for k in range(len(inpaint_area)):
            frames_scaled[k] = []

        for j in range(len(frames_hr)):
            image = frames_hr[j]
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))
                frames_scaled[k].append(image_resize)

        for k in range(len(inpaint_area)):
            comps[k] = self.inpaint(frames_scaled[k])

        FEATHER_SIGMA = 2.0

        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]
                for k in range(len(inpaint_area)):
                    comp_result = comps[k][j]
                    if comp_result is None:
                        continue
                    comp = cv2.resize(comp_result, (W_ori, split_h))
                    comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                    mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                    feathered = cv2.GaussianBlur(mask_area.astype(np.float32), (0, 0), FEATHER_SIGMA)
                    if feathered.ndim == 2:
                        feathered = feathered[:, :, None]
                    frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = (
                        feathered * comp.astype(np.float32)
                        + (1.0 - feathered) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :].astype(np.float32)
                    ).astype(np.uint8)

        result = []
        valid_idx = 0
        for i in range(len(input_frames)):
            if none_mask[i]:
                result.append(np.zeros((H_ori, W_ori, 3), dtype=np.uint8))
            else:
                result.append(frames_hr[valid_idx])
                valid_idx += 1
        return result

    STTNInpaint.__call__ = _safe_sttn_inpaint_call

    _original_video_call = STTNVideoInpaint.__call__

    def _safe_sttn_video_call(self, input_mask=None, input_sub_remover=None, tbar=None):
        reader, frame_info = self.read_frame_info_from_video()
        if input_sub_remover is not None:
            writer = input_sub_remover.video_writer
        else:
            writer = cv2.VideoWriter(
                self.video_out_path,
                cv2.VideoWriter_fourcc(*"mp4v"),
                frame_info['fps'],
                (frame_info['W_ori'], frame_info['H_ori']),
            )
        rec_time = (
            frame_info['len'] // self.clip_gap
            if frame_info['len'] % self.clip_gap == 0
            else frame_info['len'] // self.clip_gap + 1
        )
        split_h = int(frame_info['W_ori'] * 3 / 16)
        if input_mask is None:
            mask = self.sttn_inpaint.read_mask(self.mask_path)
        else:
            _, mask = cv2.threshold(input_mask, 127, 1, cv2.THRESH_BINARY)
            mask = mask[:, :, None]
        inpaint_area = self.sttn_inpaint.get_inpaint_area_by_mask(frame_info['H_ori'], split_h, mask)

        FEATHER_SIGMA = 2.0
        skipped_frames = 0

        for i in range(rec_time):
            start_f = i * self.clip_gap
            end_f = min((i + 1) * self.clip_gap, frame_info['len'])
            print('Processing:', start_f + 1, '-', end_f, ' / Total:', frame_info['len'])
            frames_hr = []
            frames = {}
            comps = {}
            for k in range(len(inpaint_area)):
                frames[k] = []

            for j in range(start_f, end_f):
                success, image = reader.read()
                if not success or image is None:
                    skipped_frames += 1
                    continue
                frames_hr.append(image)
                for k in range(len(inpaint_area)):
                    image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]
                    image_resize = cv2.resize(image_crop, (self.sttn_inpaint.model_input_width, self.sttn_inpaint.model_input_height))
                    frames[k].append(image_resize)

            if not frames_hr:
                print(f'WARNING: No valid frames in chunk {i}, skipping')
                continue

            actual_frame_count = len(frames_hr)

            for k in range(len(inpaint_area)):
                comps[k] = self.sttn_inpaint.inpaint(frames[k])

            if inpaint_area:
                for j in range(actual_frame_count):
                    if input_sub_remover is not None and input_sub_remover.gui_mode:
                        original_frame = copy.deepcopy(frames_hr[j])
                    else:
                        original_frame = None
                    frame = frames_hr[j]
                    for k in range(len(inpaint_area)):
                        comp_result = comps[k][j]
                        if comp_result is None:
                            continue
                        comp = cv2.resize(comp_result, (frame_info['W_ori'], split_h))
                        comp = cv2.cvtColor(np.array(comp).astype(np.uint8), cv2.COLOR_BGR2RGB)
                        mask_area = mask[inpaint_area[k][0]:inpaint_area[k][1], :]
                        feathered = cv2.GaussianBlur(mask_area.astype(np.float32), (0, 0), FEATHER_SIGMA)
                        if feathered.ndim == 2:
                            feathered = feathered[:, :, None]
                        frame[inpaint_area[k][0]:inpaint_area[k][1], :, :] = (
                            feathered * comp.astype(np.float32)
                            + (1.0 - feathered) * frame[inpaint_area[k][0]:inpaint_area[k][1], :, :].astype(np.float32)
                        ).astype(np.uint8)
                    writer.write(frame)
                    if input_sub_remover is not None:
                        if tbar is not None:
                            input_sub_remover.update_progress(tbar, increment=1)
                        if original_frame is not None and input_sub_remover.gui_mode:
                            input_sub_remover.preview_frame = cv2.hconcat([original_frame, frame])

        if skipped_frames > 0:
            print(f'INFO: Skipped {skipped_frames} unreadable frame(s) during STTN processing')
        writer.release()

    STTNVideoInpaint.__call__ = _safe_sttn_video_call

    _STTN_PATCHED = True
    print("INFO: Patched STTNInpaint.__call__ and STTNVideoInpaint.__call__ (None-safe + feathered blending)")


def _start_progress_poller(remover, phase, scale_end, progress_callback, poll_interval=0.5):
    """Background thread that reads remover.progress_total and rescaled
    forwards it via progress_callback until the remover is released or the
    thread is stopped."""
    stop_event = threading.Event()
    state = {"last_reported": -1}

    def _loop():
        while not stop_event.is_set():
            try:
                raw = float(getattr(remover, "progress_total", 0) or 0)
            except Exception:
                raw = 0.0
            scaled = max(0.0, min(scale_end, raw * scale_end / 100.0))
            current = int(scaled)
            if current != state["last_reported"]:
                try:
                    progress_callback(phase, scaled, "inpainting")
                except Exception:
                    pass
                state["last_reported"] = current
            stop_event.wait(poll_interval)

    thread = threading.Thread(target=_loop, name=f"progress-poller-{phase}", daemon=True)
    thread.start()
    return stop_event


def _run_subtitle_remover(video_path, area, options, refine_area=None, progress_callback=None):
    _patch_numpy_compat()
    from backend.main import SubtitleRemover, SubtitleDetect
    from backend import config

    def _report(phase, percent, stage=""):
        if progress_callback is not None:
            try:
                progress_callback(phase, percent, stage)
            except Exception:
                pass

    if options["mode"] == "sttn":
        _patch_sttn_none_safe_and_quality()

    ocr_preset_name = str(options.get("ocr_preset") or "default").strip().lower()
    ocr_preset = _OCR_PRESETS.get(ocr_preset_name, _OCR_PRESETS["default"])
    need_ocr_override = (
        not options.get("sttn_skip_detection", False)
        and ocr_preset_name != "default"
        and (ocr_preset.get("det_db_thresh") is not None or ocr_preset.get("det_db_box_thresh") is not None)
    )

    old_values = {}
    last_exc = None
    has_refine = bool(options.get("post_lama_refine") and options["mode"] == "sttn")
    main_scale_end = 70.0 if has_refine else 100.0
    remover = None
    poller_stop = None
    phase_label = "sttn" if options["mode"] == "sttn" else options["mode"]
    for attempt in range(2):
        remover = SubtitleRemover(str(video_path), sub_area=area)
        if need_ocr_override:
            print(f"INFO: _run_subtitle_remover overriding OCR detector: ocr_preset={ocr_preset_name}, det_db_thresh={ocr_preset.get('det_db_thresh')}, det_db_box_thresh={ocr_preset.get('det_db_box_thresh')}, det_limit_side_len={ocr_preset.get('det_limit_side_len')}")
            remover.sub_detector = SubtitleDetect(
                str(video_path), sub_area=area,
                det_db_thresh=ocr_preset.get("det_db_thresh"),
                det_db_box_thresh=ocr_preset.get("det_db_box_thresh"),
                det_limit_side_len=ocr_preset.get("det_limit_side_len"),
            )
        elif options.get("sttn_skip_detection", False):
            print(f"INFO: _run_subtitle_remover sttn_skip_detection=True, OCR override not needed")
        else:
            print(f"INFO: _run_subtitle_remover ocr_preset={ocr_preset_name} (default), using built-in OCR thresholds")
        old_values = _apply_config_options(config, options)
        # Detector-empty fallback: if STTN is the engine and the user did
        # NOT explicitly request skip_detection, peek at the OCR result
        # first. If sub_list is empty (PaddleOCR missed the subtitles —
        # common with thick-black-stroke + AA-fringe hardcoded subs on
        # 1080p), flip STTN_SKIP_DETECTION=True so sttn_mode() takes the
        # sttn_mode_with_no_detection branch and inpaints the entire
        # sub_area. Without this, an OCR miss means every frame gets
        # written unchanged and the user sees the original subtitle.
        if (
            options["mode"] == "sttn"
            and not options.get("sttn_skip_detection", False)
        ):
            try:
                _report(phase_label, 0.0, "detector_peek")
                peek_list = remover.sub_detector.find_subtitle_frame_no(sub_remover=remover)
            except Exception as peek_exc:
                print(f"WARN: detector peek failed ({type(peek_exc).__name__}: {peek_exc}), proceeding without fallback")
                peek_list = None
            if peek_list is not None and len(peek_list) == 0:
                print(
                    "WARN: STTN detector found no subtitle frames in sub_area — "
                    "falling back to STTN_SKIP_DETECTION=True (use the entire "
                    "sub_area as the inpaint mask). Override with "
                    "sttn_skip_detection=False to keep the empty result."
                )
                config.STTN_SKIP_DETECTION = True
            else:
                # Restore whatever the original value was so a non-empty
                # detector result doesn't accidentally trigger skip mode
                # later if a downstream pass re-reads config.
                config.STTN_SKIP_DETECTION = bool(options.get("sttn_skip_detection", False))
        _report(phase_label, 0.0, "starting")
        poller_stop = _start_progress_poller(remover, phase_label, main_scale_end, _report, poll_interval=0.5)
        try:
            if options["mode"] == "lama_area":
                _run_lama_area_remover(remover, area)
            elif options["mode"] == "blur_cover":
                ocr_preset = _OCR_PRESETS.get(ocr_preset_name, _OCR_PRESETS["default"])
                _run_blur_cover(video_path, area, options, ocr_preset=ocr_preset)
            else:
                remover.run()
            break
        except Exception as run_exc:
            last_exc = run_exc
            try:
                if hasattr(remover, "video_writer") and remover.video_writer is not None:
                    remover.video_writer.release()
                if hasattr(remover, "video_cap") and remover.video_cap is not None:
                    remover.video_cap.release()
            except Exception:
                pass
            if attempt == 0 and _is_cudnn_error(run_exc):
                print(f"WARNING: CUDNN error on attempt 1: {run_exc}. Clearing GPU cache and retrying...")
                _clear_gpu_cache()
                continue
            raise run_exc
    else:
        raise last_exc
    if poller_stop is not None:
        poller_stop.set()
    _report(phase_label, main_scale_end, "main_done")
    output_path = Path(remover.video_out_name).resolve()
    if not output_path.is_file():
        raise RequestError(HTTPStatus.INTERNAL_SERVER_ERROR, "Subtitle remover did not create an output file")
    if has_refine:
        refine_target = refine_area or area
        refine_start = main_scale_end
        refine_end = 95.0
        refine_method = options.get("post_refine_method") or "telea_text"
        if refine_method == "lama_rect":
            _report("refine", refine_start, "starting")
            refine_remover = SubtitleRemover(str(output_path), sub_area=refine_target)
            _apply_config_options(config, options)

            def _refine_report(phase, percent, stage=""):
                # Rescale the poller's 0-100 raw value into the refine bar
                # segment [refine_start, refine_end].
                raw = float(percent or 0)
                scaled = refine_start + (raw / 100.0) * (refine_end - refine_start)
                _report("refine", scaled, stage)

            refine_poller_stop = _start_progress_poller(
                refine_remover, "refine", 100.0, _refine_report,
                poll_interval=0.5,
            )
            try:
                _run_lama_area_remover(refine_remover, refine_target)
            finally:
                refine_poller_stop.set()
            new_output = Path(refine_remover.video_out_name).resolve()
        else:
            new_output = _run_text_trace_refine(
                output_path, video_path, refine_target, options,
                progress_callback=progress_callback,
                start_percent=refine_start,
                end_percent=refine_end,
            )
        # Defensive: an old / partially-deployed api-server may leave
        # output_path unset in some error path. Coerce None to a clear
        # error instead of a confusing AttributeError downstream.
        if new_output is None:
            raise RequestError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"Post refinement ({refine_method}) returned no output path",
            )
        output_path = new_output
        _report("refine", refine_end, "refine_done")
        if not output_path.is_file():
            raise RequestError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"Post refinement did not create an output file at {output_path}",
            )
    # Post-verify blur: opt-in tail pass that re-runs PaddleOCR text_detector
    # on the STTN output and blurs any residual-text bboxes. Off by default
    # because it costs another 30-60s on the PaddleOCR pass. The
    # helper itself auto-skips when no residual is found, so it only
    # adds cost when there's actually something to fix.
    if options.get("post_verify_blur", False) and output_path.is_file():
        try:
            verified_output = _run_post_verify_blur(
                output_path, area, options,
                progress_callback=progress_callback,
                start_percent=95.0, end_percent=99.0,
            )
            if verified_output != output_path:
                output_path = verified_output
                _report("refine", 99.0, "verify_done")
        except Exception as verify_exc:
            # Best-effort: log and keep the upstream output rather than
            # failing the whole job. Print the traceback so silent
            # bugs (NameError, etc.) surface immediately — not buried
            # in a one-line WARN that's easy to miss.
            import traceback
            print(
                f"WARN: post_verify_blur failed ({type(verify_exc).__name__}: "
                f"{verify_exc}); returning upstream output unchanged"
            )
            traceback.print_exc()
    # 输出完整性校验：防止 GPU 异常导致末尾 chunk 丢失却返回 200
    input_info = _probe_video_info(video_path)
    output_info = _probe_video_info(output_path)
    input_dur = input_info.get("duration") or 0
    output_dur = output_info.get("duration") or 0
    if input_dur > 3 and output_dur > 0:
        frame_diff = abs(input_info.get("nb_frames", 0) - output_info.get("nb_frames", 0))
        dur_diff = abs(input_dur - output_dur)
        # 帧数差超过 5% 或时长差超过 1.5 秒视为不完整
        if frame_diff > max(30, input_info.get("nb_frames", 0) * 0.05) or dur_diff > 1.5:
            print(
                f"ERROR: Subtitle remover output incomplete. "
                f"input={input_info['nb_frames']}f/{input_dur:.2f}s, "
                f"output={output_info['nb_frames']}f/{output_dur:.2f}s, "
                f"frame_diff={frame_diff}, dur_diff={dur_diff:.2f}s"
            )
            raise RequestError(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                f"硬字幕擦除输出不完整：输入 {input_info['nb_frames']} 帧/{input_dur:.1f}s，"
                f"输出仅 {output_info['nb_frames']} 帧/{output_dur:.1f}s。"
                f"通常是 GPU 处理末尾 chunk 时异常导致。"
            )
    # Auto residual cleanup: re-scan the upstream output for white / yellow
    # text-like pixels that survived STTN's main pass, and inpaint them
    # inside the user-defined `area` only. This catches the two main
    # badcases — entire rows of text that STTN missed, and glyph edges that
    # were outside STTN's text-trace mask. blur_cover is intentionally
    # skipped because the whole area is already blurred.
    if options["mode"] != "blur_cover" and options.get("auto_residual_cleanup", False) and output_path.is_file():
        try:
            residual_tmp = output_path.with_name(output_path.stem + "_residual.mp4")
            _report("refine", 70.0, "residual_starting")
            residual_output = _run_residual_cleanup(
                output_path, residual_tmp, area, options,
                progress_callback=progress_callback,
                start_percent=70.0, end_percent=95.0,
            )
            # Replace the upstream output with the residual-cleaned one.
            try:
                output_path.unlink()
            except OSError:
                pass
            residual_output.replace(output_path)
            _report("refine", 95.0, "residual_done")
        except Exception as residual_exc:
            # Residual cleanup is best-effort: if it fails (e.g. cv2 import
            # blip, decoder error on the upstream mp4), log and keep the
            # upstream output rather than failing the whole job.
            print(
                f"WARN: residual_cleanup failed ({type(residual_exc).__name__}: "
                f"{residual_exc}); returning upstream output unchanged"
            )
            try:
                if residual_tmp.is_file():
                    residual_tmp.unlink()
            except OSError:
                pass
    _restore_config_options(config, old_values)
    # Final pass: ensure the returned file is H.264 so browsers can play
    # the result page. STTN's `merge_audio_to_video` defaults to
    # `-vcodec copy`, which preserves cv2's mp4v fourcc and produces
    # mpeg4 (Xvid) video — Chrome/Safari can't decode that and silently
    # fall back to audio-only.
    try:
        output_path = _ensure_h264(output_path)
    except Exception as h264_exc:
        print(f"WARN: _ensure_h264 failed: {h264_exc}; returning original output")
    return output_path


def _normalize_detect_options(payload):
    raw_options = payload.get("detect_options")
    if not isinstance(raw_options, dict):
        raw_options = payload.get("options")
    if not isinstance(raw_options, dict):
        raw_options = {}
    merged = {**raw_options, **{key: payload[key] for key in payload.keys() if key in {
        "sample_count",
        "detect_sample_count",
        "min_center_y_pct",
        "max_center_y_pct",
        "preferred_center_y_pct",
        "allow_outside_band",
        "padding_x",
        "padding_y",
        "raw_area_buffer_y",
        "max_boxes",
        "ocr_preset",
    }}}
    ocr_preset_raw = str(merged.get("ocr_preset") or "default").strip().lower()
    if ocr_preset_raw not in _OCR_PRESETS:
        ocr_preset_raw = "default"
    return {
        "sample_count": _bounded_int(
            merged.get("sample_count", merged.get("detect_sample_count")),
            "sample_count",
            72,
            1,
            160,
        ),
        "min_center_y_pct": _bounded_int(merged.get("min_center_y_pct"), "min_center_y_pct", 50, 0, 95),
        "max_center_y_pct": _bounded_int(merged.get("max_center_y_pct"), "max_center_y_pct", 82, 50, 100),
        "preferred_center_y_pct": _bounded_int(merged.get("preferred_center_y_pct"), "preferred_center_y_pct", 68, 40, 98),
        "allow_outside_band": _to_bool(merged.get("allow_outside_band", False)),
        "padding_x": _bounded_int(merged.get("padding_x"), "padding_x", 16, 0, 240),
        "padding_y": _bounded_int(merged.get("padding_y"), "padding_y", 16, 0, 240),
        "raw_area_buffer_y": _bounded_int(merged.get("raw_area_buffer_y"), "raw_area_buffer_y", 60, 0, 120),
        "max_boxes": _bounded_int(merged.get("max_boxes"), "max_boxes", 240, 20, 1000),
        "ocr_preset": ocr_preset_raw,
    }


def _sample_frame_numbers(frame_count, sample_count):
    frame_count = int(frame_count or 0)
    sample_count = int(sample_count or 0)
    if frame_count <= 0 or sample_count <= 0:
        return []
    count = min(frame_count, sample_count)
    if count >= frame_count:
        return list(range(1, frame_count + 1))
    frames = {
        max(1, min(frame_count, int(round((index + 0.5) * frame_count / count))))
        for index in range(count)
    }
    return sorted(frames)


def _box_from_coords(frame_no, coord, width, height):
    xmin, xmax, ymin, ymax = coord
    xmin = max(0, min(int(width) - 1, int(xmin)))
    xmax = max(xmin + 1, min(int(width), int(xmax)))
    ymin = max(0, min(int(height) - 1, int(ymin)))
    ymax = max(ymin + 1, min(int(height), int(ymax)))
    return {
        "frame": int(frame_no),
        "x": xmin,
        "y": ymin,
        "width": xmax - xmin,
        "height": ymax - ymin,
    }


def _box_in_area(box, area):
    if area is None:
        return True
    ymin, ymax, xmin, xmax = area
    box_xmax = box["x"] + box["width"]
    box_ymax = box["y"] + box["height"]
    return xmin <= box["x"] and box_xmax <= xmax and ymin <= box["y"] and box_ymax <= ymax


def _area_from_boxes(boxes):
    if not boxes:
        return None
    left = min(box["x"] for box in boxes)
    top = min(box["y"] for box in boxes)
    right = max(box["x"] + box["width"] for box in boxes)
    bottom = max(box["y"] + box["height"] for box in boxes)
    return {"x": left, "y": top, "width": right - left, "height": bottom - top}


def _padded_area(area, width, height, pad_x, pad_y):
    if area is None:
        return None
    left = max(0, int(area["x"]) - int(pad_x))
    top = max(0, int(area["y"]) - int(pad_y))
    right = min(int(width), int(area["x"] + area["width"]) + int(pad_x))
    bottom = min(int(height), int(area["y"] + area["height"]) + int(pad_y))
    return {
        "x": left,
        "y": top,
        "width": max(1, right - left),
        "height": max(1, bottom - top),
    }


def _sub_area_from_area(area):
    if area is None:
        return None
    return [
        int(area["y"]),
        int(area["y"] + area["height"]),
        int(area["x"]),
        int(area["x"] + area["width"]),
    ]


def _choose_subtitle_area(boxes, width, height, options):
    if not boxes or width <= 0 or height <= 0:
        return None, None, [], 0.0
    min_center_y = height * float(options["min_center_y_pct"]) / 100.0
    max_center_y = height * float(options["max_center_y_pct"]) / 100.0
    preferred_center_y = float(options["preferred_center_y_pct"]) / 100.0
    aspect_ratio = float(height) / max(1.0, float(width))
    if aspect_ratio > 1.5:
        preferred_center_y = max(preferred_center_y, 0.76)
    primary_candidates = [
        box for box in boxes
        if min_center_y <= box["y"] + box["height"] / 2.0 <= max_center_y
    ]
    if primary_candidates:
        candidates = primary_candidates
    elif options.get("allow_outside_band"):
        candidates = [
            box for box in boxes
            if box["y"] + box["height"] / 2.0 >= min_center_y
        ] or boxes
    else:
        return None, None, [], 0.0
    threshold = max(18.0, height * 0.06)
    groups = []
    for box in sorted(candidates, key=lambda item: item["y"] + item["height"] / 2.0):
        center_y = box["y"] + box["height"] / 2.0
        matched = None
        for group in groups:
            group_center = group["center_sum"] / group["count"]
            if abs(center_y - group_center) <= threshold:
                matched = group
                break
        if matched is None:
            matched = {
                "boxes": [],
                "frames": set(),
                "center_sum": 0.0,
                "count": 0,
            }
            groups.append(matched)
        matched["boxes"].append(box)
        matched["frames"].add(box["frame"])
        matched["center_sum"] += center_y
        matched["count"] += 1

    def group_score(group):
        group_area = _area_from_boxes(group["boxes"]) or {"x": 0, "y": 0, "width": 0, "height": 0}
        center_x = (group_area["x"] + group_area["width"] / 2.0) / max(1.0, float(width))
        center_y = (group_area["y"] + group_area["height"] / 2.0) / max(1.0, float(height))
        bottom = (group_area["y"] + group_area["height"]) / max(1.0, float(height))
        width_ratio = group_area["width"] / max(1.0, float(width))
        height_ratio = group_area["height"] / max(1.0, float(height))
        vertical_tolerance = 0.30 if aspect_ratio > 1.5 else 0.22
        vertical_score = max(0.0, 1.0 - abs(center_y - preferred_center_y) / vertical_tolerance)
        center_score = max(0.0, 1.0 - abs(center_x - 0.5) / 0.45)
        width_score = min(1.0, width_ratio / 0.55)
        height_score = 1.0 - min(1.0, abs(height_ratio - 0.08) / 0.18)
        bottom_penalty = max(0.0, bottom - 0.96) * 35.0
        tiny_edge_penalty = 2.0 if width_ratio < 0.12 and center_score < 0.7 else 0.0
        return (
            len(group["frames"]) * 8.0
            + len(group["boxes"]) * 1.1
            + vertical_score * 6.0
            + center_score * 2.0
            + width_score * 2.0
            + height_score
            - bottom_penalty
            - tiny_edge_penalty
        )

    selected = max(groups, key=group_score)
    raw_area = _area_from_boxes(selected["boxes"])
    if raw_area:
        buffer_y = int(options.get("raw_area_buffer_y", 10))
        raw_area = {
            "x": raw_area["x"],
            "y": max(0, raw_area["y"] - buffer_y),
            "width": raw_area["width"],
            "height": min(height - max(0, raw_area["y"] - buffer_y), raw_area["height"] + buffer_y * 2),
        }
    area = _padded_area(raw_area, width, height, options["padding_x"], options["padding_y"])
    sampled_frames = {box["frame"] for box in boxes}
    hit_frames = len(selected["frames"])
    confidence = min(1.0, 0.25 + hit_frames / max(1, len(sampled_frames)) * 0.55 + min(0.2, len(selected["boxes"]) / 50.0))
    return area, raw_area, selected["boxes"], round(confidence, 3)


_OCR_PRESETS = {
    "default": {
        # Lower thresholds than PaddleDB's stock (which is det_db_thresh 0.3).
        # Subtitles with thick black stroke + AA fringe often get their
        # inner glyphs suppressed by the stroke at the default threshold,
        # so the whole row gets missed. 0.05 / 0.15 catches white-on-dark
        # with stroke reliably without producing too many false positives
        # in the typical 1080p live-action-drama footage.
        "det_db_thresh": 0.05,
        "det_db_box_thresh": 0.15,
        "det_limit_side_len": 1920,
        "detect_max_edge": 1920,
    },
    "fuzzy": {
        # 比 default 略激进(thresh 0.05→0.03,box 0.15→0.10),仍然比 ultra
        # 保守(thresh 0.02)。边长保持 1920(不增加检测成本)。
        # 适用:1080p 模糊字幕/低对比度/抗锯齿 fringe 漏字边,default 漏检
        # 但 ultra 又过于激进(误检率上升)的中间档。
        "det_db_thresh": 0.03,
        "det_db_box_thresh": 0.10,
        "det_limit_side_len": 1920,
        "detect_max_edge": 1920,
    },
    "ultra": {
        "det_db_thresh": 0.02,
        "det_db_box_thresh": 0.05,
        "det_limit_side_len": 1920,
        "detect_max_edge": 2400,
    },
    "aggressive": {
        "det_db_thresh": 0.005,
        "det_db_box_thresh": 0.02,
        "det_limit_side_len": 2560,
        "detect_max_edge": 2560,
    },
}

_MAX_DETECTION_EDGE = 1280


def _detect_subtitle_on_frame(detector, frame, width, height, max_detection_edge=None):
    """对单帧做字幕检测；如果分辨率过高先缩放，再把坐标映射回原始分辨率。"""
    import cv2

    edge_limit = max_detection_edge or _MAX_DETECTION_EDGE
    max_edge = max(frame.shape[:2])
    scale = 1.0
    detect_frame = frame
    if max_edge > edge_limit:
        scale = edge_limit / max_edge
        detect_frame = cv2.resize(frame, None, fx=scale, fy=scale)
    try:
        coords = detector.detect_subtitle(detect_frame)
    except OSError:
        # cuDNN 对大尺寸输入可能报 CUDNN_STATUS_NOT_SUPPORTED；再缩一半重试
        if max_edge > edge_limit:
            scale = (edge_limit // 2) / max_edge
            detect_frame = cv2.resize(frame, None, fx=scale, fy=scale)
            coords = detector.detect_subtitle(detect_frame)
        else:
            raise
    # `detect_subtitle` already returns (xmin, xmax, ymin, ymax) tuples
    # in `detect_frame` pixel space — no raw-box conversion needed.
    if coords:
        print(f"INFO: _detect_subtitle_on_frame found {len(coords)} boxes (scale={scale:.3f}, edge_limit={edge_limit}, frame={width}x{height})")
    if scale < 1.0:
        coords = [
            (
                int(xmin / scale),
                int(xmax / scale),
                int(ymin / scale),
                int(ymax / scale),
            )
            for xmin, xmax, ymin, ymax in coords
        ]
    return coords


def _run_subtitle_area_detection(video_path, area, options):
    _patch_numpy_compat()
    import cv2
    from backend.main import SubtitleDetect

    ocr_preset_name = str(options.get("ocr_preset") or "default").strip().lower()
    ocr_preset = _OCR_PRESETS.get(ocr_preset_name, _OCR_PRESETS["default"])
    det_db_thresh = ocr_preset.get("det_db_thresh")
    det_db_box_thresh = ocr_preset.get("det_db_box_thresh")
    det_limit_side_len = ocr_preset.get("det_limit_side_len")
    detect_max_edge = ocr_preset.get("detect_max_edge") or _MAX_DETECTION_EDGE
    print(f"INFO: _run_subtitle_area_detection ocr_preset={ocr_preset_name}, det_db_thresh={det_db_thresh}, det_db_box_thresh={det_db_box_thresh}, det_limit_side_len={det_limit_side_len}, detect_max_edge={detect_max_edge}")

    video_cap = cv2.VideoCapture(str(video_path))
    if not video_cap.isOpened():
        raise RequestError(HTTPStatus.BAD_REQUEST, "Could not open video")
    try:
        frame_count = int(video_cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(video_cap.get(cv2.CAP_PROP_FPS) or 0.0)
        width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        frame_numbers = _sample_frame_numbers(frame_count, options["sample_count"])
        detector = SubtitleDetect(
            str(video_path), sub_area=area,
            det_db_thresh=det_db_thresh, det_db_box_thresh=det_db_box_thresh,
            det_limit_side_len=det_limit_side_len,
        )
        boxes = []
        frames_sampled = []

        if frame_numbers:
            for frame_no in frame_numbers:
                video_cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_no - 1))
                ret, frame = video_cap.read()
                if not ret:
                    continue
                frames_sampled.append(frame_no)
                if width <= 0 or height <= 0:
                    height, width = frame.shape[:2]
                for coord in _detect_subtitle_on_frame(detector, frame, width, height, detect_max_edge):
                    box = _box_from_coords(frame_no, coord, width, height)
                    if box["width"] <= 1 or box["height"] <= 1 or not _box_in_area(box, area):
                        continue
                    boxes.append(box)
        else:
            current_frame = 0
            while video_cap.isOpened() and len(frames_sampled) < options["sample_count"]:
                ret, frame = video_cap.read()
                if not ret:
                    break
                current_frame += 1
                frames_sampled.append(current_frame)
                if width <= 0 or height <= 0:
                    height, width = frame.shape[:2]
                for coord in _detect_subtitle_on_frame(detector, frame, width, height, detect_max_edge):
                    box = _box_from_coords(current_frame, coord, width, height)
                    if box["width"] <= 1 or box["height"] <= 1 or not _box_in_area(box, area):
                        continue
                    boxes.append(box)
    finally:
        video_cap.release()

    detected_area, raw_area, selected_boxes, confidence = _choose_subtitle_area(boxes, width, height, options)
    max_boxes = options["max_boxes"]
    return {
        "video_size": {"width": width, "height": height},
        "frame_count": frame_count,
        "fps": round(fps, 3),
        "sample_method": "uniform_midpoint_frames",
        "random": False,
        "sample_count_requested": options["sample_count"],
        "frames_sampled": frames_sampled,
        "box_count": len(boxes),
        "area": detected_area,
        "raw_area": raw_area,
        "sub_area": _sub_area_from_area(detected_area),
        "raw_sub_area": _sub_area_from_area(raw_area),
        "confidence": confidence,
        "boxes": boxes[:max_boxes],
        "selected_boxes": selected_boxes[:max_boxes],
        "input_area": list(area) if area is not None else None,
        "detect_options": options,
    }


def _download_url(handler, job_id, filename):
    host = handler.headers.get("Host") or f"{handler.server.server_name}:{handler.server.server_port}"
    proto = handler.headers.get("X-Forwarded-Proto", "http").split(",")[0].strip() or "http"
    return f"{proto}://{host}/download/{quote(job_id)}/{quote(filename)}"


class RemoverAPIHandler(BaseHTTPRequestHandler):
    server_version = "VideoSubtitleRemoverAPI/1.1"

    def do_GET(self):
        parsed = urlparse(self.path)
        if parsed.path == "/health":
            self._send_json(HTTPStatus.OK, {"status": "ok"})
            return
        if parsed.path == "/":
            self._send_json(
                HTTPStatus.OK,
                {
                    "name": "video-subtitle-remover-api",
                    "endpoints": {
                        "submit": "POST /api/remove-subtitle",
                        "detect": "POST /api/detect-subtitle-area",
                        "download": "GET /download/{job_id}/{filename}",
                        "progress": "GET /api/progress/{job_id}",
                        "health": "GET /health",
                    },
                },
            )
            return
        if parsed.path.startswith("/api/progress/"):
            _prune_active_jobs()
            job_id = parsed.path[len("/api/progress/"):].strip("/")
            if not job_id:
                self._send_json(HTTPStatus.BAD_REQUEST, {"error": "job_id is required"})
                return
            entry = _get_job_progress(job_id)
            if entry is None:
                self._send_json(HTTPStatus.OK, {"job_id": job_id, "phase": "unknown", "percent": 0.0, "stage": "not_found", "found": False})
                return
            entry["found"] = True
            self._send_json(HTTPStatus.OK, entry)
            return
        if parsed.path.startswith("/download/"):
            self._handle_download(parsed.path)
            return
        self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        if parsed.path in ("/api/detect-subtitle-area", "/api/detect", "/detect-subtitle-area"):
            self._handle_detect_subtitle_area()
            return
        if parsed.path not in ("/api/remove-subtitle", "/api/remove", "/remove"):
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        started_at = time.time()
        # job_id is provided by the caller (e.g. the dub web app) so it can
        # poll /api/progress/{job_id} for in-flight progress updates. We
        # fall back to a fresh uuid when the caller doesn't supply one.
        # self.headers is an http.client.HTTPMessage; iterating it yields
        # "Key: Value" strings, so we must use .get() / .items() instead
        # of tuple-unpacking the iterator.
        provided_job_id = ""
        try:
            provided_job_id = str(self.headers.get("X-Job-Id", "") or "").strip()
        except Exception:
            try:
                provided_job_id = ""
                for key in self.headers:
                    if str(key).lower() == "x-job-id":
                        provided_job_id = str(self.headers.get(key, "") or "").strip()
                        break
            except Exception:
                provided_job_id = ""

        try:
            payload = _read_json(self)
        except RequestError as exc:
            # Body parse failed with a structured 4xx (missing/oversized
            # Content-Length, bad JSON, body not an object). The previous
            # implementation re-raised here, which escaped do_POST entirely
            # and left the client with a connection reset / no HTTP body.
            # Send a real response so the dub web app can surface the
            # upstream cause to the user.
            self._send_json(exc.status, {"error": exc.message})
            return
        except Exception as exc:
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": f"Invalid request body: {exc}"})
            return
        if not provided_job_id and isinstance(payload, dict):
            provided_job_id = str(payload.get("progress_job_id") or "").strip()
        job_id = provided_job_id or uuid.uuid4().hex
        job_dir = WORK_DIR / "jobs" / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            # Caller reused a job_id; reuse the existing directory.
            pass

        try:
            video_url = payload.get("video_url") or payload.get("url")
            if not isinstance(video_url, str) or not video_url.strip():
                raise RequestError(HTTPStatus.BAD_REQUEST, "video_url is required")

            filename_hint = _safe_filename(str(payload.get("filename") or "input.mp4"))
            area = _normalize_area(payload)
            options = _normalize_options(payload)
            refine_area = _normalize_refine_area(payload) if options.get("post_lama_refine") else None
            source_path = _stage_video(video_url.strip(), job_dir, filename_hint)

            def _job_progress(phase, percent, stage=""):
                _set_job_progress(job_id, phase, percent, stage)
                # Also surface progress in the api-server stdout so operators
                # can see main pass / STTN 修边 pass / done in the log. The
                # dub web app gets the structured data via /api/progress;
                # this print is for the server log only. Throttle to phase
                # changes + 2% steps + terminal states. 2% (50 lines/phase)
                # is the sweet spot — 10% felt jumpy, 1% (100 lines/phase)
                # is too noisy in journalctl / docker logs.
                pct_int = int(round(percent))
                bucket = pct_int // 2
                terminal = phase in ("done", "error")
                with _last_logged_progress_LOCK:
                    last = _last_logged_progress.get(job_id)
                    should_log = (
                        last is None
                        or last[0] != phase
                        or last[1] != bucket
                        or terminal
                    )
                    if should_log:
                        _last_logged_progress[job_id] = (phase, bucket)
                        if terminal:
                            _last_logged_progress.pop(job_id, None)
                if should_log:
                    label = {
                        "sttn": "STTN 擦除",
                        "lama_area": "LAMA 擦除",
                        "blur_cover": "模糊覆盖",
                        "lama": "LAMA 擦除",
                        "refine": "STTN 修边",
                        "stitch": "合并片段",
                        "finalize": "回灌音频",
                        "done": "完成",
                        "error": "出错",
                    }.get(phase, phase)
                    print(
                        f"PROGRESS job={job_id} {label} {pct_int}%{f' stage={stage}' if stage else ''}"
                    )

            with PROCESS_LOCK:
                output_path = _run_subtitle_remover(
                    source_path, area, options, refine_area,
                    progress_callback=_job_progress,
                )
            _set_job_progress(job_id, "done", 100.0, "completed")

            download_url = _download_url(self, job_id, output_path.name)
            print(f"INFO: job={job_id} download_url={download_url}")
            self._send_json(
                HTTPStatus.OK,
                {
                    "url": download_url,
                    "download_url": download_url,
                    "job_id": job_id,
                    "filename": output_path.name,
                    "area": list(area) if area is not None else None,
                    "refine_area": list(refine_area) if refine_area is not None else None,
                    "options": options,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                },
            )
        except RequestError as exc:
            # Surface the failure in the progress registry so a polling
            # client (e.g. the dub web app's progress poller) sees
            # phase="error" instead of the last successful phase / percent
            # it observed before the request blew up. Best-effort: a write
            # failure here must not mask the original exception.
            try:
                _set_job_progress(job_id, "error", 0.0, type(exc).__name__)
            except Exception:
                pass
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:
            # Print the full traceback to BOTH stderr (default) and stdout
            # (where PROGRESS lines live) so the operator can see the
            # failure next to the progress trail in journalctl / docker
            # logs / a tty. Also tag it with the job id for grep.
            tb_text = traceback.format_exc()
            print(
                f"ERROR job={job_id} {type(exc).__name__}: {exc}\n{tb_text}",
                flush=True,
            )
            try:
                _set_job_progress(job_id, "error", 0.0, type(exc).__name__)
            except Exception:
                pass
            try:
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": tb_text,
                    "job_id": job_id,
                },
            )

    def _handle_detect_subtitle_area(self):
        started_at = time.time()
        job_id = uuid.uuid4().hex
        job_dir = WORK_DIR / "jobs" / job_id
        try:
            job_dir.mkdir(parents=True, exist_ok=False)
        except FileExistsError:
            pass

        try:
            payload = _read_json(self)
            video_url = payload.get("video_url") or payload.get("url")
            if not isinstance(video_url, str) or not video_url.strip():
                raise RequestError(HTTPStatus.BAD_REQUEST, "video_url is required")

            filename_hint = _safe_filename(str(payload.get("filename") or "input.mp4"))
            area = _normalize_area(payload)
            detect_options = _normalize_detect_options(payload)
            source_path = _stage_video(video_url.strip(), job_dir, filename_hint)

            with PROCESS_LOCK:
                result = _run_subtitle_area_detection(source_path, area, detect_options)

            result.update(
                {
                    "job_id": job_id,
                    "filename": source_path.name,
                    "elapsed_seconds": round(time.time() - started_at, 3),
                }
            )
            self._send_json(HTTPStatus.OK, result)
        except RequestError as exc:
            self._send_json(exc.status, {"error": exc.message})
        except Exception as exc:
            # Print the full traceback to BOTH stderr (default) and stdout
            # (where PROGRESS lines live) so the operator can see the
            # failure next to the progress trail in journalctl / docker
            # logs / a tty. Also tag it with the job id for grep.
            tb_text = traceback.format_exc()
            print(
                f"ERROR job={job_id} {type(exc).__name__}: {exc}\n{tb_text}",
                flush=True,
            )
            try:
                if job_dir.exists():
                    shutil.rmtree(job_dir, ignore_errors=True)
            except Exception:
                pass
            self._send_json(
                HTTPStatus.INTERNAL_SERVER_ERROR,
                {
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": tb_text,
                    "job_id": job_id,
                },
            )

    def _handle_download(self, path):
        parts = path.split("/")
        if len(parts) != 4 or not parts[2] or not parts[3]:
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "not_found"})
            return

        job_id = parts[2]
        filename = _safe_filename(parts[3])
        if not re.fullmatch(r"[A-Fa-f0-9]{32}", job_id):
            self._send_json(HTTPStatus.BAD_REQUEST, {"error": "invalid job_id"})
            return

        file_path = (WORK_DIR / "jobs" / job_id / filename).resolve()
        job_root = (WORK_DIR / "jobs" / job_id).resolve()
        if job_root not in file_path.parents or not file_path.is_file():
            self._send_json(HTTPStatus.NOT_FOUND, {"error": "file_not_found"})
            return

        content_type = mimetypes.guess_type(file_path.name)[0] or "application/octet-stream"
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(file_path.stat().st_size))
        self.send_header("Content-Disposition", f'attachment; filename="{file_path.name}"')
        self.end_headers()
        with file_path.open("rb") as input_file:
            shutil.copyfileobj(input_file, self.wfile)

    def _send_json(self, status, payload):
        body = _json_bytes(payload)
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        # Skip the high-frequency /api/progress/<job_id> poll requests so the
        # access log stays readable. Other endpoints still log normally.
        try:
            if self.path.startswith("/api/progress/"):
                return
        except Exception:
            pass
        print(f"{self.address_string()} - {fmt % args}")


def main(argv=None):
    parser = argparse.ArgumentParser(description="HTTP API for backend.main.SubtitleRemover")
    parser.add_argument("--host", default=os.environ.get("HOST", "127.0.0.1"))
    parser.add_argument("--port", type=int, default=int(os.environ.get("PORT", "6006")))
    args = parser.parse_args(argv)

    try:
        multiprocessing.set_start_method("spawn")
    except RuntimeError:
        pass

    WORK_DIR.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), RemoverAPIHandler)
    print(f"Serving on http://{args.host}:{args.port}")
    print(f"Work dir: {WORK_DIR}")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
