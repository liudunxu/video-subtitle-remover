#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-6006}"
VSR_API_WORK_DIR="${VSR_API_WORK_DIR:-/workspace/video_subtitle_remover_api}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
MPLCONFIGDIR="${MPLCONFIGDIR:-${XDG_CACHE_HOME}/matplotlib}"

export HOST PORT VSR_API_WORK_DIR XDG_CACHE_HOME MPLCONFIGDIR
mkdir -p "${VSR_API_WORK_DIR}" "${XDG_CACHE_HOME}" "${MPLCONFIGDIR}"

PY_NVIDIA_LIBS="$(python - <<'PY'
from pathlib import Path
import site

paths = []
for base in site.getsitepackages():
    root = Path(base) / "nvidia"
    if not root.is_dir():
        continue
    for lib_dir in sorted(root.glob("*/lib")):
        if lib_dir.is_dir():
            paths.append(str(lib_dir))
print(":".join(paths))
PY
)"

if [ -n "${PY_NVIDIA_LIBS}" ]; then
    export LD_LIBRARY_PATH="${PY_NVIDIA_LIBS}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
fi

if [ "${VSR_REQUIRE_GPU:-1}" = "1" ]; then
    python - <<'PY'
import sys

errors = []

try:
    import torch
    if not torch.cuda.is_available():
        errors.append("torch.cuda.is_available() is false")
except Exception as exc:
    errors.append(f"torch CUDA check failed: {exc}")

try:
    import paddle
    if not paddle.device.is_compiled_with_cuda():
        errors.append("paddle is not compiled with CUDA")
    elif paddle.device.cuda.device_count() <= 0:
        errors.append("paddle sees zero CUDA devices")
except Exception as exc:
    errors.append(f"paddle GPU check failed: {exc}")

try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" not in providers:
        errors.append(f"onnxruntime CUDAExecutionProvider missing: {providers}")
except Exception as exc:
    errors.append(f"onnxruntime-gpu check failed: {exc}")

try:
    import paddlex  # noqa: F401
    import paddleocr  # noqa: F401
    from paddle import inference  # noqa: F401
except Exception as exc:
    errors.append(f"PaddleOCR/PaddleX/HPI import check failed: {exc}")

if errors:
    print("GPU runtime check failed:", file=sys.stderr)
    for error in errors:
        print(f"  - {error}", file=sys.stderr)
    sys.exit(1)

print("GPU runtime check passed")
PY
fi

exec python /app/api.py --host "${HOST}" --port "${PORT}"
