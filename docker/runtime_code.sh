#!/usr/bin/env bash
# Git-backed application source for the long-lived GPU runtime image
# (OmniVoice 同款模式：镜像只当环境，启动时同步最新代码运行，
# 代码更新只需 git push + 实例重启，不必重建镜像)。
# Shell output is deliberately reserved for the resolved source directory;
# diagnostics go to stderr so callers can safely capture the path.

set -euo pipefail

runtime_log() {
  printf '[runtime-code] %s\n' "$*" >&2
}

sync_git_checkout() {
  local code_dir="$1"
  local repository="$2"
  local ref="$3"
  local askpass=""

  mkdir -p "$code_dir" || return
  if ! git -C "$code_dir" rev-parse --is-inside-work-tree >/dev/null 2>&1; then
    git -C "$code_dir" init -q || return
  fi
  if git -C "$code_dir" remote get-url origin >/dev/null 2>&1; then
    git -C "$code_dir" remote set-url origin "$repository" || return
  else
    git -C "$code_dir" remote add origin "$repository" || return
  fi

  # 本仓库把模型权重（~600MB）、ffmpeg 二进制（~250MB）和素材放在 git 里。
  # --filter=blob:none（blobless 克隆）+ sparse-checkout 让启动只下载代码
  # blob；这些大目录在 checkout 后用镜像内烘焙的副本软链进来。
  git -C "$code_dir" sparse-checkout set --no-cone \
    '/*' '!/backend/models/' '!/backend/ffmpeg/' '!/design/' '!/test/' || return

  if [[ -n "${APP_GIT_TOKEN:-}" ]]; then
    askpass="$(mktemp)" || return
    chmod 700 "$askpass" || return
    cat >"$askpass" <<'EOF' || return
#!/bin/sh
case "$1" in
  *Username*) printf '%s\n' 'x-access-token' ;;
  *Password*) printf '%s\n' "$APP_GIT_TOKEN" ;;
  *) exit 1 ;;
esac
EOF
  fi

  runtime_log "syncing ref ${ref}"
  if [[ -n "$askpass" ]]; then
    if ! GIT_ASKPASS="$askpass" GIT_TERMINAL_PROMPT=0 \
      git -C "$code_dir" -c credential.helper= fetch --depth=1 --filter=blob:none origin "$ref"; then
      rm -f "$askpass"
      return 1
    fi
  else
    GIT_TERMINAL_PROMPT=0 git -C "$code_dir" fetch --depth=1 --filter=blob:none origin "$ref" || return
  fi
  rm -f "$askpass"
  git -C "$code_dir" checkout -q --detach --force FETCH_HEAD || return
  git -C "$code_dir" clean -ffdq -e /work/ || return

  # 链接镜像内烘焙的大文件目录（模型权重 / ffmpeg 二进制）
  local baked_app="${BAKED_APP_DIR:-/app}"
  local d
  for d in models ffmpeg; do
    if [[ -d "${baked_app}/backend/${d}" && ! -e "${code_dir}/backend/${d}" ]]; then
      ln -sfn "${baked_app}/backend/${d}" "${code_dir}/backend/${d}" || return
    fi
  done
}

resolve_runtime_app_path() {
  local code_dir="${APP_CODE_DIR:-/workspace/video-subtitle-remover}"
  local repository="${APP_GIT_URL:-https://github.com/liudunxu/video-subtitle-remover.git}"
  local ref="${APP_GIT_REF:-main}"
  local revision

  sync_git_checkout "$code_dir" "$repository" "$ref" || return
  [[ -f "$code_dir/api.py" ]] || { runtime_log "synced ref has no api.py"; return 1; }
  [[ -d "$code_dir/backend" ]] || { runtime_log "synced ref has no backend/"; return 1; }
  revision="$(git -C "$code_dir" rev-parse HEAD)" || return
  runtime_log "using source=git ref=${ref} commit=${revision}"
  printf '%s\n' "$code_dir"
}

if [[ "${BASH_SOURCE[0]}" == "$0" ]]; then
  case "${1:-}" in
    resolve-runtime-path) resolve_runtime_app_path ;;
    *) printf 'usage: %s resolve-runtime-path\n' "$0" >&2; exit 2 ;;
  esac
fi
