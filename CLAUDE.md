# CLAUDE.md

Behavioral guidelines to reduce common LLM coding mistakes. Merge with project-specific instructions as needed.

**Tradeoff:** These guidelines bias toward caution over speed. For trivial tasks, use judgment.

## 1. Think Before Coding

**Don't assume. Don't hide confusion. Surface tradeoffs.**

Before implementing:
- State your assumptions explicitly. If uncertain, ask.
- If multiple interpretations exist, present them - don't pick silently.
- If a simpler approach exists, say so. Push back when warranted.
- If something is unclear, stop. Name what's confusing. Ask.

## 2. Simplicity First

**Minimum code that solves the problem. Nothing speculative.**

- No features beyond what was asked.
- No abstractions for single-use code.
- No "flexibility" or "configurability" that wasn't requested.
- No error handling for impossible scenarios.
- If you write 200 lines and it could be 50, rewrite it.

Ask yourself: "Would a senior engineer say this is overcomplicated?" If yes, simplify.

## 3. Surgical Changes

**Touch only what you must. Clean up only your own mess.**

When editing existing code:
- Don't "improve" adjacent code, comments, or formatting.
- Don't refactor things that aren't broken.
- Match existing style, even if you'd do it differently.
- If you notice unrelated dead code, mention it - don't delete it.

When your changes create orphans:
- Remove imports/variables/functions that YOUR changes made unused.
- Don't remove pre-existing dead code unless asked.

The test: Every changed line should trace directly to the user's request.

## 4. Goal-Driven Execution

**Define success criteria. Loop until verified.**

Transform tasks into verifiable goals:
- "Add validation" → "Write tests for invalid inputs, then make them pass"
- "Fix the bug" → "Write a test that reproduces it, then make it pass"
- "Refactor X" → "Ensure tests pass before and after"

For multi-step tasks, state a brief plan:

    1. [Step] → verify: [check]
    2. [Step] → verify: [check]
    3. [Step] → verify: [check]

Strong success criteria let you loop independently. Weak criteria ("make it work") require constant clarification.

---

**These guidelines are working if:** fewer unnecessary changes in diffs, fewer rewrites due to overcomplication, and clarifying questions come before implementation rather than after mistakes.

## 5. Vast.ai Instance Operations

Use this convention for this project's GPU API deployment:

- Stable Vast.ai label prefix: `video-subtitle-remover-api`
- Default instance label: `video-subtitle-remover-api-mvp`
- Service port inside container: `6006`
- Docker Hub image tag: `liudunxu/video-subtitle-remover:vast-gpu`
- Prefer resolving the Docker Hub tag to the current linux/amd64 digest before creating a Vast.ai instance, then pass `liudunxu/video-subtitle-remover@sha256:<digest>` to Vast.ai.
- Runtime type: `args`
- Default disk: `70`
- Required env:
  - `-p 6006:6006=1`
  - `HOST=0.0.0.0`
  - `PORT=6006`
  - `VSR_REQUIRE_GPU=1`
  - `VSR_API_WORK_DIR=/workspace/video_subtitle_remover_api`
  - `XDG_CACHE_HOME=/workspace/.cache`
  - `MPLCONFIGDIR=/workspace/.cache/matplotlib`
  - `PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK=True`

When the user says "关闭vast.ai实例" or equivalent:

- Query current Vast.ai instances with `VAST_API_KEY`.
- Delete every running/loading/stopped Vast.ai instance whose `label` starts with `video-subtitle-remover-api`.
- Do not delete instances with other labels.
- Report the deleted instance ids and confirm no matching instances remain.

When the user says "开启vast.ai实例" or equivalent:

- Query current Vast.ai instances first.
- If a matching `video-subtitle-remover-api*` instance is already running, reuse it and report its public URL.
- Otherwise search verified rentable on-demand offers for one RTX 3090 with at least 24GB VRAM, at least one direct port, and enough disk space.
- Prefer Asia-region offers that can pull from Docker Hub reliably; if no suitable Asia offer is available, fall back to Americas-region offers with good reliability and network.
- Avoid machines that recently failed Docker Hub pulls.
- Create exactly one instance using label `video-subtitle-remover-api-mvp`.
- After creation, poll until `actual_status=running` and port `6006/tcp` is assigned.
- Verify `GET /health` on the public URL and read Vast logs to confirm `GPU runtime check passed`.
