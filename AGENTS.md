# AGENTS.md

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
- Minimum GPU VRAM: `12000` MB (12 GB); RTX 3060/3060 Ti/4060 Ti/4070/3090 class GPUs are acceptable.
- CPU preference: modern x86-64 CPUs with AVX2 and FMA support. Avoid very old Intel Xeon E5 v2/v3 without AVX2, and avoid `Common KVM processor` / unknown generic CPUs. AMD EPYC 7xx2/7xx3/7K62/7402/7452/7502/7532/7551P or newer, and Intel Xeon E5-2673 v4 / E5-2686 v4 / E5-2690 v4 / E5-2695 v4 / E5-2696 v4 / Xeon W / Xeon Gold/Silver / Core i5/i7/i9 6th-gen or newer are preferred.
- Region preference: prefer East Asia, Southeast Asia, and West Asia (CN/HK/TW/SG/JP/KR/VN/TH/IN/TR/AE). Fall back to Americas (US/CA) only when no suitable Asia offer is available.
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
- Otherwise search verified rentable on-demand offers for one GPU with at least 12000 MB VRAM, at least one direct port, and enough disk space.
- Prefer offers with modern x86-64 CPUs that support AVX2 and FMA (e.g. AMD EPYC 7xx2/7xx3/7K62/7402/7452/7502/7532/7551P or newer; Intel Xeon E5-2673 v4 / E5-2686 v4 / E5-2690 v4 / E5-2695 v4 / E5-2696 v4 / Xeon W / Xeon Gold/Silver / Core i5/i7/i9 6th-gen or newer). Avoid very old CPUs without AVX2 and generic `Common KVM processor` labels.
- Prefer East Asia, Southeast Asia, and West Asia offers (CN/HK/TW/SG/JP/KR/VN/TH/IN/TR/AE) that can pull from Docker Hub reliably; fall back to Americas-region offers (US/CA) only when no suitable Asia offer is available.
- Avoid machines that recently failed Docker Hub pulls.
- Create exactly one instance using label `video-subtitle-remover-api-mvp`.
- After creation, poll until `actual_status=running` and port `6006/tcp` is assigned.
- Verify `GET /health` on the public URL and read Vast logs to confirm `GPU runtime check passed`.

## 6. Reusable Vast.ai Manager Script

A reusable Python script encapsulates the operations above:

- Path: `scripts/vast_ai_manager.py`
- Authentication: reads `VAST_API_KEY` from the environment (or `--api-key`)

Common commands:

```bash
# List matching instances
python scripts/vast_ai_manager.py status

# Start (or reuse) an instance with the latest Docker Hub digest
python scripts/vast_ai_manager.py start

# Stop and delete all matching instances
python scripts/vast_ai_manager.py stop

# Fetch logs for an instance
python scripts/vast_ai_manager.py logs <instance_id>
```

The `start` command resolves `liudunxu/video-subtitle-remover:vast-gpu` to its current linux/amd64 digest, searches for a GPU with at least 12GB VRAM (preferring modern AVX2/FMA CPUs and Asia-region offers), creates the instance, polls until it is running, checks `/health`, verifies "GPU runtime check passed" in the logs, and appends the successful instance to `scripts/vast_known_good.json`. Pass `--no-save-known-good` to skip persistence.

## 7. Known-good Vast.ai Instances

Keep a ledger of Vast.ai instances that started successfully and passed the GPU runtime check so future `start` commands can prefer proven machines.

Storage: `scripts/vast_known_good.json`

Schema for each entry:

| Field | Description |
|---|---|
| `instance_id` | Vast.ai contract ID from the successful run |
| `machine_id` | Stable Vast.ai machine ID; this is the matching key |
| `gpu_name` | GPU model, e.g. `RTX 3090` |
| `cpu_name` | CPU model reported by Vast.ai |
| `country_code` | Two-letter country code, e.g. `SG` |
| `region` | `asia`, `americas`, or `other` |
| `image` | Exact Docker image digest used |
| `started_at` | ISO-8601 UTC timestamp |
| `notes` | e.g. `GPU runtime check passed` |

Operation rules:

- Only append an entry after `python scripts/vast_ai_manager.py start` reports success: `actual_status=running`, public URL assigned, `GET /health` returns 200, and logs contain `GPU runtime check passed`.
- The manager loads `scripts/vast_known_good.json` automatically and ranks any offer whose `machine_id` matches a known-good entry above other offers.
- If no known-good machine is currently rentable, fall back to the normal scoring in section 5.
- To disable automatic persistence, pass `--no-save-known-good`.
- When adding entries manually, prefer `machine_id` over `offer_id` because the same physical machine can appear under multiple offers.
