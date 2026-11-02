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

- Stable Vast.ai label prefix: `video-subtitle-remover-api-`
- Default instance label: `video-subtitle-remover-api-mvp`
- Service port inside container: `6006`
- Docker Hub image tag: `liudunxu/video-subtitle-remover:vast-gpu`
- Prefer resolving the Docker Hub tag to the current linux/amd64 digest before creating a Vast.ai instance, then pass `liudunxu/video-subtitle-remover@sha256:<digest>` to Vast.ai.
- Runtime type: `args`
- **Startup code sync (OmniVoice-style)**: at container start the entrypoint shallow-fetches the latest code (blobless + sparse, ~2MB; `backend/models` and `backend/ffmpeg` are excluded and symlinked from the baked `/app` copies) into `/workspace/video-subtitle-remover` and runs from there — code changes only need a git push + instance restart, NOT an image rebuild. Rebuild is still required for dependency/env/model-weight changes. Optional env: `APP_GIT_REF` (default `main`), `APP_GIT_URL`, `APP_CODE_DIR`, `APP_GIT_TOKEN` (private repo). If the sync fails (e.g. GitHub unreachable), the entrypoint falls back to the baked `/app` code. `/health` and `/api/status` report `runtime_commit_id` / `runtime_source` (`git` vs `image`) and `image_commit_id` so callers can verify which code is actually live.
- Default disk: `70`
- Minimum GPU VRAM: `12000` MB (12 GB); RTX 3060/3060 Ti/4060 Ti/4070/3090 class GPUs are acceptable.
- CPU preference: modern x86-64 CPUs with AVX2 and FMA support. Avoid very old Intel Xeon E5 v2/v3 without AVX2, and avoid `Common KVM processor` / unknown generic CPUs. AMD EPYC 7xx2/7xx3/7K62/7402/7452/7502/7532/7551P or newer, and Intel Xeon E5-2673 v4 / E5-2686 v4 / E5-2690 v4 / E5-2695 v4 / E5-2696 v4 / Xeon W / Xeon Gold/Silver / Core i5/i7/i9 6th-gen or newer are preferred.
- Region preference (aligned with OmniVoice): **Southeast Asia first** (SG/TH/VN/MY/ID/PH/KH/MM/LA/BN/TL), then **East Asia** (KR/TW/JP/HK/MO), then other Asia (IN/TR/AE/SA/IL/QA/KW/BH/OM). Fall back to Americas (US/CA) only when no suitable Asia offer is available. **CN offers are blacklisted and never selected** (aligned with OmniVoice's never-CN rule). (A previous Oman pick downloaded at ~6KB/s — West Asia is rank 2, not rank 0.)
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
- Delete every running/loading/stopped Vast.ai instance whose `label` starts with `video-subtitle-remover-api-`.
- Do not delete instances with other labels.
- Report the deleted instance ids and confirm no matching instances remain.

When the user says "开启vast.ai实例" or equivalent:

- Query current Vast.ai instances first.
- If a matching `video-subtitle-remover-api-*` instance is already running, reuse it and report its public URL.
- Otherwise search verified rentable on-demand offers for a **single-GPU** machine (`num_gpus=1` — never rent multi-GPU machines) with at least 12000 MB VRAM, at least one direct port, and enough disk space, priced **within the hourly cap** (default `$0.35`/hr, `--max-price`; `0` disables it). GPU model preference order (aligned with OmniVoice's try-next-model-down pattern; fall back to the next model only when the current one has no offer inside the cap): **RTX 3090 Ti → RTX 3090 → RTX 4090**. Never rent above the cap silently: if no model has an offer inside it, report the failure and let the operator raise `--max-price`.
- Prefer offers with modern x86-64 CPUs that support AVX2 and FMA (e.g. AMD EPYC 7xx2/7xx3/7K62/7402/7452/7502/7532/7551P or newer; Intel Xeon E5-2673 v4 / E5-2686 v4 / E5-2690 v4 / E5-2695 v4 / E5-2696 v4 / Xeon W / Xeon Gold/Silver / Core i5/i7/i9 6th-gen or newer). Avoid very old CPUs without AVX2 and generic `Common KVM processor` labels. Matching runs on a normalized `cpu_name` because Vast embeds trademark glyphs (`Xeon® E5-2680 v4 `, `Core™ i7-9700F`) that made unnormalized substring matching miss ~46% of live offers; the whitelist covers the whole Xeon E5 **v4** family, Xeon Gold/Silver/Platinum/W, Xeon 6xxx, AMD Ryzen/EPYC 7xxx–9xxx and Core i5/i7/i9/Ultra. E5 v2/v3, `AMD FX`, engineering samples, bare/unknown CPU strings and `Common KVM processor` stay penalized.
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

The `start` command resolves `liudunxu/video-subtitle-remover:vast-gpu` to its current linux/amd64 digest, searches for a GPU with at least 12GB VRAM priced within `--max-price` (default `$0.35`/hr; `0` disables the cap), ranks offers by **known-good machine → region (SEA > East Asia > other Asia > Americas) → CPU tier → lowest price → higher reliability → fewer cores**, creates the instance, polls until it is running, checks `/health`, verifies "GPU runtime check passed" in the logs, and appends the successful instance to `scripts/vast_known_good.json`. Pass `--no-save-known-good` to skip persistence. Price is ranked **above** reliability on purpose: reliability is a near-continuous float, so putting it first means price never actually decides (it once picked a 4% more expensive machine over an identical-region, identical-CPU-tier one). If nothing fits the cap, `start` exits with an error rather than renting a pricier machine.

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

## 8. VSR API / 算法层行为说明（2026-10-14 更新）

These notes cover behavior that changed in the OCR-sensitivity / vertical-video quality pass. Keep them in sync when touching the corresponding code.

- **Glyph v5 retry ordering / line merging (2026-11-02)**: Preserve separate OCR boxes when their height ratio exceeds 2, so tall background detections cannot turn a valid subtitle line into an oversized anomaly. Run short-text recovery before automatic narrow-row retries; only initial overlap-local frames are excluded at this stage. Recovered short captions must retain their masks and avoid empty narrow retry replacement. Keep the strict overlap-local outline thresholds unchanged. Regression tests cover the shared STTN/ProPainter route, true blank frames, and tall side boxes alongside normal captions. No new profile or switch.

- **Glyph v5 automatic narrow OCR (2026-10-31)**: The shared STTN/ProPainter mask stage derives one area-clipped row per shot from at least two aligned horizontal detections verified against current outlined-white pixels. Retry empty/oversized results only; preserve plausible normal/short captions. Abstain without anchors or with multiple supported rows. Current-frame local OCR and outlined pixels are mandatory; never copy neighbor masks. A confirmed detection renews the row for one second, with no propagation across shots. Add retry frames to the existing constrained-mask path before gap recovery/stabilization, and keep narrow bounds during isolated-gap recovery. Audit `narrow_ocr` groups/anchors/regions/frames/confirmed_frames/empty_frames/ocr_calls, using band-relative yxyx coordinates. No new option/profile or hardcoded sample coordinates. At most two sequential decode passes plus target OCR; no full-frame cache or new model invocation. Test both model routes, moved rows, manual areas, short/multiple rows, missing anchors, false OCR, expiry, cuts and truncated decodes. The E01 ten-second STTN experiment does not establish ProPainter visual quality or general recovery.

- **Glyph v5 short text / photometric context (2026-10-30)**: Keep the uncommitted `glyph_v5` profile; no new switch. Repeated aligned short OCR boxes within the same shot and half-second radius (max 15 frames) propose area-clipped local crops. Re-detect each current frame; if no valid short box exists, retry that small crop once at 2x and map boxes back to source coordinates. Require strong-white, dark-outlined body components before adding short glyphs and detached upper marks. Exclude ordinary long captions and overlap-local recovery frames; do not guess entirely undetected captions or hardcode sample coordinates. Record `short_ocr` retry/update indices and band-relative yxyx regions. ProPainter independently splits existing windows when unmasked median luma changes by >6 with >=75% coherent same-direction changes >3 (>=20% unmasked context). Duplicate singleton frames for RAFT and retain one prediction; pass empty segments through. Preserve job spans, output count, overlap rendering and STTN behavior. Record `jobs[].photometric_cuts` as zero-based video indices. Keep regression tests for blank frames, pale/unoutlined background, changed captions, cuts, local motion, coherent flashes and complete segment outputs; no full-video quality claim.

- **Glyph v5 overlap recovery (2026-10-30)**: No new option. `glyph_context.py` handles tall OCR boxes crossing `sub_area` by actual area-cropped OCR; empty accepted wide-band detection probes only areas <=15% of frame height. An anomaly or successful empty probe activates per-frame local OCR until the shot ends. Reject clipped/oversized/non-horizontal boxes; use dark-outline-constrained white components clipped to area, bypass ordinary border recovery and shrink retries for these frames. Isolated local OCR gaps require agreeing immediate same-shot neighbors AND current-image outlined pixels with IoU>=0.75 to both; never copy their masks. STTN may add one verified empty-local-OCR reference within its existing same-shot model window, using resized source context similarity, edge retention and subtitle-pattern disappearance. Reject changing mask layouts; do not cross shots/windows, load full-resolution frame batches or add references to ProPainter. `STTNDetInpaint.inpaint(..., extra_reference_indices=())` is per-call, deduplicated and default-neutral for legacy callers. Audit profile/capabilities are `glyph_v5`; record `local_ocr` trigger/processed/pixel-recovered frames and per-job reference selection/rejections. Keep tests for false-empty OCR, actual empty frames, scene cuts, subtitle changes and movement. E01 short-clip validation is not evidence of general quality across fonts/scenes.

- **Glyph output encoding (2026-10-30)**: Both STTN and ProPainter render final composited frames directly with `FFmpegVideoWriter(crf=18, preset='fast', strict=True)` (previously 12/veryfast). OCR/inference still use decoded source frames at the existing model sizes; audio muxing stream-copies video. Preserve this single-encode path rather than introducing a high-bitrate intermediate followed by another lossy encode. Legacy rectangle/tail encoding settings are separate.

- **Glyph v4 area / clipped symbols (2026-10-30)**: Only the remove endpoint's auto-detected glyph area expands to full frame width; preserve its vertical bounds, caller-supplied areas, legacy rectangle routing and the standalone detect endpoint. `glyph_mask` additionally recovers small outlined components crossing the old search window's upper/lower edge, aligned with existing glyph columns and with strong white cores (V>=230, S<=55, >=30% of component and >=3 pixels). Keep the pale-fabric rejection gate; do not replace whole lines with rectangles. Both models and OCR retries share this path. Audit profile is `glyph_v5`, with `clipped_symbol_recovery=aligned_strong_white_components`. Validated on three 50-frame My Poet clips near 31/90/119 seconds. The dubbing frontend allows users to adjust the removal area; caller-supplied selections retain their bounds. Glyph remains opt-in.

- **Opt-in glyph pipeline (2026-10-30, glyph_v5)**: `mask_strategy=glyph` dispatches to `backend/inpaint/glyph_pipeline.py` for STTN/ProPainter; default `rectangle` retains the legacy path. Reuse `backend/tools/glyph_masks.py` for white outlined text extraction, clipped-border recovery, scene-local ±2/IoU≥0.75 stabilization and 80/20 windows. Recover disconnected accents and grow masks locally around uncovered outlined bright pixels inside OCR lines; do not replace entire lines with rectangles. Inference must decode sequentially like OCR/render (MP4 random seeks can land one frame early); cache resized overlap frames only. Resize masks conservatively (INTER_AREA >0 plus 1 model-pixel dilation); composite with opaque mask cores and external sigma-2 feather. Both models use 432×240, ProPainter FP32 and zero internal dilation; per-frame masks are a supported `read_mask` input, not a monkey patch. Empty glyphs return 422; unsupported modes, extra cleanup/blur, whole-area fallback, larger inputs and explicit legacy mask/chunk overrides return 400. The multiline OCR preset does not enable tails on this path. No experiment-specific coordinates or frame numbers are allowed. `sub_area` is the OCR acceptance window; default OCR step is 2 (only 0/1/2 allowed). `glyph_ocr_max_side` defaults to 640 (range 640–1280); 960 recovered missed short text in a local case, without full-video validation. Below base OCR size 960, use lazy 960 retries only for lines at least 15% narrower than agreeing wider detections on both sides within 2 frames in the same shot. Only actual current-frame detections may expand the line; never copy neighbor boxes into masks. Audit retry/update frame indices. Empty/persistent misses and clip edges are not recovered. Sampled identical layouts may miss short intermediate subtitles; step 1 avoids that interpolation. The response exposes `processing` and an audit JSON with masks/input hashes. Temp disk maps are cleaned by ExitStack/TemporaryDirectory; strict FFmpeg writer errors fail the job. Source A/V start offsets are restored during muxing; encoding remains CFR. CUDA when enabled, otherwise CPU (MPS/DirectML unvalidated). See README for scope and options.

- **OCR presets actually reach the detector.** `_OCR_PRESETS` values (`det_db_thresh` / `det_db_box_thresh` / `det_limit_side_len`) are passed through `SubtitleDetect._kwargs` and mapped to the parameter names the installed PaddleOCR really accepts: 2.x uses the same `det_db_*` names, 3.x/PaddleX uses `thresh` / `box_thresh` / `limit_side_len` (resolved via `inspect.signature` on `TextDetection.__init__`). Construction logs the applied values; if the constructor rejects them, detection falls back to model defaults plus predict-time score filtering (`dt_scores >= det_db_box_thresh`).
- **`_apply_config_options` resolves the config singleton** (`backend.config.config`). `backend.config` is a plain-Python `Config` class (the qfluentwidgets `QConfig` dependency was removed together with the GUI). ConfigItems are attributes of the instance, not the module, so overrides must go through `getattr(config, "config", config)` and `cfg.set(item, value, save=False)` — the old module-level lookup silently no-op'd every override, so API `mode=sttn` used to run the default `STTN_AUTO` (whole-area repaint) for every request. It now really switches to `STTN_DET` (detection-driven masks) and back.
- **New `/api/remove-subtitle` options**: `bbox_min_inside_ratio` (0–1, default 1.0 = box must be fully inside `sub_area`; lower values keep boxes straddling the area edge), `ocr_sample_step` (0 = fps-adaptive, 1 = every frame, max 30), `empty_detection_full_inpaint` (default false = empty STTN OCR result returns HTTP 422; true opts into whole-area STTN_AUTO). The response includes `fallback_triggered`. `/api/detect-subtitle-area` also accepts `bbox_min_inside_ratio`.
- **Measured quality defaults (2026-10-14, Vast 3090 Ti)**: API mask expansion now defaults to X=12/Y=16, `sttn_band_margin=80`, `post_lama_refine=false`, and `empty_detection_full_inpaint=false`; CLI/model defaults stay unchanged. Ten runs on two 6-second 1080p/480p samples with 432×240 STTN input showed less smearing than 44/44 masks with refinement, but 1080p knit textures remained soft. Expanding auto-area padding to 160/64 produced byte-identical outputs to the baseline combination on both samples; area is not the model context margin. LaMa gave better low-res still-frame texture at roughly twice the cost; long videos and temporal flicker were not validated. `split_h` remains clamped to ≥ model input height so low-res crops are not vertically upscaled into the model.
- **Portrait follow-up (2026-10-14, Vast RTX 3090, `e0a2f36`)**: on one 6-second 1080×1920 sample, 432×240/margin 80 took 63.2s; 864×240/margin 80 took 70.6s (~12% more) with slightly better knit structure but remaining grid artifacts. Recommend explicit `sttn_input_width=864`, `sttn_input_height=240`, margin 80 for this quality/performance tradeoff, not as a global default (omitted sizes still use env/default 432×240). 864×480/margin 80 took 112.8s with limited visual gain; margin 160 took 102.7s with no clear advantage (single-run timing noise). LaMa took 65.1s but introduced visible seams on this portrait sample. New outputs preserved resolution, 180 frames and audio, with duration differences under 50ms; logs confirmed model sizes. Assessment used sampled frames, not clean-background ground truth or temporal-flicker metrics; do not infer that larger margins or LaMa always improve high-resolution portrait video.
- **STTN_DET crop is mask-driven** (`backend/inpaint/sttn_det_inpaint.py`): `split_h` = tallest mask band + 2×`VSR_STTN_BAND_MARGIN` (default 40 px, clamp [96, H]); empty masks fall back to the old proportional split. Model output is upscaled back and blended **only inside the Gaussian-feathered mask** (sigma 2.0) — the rest of the band keeps the original pixels instead of the old full-band 432×240 round-trip. Model input size is tunable via `VSR_STTN_INPUT_WIDTH` / `VSR_STTN_INPUT_HEIGHT` **or per-request via the `sttn_input_width` / `sttn_input_height` options (request wins over env, injected as config dynamic attrs)** but constrained by the attention patch grid: width must be a multiple of 432, height a multiple of 240, capped at 864×480; invalid values fall back to 432×240 with a warning. Larger inputs can reduce scaling loss but do not guarantee sharper fills; see the measured 864×240 versus 864×480 tradeoff above. `sttn_band_margin` is likewise a per-request option (request wins over env, injected as the `STTN_BAND_MARGIN` config dynamic attr; API default 80, range 0–400) — sttn_det_inpaint reads these attrs from the `backend.config` **module**, not the singleton.
- **Tail chain encodes once per pass and preserves source duration/audio.** `_run_text_trace_refine` / `_run_residual_cleanup` / `_run_post_verify_blur` write intermediates with `FFmpegVideoWriter` (libx264, crf 12, preset veryfast, yuv420p) and mux audio with `-c:v copy`; a pass that modified zero frames returns the upstream file unchanged instead of adding a wasted lossy generation. Audio muxing probes the source codec (AAC stream-copied, anything else transcoded to AAC 320k) and keeps the FULL source audio length — no `-shortest`, so the output duration matches the source. `_pad_video_duration` pads by concat-ing a freeze-frame clip with `-c copy` (main body not re-encoded; tpad re-encode only as fallback); `_ensure_h264` only remuxes (+faststart) when the stream is already H.264. Any ffmpeg failure raises instead of silently returning a video without audio.
- **`post_verify_blur` detector uses `sub_area=None` on purpose**: it feeds the detector area-cropped frames, whose crop-local coordinates must not be compared against the full-frame area (doing so used to drop every box when an area was set). Blur regions are still clamped to the area downstream.
- **API hygiene (2026-10)**: multipart uploads are parsed with `python-multipart` (stdlib `cgi` was removed in Python 3.13); file fields stream to disk in 1MB chunks. `blur_kernel` / `blur_passes` / `blur_pad` / `blur_feather` / `blur_temporal_window` are real `/api/remove-subtitle` options now (they used to fall outside the options whitelist). On detection failure or empty detection, `blur_cover` returns HTTP 422 unless `blur_cover_allow_passthrough=true` (the old copy-source-through behavior). Caller-supplied job ids (`X-Job-Id` / `progress_job_id`) must match `[A-Za-z0-9_-]{1,64}` — `/download/{job_id}/...` accepts the same charset (it required 32-hex before, which made custom job ids undownloadable). Job dirs under `VSR_API_WORK_DIR/jobs` are swept after `VSR_API_JOB_TTL_SECONDS` (default 6h) once they leave the active registry. 500 responses no longer include the server traceback (still printed server-side).
- **API contract follows OmniVoice conventions (2026-10)**: all JSON responses carry the `{"ok": true|false, ...}` envelope (`ok` is injected centrally in `_send_json`), `api_schema_version` (`vsr-removal-v1`) is returned by `/health`, `/api/capabilities`, `/api/status` and the submit/detect responses, and `X-Request-ID` is accepted and echoed as a response header. `/health` (alias `/api/health`) returns `{"ok": true, "service": "video_subtitle_remover_api", ...}`; `/api/capabilities` publishes modes / ocr_presets / feature flags / deprecated-alias usage; `/api/status` reports uptime / work_dir / disk / active_jobs. Route aliases (`/api/remove`, `/remove`, `/api/detect`, `/detect-subtitle-area`) send `Deprecation: true` + `Link` headers. Client disconnects mid-upload map to 499 (not 500/502). Deliberate deviation: concurrent submissions still queue on `PROCESS_LOCK` — OmniVoice returns 503+Retry-After when busy, VSR keeps queueing to avoid changing caller semantics.
- **Default chain cost/residue balance (2026-10)**: `find_subtitle_frame_no` is memoized per `SubtitleDetect` instance — the default sttn chain calls it twice (detector peek + main inpaint pass), and the second call now reuses the cached list instead of re-running a full sampled-OCR pass. `post_lama_refine` defaults to **false** to reduce texture flattening; explicit true or `mode=sttn_then_lama` enables it. The `multiline` preset still enables refinement/cleanup on 3+ line frames unless explicitly disabled. **Refine seeds are computed on the STTN output, not the original frame**: only areas where light/yellow residue survived STTN get re-inpainted with telea. (The earlier version seeded on the original frame, where glyphs are always bright — and since STTN changes every glyph pixel, the diff gate was vacuous, so the whole glyph region got textureless telea refills on every subtitle frame: that was the main source of the frosted-glass look.) `auto_residual_cleanup` / `post_verify_blur` stay opt-in (higher false-positive / blur-visibility cost).
- **`auto_detect_area` defaults to true**: when `/api/remove-subtitle` is called without `sub_area`, the server first runs area detection and narrows processing to the detected subtitle band (response `area` shows it, `auto_detected_area: true`) — less full-frame false detection and cheaper tail passes. The removal path widens detection `padding_x` to ≥96 px (the detect endpoint keeps 16 px) because the default 16 px used to clip subtitle edge fragments outside the area; wide areas are nearly free in STTN_DET since masks stay box-driven. The removal path is also **accuracy-first, not identical to `/api/detect-subtitle-area` defaults**: it forces `multi_group_union=true`, `crop_to_band=false` (unless the caller set it explicitly) and `sample_count=48` (unless explicit) — slower by a few seconds but catches side vertical credits and other out-of-band text that the fast detect defaults would cut. Detection failure or an empty result falls back to full-frame processing (the old behavior); pass `auto_detect_area=false` to force full-frame.
- **Caller-supplied `sub_area` is used as-is; `sub_area_margin` opts into expansion.** Every area padding step (`raw_area_buffer_y=60`, `_padded_area(16,16)`, `_DETECT_RESULT_V_PAD=50`, forced `padding_x≥96`) lives in the auto-detect branch only — a caller-passed area used to skip all of them and was not even clamped to the frame. It now goes through `_expand_sub_area`, which applies `sub_area_margin` (default **0**, range 0–400) and clamps to the ffprobe'd frame size; an area entirely outside the frame returns HTTP 400 instead of the old 422 (DET, empty detection) or silent no-op (STTN_AUTO, mask clips to nothing so the source frames come back untouched). The zero default is deliberate: in STTN_DET the area is only the OCR acceptance window — masks are box-driven (X=12/Y=16 + dilation) and the crop band is mask-driven — so a tight area does not clip glyphs, while silently widening it would let adjacent watermarks/channel logos be erased. Margin only matters on the paths where the rectangle *is* the mask: `sttn_skip_detection=true` and the empty-detection STTN_AUTO fallback (plus `post_verify_blur`, which clamps blur regions inside `sub_area`). For a tight area in DET mode, callers can explicitly widen the selection or lower `bbox_min_inside_ratio` (default 1.0 drops boxes not fully inside). Glyph auto areas now use full width as described above; this does not change manual areas.
- **`/api/detect-subtitle-area` `multi_group_union` option** (default false): skips the y-band filter, groups all boxes and unions every group scoring ≥60% of the top group — for videos with bottom horizontal subtitles plus side vertical credits coexisting, instead of picking one band. Position priors still score groups, but strong evidence outside the preferred band is no longer discarded.
