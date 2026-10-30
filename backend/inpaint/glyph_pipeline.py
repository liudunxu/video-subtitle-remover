"""Opt-in OCR -> per-frame glyph masks -> scene-local STTN/ProPainter pipeline.

Full-resolution masks live in temporary disk maps, not a whole-video RAM cache.
Only 80 resized frames are passed to a model at once. No sample-specific paths,
frame numbers, subtitle coordinates, or monkey patches belong here.
"""
import hashlib
import json
import shutil
import tempfile
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from backend.tools.glyph_masks import (
    same_layout, merge_line_boxes, glyph_mask, recover_vertical_border, find_shrunk_lines, box_iou,
    stabilize_masks, build_windows, overlap_weight, blend_prediction, resize_model_mask,
)


class NoGlyphsError(ValueError):
    pass


def _read_frames(path):
    cap = cv2.VideoCapture(str(path))
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()


def _disk_map(stack, path, shape, mode='w+'):
    result = np.memmap(path, mode=mode, dtype=np.uint8, shape=shape)
    stack.callback(result._mmap.close)
    return result


def _mask_hash(masks):
    digest = hashlib.sha256()
    for mask in masks:
        digest.update(mask.tobytes())
    return digest.hexdigest()


def _accepted_boxes(coords, detector, area, band, width):
    by0, by1 = band
    accepted = []
    for x0, x1, y0, y1 in coords:
        box = (int(x0), int(x1), int(y0+by0), int(y1+by0))
        if x1 <= x0 or y1 <= y0 or detector._box_inside_ratio(box, area) < detector.bbox_min_inside_ratio:
            continue
        x0, x1 = max(0, box[0]), min(width, box[1])
        y0, y1 = max(by0, box[2]), min(by1, box[3])
        if x1 > x0 and y1 > y0:
            accepted.append([x0, x1, y0-by0, y1-by0])
    return accepted


def _scan(video_path, detector, area, band, step, raw_path, progress, expected_count):
    """Detect alternate frames; inspect intervening frames at layout/cut changes."""
    by0, by1 = band
    boxes, cuts, cache = [], [0], {}
    previous = previous_thumb = None
    detected_count = 0

    def detect(index, frame):
        nonlocal detected_count
        if index not in cache:
            # Detector sees original pixels and applies its own max-side resize.
            coords = detector.detect_subtitle(frame[by0:by1])
            cache[index] = _accepted_boxes(coords, detector, area, band, frame.shape[1])
            detected_count += 1
        return cache[index]

    def emit(index, frame, candidates, output):
        lines = merge_line_boxes(candidates)
        boxes.append(lines)
        output.write(glyph_mask(frame[by0:by1], lines).tobytes())

    with raw_path.open('wb') as output:
        count = 0
        for i, frame in enumerate(_read_frames(video_path)):
            thumb = cv2.resize(frame, (96, 54)).astype(np.int16)
            cut = previous_thumb is not None and np.mean(np.abs(thumb-previous_thumb)) > 40
            if cut:
                cuts.append(i)
            sampled = i % step == 0 or cut
            if sampled:
                current_boxes = detect(i, frame)
            if previous is not None and len(boxes) == i-1:
                # Pending odd frame. Its adjacent even frames must share a shot
                # and layout; otherwise actually run OCR rather than bridging it.
                if not cut and sampled and same_layout(cache.get(i-2, []), current_boxes):
                    candidates = sorted(set(tuple(b) for b in cache.get(i-2, [])+current_boxes))
                else:
                    candidates = detect(i-1, previous)
                emit(i-1, previous, candidates, output)
            if sampled:
                emit(i, frame, current_boxes, output)
            previous, previous_thumb = frame, thumb
            # Only neighboring OCR samples are needed for interpolation.
            cache = {k: v for k, v in cache.items() if k >= i-1}
            count = i+1
            if i % 24 == 0:
                progress('ocr', min(24, 25*count/max(1, expected_count)), f'frame={count}')
                if shutil.disk_usage(raw_path.parent).free < frame[by0:by1].size * 4 + 256*1024**2:
                    raise OSError('Insufficient temporary disk space for glyph masks')
        if len(boxes) < count:
            emit(count-1, previous, detect(count-1, previous), output)
    if not count:
        raise ValueError('Video has no decodable frames')
    cuts.append(count)
    return boxes, list(zip(cuts, cuts[1:])), detected_count


def _retry_shrunk_lines(video_path, boxes, shots, raw, detector, area, band, progress):
    """Re-detect selected current frames; never copy adjacent subtitle boxes."""
    targets = find_shrunk_lines(boxes, shots) if detector is not None else {}
    retried, updated = [], []
    if not targets:
        return dict(frames=retried, updated_frames=updated)
    frames = _read_frames(video_path)
    try:
        for i, frame in enumerate(frames):
            if i not in targets:
                continue
            crop = frame[band[0]:band[1]]
            detected = merge_line_boxes(_accepted_boxes(
                detector.detect_subtitle(crop), detector, area, band, frame.shape[1]))
            additions = []
            for narrow, bounds in targets[i]:
                for line in detected:
                    # Actual current-frame detection must cover the old line
                    # and agree with the neighboring envelope. Ignore other text.
                    overlap = max(0, min(line[1], narrow[1])-max(line[0], narrow[0]))
                    if (line[1]-line[0] >= 1.1*(narrow[1]-narrow[0]) and
                            overlap >= .9*(narrow[1]-narrow[0]) and box_iou(line, bounds) >= .7):
                        additions.append(line)
            retried.append(i)
            if additions:
                boxes[i] = merge_line_boxes(boxes[i]+additions)
                raw[i] = glyph_mask(crop, boxes[i])
                updated.append(i)
            progress('ocr_retry', 25+4*len(retried)/len(targets), f'frame={i+1}, updated={len(updated)}')
            if len(retried) == len(targets):
                break
    finally:
        frames.close()
    if len(retried) != len(targets):
        raise RuntimeError('Incomplete decode during OCR retry')
    return dict(frames=retried, updated_frames=updated)


def _make_model(mode, options):
    import torch
    from backend.tools.model_config import ModelConfig
    from backend.tools.hardware_accelerator import HardwareAccelerator
    from backend.config import config

    accelerator = HardwareAccelerator.instance()
    accelerator.set_enabled(config.hardwareAcceleration.value)
    device = accelerator.device
    # The validated local run used CPU. MPS/DirectML support is not established
    # for this joint profile (MPS previously stalled on the larger experiment).
    if device.type != 'cuda':
        device = torch.device('cpu')
    mc = ModelConfig()
    if mode == 'sttn':
        from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
        model = STTNDetInpaint(device, mc.STTN_DET_MODEL_PATH)
        model.model_input_width, model.model_input_height = 432, 240
        model.neighbor_stride = options['sttn_neighbor_stride']
        model.ref_length = options['sttn_reference_length']
    else:
        from backend.inpaint.propainter_inpaint import PropainterInpaint
        model = PropainterInpaint(device, mc.PROPAINTER_MODEL_DIR, sub_video_length=80, use_fp16=False)
        model.mask_dilation = 0
        model.neighbor_length = 10
        model.ref_stride = options['sttn_reference_length']
    return model, str(device)


def _infer(video_path, masks, jobs, band_y, options, folder, progress):
    model, device = _make_model(options['mode'], options)
    # Frame seeking is not exact for every MP4 (including files with a nonzero
    # start timestamp). OCR/render decode sequentially, so inference must use
    # the same frame numbering. Cache only resized overlapping window frames.
    source_frames = _read_frames(video_path)
    next_frame = 0
    cached = {}
    records = []
    try:
        for number, job in enumerate(jobs):
            y0, y1 = job['crop_y']
            cached = {i: value for i, value in cached.items() if i >= job['start']}
            frames, small_masks = [], []
            for i in range(job['start'], job['stop']):
                if i not in cached:
                    if i < next_frame:
                        raise RuntimeError('Model windows must be ordered by frame')
                    try:
                        while next_frame <= i:
                            frame = next(source_frames)
                            next_frame += 1
                    except StopIteration as exc:
                        raise RuntimeError(f'Decode failed in inference at frame {i}') from exc
                    cached[i] = ((y0, y1), cv2.resize(frame[y0:y1], (432, 240)))
                crop, resized = cached[i]
                if crop != (y0, y1):
                    raise RuntimeError('Overlapping windows have different crops')
                frames.append(resized)
                mask = masks[i, y0-band_y:y1-band_y]
                small_masks.append(resize_model_mask(mask))
            frame_hash = _mask_hash(frames)
            mask_hash = _mask_hash(small_masks)
            # RAFT needs a pair for a one-frame shot. Duplicate within the shot;
            # never borrow a frame from across the cut.
            single_frame = len(frames) == 1 and options['mode'] == 'propainter'
            if single_frame:
                frames = frames*2
                small_masks = small_masks*2
            predictions = model.inpaint(frames, small_masks)
            if single_frame:
                predictions = predictions[:1]
            if len(predictions) != job['stop']-job['start'] or any(p is None for p in predictions):
                raise RuntimeError('Model returned incomplete predictions')
            predictions = np.stack(predictions)
            if not np.isfinite(predictions).all():
                raise RuntimeError('Model returned non-finite predictions')
            predictions = predictions.astype(np.uint8)
            if options['mode'] == 'sttn':
                predictions = predictions[:, :, :, ::-1].copy()  # STTN returns RGB.
            if predictions.shape[1:] != (240, 432, 3):
                raise RuntimeError('Unexpected model output geometry')
            np.save(folder/f'job{number:05}.npy', predictions)
            records.append(dict(**job, input_sha256=frame_hash, mask_sha256=mask_hash))
            progress(options['mode'], 45+40*(number+1)/len(jobs), f'chunk={number+1}/{len(jobs)}')
    finally:
        source_frames.close()
    return records, device


def _render(video_path, masks, jobs, band_y, folder, silent, fps, size, progress):
    from backend.tools.video_io import FFmpegVideoWriter
    by_frame = [[] for _ in range(len(masks))]
    for job in jobs:
        for i in range(job['start'], job['stop']):
            by_frame[i].append(job)
    predictions = {}
    writer = FFmpegVideoWriter(str(silent), fps, size, crf=12, preset='veryfast', strict=True)
    count = active_count = 0
    try:
        for i, source in enumerate(_read_frames(video_path)):
            if i >= len(masks):
                raise RuntimeError('Frame count changed between decoding passes')
            live = {j['id'] for j in by_frame[i]}
            for key in list(predictions):
                if key not in live:
                    del predictions[key]
            if np.any(masks[i]):
                if not by_frame[i]:
                    raise RuntimeError(f'No model window covers active frame {i}')
                numerator = np.zeros((240, 432, 3), np.float32)
                denominator = 0.
                crops = {tuple(j['crop_y']) for j in by_frame[i]}
                if len(crops) != 1:
                    raise RuntimeError('Overlapping windows have different crops')
                y0, y1 = next(iter(crops))
                for job in by_frame[i]:
                    key = job['id']
                    if key not in predictions:
                        predictions[key] = np.load(folder/f'job{key:05}.npy')
                    weight = overlap_weight(i, job)
                    numerator += predictions[key][i-job['start']]*weight
                    denominator += weight
                source[y0:y1] = blend_prediction(source[y0:y1], numerator/denominator,
                                                 masks[i, y0-band_y:y1-band_y])
                active_count += 1
            writer.write(source)
            count = i+1
            if i % 24 == 0:
                progress('stitch', 85+13*count/len(masks), f'frame={count}/{len(masks)}')
    finally:
        writer.release()
    if count != len(masks):
        raise RuntimeError('Incomplete rendered video')
    return active_count


def run_glyph_removal(video_path, area, options, detector, progress, mux_audio, retry_detector=None):
    """Produce H.264 and an audit sidecar, without failure-driven whole-area repaint."""
    video_path = Path(video_path)
    cap = cv2.VideoCapture(str(video_path))
    width, height = (int(cap.get(k)) for k in (cv2.CAP_PROP_FRAME_WIDTH, cv2.CAP_PROP_FRAME_HEIGHT))
    fps, expected = cap.get(cv2.CAP_PROP_FPS), int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()
    if min(width, height) <= 0 or not np.isfinite(fps) or fps <= 0:
        raise ValueError('Invalid video geometry or frame rate')
    area = area or (0, height, 0, width)
    margin = max(100, options['sttn_band_margin'])
    band = (max(0, area[0]-margin), min(height, area[1]+margin))
    # Four disk-backed mask planes plus output headroom; no full-resolution
    # source-frame cache. Temp files are removed on success and on failure.
    needed = max(1, expected)*(band[1]-band[0])*width*4 + 512*1024**2
    if shutil.disk_usage(video_path.parent).free < needed:
        raise OSError(f'Glyph pipeline needs about {needed/1024**3:.1f} GiB temporary free space')
    output = video_path.with_name(video_path.stem+'_glyph_'+options['mode']+'.mp4')
    with tempfile.TemporaryDirectory(prefix='glyph-', dir=video_path.parent) as tmp, ExitStack() as stack:
        folder = Path(tmp)
        boxes, shots, ocr_frames = _scan(video_path, detector, area, band,
                                        options['ocr_sample_step'] or 2, folder/'raw.bin', progress, expected)
        shape = (len(boxes), band[1]-band[0], width)
        raw = _disk_map(stack, folder/'raw.bin', shape, 'r+')
        retries = _retry_shrunk_lines(video_path, boxes, shots, raw, retry_detector, area, band, progress)
        stable = _disk_map(stack, folder/'stable.bin', shape)
        stabilize_masks(raw, shots, stable)
        progress('mask', 30, 'temporal_stabilization')
        revised = _disk_map(stack, folder/'revised.bin', shape)
        decoded = 0
        for i, frame in enumerate(_read_frames(video_path)):
            if i >= len(boxes):
                raise RuntimeError('Frame count changed during border recovery')
            revised[i] = raw[i] | recover_vertical_border(frame[band[0]:band[1]], boxes[i], stable[i])
            decoded += 1
            if i % 24 == 0:
                progress('mask', 30+10*decoded/len(boxes), 'recover_outline')
        if decoded != len(boxes):
            raise RuntimeError('Incomplete decode during border recovery')
        masks = _disk_map(stack, folder/'final.bin', shape)
        stabilize_masks(revised, shots, masks, base=stable, original_raw=raw)
        jobs = build_windows(masks, shots, band[0], options['sttn_band_margin'])
        if not jobs:
            raise NoGlyphsError('OCR/glyph detection produced no editable white outlined text; check sub_area or use mask_strategy=rectangle')
        progress('mask', 45, f'ready chunks={len(jobs)}')
        records, device = _infer(video_path, masks, jobs, band[0], options, folder, progress)
        silent = folder/'silent.mp4'
        active = _render(video_path, masks, jobs, band[0], folder, silent, fps, (width, height), progress)
        progress('finalize', 98, 'audio_mux')
        mux_audio(silent, video_path, output)
        audit = dict(profile='glyph_v3', mode=options['mode'], device=device, frames=len(boxes),
                     fps=fps, size=[width, height], area=list(area), band=list(band),
                     ocr_frames=ocr_frames, shots=shots, jobs=records, active_mask_frames=active,
                     ocr_max_side=options.get('glyph_ocr_max_side', 640),
                     ocr_retry_max_side=960 if retry_detector is not None else None,
                     ocr_retry_frames=retries['frames'], ocr_retry_updated_frames=retries['updated_frames'],
                     masks_sha256=_mask_hash(masks), model_input_size=[432, 240],
                     chunk_size=80, overlap=20, feather_sigma=2, glyph_dilation=5,
                     frame_decode='sequential', mask_coverage_guard='outlined_pixels_in_ocr_line',
                     model_mask_resize='area_nonzero_then_1px_dilation', feather_policy='opaque_core',
                     scene_metric='mean absolute BGR difference at 96x54 > 40',
                     precision='fp32', post_processing=False)
        output.with_suffix('.json').write_text(json.dumps(audit, indent=2))
    progress('finalize', 100, 'complete')
    return output
