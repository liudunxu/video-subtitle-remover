"""Opt-in OCR -> per-frame glyph masks -> scene-local STTN/ProPainter pipeline.

Full-resolution masks live in temporary disk maps, not a whole-video RAM cache.
Only 80 resized frames are passed to a model at once. No sample-specific paths,
frame numbers, subtitle coordinates, or monkey patches belong here.
"""
import hashlib
import json
import shutil
import tempfile
from bisect import bisect_left
from contextlib import ExitStack
from pathlib import Path

import cv2
import numpy as np

from backend.tools.glyph_masks import (
    same_layout, merge_line_boxes, glyph_mask, recover_vertical_border, find_shrunk_lines, box_iou,
    stabilize_masks, build_windows, overlap_weight, blend_prediction, resize_model_mask,
)
from backend.tools.glyph_context import (
    needs_local_ocr, local_ocr_boxes, outlined_local_mask, select_reference,
    short_text_regions, short_text_mask, photometric_boundaries,
    subtitle_line_shape, narrow_text_regions,
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


def _scan(video_path, detector, area, band, step, raw_path, progress, expected_count, local_audit=None):
    """Detect alternate frames; inspect intervening frames at layout/cut changes."""
    by0, by1 = band
    boxes, cuts, cache = [], [0], {}
    previous = previous_thumb = None
    detected_count = 0
    local_frames, trigger_frames = set(), []
    local_area = (area[0]-by0, area[1]-by0, area[2], area[3])
    overlap_mode = False

    def detect(index, frame, force_local=False):
        nonlocal detected_count
        if index not in cache:
            # Detector sees original pixels and applies its own max-side resize.
            crop = frame[by0:by1]
            coords = [] if force_local else detector.detect_subtitle(crop)
            detected_count += int(not force_local)
            anomaly = needs_local_ocr(coords, local_area)
            accepted = _accepted_boxes(coords, detector, area, band, frame.shape[1])
            # Missing wide-band OCR can also hide a subtitle on the first frame.
            # Probe only a narrow selection, not the full image or a broad area.
            empty_probe = not force_local and not accepted and area[1]-area[0] <= .15*frame.shape[0]
            if force_local or anomaly or empty_probe:
                local = local_ocr_boxes(crop, detector, local_area)
                detected_count += 1
                use_local = force_local or anomaly or bool(local)
                cache[index] = local if use_local else accepted
                if use_local:
                    local_frames.add(index)
                if anomaly or (empty_probe and local):
                    trigger_frames.append(index)
            else:
                cache[index] = accepted
        return cache[index]

    def emit(index, frame, candidates, output):
        lines = merge_line_boxes(candidates)
        boxes.append(lines)
        crop = frame[by0:by1]
        mask = outlined_local_mask(crop, lines, local_area) if index in local_frames else glyph_mask(crop, lines)
        output.write(mask.tobytes())

    with raw_path.open('wb') as output:
        count = 0
        for i, frame in enumerate(_read_frames(video_path)):
            thumb = cv2.resize(frame, (96, 54)).astype(np.int16)
            cut = previous_thumb is not None and np.mean(np.abs(thumb-previous_thumb)) > 40
            if cut:
                cuts.append(i)
            previous_overlap = overlap_mode
            if cut:
                overlap_mode = False
            sampled = i % step == 0 or cut or overlap_mode
            if sampled:
                current_boxes = detect(i, frame, overlap_mode)
                overlap_mode = i in local_frames
            if previous is not None and len(boxes) == i-1:
                # Pending odd frame. Its adjacent even frames must share a shot
                # and layout; otherwise actually run OCR rather than bridging it.
                if not cut and not overlap_mode and not previous_overlap and sampled and same_layout(cache.get(i-2, []), current_boxes):
                    candidates = sorted(set(tuple(b) for b in cache.get(i-2, [])+current_boxes))
                else:
                    candidates = detect(i-1, previous, previous_overlap or (overlap_mode and not cut))
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
    if local_audit is not None:
        local_audit.update(frames=sorted(local_frames), trigger_frames=trigger_frames)
    return boxes, list(zip(cuts, cuts[1:])), detected_count


def _retry_narrow_lines(video_path, boxes, shots, raw, detector, area, band, height, fps, progress):
    """Retry anomalous/empty frames in a row inferred from current-video pixels.

    Existing plausible lines (including short captions) keep their masks. A
    current-frame confirmation renews the row for one second; it never crosses
    a shot boundary. No full-resolution frames are cached between passes.
    """
    local_area = (area[0]-band[0], area[1]-band[0], area[2], area[3])
    audit = dict(groups=[], frames=[], confirmed_frames=[], empty_frames=[], regions={},
                 ocr_calls=0, region_coordinate_space='ocr_band_yxyx')
    candidates = {i: [b for b in lines if subtitle_line_shape(b, height)
                     and area[2] <= b[0] < b[1] <= area[3]
                     and local_area[0] <= b[2] < b[3] <= local_area[1]]
                  for i, lines in enumerate(boxes)}
    normal = {i for i, lines in enumerate(boxes)
              if any(subtitle_line_shape(b, height, min_aspect=.3) for b in lines)}
    # Avoid even decoding again if there is no supported row or no anomaly.
    eligible = [(a, b) for a, b in shots
                if sum(bool(candidates[i]) for i in range(a, b)) >= 2
                and any(i not in normal for i in range(a, b))]
    candidates = {i: candidates[i] for a, b in eligible for i in range(a, b) if candidates[i]}
    if not candidates:
        return audit
    anchors = {}
    last_anchor = max(candidates)
    last_decoded = -1
    frames = _read_frames(video_path)
    try:
        for i, frame in enumerate(frames):
            last_decoded = i
            if i in candidates:
                crop = frame[band[0]:band[1]]
                anchors[i] = [b for b in candidates[i]
                              if np.count_nonzero(outlined_local_mask(crop, [b], local_area)) >= 2*(b[3]-b[2])]
            if i % 24 == 0:
                progress('ocr_narrow', 25, f'verify_anchor_frame={i+1}')
            if i >= last_anchor:
                break
    finally:
        frames.close()
    if last_decoded < last_anchor:
        raise RuntimeError('Incomplete decode during narrow OCR anchor verification')
    audit['groups'] = narrow_text_regions(anchors, shots, local_area)
    if not audit['groups']:
        return audit
    groups = {g['shot']: g for g in audit['groups']}
    ttl = max(1, round(fps))
    shot = 0
    last_confirmed = None
    last_target = max(g['bounds'][1]-1 for g in groups.values())
    last_decoded = -1
    frames = _read_frames(video_path)
    try:
        for i, frame in enumerate(frames):
            last_decoded = i
            while i >= shots[shot][1]:
                shot += 1
                last_confirmed = None
            group = groups.get(shot)
            if group is not None:
                anchor_frames = group['anchor_frames']
                pos = bisect_left(anchor_frames, i)
                nearby = anchor_frames[max(0, pos-1):pos+1]
                nearest = min(abs(i-j) for j in nearby)
                if nearest == 0:
                    last_confirmed = i
                if i not in normal and (nearest <= ttl or (
                        last_confirmed is not None and i-last_confirmed <= ttl)):
                    region = group['region']
                    crop = frame[band[0]:band[1]]
                    found = merge_line_boxes(local_ocr_boxes(crop, detector, region))
                    mask = outlined_local_mask(crop, found, region)
                    boxes[i] = found if mask.any() else []
                    raw[i] = mask
                    audit['frames'].append(i)
                    audit['regions'][i] = region
                    audit['ocr_calls'] += 1
                    if mask.any():
                        last_confirmed = i
                        audit['confirmed_frames'].append(i)
                    else:
                        audit['empty_frames'].append(i)
            if i % 24 == 0:
                progress('ocr_narrow', 25, f'frame={i+1}, confirmed={len(audit["confirmed_frames"])}')
            if i >= last_target:
                break
    finally:
        frames.close()
    if last_decoded < last_target:
        raise RuntimeError('Incomplete decode during narrow OCR retry')
    return audit


def _recover_local_gaps(video_path, boxes, shots, raw, area, band, local_frames, regions=None):
    """Repair isolated OCR blanks only when current outlined pixels agree with both sides.

    Neighbor boxes provide a search region, never replacement mask pixels. This
    prevents a real blank frame (potential clean reference) inheriting subtitles.
    """
    targets = {}
    for a, b in shots:
        for i in range(a+1, b-1):
            if (i in local_frames and i-1 in local_frames and i+1 in local_frames
                    and not boxes[i] and boxes[i-1] and same_layout(boxes[i-1], boxes[i+1])):
                targets[i] = merge_line_boxes(boxes[i-1]+boxes[i+1])
    if not targets:
        return []
    local_area = (area[0]-band[0], area[1]-band[0], area[2], area[3])
    recovered = []
    last_target = max(targets)
    last_decoded = -1
    frames = _read_frames(video_path)
    try:
        for i, frame in enumerate(frames):
            last_decoded = i
            if i in targets:
                current = outlined_local_mask(frame[band[0]:band[1]], targets[i], (regions or {}).get(i, local_area))
                def agrees(other):
                    union = np.count_nonzero((current > 0) | (other > 0))
                    return union > 0 and np.count_nonzero((current > 0) & (other > 0))/union >= .75
                if agrees(raw[i-1]) and agrees(raw[i+1]):
                    raw[i] = current
                    boxes[i] = targets[i]
                    recovered.append(i)
            if i >= last_target:
                break
    finally:
        frames.close()
    if last_decoded < last_target:
        raise RuntimeError('Incomplete decode during local OCR gap recovery')
    return recovered


def _retry_shrunk_lines(video_path, boxes, shots, raw, detector, area, band, progress, local_frames=()):
    """Re-detect selected current frames; never copy adjacent subtitle boxes."""
    targets = find_shrunk_lines(boxes, shots) if detector is not None else {}
    targets = {i: value for i, value in targets.items() if i not in local_frames}
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


def _retry_short_lines(video_path, boxes, shots, raw, detector, area, band, height, fps, excluded, progress):
    local_area = (area[0]-band[0], area[1]-band[0], area[2], area[3])
    targets = short_text_regions(boxes, shots, local_area, height, fps, excluded)
    audit = dict(frames=[], updated_frames=[], regions={}, region_coordinate_space='ocr_band_yxyx')
    if not targets:
        return audit
    frames = _read_frames(video_path)
    last_decoded = -1
    last_target = max(targets)
    try:
        for i, frame in enumerate(frames):
            last_decoded = i
            if i in targets:
                detected, extra = short_text_mask(frame[band[0]:band[1]], detector, targets[i])
                audit['frames'].append(i)
                audit['regions'][i] = targets[i]
                if np.any((extra > 0) & (raw[i] == 0)):
                    raw[i] |= extra
                    boxes[i] = merge_line_boxes(boxes[i]+detected)
                    audit['updated_frames'].append(i)
                progress('ocr_short', 29, f'frame={i+1}, updated={len(audit["updated_frames"])}')
            if i >= last_target:
                break
    finally:
        frames.close()
    if last_decoded < last_target:
        raise RuntimeError('Incomplete decode during short subtitle recovery')
    return audit


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


def _infer(video_path, masks, jobs, band_y, options, folder, progress, local_frames=(), clean_candidates=()):
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
            reference, reference_audit = None, []
            if options['mode'] == 'sttn' and any(i in local_frames for i in range(job['start'], job['stop'])):
                eligible = {i-job['start'] for i in clean_candidates if job['start'] <= i < job['stop']}
                reference, reference_audit = select_reference(frames, small_masks, eligible)
            boundaries = [0, len(frames)]
            if options['mode'] == 'propainter':
                boundaries = photometric_boundaries(frames, small_masks)
                predictions = []
                for a, b in zip(boundaries, boundaries[1:]):
                    segment, segment_masks = frames[a:b], small_masks[a:b]
                    if not any(m.any() for m in segment_masks):
                        predictions.extend(segment)
                        continue
                    # RAFT needs a pair; duplicate the same frame, not a neighbor
                    # across the brightness boundary. No new pixels are composited.
                    single = b-a == 1
                    result = model.inpaint(segment*2 if single else segment,
                                           segment_masks*2 if single else segment_masks)
                    if len(result) != (2 if single else b-a):
                        raise RuntimeError('Model returned incomplete segment predictions')
                    predictions.extend(result[:b-a])
            elif reference is None:
                predictions = model.inpaint(frames, small_masks)
            else:
                predictions = model.inpaint(frames, small_masks, extra_reference_indices=[reference])
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
            records.append(dict(**job, input_sha256=frame_hash, mask_sha256=mask_hash,
                                clean_reference_frame=None if reference is None else job['start']+reference,
                                reference_candidates=reference_audit,
                                photometric_cuts=[job['start']+i for i in boundaries[1:-1]]))
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
    # Encode only the final composited frames; OCR/inference use source pixels.
    # Audio muxing copies this stream, so CRF 18 needs no second lossy encode.
    writer = FFmpegVideoWriter(str(silent), fps, size, crf=18, preset='fast', strict=True)
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
        local_audit = {}
        boxes, shots, ocr_frames = _scan(video_path, detector, area, band,
                                        options['ocr_sample_step'] or 2, folder/'raw.bin', progress, expected, local_audit)
        local_frames = set(local_audit.get('frames', []))
        shape = (len(boxes), band[1]-band[0], width)
        raw = _disk_map(stack, folder/'raw.bin', shape, 'r+')
        # Recover short captions before narrow-row retries can label their
        # missed frames as empty/local and exclude them from short recovery.
        short_retries = _retry_short_lines(video_path, boxes, shots, raw, detector, area, band,
                                           height, fps, local_frames, progress)
        narrow_retries = _retry_narrow_lines(video_path, boxes, shots, raw, detector, area, band,
                                             height, fps, progress)
        local_frames.update(narrow_retries['frames'])
        local_audit['frames'] = sorted(local_frames)
        ocr_frames += narrow_retries['ocr_calls']
        local_audit['pixel_recovered_frames'] = _recover_local_gaps(
            video_path, boxes, shots, raw, area, band, local_frames, narrow_retries['regions'])
        retries = _retry_shrunk_lines(video_path, boxes, shots, raw, retry_detector, area, band, progress, local_frames)
        stable = _disk_map(stack, folder/'stable.bin', shape)
        stabilize_masks(raw, shots, stable)
        progress('mask', 30, 'temporal_stabilization')
        revised = _disk_map(stack, folder/'revised.bin', shape)
        decoded = 0
        for i, frame in enumerate(_read_frames(video_path)):
            if i >= len(boxes):
                raise RuntimeError('Frame count changed during border recovery')
            # Unconstrained recovery would re-admit the background title strokes.
            revised[i] = raw[i] if i in local_frames else raw[i] | recover_vertical_border(frame[band[0]:band[1]], boxes[i], stable[i])
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
        clean_candidates = {i for i in local_frames if not boxes[i] and not masks[i].any()}
        records, device = _infer(video_path, masks, jobs, band[0], options, folder, progress,
                                 local_frames, clean_candidates)
        silent = folder/'silent.mp4'
        active = _render(video_path, masks, jobs, band[0], folder, silent, fps, (width, height), progress)
        progress('finalize', 98, 'audio_mux')
        mux_audio(silent, video_path, output)
        audit = dict(profile='glyph_v5', mode=options['mode'], device=device, frames=len(boxes),
                     fps=fps, size=[width, height], area=list(area), band=list(band),
                     ocr_frames=ocr_frames, shots=shots, jobs=records, active_mask_frames=active,
                     ocr_max_side=options.get('glyph_ocr_max_side', 640),
                     ocr_retry_max_side=960 if retry_detector is not None else None,
                     ocr_retry_frames=retries['frames'], ocr_retry_updated_frames=retries['updated_frames'],
                     local_ocr=local_audit, clean_reference_policy='sttn_local_ocr_same_window',
                     narrow_ocr=narrow_retries,
                     short_ocr=short_retries, photometric_policy='propainter_coherent_luma_shift',
                     masks_sha256=_mask_hash(masks), model_input_size=[432, 240],
                     chunk_size=80, overlap=20, feather_sigma=2, glyph_dilation=5,
                     frame_decode='sequential', mask_coverage_guard='outlined_pixels_in_ocr_line',
                     clipped_symbol_recovery='aligned_strong_white_components',
                     model_mask_resize='area_nonzero_then_1px_dilation', feather_policy='opaque_core',
                     scene_metric='mean absolute BGR difference at 96x54 > 40',
                     precision='fp32', post_processing=False)
        output.with_suffix('.json').write_text(json.dumps(audit, indent=2))
    progress('finalize', 100, 'complete')
    return output
