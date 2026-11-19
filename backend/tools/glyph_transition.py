"""仅在确认字幕随水平转场移动时补齐拖影掩膜。"""
import cv2
import numpy as np

from backend.tools.glyph_context import subtitle_line_shape
from backend.tools.glyph_masks import same_layout


def transition_candidates(masks, boxes, shots, area, height, fps, excluded=()):
    """寻找切镜附近短暂漏检，要求两侧均有连续可靠字幕。"""
    excluded = set(excluded)
    radius = min(15, int(fps * .5))
    if radius < 4:
        return []
    ay0, ay1, ax0, ax1 = area
    counts = {}

    def valid(i):
        if i < 0 or i >= len(boxes) or i in excluded or not boxes[i]:
            return False
        return all(subtitle_line_shape(b, height) and ax0 <= b[0] < b[1] <= ax1
                   and ay0 <= b[2] < b[3] <= ay1 for b in boxes[i])

    def count(i):
        if i not in counts:
            counts[i] = int(np.count_nonzero(masks[i, ay0:ay1, ax0:ax1]))
        return counts[i]

    def stable(i):
        if not valid(i) or count(i) < sum(3*(b[3]-b[2]) for b in boxes[i]):
            return False
        matches = [k for k in range(max(0, i-3), min(len(boxes), i+4))
                   if valid(k) and same_layout(boxes[i], boxes[k])
                   and count(k) >= .6*count(i)]
        return len(matches) >= 3

    found = {}
    for boundary, _ in shots[1:]:
        left = next((i for i in range(boundary-1, max(-1, boundary-radius-1), -1)
                     if stable(i)), None)
        right = next((i for i in range(boundary, min(len(boxes), boundary+radius+1))
                      if stable(i)), None)
        if left is None or right is None or not 2 <= right-left <= radius:
            continue
        if any(i in excluded for i in range(left, right+1)):
            continue
        threshold = .6*min(count(left), count(right))
        weak = [i for i in range(left+1, right) if count(i) < threshold]
        if len(weak) >= 2:
            found[(left, right)] = weak
    # 重叠候选不互相传递补齐结果，避免重复推断和累积扩大。
    result, last = [], -1
    for (left, right), targets in sorted(found.items()):
        if left <= last:
            continue
        result.append((left, right, targets))
        last = right
    return result


def _background_thumbnail(frame, area):
    """仅用选区外较大的背景估计位移，不以字幕形状驱动匹配。"""
    y0, y1, _, _ = area
    part = frame[:y0] if y0 >= frame.shape[0]-y1 else frame[y1:]
    if part.shape[0] < .15*frame.shape[0]:
        return None
    gray = cv2.cvtColor(part, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(cv2.resize(gray, (180, 120)), (9, 3), 0)


def _align(reference, current, sign):
    best = (float('inf'), 0)
    a, b = reference.astype(np.float32), current.astype(np.float32)
    # 保留至少四分之一画面重叠，避免极窄背景的偶然匹配。
    for d in range(135):
        x, y = (a[:, d:], b[:, :180-d]) if sign < 0 else (a[:, :180-d], b[:, d:])
        score = float(np.mean((x-y)**2))
        if score < best[0]:
            best = (score, sign*d)
    return best


def _text_evidence(current, reference, mask, dx):
    """比较候选范围内的纵向笔画分布，拒绝只有背景相似的帧。"""
    height, width = current.shape
    transform = np.float32([[1, 0, dx], [0, 1, 0]])
    ref = cv2.warpAffine(reference, transform, (width, height))
    shifted = cv2.warpAffine(mask, transform, (width, height))
    span = max(3, round(width * .155)) | 1
    support = cv2.dilate(shifted, cv2.getStructuringElement(cv2.MORPH_RECT, (span, 5))) > 0
    rows = support.sum(axis=1)
    valid = rows >= max(10, width * .025)
    if valid.sum() < 10:
        return 0., 0.

    def profile(image):
        data = image.astype(np.float32)
        high = data-cv2.GaussianBlur(data, (1, 41), 0)
        return (high*support).sum(axis=1)/np.maximum(rows, 1)

    a, b = profile(ref)[valid], profile(current)[valid]
    correlation = float(np.dot(a, b)/max(1., np.linalg.norm(a)*np.linalg.norm(b)))
    strength = float(np.linalg.norm(b)/np.sqrt(len(b)))
    return correlation, strength


def _process_event(event, cache, masks, area, band, audit):
    left, right, targets = event
    width = masks.shape[2]
    ay0, ay1, ax0, ax1 = area
    local_y0, local_y1 = ay0-band[0], ay1-band[0]
    if any(cache[i][1] is None for i in (left, right, *targets)):
        return
    # 分别验证向左和向右的整组转场，两侧字幕各需至少两帧支持。
    choices = []
    for direction in (-1, 1):
        matches = {}
        for ref, sign in ((left, direction), (right, -direction)):
            matches[ref] = []
            for i in targets:
                score, shift = _align(cache[ref][1], cache[i][1], sign)
                dx = int(round(shift*width/180))
                if score >= 700 or abs(dx) < width*.035:
                    continue
                corr, strength = _text_evidence(cache[i][0], cache[ref][0], masks[ref], dx)
                if corr >= .75 and strength >= 2.5:
                    matches[ref].append((i, dx, score, corr, strength))
        if min(len(matches[left]), len(matches[right])) >= 2:
            choices.append((sum(x[2] for v in matches.values() for x in v)/
                            sum(len(v) for v in matches.values()), matches))
    if not choices:
        return
    matches = min(choices, key=lambda item: item[0])[1]
    additions = {}
    for ref, entries in matches.items():
        previous_dx, previous_frame = 0, ref
        for i, dx, score, corr, strength in sorted(entries, reverse=ref == right):
            # 以帧间位移估算拖影扩散，设上限，且只在用户选区内补齐。
            speed = abs(dx-previous_dx)/max(1, abs(i-previous_frame))
            radius = int(min(width*.2, max(width*.075, speed*.7)))
            previous_dx, previous_frame = dx, i
            shifted = cv2.warpAffine(masks[ref], np.float32([[1, 0, dx], [0, 1, 0]]),
                                     (width, masks.shape[1]))
            extra = cv2.dilate(shifted, cv2.getStructuringElement(cv2.MORPH_RECT, (2*radius+1, 5)))
            extra[:local_y0] = 0
            extra[local_y1:] = 0
            extra[:, :ax0] = 0
            extra[:, ax1:] = 0
            additions[i] = additions.get(i, np.zeros_like(extra)) | extra
            audit['matches'].append(dict(frame=i, reference=ref, dx=dx,
                                         background_error=round(score, 2),
                                         text_correlation=round(corr, 3),
                                         text_strength=round(strength, 2), radius=radius))
    for i, extra in additions.items():
        added = int(np.count_nonzero((extra > 0) & (masks[i] == 0)))
        if added:
            masks[i] |= extra
            audit['updated_frames'].append(i)
            audit['added_pixels'][i] = added


def supplement_transition_glyphs(video_path, masks, boxes, shots, area, band, height, fps,
                                  excluded, read_frames, progress):
    """按事件顺序缓存少量灰度帧；补齐结果不参与其他事件的参考。"""
    local = (area[0]-band[0], area[1]-band[0], area[2], area[3])
    events = transition_candidates(masks, boxes, shots, local, height, fps, excluded)
    audit = dict(events=[dict(left=a, right=b, frames=indices) for a, b, indices in events],
                 updated_frames=[], added_pixels={}, matches=[])
    if not events:
        return audit
    index, cache, last_decoded = 0, {}, -1
    stream = read_frames(video_path)
    try:
        for i, frame in enumerate(stream):
            last_decoded = i
            left, right, targets = events[index]
            if i == left or i == right or i in targets:
                cache[i] = (cv2.cvtColor(frame[band[0]:band[1]], cv2.COLOR_BGR2GRAY),
                            _background_thumbnail(frame, area))
            if i == right:
                _process_event(events[index], cache, masks, area, band, audit)
                cache.clear()
                progress('mask', 44, f'transition_event={index+1}, updated={len(audit["updated_frames"])}')
                index += 1
                if index == len(events):
                    break
    finally:
        stream.close()
    if index < len(events):
        raise RuntimeError(f'Incomplete decode during transition recovery at frame {last_decoded}')
    audit['updated_frames'].sort()
    return audit
