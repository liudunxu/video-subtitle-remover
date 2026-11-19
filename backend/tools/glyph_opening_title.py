"""静止片头的金色背景片名保护，证据不足时保留现有掩膜。"""
import cv2
import numpy as np

from backend.tools.glyph_masks import glyph_mask


def _colors(frame, box):
    x0, x1, y0, y1 = map(int, box)
    crop = frame[max(0, y0):min(frame.shape[0], y1), max(0, x0):min(frame.shape[1], x1)]
    if not crop.size:
        return 0., 0.
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    bright = v >= 165
    gold = bright & (s > 55) & (h >= 5) & (h <= 35)
    return float(np.count_nonzero(gold) / max(1, np.count_nonzero(bright))), float(np.mean((v >= 210) & (s <= 55)))


def _gold_titles(frame, boxes, area, height):
    ay0, ay1, ax0, ax1 = area
    result = []
    for box in boxes:
        x0, x1, y0, y1 = map(int, box)
        if not (ax0 <= x0 < x1 <= ax1 and ay0 <= y0 < y1 <= ay1):
            continue
        if y1-y0 < .06*height or x1-x0 < .45*frame.shape[1]:
            continue
        if _colors(frame, box)[0] >= .6:
            result.append(box)
    if len(result) < 2 or not any(abs((a[2]+a[3]-b[2]-b[3])/2) >= .04*height
                                    for a in result for b in result):
        return []
    if max(b[3]-b[2] for b in result) < .1*height:
        return []
    return result


def _inside_fraction(box, area):
    x0, x1, y0, y1 = map(int, box)
    ay0, ay1, ax0, ax1 = area
    return (max(0, min(x1, ax1)-max(x0, ax0)) * max(0, min(y1, ay1)-max(y0, ay0)) /
            max(1, (x1-x0)*(y1-y0)))


def _uncovered_white_stroke(frame, area, mask, height):
    """OCR 未确认的强白描边短笔画会否决保护，包括单字字幕。"""
    y0, y1, x0, x1 = area
    hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
    s, v = hsv[:, :, 1], hsv[:, :, 2]
    count, labels, stats, _ = cv2.connectedComponentsWithStats(((v > 235) & (s < 32)).astype('uint8'), 8)
    for k in range(1, count):
        x, y, w, h, size = map(int, stats[k])
        if not (10 <= h <= .055*height and 1 <= w <= 1.5*h and size >= 4):
            continue
        pixels = (labels[y:y+h, x:x+w] == k).astype('uint8')
        # 给组件保留描边环空间，避免对整张选区逐组件膨胀。
        rx0, rx1 = max(0, x-2), min(v.shape[1], x+w+2)
        ry0, ry1 = max(0, y-2), min(v.shape[0], y+h+2)
        component = (labels[ry0:ry1, rx0:rx1] == k).astype('uint8')
        ring = (cv2.dilate(component, np.ones((5, 5), np.uint8)) > 0) & (component == 0)
        if not ring.any() or np.mean(v[ry0:ry1, rx0:rx1][ring] < 100) < .5:
            continue
        covered = mask[y0+y:y0+y+h, x0+x:x0+x+w] > 0
        if np.count_nonzero(covered & (pixels > 0)) < .95*size:
            return True
    return False


def _uncovered_outlined_stroke(frame, boxes, mask, area, height):
    """字幕框内若有未覆盖的亮色描边笔画，不能把它误认为白色字幕。"""
    ay0, ay1, ax0, ax1 = area
    for bx0, bx1, by0, by1 in boxes:
        x0, x1 = max(ax0, bx0-6), min(ax1, bx1+6)
        y0, y1 = max(ay0, by0-6), min(ay1, by1+6)
        hsv = cv2.cvtColor(frame[y0:y1, x0:x1], cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        # 区分白/灰、金色与其他颜色，避免不同文字连成一个组件。
        colors = ((v >= 145) & (s <= 80),
                  (v >= 145) & (s > 80) & (h >= 5) & (h <= 35),
                  (v >= 140) & (s > 80) & ((h < 5) | (h > 35)))
        for color in colors:
            count, labels, stats, _ = cv2.connectedComponentsWithStats(color.astype('uint8'), 8)
            for k in range(1, count):
                x, y, w, ch, size = map(int, stats[k])
                if not (5 <= ch <= .055*height and 1 <= w <= 2*ch and size >= 8):
                    continue
                if x == 0 or y == 0 or x+w >= color.shape[1] or y+ch >= color.shape[0]:
                    continue
                rx0, rx1 = max(0, x-2), min(color.shape[1], x+w+2)
                ry0, ry1 = max(0, y-2), min(color.shape[0], y+ch+2)
                component = (labels[ry0:ry1, rx0:rx1] == k).astype('uint8')
                ring = (cv2.dilate(component, np.ones((5, 5), np.uint8)) > 0) & (component == 0)
                if not ring.any() or np.mean(v[ry0:ry1, rx0:rx1][ring] < 100) <= .4:
                    continue
                pixels = labels[y:y+ch, x:x+w] == k
                covered = mask[y0+y:y0+y+ch, x0+x:x0+x+w] > 0
                if np.count_nonzero(covered & pixels) < .8*size:
                    return True
    return False


def _subtitle_mask(frame, primary, detected, area, height):
    """只有全部检测结果可分类、当前白色描边笔画均被覆盖才缩小掩膜。"""
    small = []
    for box in list(primary)+list(detected):
        x0, x1, y0, y1 = map(int, box)
        fraction = _inside_fraction(box, area)
        if fraction == 0:
            continue
        h = y1-y0
        # 大字在选区边缘可能只有一小部分；正常字号越界时不能忽略。
        if fraction < .5 and h > .055*height:
            continue
        if fraction < .995:
            return None
        gold, white = _colors(frame, box)
        if h >= .06*height and x1-x0 >= .45*frame.shape[1] and gold >= .6:
            continue
        if not (10 <= h <= .055*height and white >= .025):
            return None
        small.append((x0, x1, y0, y1))
    small = sorted(set(small))
    mask = glyph_mask(frame, small) if small else np.zeros(frame.shape[:2], np.uint8)
    for x0, x1, y0, y1 in small:
        if np.count_nonzero(mask[y0:y1, x0:x1]) < 2*(y1-y0):
            return None
    ay0, ay1, ax0, ax1 = area
    mask[:ay0] = mask[ay1:] = 0
    mask[:, :ax0] = mask[:, ax1:] = 0
    if (_uncovered_white_stroke(frame, area, mask, height) or
            _uncovered_outlined_stroke(frame, small, mask, area, height)):
        return None
    return mask


def protect_opening_title(video_path, masks, boxes, shots, area, band, height, fps,
                          retry_detector, excluded, read_frames, progress):
    """首镜头仅处理连续稳定且经逐帧复检的片名，不跨镜头或修改检测框。"""
    audit = dict(checked_frames=[], candidate_frames=[], protected_frames=[],
                 subtitle_frames=[], veto_frames=[], ocr_calls=0, removed_pixels={})
    if not shots or shots[0][0] != 0 or area[1]-area[0] <= .15*height:
        return audit
    stop = min(shots[0][1], max(3, int(round(5*fps))))
    targets = {i for i in range(stop) if i not in excluded and masks[i].any() and
               sum(b[3]-b[2] >= .06*height and b[1]-b[0] >= .45*masks.shape[2]
                   for b in boxes[i]) >= 2}
    if len(targets) < 3:
        return audit
    local = (area[0]-band[0], area[1]-band[0], area[2], area[3])
    pending, previous, last = {}, None, -1
    frames = read_frames(video_path)
    try:
        for i, source in enumerate(frames):
            last = i
            if i >= stop:
                break
            if i not in targets:
                previous = None
                continue
            frame = source[band[0]:band[1]]
            audit['checked_frames'].append(i)
            if not _gold_titles(frame, boxes[i], local, height):
                previous = None
                continue
            # 仅比较选区缩略图，帧变化时重新积累连续证据。
            thumb = cv2.resize(frame[local[0]:local[1], local[2]:local[3]], (90, 51), interpolation=cv2.INTER_AREA)
            stable = previous is None or np.max(cv2.absdiff(previous, thumb)) <= 8
            previous = thumb
            if not stable:
                audit['veto_frames'].append(i)
                continue
            audit['candidate_frames'].append(i)
            found = retry_detector.detect_subtitle(frame)
            audit['ocr_calls'] += 1
            current = _subtitle_mask(frame, boxes[i], found, local, height)
            if current is None:
                audit['veto_frames'].append(i)
                continue
            # 位图暂存直到完整解码成功，避免异常后留下部分修改。
            pending[i] = np.packbits(current > 0)
            if current.any():
                audit['subtitle_frames'].append(i)
            if i % 24 == 0:
                progress('title_guard', 44, f'opening_frame={i+1}')
    finally:
        frames.close()
    if last < stop-1:
        raise RuntimeError('Incomplete decode during opening title verification')
    run = []
    for i in range(stop+1):
        if i < stop and i in pending:
            run.append(i)
            continue
        if len(run) >= 3:
            for j in run:
                replacement = np.unpackbits(pending[j], count=masks[j].size).reshape(masks[j].shape).astype('uint8')*255
                audit['removed_pixels'][j] = int(np.count_nonzero((masks[j] > 0) & (replacement == 0)))
                masks[j] = replacement
                audit['protected_frames'].append(j)
        run = []
    return audit
