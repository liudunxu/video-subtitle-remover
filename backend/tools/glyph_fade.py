"""以当前帧笔画证据恢复短暂淡出的字幕，不复制邻帧擦除范围。"""
import cv2
import numpy as np

from backend.tools.glyph_context import subtitle_line_shape
from backend.tools.glyph_masks import box_iou


def fading_targets(masks, boxes, shots, area, height, fps, excluded=()):
    """仅提出同镜头、同位置、近期至少三帧支持的掩膜骤减目标。"""
    excluded = set(excluded)
    radius = max(1, min(8, round(fps * .25)))
    ay0, ay1, ax0, ax1 = area
    targets = {}
    for start, stop in shots:
        recent = []
        for i in range(start, stop):
            recent = [item for item in recent if i-item[0] <= radius]
            if i in excluded:
                continue
            current = []
            for box in boxes[i]:
                x0, x1, y0, y1 = box
                if not (subtitle_line_shape(box, height) and
                        ax0 <= x0 < x1 <= ax1 and ay0 <= y0 < y1 <= ay1):
                    continue
                count = int(np.count_nonzero(masks[i, y0:y1, x0:x1]))
                current.append((i, box, count))
                anchors = [(j, b) for j, b, pixels in recent
                           if pixels >= 2*(b[3]-b[2]) and count < .6*pixels
                           and box_iou(box, b) >= .8]
                if len({j for j, _ in anchors}) >= 3:
                    targets.setdefault(i, []).append((box, anchors))
            # 在任何补齐前确定候选；补回的像素不能续期或成为新依据。
            recent.extend(current)
    return targets


def _highpass(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    return gray-cv2.GaussianBlur(gray, (0, 0), 2)


def _template(frame, mask, line_height):
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hp = _highpass(frame)
    core = (hsv[:, :, 2] >= 230) & (hsv[:, :, 1] <= 55) & (hp >= 12) & (mask > 0)
    n, labels, stats, _ = cv2.connectedComponentsWithStats(core.astype(np.uint8), 8)
    bodies = [k for k in range(1, n) if stats[k, cv2.CC_STAT_HEIGHT] >= .3*line_height
              and stats[k, cv2.CC_STAT_AREA] >= 8]
    # 独立亮点、大片反光或只有少量残字不能作为整行字幕依据。
    if len(bodies) < 3 or core.sum() < .025*frame.shape[0]*frame.shape[1]:
        return None
    support = cv2.dilate(core.astype(np.uint8), np.ones((11, 11), np.uint8)) > 0
    return hp, core, support, labels, bodies


def fading_mask(frame, reference, reference_mask, line_height):
    """要求淡字保留原笔画排列，再从当前图像提取种子并有限膨胀。"""
    result = np.zeros(frame.shape[:2], np.uint8)
    template = _template(reference, reference_mask, line_height)
    if template is None:
        return result
    ref_hp, core, support, labels, bodies = template
    hp = _highpass(frame)
    a, b = ref_hp[support], hp[support]
    cosine = float(np.dot(a, b) / max(1., np.linalg.norm(a)*np.linalg.norm(b)))
    strength = float(np.median(hp[core]) / max(1., np.median(ref_hp[core])))
    # 只补仍能看见的淡出，不把亮度不变的漏框、换字或空背景混入。
    if cosine < .9 or not .15 <= strength <= .8 or np.mean(hp[core] >= 6) < .75:
        return result
    for label in bodies:
        pixels = labels == label
        if np.mean(hp[pixels] >= 6) < .6:
            return result
    seeds = (hp >= 4) & (cv2.dilate(core.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0)
    grown = cv2.dilate(seeds.astype(np.uint8)*255,
                       cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    # 邻帧只约束搜索范围，实际覆盖必须来自当前笔画；不涂满文字框。
    result[(grown > 0) & (reference_mask > 0)] = 255
    return result


def supplement_fading_glyphs(video_path, masks, boxes, shots, area, band, height, fps,
                              excluded, read_frames, progress):
    """一次顺序解码，缓存近期候选行的小裁剪；无目标时不解码。"""
    local = (area[0]-band[0], area[1]-band[0], area[2], area[3])
    targets = fading_targets(masks, boxes, shots, local, height, fps, excluded)
    audit = dict(frames=sorted(targets), updated_frames=[], rejected_frames=[],
                 added_pixels={}, references={}, ocr_calls=0)
    if not targets:
        return audit
    needed = {}
    regions = {}
    for i, rows in targets.items():
        for row_no, (box, anchors) in enumerate(rows):
            bounds = [box]+[b for _, b in anchors]
            x0 = max(local[2], min(b[0] for b in bounds)-8)
            x1 = min(local[3], max(b[1] for b in bounds)+8)
            pad = max(12, round((box[3]-box[2])*.22))
            y0 = max(local[0], min(b[2] for b in bounds)-pad)
            y1 = min(local[1], max(b[3] for b in bounds)+pad)
            key = (i, row_no)
            regions[key] = (x0, x1, y0, y1)
            for j, _ in anchors:
                needed.setdefault(j, []).append(key)
    cache = {}
    last_target = max(targets)
    last_decoded = -1
    frames = read_frames(video_path)
    try:
        for i, full in enumerate(frames):
            last_decoded = i
            frame = full[band[0]:band[1]]
            for key in needed.get(i, []):
                x0, x1, y0, y1 = regions[key]
                crop = frame[y0:y1, x0:x1].copy()
                mask = masks[i, y0:y1, x0:x1].copy()
                box = targets[key[0]][key[1]][0]
                template = _template(crop, mask, box[3]-box[2])
                if template is not None:
                    score = int(template[1].sum())
                    # 只留白字依据最充分的一帧，淡出中的残亮笔画不能替换它。
                    if key not in cache or score > cache[key][0]:
                        cache[key] = (score, i, crop, mask)
            if i in targets:
                additions = 0
                for row_no, (box, _) in enumerate(targets[i]):
                    key = (i, row_no)
                    reference = cache.pop(key, None)
                    if reference is None:
                        continue
                    _, j, crop, mask = reference
                    x0, x1, y0, y1 = regions[key]
                    extra = fading_mask(frame[y0:y1, x0:x1], crop, mask, box[3]-box[2])
                    current = masks[i, y0:y1, x0:x1]
                    added = int(np.count_nonzero((extra > 0) & (current == 0)))
                    if added:
                        current |= extra
                        additions += added
                        audit['references'].setdefault(i, []).append(j)
                if additions:
                    audit['updated_frames'].append(i)
                    audit['added_pixels'][i] = additions
                else:
                    audit['rejected_frames'].append(i)
                progress('mask', 44, f'fade_frame={i+1}, updated={len(audit["updated_frames"])}')
            if i >= last_target:
                break
    finally:
        frames.close()
    if last_decoded < last_target:
        raise RuntimeError('Incomplete decode during fading glyph recovery')
    return audit
