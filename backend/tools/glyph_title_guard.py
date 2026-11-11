"""双行字幕选区内的背景大字保护；证据不足时保留原掩膜。"""
import cv2
import numpy as np


def _white_fraction(frame, box):
    x0, x1, y0, y1 = map(int, box)
    crop = frame[max(0, y0):min(frame.shape[0], y1), max(0, x0):min(frame.shape[1], x1)]
    if not crop.size:
        return 0
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    return float(np.mean((hsv[:, :, 2] >= 210) & (hsv[:, :, 1] <= 55)))


def _learn_rows(groups, frame, lines, mask, area, height, index):
    """用当前帧白色像素和已有掩膜验证字幕行，不学习背景大字。"""
    ay0, ay1, ax0, ax1 = area
    for box in lines:
        x0, x1, y0, y1 = box
        h = y1-y0
        if not (10 <= h <= .08*height and ax0 <= x0 < x1 <= ax1 and ay0 <= y0 < y1 <= ay1):
            continue
        if _white_fraction(frame, box) < .03 or np.count_nonzero(mask[y0:y1, x0:x1]) < 2*h:
            continue
        for group in groups:
            old = np.median(group['boxes'], axis=0)
            old_h = old[3]-old[2]
            if abs((y0+y1-old[2]-old[3])/2) < .4*min(h, old_h) and .65 < h/old_h < 1.5:
                if group['last_frame'] != index:
                    group['count'] += 1
                    group['last_frame'] = index
                    group['boxes'] = (group['boxes']+[box])[-15:]
                break
        else:
            # 行位过多时放弃本镜头学习，避免复杂画面的无界缓存。
            if len(groups) < 32:
                groups.append(dict(boxes=[box], count=1, last_frame=index))
    supported = [np.median(g['boxes'], axis=0).tolist() for g in groups if g['count'] >= 3]
    if len(supported) != 2:
        return None
    heights = [b[3]-b[2] for b in supported]
    if abs(sum(supported[0][2:])/2-sum(supported[1][2:])/2) < 1.8*max(heights):
        return None
    if ay1-ay0 < 4*np.median(heights):
        return None
    return supported


def _title_candidate(frame, coords, rows, area):
    h = float(np.median([b[3]-b[2] for b in rows]))
    ay0, ay1, ax0, ax1 = area
    relevant = [b for b in coords if min(b[1], ax1) > max(b[0], ax0)
                and min(b[3], ay1) > max(b[2], ay0)]
    large = [b for b in relevant if b[3]-b[2] >= 2.5*h and b[1]-b[0] >= 4*h
             and _white_fraction(frame, b) >= .03]
    if not any(abs(sum(a[2:])/2-sum(b[2:])/2) >= h for a in large for b in large):
        return False
    # 正常字号的白字即使移位，也应阻止整帧保护，优先保证字幕擦除。
    return not any(10 <= b[3]-b[2] <= 1.5*h and _white_fraction(frame, b) >= .03 for b in relevant)


def protect_background_titles(video_path, masks, boxes, shots, area, band, height,
                              detector, retry_detector, excluded, read_frames, progress):
    """在最终掩膜上保护已核实的大字，不跨镜头传播，不缓存整片画面。"""
    audit = dict(checked_frames=[], candidate_frames=[], eligible_frames=[], protected_frames=[],
                 veto_frames=[], ocr_calls=0, regions={}, profiles=[], removed_pixels={},
                 region_coordinate_space='ocr_band_yxyx')
    if area[1]-area[0] <= .15*height:
        return audit
    targets = {i for i, lines in enumerate(boxes) if i not in excluded and
               any(b[3]-b[2] > .08*height and b[1]-b[0] > b[3]-b[2] for b in lines)}
    if not targets:
        return audit
    last_target = max(targets)
    local = (area[0]-band[0], area[1]-band[0], area[2], area[3])
    retry = retry_detector if retry_detector is not None else detector
    rows, groups, shot, last_decoded = None, [], 0, -1
    eligible = set()
    frames = read_frames(video_path)
    try:
        for i, source in enumerate(frames):
            last_decoded = i
            while i >= shots[shot][1]:
                shot += 1
                groups = []
            frame = source[band[0]:band[1]]
            if i not in excluded:
                learned = _learn_rows(groups, frame, boxes[i], masks[i], local, height, i)
                # 出现三条以上稳定行时，旧的双行模板已不可靠，停止整帧保护。
                if sum(g['count'] >= 3 for g in groups) > 2:
                    rows = None
                elif learned is not None:
                    rows = learned
            if i in targets and rows is not None and masks[i].any():
                audit['checked_frames'].append(i)
                coords = detector.detect_subtitle(frame)
                audit['ocr_calls'] += 1
                if _title_candidate(frame, coords, rows, local):
                    audit['candidate_frames'].append(i)
                    regions, normal = [], False
                    for box in rows:
                        h = box[3]-box[2]
                        y0, y1 = max(local[0], int(box[2]-.4*h)), min(local[1], int(box[3]+.4*h))
                        region = [y0, y1, local[2], local[3]]
                        regions.append(region)
                        crop = frame[y0:y1, local[2]:local[3]]
                        found = retry.detect_subtitle(crop)
                        audit['ocr_calls'] += 1
                        # 必须复查两行；正常字号且未贴裁剪边缘的白字阻止保护。
                        normal |= any(b[2] > 2 and b[3] < crop.shape[0]-2 and
                                      .65*h <= b[3]-b[2] <= 1.5*h and _white_fraction(crop, b) >= .03
                                      for b in found)
                    audit['regions'][i] = regions
                    if normal:
                        audit['veto_frames'].append(i)
                    else:
                        eligible.add(i)
                        audit['profiles'].append(dict(frame=i, rows=rows))
            if i % 24 == 0:
                progress('title_guard', 44, f'frame={i+1}, candidates={len(eligible)}')
            if i >= last_target:
                break
    finally:
        frames.close()
    if last_decoded < last_target:
        raise RuntimeError('Incomplete decode during background title verification')
    audit['eligible_frames'] = sorted(eligible)
    # 要求同镜头至少连续三帧确认；检查全部完成后才清除掩膜，失败不能留下部分修改。
    for start, stop in shots:
        run = []
        for i in range(start, stop+1):
            if i < stop and i in eligible:
                run.append(i)
                continue
            if len(run) >= 3:
                for j in run:
                    audit['removed_pixels'][j] = int(np.count_nonzero(masks[j]))
                    masks[j] = 0
                    audit['protected_frames'].append(j)
            run = []
    return audit
