"""Conservative local OCR recovery and model context safeguards."""
import cv2
import numpy as np


def subtitle_line_shape(box, frame_height, min_aspect=2):
    width, height = box[1]-box[0], box[3]-box[2]
    return 10 <= height <= .08*frame_height and width >= min_aspect*height


def narrow_text_regions(anchors, shots, area):
    """Derive one row per shot from pixel-verified repeated horizontal lines.

    Ambiguous/multiple rows retain the existing path. Coordinates are relative
    to the OCR band, and even a full-width retry stays inside the caller's area.
    """
    regions = []
    for shot, (start, stop) in enumerate(shots):
        groups = []
        for i in range(start, stop):
            for box in anchors.get(i, []):
                height = box[3]-box[2]
                center = (box[2]+box[3])/2
                for group in groups:
                    old = np.median([b for _, b in group], axis=0)
                    old_height = old[3]-old[2]
                    if (abs(center-(old[2]+old[3])/2) <= .3*min(height, old_height)
                            and .65 <= height/old_height <= 1.5):
                        group.append((i, box))
                        break
                else:
                    groups.append([(i, box)])
        supported = [g for g in groups if len({i for i, _ in g}) >= 2]
        if len(supported) != 1:
            continue
        group = supported[0]
        box = np.median([b for _, b in group], axis=0)
        pad = max(12, round((box[3]-box[2])*.25))
        region = [max(area[0], int(box[2])-pad), min(area[1], int(box[3])+pad), area[2], area[3]]
        if region[0] < region[1] and region[2] < region[3]:
            regions.append(dict(shot=shot, bounds=[start, stop], region=region,
                                anchor_frames=sorted({i for i, _ in group}), median_box=box.tolist()))
    return regions


def short_text_regions(boxes, shots, area, frame_height, fps, excluded=()):
    """Propose bounded OCR crops from repeated short detections in the same shot.

    A neighboring box only proposes a search, never supplies replacement text.
    Entirely undetected captions have no anchor and are intentionally not guessed.
    """
    radius = max(1, min(15, round(fps*.5)))
    def short(b):
        h, w = b[3]-b[2], b[1]-b[0]
        return 10 <= h <= .08*frame_height and .3*h <= w <= 1.5*h
    regions = {}
    for start, stop in shots:
        anchors = {i: [b for b in boxes[i] if short(b)]
                   for i in range(start, stop) if i not in excluded}
        for i in range(start, stop):
            if i in excluded or any(not short(b) for b in boxes[i]):
                continue
            nearby = [(j, b) for j in range(max(start, i-radius), min(stop, i+radius+1))
                      for b in anchors.get(j, [])]
            if not nearby:
                continue
            _, nearest = min(nearby, key=lambda pair: abs(pair[0]-i))
            h = nearest[3]-nearest[2]
            aligned = [(j, b) for j, b in nearby
                       if abs((b[0]+b[1]-nearest[0]-nearest[1])/2) <= h
                       and abs((b[2]+b[3]-nearest[2]-nearest[3])/2) <= h]
            if len({j for j, _ in aligned}) < 2:
                continue
            x0, x1, y0, y1 = np.median([b for _, b in aligned], axis=0)
            h = y1-y0
            region = [max(area[0], int(y0-1.5*h)), min(area[1], int(y1+h)),
                      max(area[2], int(x0-1.4*h)), min(area[3], int(x1+1.4*h))]
            if region[0] < region[1] and region[2] < region[3]:
                regions[i] = region
    return regions


def short_text_mask(frame, detector, area):
    """Re-detect current pixels and cover outlined short glyphs plus upper marks."""
    ay0, ay1, ax0, ax1 = area
    search = frame[ay0:ay1, ax0:ax1]
    boxes = []
    # Tiny crops are sensitive to detector resizing/rounding. One enlarged
    # retry may recover a missed body; all pixel tests still use the source.
    for scale in (1, 2):
        image = search if scale == 1 else cv2.resize(search, None, fx=scale, fy=scale)
        for coords in detector.detect_subtitle(image):
            x0, x1, y0, y1 = (v/scale for v in coords)
            h, w = y1-y0, x1-x0
            if (min(x0, y0) >= 3 and x1 <= ax1-ax0-3 and y1 <= ay1-ay0-3
                    and h >= 10 and .3*h < w < 1.5*h):
                boxes.append([int(x0+ax0), int(x1+ax0), int(y0+ay0), int(y1+ay0)])
        if boxes:
            break
    mask = np.zeros(frame.shape[:2], np.uint8)
    accepted = []
    for bx0, bx1, by0, by1 in boxes:
        h = by1-by0
        x0, x1 = max(ax0, bx0-10), min(ax1, bx1+10)
        y0, y1 = max(ay0, by0-h), min(ay1, by1+10)
        crop = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(((v >= 165) & (s <= 80)).astype('uint8'), 8)
        keep = np.zeros(crop.shape[:2], np.uint8)
        for k in range(1, n):
            x, y, w, ch, size = stats[k]
            if size < 3 or w > 1.5*h or ch > 1.5*h:
                continue
            if x == 0 or y == 0 or x+w >= crop.shape[1] or y+ch >= crop.shape[0]:
                continue
            pixels = (labels == k).astype('uint8')
            if np.count_nonzero((v >= 230) & (s <= 55) & (pixels > 0)) < max(3, .3*size):
                continue
            ring = (cv2.dilate(pixels, np.ones((5, 5), np.uint8)) > 0) & (pixels == 0)
            if np.mean(v[ring] < 120) < .5:
                continue
            keep[pixels > 0] = 255
        # Detached marks alone cannot confirm an otherwise absent subtitle.
        if not keep[by0-y0:by1-y0, bx0-x0:bx1-x0].any():
            continue
        accepted.append([bx0, bx1, by0, by1])
        mask[y0:y1, x0:x1] |= keep
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11)))
    mask[:ay0] = mask[ay1:] = 0
    mask[:, :ax0] = mask[:, ax1:] = 0
    return accepted, mask


def photometric_boundaries(frames, masks):
    """Separate coherent brightness transitions, not ordinary local object motion.

    Use unmasked model-input pixels so changing subtitle text cannot trigger cuts.
    Returning boundaries within a window also isolates gradual flash transitions
    whose individual frames are below the ordinary scene-cut threshold.
    """
    cuts = [0]
    previous = cv2.cvtColor(frames[0], cv2.COLOR_BGR2GRAY).astype(np.float32)
    for i in range(1, len(frames)):
        gray = cv2.cvtColor(frames[i], cv2.COLOR_BGR2GRAY).astype(np.float32)
        valid = (np.asarray(masks[i-1]) == 0) & (np.asarray(masks[i]) == 0)
        delta = (gray-previous)[valid]
        if len(delta) >= .2*gray.size:
            shift = float(np.median(delta))
            if abs(shift) > 6 and np.mean(delta*np.sign(shift) > 3) >= .75:
                cuts.append(i)
        previous = gray
    return cuts+[len(frames)]


def select_reference(frames, masks, eligible,
                     min_edge_recall=.65, max_context_mae=25.,
                     max_subtitle_retention=.20):
    # Caller supplies one shot-local model window (at most 80 resized frames).
    masks = np.asarray(masks)
    active = np.any(masks, axis=(1, 2))
    indices = np.flatnonzero(active)
    if not len(indices):
        return None, []
    anchor = int(indices[len(indices) // 2])
    # Different subtitles within a window are not a reliable common template.
    template_mask = masks[anchor] > 0
    for i in indices:
        current = masks[i] > 0
        if np.count_nonzero(current & template_mask) / max(1, np.count_nonzero(current | template_mask)) < .65:
            return None, [dict(reason='changing_subtitle_layout', accepted=False)]
    template = frames[anchor]
    union = np.any(masks[active], axis=0).astype('uint8')
    ys, xs = np.where(union)
    roi = np.zeros(union.shape, np.uint8)
    roi[max(0, ys.min()-80):ys.max()+81,
        max(0, xs.min()-80):xs.max()+81] = 1
    outside = (roi > 0) & (cv2.dilate(union, np.ones((15, 15), np.uint8)) == 0)
    gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    edges = (cv2.Canny(gray, 80, 160) > 0) & outside
    hp = gray.astype(float) - cv2.GaussianBlur(gray, (0, 0), 3)
    core = (masks[anchor] > 0) & (hp > 30)
    ring = (masks[anchor] > 0) & (hp < -30)
    if edges.sum() < 50 or core.sum() < 30 or ring.sum() < 30:
        return None, []
    rows = []
    for index, candidate in enumerate(frames):
        if masks[index].any():
            continue
        row = dict(index=index, anchor=anchor)
        if index not in eligible:
            rows.append(dict(row, accepted=False, reason='not_verified_by_local_ocr'))
            continue
        gray = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
        nearby_edges = cv2.dilate(cv2.Canny(gray, 80, 160), np.ones((5, 5), np.uint8)) > 0
        recall = float(np.mean(nearby_edges[edges]))
        mae = float(np.mean(np.abs(template.astype(float)-candidate)[outside]))
        hp = gray.astype(float) - cv2.GaussianBlur(gray, (0, 0), 3)
        # Independent of the candidate mask: an OCR miss must not imply clean.
        retention = float(max(np.mean(hp[core] > 30), np.mean(hp[ring] < -30)))
        reason = ('subtitle_pattern_remains' if retention > max_subtitle_retention
                  else 'background_changed' if recall < min_edge_recall or mae > max_context_mae
                  else 'accepted')
        rows.append(dict(row, edge_recall=recall, context_mae=mae,
                         subtitle_retention=retention,
                         accepted=reason == 'accepted', reason=reason,
                         score=recall-mae/255.))
    accepted = [r for r in rows if r['accepted']]
    selected = max(accepted, key=lambda r: r['score'])['index'] if accepted else None
    return selected, rows


def needs_local_ocr(coords, area):
    """A tall detection crossing the selection may merge subtitle/background text.

    Coordinates and area are both relative to the OCR band. An ordinary small
    box just outside the selection must not activate recovery for the shot.
    """
    ay0, ay1, ax0, ax1 = area
    for x0, x1, y0, y1 in coords:
        overlap_x = max(0, min(x1, ax1)-max(x0, ax0))
        overlap_y = max(0, min(y1, ay1)-max(y0, ay0))
        if (y0 < ay0 or y1 > ay1) and y1-y0 >= ay1-ay0 and (
                overlap_y >= .5*(ay1-ay0) and overlap_x >= .5*max(1, x1-x0)):
            return True
    return False


def local_ocr_boxes(frame, detector, area):
    """Crop to the selection before detection; reject clipped/background boxes."""
    y0, y1, x0, x1 = area
    height = y1-y0
    coords = detector.detect_subtitle(frame[y0:y1, x0:x1])
    return [[int(a+x0), int(b+x0), int(c+y0), int(d+y0)]
            for a, b, c, d in coords
            if a >= 0 and b <= x1-x0 and c > 2 and d < height-2
            and 0 < d-c <= .84*height and b-a > 1.5*(d-c)]


def outlined_local_mask(frame, boxes, area, radius=5):
    """Keep small white components surrounded by a dark outline, within area."""
    ay0, ay1, ax0, ax1 = area
    mask = np.zeros(frame.shape[:2], np.uint8)
    for bx0, bx1, by0, by1 in boxes:
        h = by1-by0
        pad = max(12, round(h*.4))
        x0, x1 = max(ax0, bx0-pad), min(ax1, bx1+pad)
        y0, y1 = max(ay0, by0-12), min(ay1, by1+12)
        crop = frame[y0:y1, x0:x1]
        if not crop.size:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        s, v = hsv[:, :, 1], hsv[:, :, 2]
        n, labels, stats, _ = cv2.connectedComponentsWithStats(((v >= 165) & (s <= 80)).astype('uint8'), 8)
        for k in range(1, n):
            x, y, w, ch, size = stats[k]
            if size < 3 or ch > 1.25*h or w > 1.25*h:
                continue
            if x == 0 or y == 0 or x+w >= crop.shape[1] or y+ch >= crop.shape[0]:
                continue
            if not by0-3 <= y0+y+ch/2 <= by1+3:
                continue
            pix = (labels == k).astype('uint8')
            if np.count_nonzero((v >= 210) & (s <= 55) & (pix > 0)) < max(3, .25*size):
                continue
            ring = (cv2.dilate(pix, np.ones((5, 5), np.uint8)) > 0) & (pix == 0)
            if np.mean(v[ring] < 100) < .6:
                continue
            mask[y0:y1, x0:x1][pix > 0] = 255
    mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))
    mask[:ay0] = 0
    mask[ay1:] = 0
    mask[:, :ax0] = 0
    mask[:, ax1:] = 0
    return mask
