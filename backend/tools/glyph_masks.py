"""OCR-constrained white subtitle masks and scene-local temporal windows.

Glyph extraction is guarded by local outlined-pixel recovery inside OCR lines.
This is still not a general model for colored/moving subtitles.
Coordinates are (xmin, xmax, ymin, ymax), intervals are zero-based [start, stop).
"""
import cv2
import numpy as np

def box_iou(a, b):
    x0,x1,y0,y1=a; u0,u1,v0,v1=b
    intersect=max(0,min(x1,u1)-max(x0,u0))*max(0,min(y1,v1)-max(y0,v0))
    return intersect / max(1,(x1-x0)*(y1-y0)+(u1-u0)*(v1-v0)-intersect)

def same_layout(a,b):
    return len(a) == len(b) and all(
        max((box_iou(x, y) for y in right), default=0) >= .8
        for left, right in ((a, b), (b, a)) for x in left
    )

def merge_line_boxes(boxes):
    """Unify word/character detections before pixel segmentation, not a fill mask.

    The glyph filter rejects dense blobs in tiny single-character boxes (e.g. I).
    A shared text-line candidate region makes that test independent of OCR splits.
    """
    boxes=[list(b) for b in boxes]
    changed=True
    while changed:
        changed=False
        for i,a in enumerate(boxes):
            for j in range(i+1,len(boxes)):
                b=boxes[j]
                overlap=max(0,min(a[3],b[3])-max(a[2],b[2]))
                gap=max(0,max(a[0],b[0])-min(a[1],b[1]))
                if overlap>=.5*min(a[3]-a[2],b[3]-b[2]) and gap<=2*max(a[3]-a[2],b[3]-b[2]):
                    boxes[i]=[min(a[0],b[0]),max(a[1],b[1]),min(a[2],b[2]),max(a[3],b[3])]
                    boxes.pop(j);changed=True;break
            if changed:break
    return sorted(boxes)


def find_shrunk_lines(boxes, shots):
    """Find brief line shrinkage bracketed by agreeing wider OCR detections.

    Neighbor boxes only select frames for real OCR retries. They must never
    become erase masks: a shorter caption may be a legitimate text change.
    Empty frames and shot edges are deliberately not inferred from neighbors.
    """
    targets = {}
    for start, stop in shots:
        for i in range(start+1, stop-1):
            for line in boxes[i]:
                x0, x1, y0, y1 = line
                def wider(other):
                    width = other[1]-other[0]
                    return (x1-x0 <= .85*width and
                            min(x1, other[1])-max(x0, other[0]) >= .9*(x1-x0) and
                            min(y1, other[3])-max(y0, other[2]) >= .7*(y1-y0))
                before = [b for j in range(max(start, i-2), i) for b in boxes[j] if wider(b)]
                after = [b for j in range(i+1, min(stop, i+3)) for b in boxes[j] if wider(b)]
                pairs = [(a, b) for a in before for b in after if box_iou(a, b) >= .8]
                if pairs:
                    a, b = max(pairs, key=lambda pair: box_iou(*pair))
                    bounds = [min(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), max(a[3], b[3])]
                    targets.setdefault(i, []).append((line, bounds))
    return targets

def _glyph_mask_core(frame, boxes, radius=5):
    """Sample-specific white lettering: bright seeds, connected AA, outline padding.

    Not a general text segmentation model. OCR bounds confine candidates, then
    bright seeds must have a nearby dark border to reject clothing highlights.
    The 5px radius covers this clip's black outline/shadow; no rectangle fallback.
    """
    mask = np.zeros(frame.shape[:2], np.uint8)
    for x0, x1, y0, y1 in boxes:
        x0, y0 = max(0, x0-3), max(0, y0-3)
        x1, y1 = min(frame.shape[1], x1+4), min(frame.shape[0], y1+4)
        crop = frame[y0:y1, x0:x1]
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation, value = hsv[:, :, 1], hsv[:, :, 2]
        seeds = (value >= 210) & (saturation <= 55)
        support = ((value >= 165) & (saturation <= 80)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
        dark_near = cv2.dilate((value < 140).astype(np.uint8), np.ones((9, 9), np.uint8))
        keep = np.zeros_like(support)
        for label in range(1, n):
            x, y, w, h, area = stats[label]
            pixels = labels == label
            seed_count = np.count_nonzero(seeds & pixels)
            if area < 4 or seed_count < 3 or h < 3:
                continue
            # Bright shirt/backlight regions continue across OCR crop edges;
            # real glyph interiors sit inside the padded box in this sample.
            if x == 0 or y == 0 or x+w >= crop.shape[1] or y+h >= crop.shape[0]:
                continue
            if area > 0.25 * crop.shape[0] * crop.shape[1]:
                continue
            if np.count_nonzero(pixels & (dark_near > 0)) / area < 0.15:
                continue
            keep[pixels] = 255
        mask[y0:y1, x0:x1] |= keep
    return cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1)))


def glyph_mask(frame, boxes, radius=5):
    """Preserve glyph gaps and locally recover uncovered outlined strokes.

    Disconnected accents can lie outside the OCR box. Bright skin/clothing can
    also join a white letter into a component rejected by the core extractor.
    Recover small outlined components first. If outlined bright pixels still
    escape the mask inside an OCR line, grow those pixels locally, without
    turning the entire line (and its glyph gaps) into a repaint rectangle.
    Uniform highlights alone do not trigger this coverage guard.
    """
    mask = _glyph_mask_core(frame, boxes, radius)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2*radius+1, 2*radius+1))
    for bx0, bx1, by0, by1 in boxes:
        height = by1-by0
        pad_y = max(12, round(height*.22))
        x0, x1 = max(0, bx0-8), min(frame.shape[1], bx1+8)
        y0, y1 = max(0, by0-pad_y), min(frame.shape[0], by1+pad_y)
        crop = frame[y0:y1, x0:x1]
        if not crop.size:
            continue
        hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
        saturation, value = hsv[:, :, 1], hsv[:, :, 2]
        seeds = (value >= 200) & (saturation <= 80)
        contrast = value.astype(np.int16)-cv2.erode(value, np.ones((7, 7), np.uint8)).astype(np.int16)
        outlined = seeds & (contrast >= 30)
        support = ((value >= 165) & (saturation <= 80)).astype(np.uint8)
        n, labels, stats, _ = cv2.connectedComponentsWithStats(support, 8)
        symbols = np.zeros_like(support)
        for label in range(1, n):
            x, y, w, h, area = stats[label]
            pixels = labels == label
            if area < 2 or h > .6*height or area > .1*height*(bx1-bx0):
                continue
            if x == 0 or y == 0 or x+w >= crop.shape[1] or y+h >= crop.shape[0]:
                continue
            if np.any(pixels & outlined):
                symbols[pixels] = 255
        local = mask[y0:y1, x0:x1]
        local |= cv2.dilate(symbols, kernel)
        iy0, iy1 = max(0, by0-y0), min(y1-y0, by1-y0)
        ix0, ix1 = max(0, bx0-x0), min(x1-x0, bx1-x0)
        missing = outlined[iy0:iy1, ix0:ix1] & (local[iy0:iy1, ix0:ix1] == 0)
        if np.count_nonzero(missing) >= max(6, .003*(bx1-bx0)*height):
            guard = np.zeros_like(support)
            guard[iy0:iy1, ix0:ix1] = missing.astype(np.uint8)*255
            local |= cv2.dilate(guard, kernel)
    return mask


def resize_model_mask(mask, size=(432, 240)):
    """Conservative coverage through resizing and the output interpolation edge."""
    small = cv2.resize((mask > 0).astype(np.float32), size, interpolation=cv2.INTER_AREA)
    return cv2.dilate((small > 0).astype(np.uint8)*255, np.ones((3, 3), np.uint8))


def recover_vertical_border(frame,boxes,existing):
    result=np.zeros(frame.shape[:2],np.uint8)
    for bx0,bx1,by0,by1 in boxes:
        x0=max(0,bx0-3);x1=min(frame.shape[1],bx1+4)
        y0=max(0,by0-12);y1=min(frame.shape[0],by1+13)
        crop=frame[y0:y1,x0:x1];hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
        saturation,value=hsv[:,:,1],hsv[:,:,2]
        seeds=(value>=210)&(saturation<=55)
        inside=np.zeros(seeds.shape,bool)
        inside[max(0,by0-y0):by1-y0,max(0,bx0-x0):bx1-x0]=True
        support=((value>=165)&(saturation<=80)).astype(np.uint8)
        n,labels,stats,_=cv2.connectedComponentsWithStats(support,8)
        dark=cv2.dilate((value<140).astype(np.uint8),np.ones((9,9),np.uint8))
        keep=np.zeros_like(support)
        for k in range(1,n):
            x,y,w,h,area=stats[k];pixels=labels==k
            if area<4 or h<3 or np.count_nonzero(pixels&seeds&inside)<3:continue
            if h<.5*(by1-by0) or not by0<=y0+y+h/2<=by1:continue
            if x==0 or y==0 or x+w>=crop.shape[1] or y+h>=crop.shape[0]:continue
            if area>.25*crop.shape[0]*crop.shape[1]:continue
            if np.count_nonzero(pixels&(dark>0))/area<.15:continue
            # Local coverage recovery may cover most of a clipped letter but
            # still miss its descender. Skip only fully covered components.
            if np.all(existing[y0:y1,x0:x1][pixels]>0):continue
            # Only recover components crossing the old padded crop's horizontal
            # border; unrelated candidates newly exposed by wider context stay out.
            borders=[by0-3-y0,by1+3-y0]
            if not any(0<=r<len(pixels) and np.any(pixels[r]) for r in borders):continue
            keep[pixels]=255
        result[y0:y1,x0:x1]|=keep
    return cv2.dilate(result,cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(11,11)))

def stabilize_masks(raw, shots, destination, base=None, original_raw=None):
    """Keep empty frames empty, and never propagate through a shot boundary.

    Revision pass propagates only newly recovered border pixels, preserving the
    first pass exactly as in the validated experiment.
    """
    for a, b in shots:
        for i in range(a, b):
            destination[i] = raw[i] if base is None else base[i] | raw[i]
            if not np.any(raw[i]):
                continue
            for j in range(max(a, i-2), min(b, i+3)):
                union = np.count_nonzero(raw[i] | raw[j])
                if union and np.count_nonzero(raw[i] & raw[j]) / union >= .75:
                    delta = raw[j] if original_raw is None else raw[j] & ~original_raw[j]
                    destination[i] |= delta


def build_windows(masks, shots, band_y, margin=80):
    """At most 80 frames, 20 overlap; one crop per shot, 10 context frames."""
    jobs = []
    for shot, (a, b) in enumerate(shots):
        counts = np.array([np.count_nonzero(masks[i]) for i in range(a, b)])
        active = np.flatnonzero(counts)
        if not len(active):
            continue
        rows = np.zeros(masks.shape[1], bool)
        for i in range(a, b):
            rows |= np.any(masks[i], axis=1)
        ys = np.flatnonzero(rows)
        y0, y1 = int(ys[0])-margin, int(ys[-1])+margin+1
        if y1-y0 < 240:
            mid = (y0+y1)//2
            y0, y1 = mid-120, mid+120
        y0, y1 = max(0, y0), min(masks.shape[1], y1)
        first, last = max(a, a+int(active[0])-10), min(b, a+int(active[-1])+11)
        for start in range(first, last, 60):
            stop = min(start+80, last)
            jobs.append(dict(id=len(jobs), shot=shot, start=start, stop=stop,
                             active_span=[first, last], crop_y=[y0+band_y, y1+band_y]))
            if stop == last:
                break
    return jobs


def overlap_weight(index, job):
    weight = 1.0
    if job['start'] > job['active_span'][0]:
        weight = min(weight, (index-job['start']+1)/21)
    if job['stop'] < job['active_span'][1]:
        weight = min(weight, (job['stop']-index)/21)
    return weight


def blend_prediction(source, prediction, mask):
    """Only modify glyph pixels and their sigma-2 feather; BGR in and out."""
    if not np.any(mask):
        return source.copy()
    fill = cv2.resize(prediction, (source.shape[1], source.shape[0]))
    core = (mask > 127).astype(np.float32)
    # Feather outside the erase region. Blurring inward used to mix original
    # subtitle pixels back into narrow strokes and disconnected accents.
    alpha = np.maximum(core, cv2.GaussianBlur(core, (0, 0), 2))[:, :, None]
    return np.clip(alpha*fill + (1-alpha)*source, 0, 255).astype(np.uint8)
