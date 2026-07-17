import multiprocessing
import os
import cv2
import numpy as np

from backend.config import config

def batch_generator(data, max_batch_size):
    """
    根据data大小，生成最大长度不超过max_batch_size的均匀批次数据
    """
    n_samples = len(data)
    # 尝试找到一个比MAX_BATCH_SIZE小的batch_size，以使得所有的批次数量尽量接近
    batch_size = max_batch_size
    num_batches = n_samples // batch_size

    # 处理最后一批可能不足batch_size的情况
    # 如果最后一批少于其他批次，则减小batch_size尝试平衡每批的数量
    while n_samples % batch_size < batch_size / 2.0 and batch_size > 1:
        batch_size -= 1  # 减小批次大小
        num_batches = n_samples // batch_size

    # 生成前num_batches个批次
    for i in range(num_batches):
        yield data[i * batch_size:(i + 1) * batch_size]

    # 将剩余的数据作为最后一个批次
    last_batch_start = num_batches * batch_size
    if last_batch_start < n_samples:
        yield data[last_batch_start:]

def create_mask(size, coords_list):
    """Build inpaint mask from detected subtitle bboxes.

    Two phases:
      1. Axis-aware rectangle expansion using independent X / Y deviation pixels,
         with frame-edge clipping.
      2. Per-axis morphology (cv2.dilate) to absorb anti-aliased edge pixels,
         ascender/descender artifacts, and the STTN model's reduced inpainting
         quality near the mask boundary.  Kernel size and iterations scale with
         deviation so taller (3+ line) subtitles — which need a larger dev_y —
         automatically get wider morphology:
           vertical: kernel (1, 1 + 2*max(1, dev_y//8)), iter=2 (dev_y<=30)
                     or iter=3 (dev_y>30).  At dev_y=22 → (1,5)/iter=2, the
                     same as the previous hardcoded values.
           horizontal: kernel (1 + 2*max(1, dev_x//8), 1), iter=1.  Catches
                     CJK stroke-end and ā/ě/ó fringe on the left/right edges.

    Both phases are skipped when the corresponding dev is 0.
    """
    mask = np.zeros(size, dtype="uint8")
    if not coords_list:
        return mask

    dev_x = config.subtitleAreaDeviationPixelX.value
    dev_y = config.subtitleAreaDeviationPixelY.value

    h, w = mask.shape[:2]
    for coords in coords_list:
        xmin, xmax, ymin, ymax = coords
        x1 = max(0, xmin - dev_x)
        y1 = max(0, ymin - dev_y)
        x2 = min(w - 1, xmax + dev_x)
        y2 = min(h - 1, ymax + dev_y)
        cv2.rectangle(mask, (x1, y1), (x2, y2), (255, 255, 255), thickness=-1)

    # Per-axis morphology.  Each axis is gated on its own dev so a user
    # who sets dev_x=0 to keep the mask tight horizontally isn't paying
    # for an unwanted horizontal dilate.  Growth is continuous in `dev`
    # (no upper cap) — at very large dev values the morphology extends
    # further, which is desirable for tall multi-line subtitles whose
    # descender/ascender edges need proportionally more padding.
    if dev_y > 0:
        morph_h = 1 + 2 * max(1, dev_y // 8)
        morph_iters = 2 if dev_y <= 30 else 3
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, morph_h))
        mask = cv2.dilate(mask, kernel, iterations=morph_iters)
    if dev_x > 0:
        morph_w = 1 + 2 * max(1, dev_x // 8)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_w, 1))
        mask = cv2.dilate(mask, kernel, iterations=1)

    # Vertical close: fill small gaps between multi-line subtitle rows.
    # When 3+ lines of text are detected, the per-bbox expansion (dev_y on
    # each side) may not fully bridge the gap between adjacent lines if
    # the inter-line spacing exceeds 2*dev_y.  A vertical close with a
    # tall kernel merges these gaps so STTN processes the entire subtitle
    # block as one inpaint region instead of splitting into separate
    # areas (which reduces per-line context and inpaint quality).
    # The kernel height is scaled to dev_y: at dev_y=22 → close_h=25,
    # which bridges gaps up to ~25px (typical for 1080p 3-line subs).
    if len(coords_list) >= 2 and dev_y > 0:
        close_h = max(5, dev_y + 3)
        if close_h % 2 == 0:
            close_h += 1
        close_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, close_h))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, close_kernel)

    return mask

def get_band_driven_split_h(mask, frame_h, frame_w, margin, min_h=96):
    """根据 mask 非零行的纵向分布计算 STTN crop 高度（split_h）。

    旧逻辑用固定比例（竖屏 H*5/9、横屏 W*5/18），竖屏 1080x1920 时 crop
    接近正方形，硬压到模型输入 432x240 形变/降质严重。改为 mask 驱动：
    把含 mask 像素的行按 margin 间隔聚成若干字幕带，取最高字幕带高度
    + 2*margin，并 clamp 到 [min_h, frame_h]。mask 为空时回退旧比例逻辑。

    Args:
        mask: (H, W) 或 (H, W, 1) 的 uint8 mask（非零表示待修复区域）
        frame_h / frame_w: 帧尺寸
        margin: 字幕带上下各扩的像素，同时作为相邻行带合并的最大间隙
        min_h: split_h 下限，避免 crop 太小丢失上下文

    Returns:
        int split_h（保证 1 <= split_h <= frame_h）
    """
    if mask is None or mask.size == 0 or not np.any(mask > 0):
        # 空 mask 回退到旧的比例逻辑
        if frame_h > frame_w:
            return int(frame_h * 5 / 9)
        return int(frame_w * 5 / 18)
    rows = np.any(mask > 0, axis=tuple(i for i in range(1, mask.ndim)))
    row_indices = np.flatnonzero(rows)
    # 相邻非零行间隔 <= margin 的合并为同一字幕带
    max_band_h = 1
    band_start = row_indices[0]
    prev_row = row_indices[0]
    for row in row_indices[1:]:
        if row - prev_row > margin:
            max_band_h = max(max_band_h, prev_row - band_start + 1)
            band_start = row
        prev_row = row
    max_band_h = max(max_band_h, prev_row - band_start + 1)
    split_h = max_band_h + 2 * int(margin)
    return max(int(min_h), min(int(frame_h), split_h))


def resolve_sttn_det_input_size(environ=None):
    """解析 STTN_DET 模型输入尺寸（env VSR_STTN_INPUT_WIDTH / VSR_STTN_INPUT_HEIGHT）。

    STTN 的 MultiHeadedAttention 用固定 patchsize [(108, 60), (36, 20),
    (18, 10), (9, 5)] 对特征图（输入的 1/4）做 reshape，因此特征图宽必须
    被 108 整除、高必须被 60 整除，即输入宽必须是 432 的整数倍、高必须是
    240 的整数倍，否则 view() 会直接抛错。合法且封顶 864x480（2x）以内的
    组合：432x240 / 864x240 / 432x480 / 864x480。校验不过时打 warning 并
    回退默认 432x240。

    Returns:
        (width, height) 二元组
    """
    environ = os.environ if environ is None else environ
    default_w, default_h = 432, 240
    max_w, max_h = 864, 480
    try:
        width = int(environ.get("VSR_STTN_INPUT_WIDTH", default_w))
        height = int(environ.get("VSR_STTN_INPUT_HEIGHT", default_h))
    except (TypeError, ValueError):
        print(f"WARNING: invalid VSR_STTN_INPUT_WIDTH/HEIGHT, fallback to {default_w}x{default_h}")
        return default_w, default_h
    if (width, height) == (default_w, default_h):
        return width, height
    if width % 432 == 0 and height % 240 == 0 and 0 < width <= max_w and 0 < height <= max_h:
        print(f"INFO: STTN_DET model input size overridden by env: {width}x{height}")
        return width, height
    print(
        f"WARNING: VSR_STTN_INPUT_WIDTH/HEIGHT={width}x{height} rejected "
        f"(width must be a multiple of 432 and <= {max_w}, height a multiple "
        f"of 240 and <= {max_h}; the STTN attention patch grid requires it). "
        f"Fallback to {default_w}x{default_h}."
    )
    return default_w, default_h


def get_inpaint_area_by_mask(W, H, h, mask, multiple=1):
    """
    获取字幕去除区域，根据mask来确定需要填补的区域和高度，
    并根据模型要求调整区域大小为指定倍数
    
    Args:
        W: 图像宽度
        H: 图像高度
        h: 检测区域高度
        mask: 遮罩图像
        multiple: 区域尺寸需要满足的倍数，默认为1
    
    Returns:
        调整后的绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]
    """
    # 存储绘画区域的列表
    inpaint_area = []
    
    # 如果mask全为0，直接返回空列表
    if np.all(mask == 0):
        return inpaint_area
    
    # 使用连通组件分析找出mask中的所有孤岛
    # 首先确保mask是二值图像
    binary_mask = (mask > 0).astype(np.uint8) * 255
    
    # 查找连通组件
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(binary_mask, connectivity=8)
    
    # 跳过背景（标签0）
    island_info = []
    for i in range(1, num_labels):
        # 获取当前孤岛的统计信息
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        height = stats[i, cv2.CC_STAT_HEIGHT]
        area = stats[i, cv2.CC_STAT_AREA]
        
        # 忽略太小的区域（可能是噪点）
        if area < 10:
            continue
        
        # 保存孤岛信息：顶部y坐标，底部y坐标，中心点y坐标，面积，标签
        center_y = int(centroids[i][1])
        island_info.append((y, y + height, center_y, area, i))
    
    # 如果没有有效孤岛，返回空列表
    if not island_info:
        return inpaint_area
    
    # 按中心点y坐标排序孤岛
    island_info.sort(key=lambda x: x[2])
    
    # 尝试合并孤岛
    merged_islands = []
    current_group = [island_info[0]]
    
    for i in range(1, len(island_info)):
        # 当前组的范围
        min_y = min([island[0] for island in current_group])
        max_y = max([island[1] for island in current_group])
        
        # 当前孤岛
        top_y, bottom_y, center_y, _, _ = island_info[i]
        
        # 计算如果添加当前孤岛，新组的范围
        new_min_y = min(min_y, top_y)
        new_max_y = max(max_y, bottom_y)
        
        # 检查是否有mask连接当前组和新孤岛
        has_connection = False
        gap = 0
        if max_y < top_y:  # 只有当前组在新孤岛上方时才需要检查连接
            gap = top_y - max_y
            # 检查两个区域之间是否有mask像素
            middle_region = binary_mask[max_y:top_y, :]
            if np.any(middle_region > 0):
                has_connection = True
            # Also merge when the gap is small (< dev_y pixels) — the
            # vertical close in create_mask should have connected them,
            # but a conservative close kernel may leave a 1-3 px gap.
            # Merging here lets STTN process the full subtitle block as
            # one region, giving it better vertical context.
            elif gap < config.subtitleAreaDeviationPixelY.value:
                has_connection = True
        else:  # 重叠或相邻
            has_connection = True
        
        # 检查合并后的高度是否在h范围内，并且有连接
        if new_max_y - new_min_y <= h and has_connection:
            # 可以合并
            current_group.append(island_info[i])
        else:
            # 无法合并，保存当前组并开始新组
            merged_islands.append(current_group)
            current_group = [island_info[i]]
    
    # 添加最后一个组
    merged_islands.append(current_group)
    
    # 为每个合并后的组创建区域
    for group in merged_islands:
        # 获取组内所有孤岛的范围
        min_y = min([island[0] for island in group])
        max_y = max([island[1] for island in group])
        
        # 计算组的中心点
        center_y = sum([island[2] for island in group]) // len(group)
        
        # 确保区域高度精确等于h
        half_h = h // 2
        
        # 从中心点向上下扩展，确保高度为h
        ymin = max(0, center_y - half_h)
        ymax = ymin + h  # 确保高度精确等于h
        
        # 如果超出图像底部，从底部向上调整
        if ymax > H:
            ymax = H
            ymin = max(0, H - h)  # 确保高度为h
        
        # 检查是否包含了所有孤岛
        if ymin > min_y or ymax < max_y:
            # 如果区域不能完全包含所有孤岛，尝试调整位置但保持高度为h
            if max_y - min_y <= h:
                # 孤岛总高度不超过h，可以调整位置使其完全包含
                ymin = min_y
                ymax = ymin + h
                # 如果超出底部，从底部向上调整
                if ymax > H:
                    ymax = H
                    ymin = max(0, H - h)
            else:
                # 孤岛总高度超过h，无法完全包含，优先包含中心区域
                # 计算孤岛的中心
                island_center = (min_y + max_y) // 2
                ymin = max(0, island_center - half_h)
                ymax = ymin + h
                # 如果超出底部，从底部向上调整
                if ymax > H:
                    ymax = H
                    ymin = max(0, H - h)
        
        # 使用完整宽度
        xmin = 0
        xmax = W
        
        # 调整区域大小为指定倍数
        if multiple > 1:
            # 计算区域高度
            height = ymax - ymin
            # 计算需要调整的高度，使其成为multiple的倍数
            remainder = height % multiple
            
            if remainder != 0:
                # 需要调整的像素数
                adjust_pixels = multiple - remainder
                
                # 计算区域中心点
                center_y = (ymin + ymax) / 2
                
                # 优先对称扩展
                if ymin - adjust_pixels/2 >= 0 and ymax + adjust_pixels/2 <= H:
                    # 对称扩展
                    ymin = int(center_y - height/2 - adjust_pixels/2)
                    ymax = int(center_y + height/2 + adjust_pixels/2)
                # 如果对称扩展会超出边界，尝试对称缩小
                elif height > multiple:  # 确保缩小后高度至少为multiple
                    # 对称缩小
                    ymin = int(center_y - (height - remainder)/2)
                    ymax = int(center_y + (height - remainder)/2)
                # 如果无法对称调整，则尝试单边调整
                else:
                    # 向下扩展
                    if ymax + adjust_pixels <= H:
                        ymax += adjust_pixels
                    # 向上扩展
                    elif ymin - adjust_pixels >= 0:
                        ymin -= adjust_pixels
                    # 如果都不行，则尝试缩小区域
                    elif height > multiple:
                        ymax = ymin + height - remainder
            
            # 调整宽度，确保是multiple的倍数
            width = xmax - xmin
            remainder_w = width % multiple
            
            if remainder_w != 0:
                # 需要调整的像素数
                adjust_pixels_w = multiple - remainder_w
                
                # 计算中心点，对称缩小
                center_x = (xmin + xmax) / 2
                xmin = int(center_x - (width - remainder_w)/2)
                xmax = int(center_x + (width - remainder_w)/2)
        
        # 将该区域添加到列表中，格式为(ymin, ymax, xmin, xmax)
        area = (int(ymin), int(ymax), int(xmin), int(xmax))
        if area not in inpaint_area:
            inpaint_area.append(area)
    
    return inpaint_area  # 返回绘画区域列表，格式为[(ymin, ymax, xmin, xmax), ...]
    
def expand_frame_ranges(frame_ranges, backward_frame_count, forward_frame_count):
    """
    扩展帧区间列表，向前和向后扩展指定的帧数，并确保区间连续性
    
    Args:
        frame_ranges: 帧区间列表，格式为[(start1, end1), (start2, end2), ...]
        backward_frame_count: 向前扩展的帧数
        forward_frame_count: 向后扩展的帧数
        
    Returns:
        扩展后的帧区间列表，保证连续性
    """
    if not frame_ranges:
        return []
    
    # 按起始帧排序
    sorted_ranges = sorted(frame_ranges)
    expanded_ranges = []
    
    for i, (start, end) in enumerate(sorted_ranges):
        # 向前扩展，但不能小于1
        new_start = max(1, start - backward_frame_count)
        
        # 向后扩展
        new_end = end + forward_frame_count
        
        # 检查是否与下一个区间重叠
        if i < len(sorted_ranges) - 1:
            next_start = sorted_ranges[i + 1][0]
            
            # 如果扩展后的结束帧超过了下一个区间的起始帧
            if new_end >= next_start:
                # 计算中点
                mid_point = (end + next_start) // 2
                
                # 如果区间是连续的(相差1)，则对半平分
                if next_start - end == 1:
                    new_end = end  # 保持原结束帧
                else:
                    # 非连续区间，限制扩展到下一个区间起始帧减去backward_frame_count
                    max_expand = next_start - 1  # 确保不会与下一个区间重叠
                    new_end = min(new_end, max_expand)
        
        # 确保与前一个区间不重叠
        if expanded_ranges:
            prev_end = expanded_ranges[-1][1]
            if new_start <= prev_end:
                # 如果新区间的开始小于等于前一个区间的结束，调整开始位置
                new_start = prev_end + 1
        
        # 确保区间有效（开始不大于结束）
        if new_start <= new_end:
            expanded_ranges.append((new_start, new_end))
        else:
            # 如果调整后区间无效，保留原始区间
            expanded_ranges.append((start, end))
    
    return expanded_ranges

def is_frame_number_in_ab_sections(frame_no, ab_sections):
    """
    检查给定的帧号是否在指定的A/B区间内。

    Args:
        frame_no: 要检查的帧号
        ab_sections: 包含A/B区间的列表，格式为[range(start, end), ...]

    Returns:
        如果帧号在A/B区间内，返回True；否则返回False。
    """
    if ab_sections is None:
        return True
    if len(ab_sections) <= 0:
        return True
    for section in ab_sections:
        if frame_no in section:
            return True
    return False

if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
