import time

import cv2
import numpy as np
import torch
from torchvision import transforms
from typing import List
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from backend.config import config
from backend.inpaint.sttn.network_sttn import InpaintGenerator
from backend.inpaint.utils.sttn_utils import Stack, ToTorchFormatTensor
from backend.tools.inpaint_tools import get_inpaint_area_by_mask, get_band_driven_split_h, resolve_sttn_det_input_size

# 合成时 mask 边缘的高斯羽化强度（sigma≈2，过渡带约 5-6px）
FEATHER_SIGMA = 2.0

# 定义图像预处理方式
_to_tensors = transforms.Compose([
    Stack(),  # 将图像堆叠为序列
    ToTorchFormatTensor()  # 将堆叠的图像转化为PyTorch张量
])

class STTNDetInpaint:
    def __init__(self, device, model_path):
        self.device = device
        # 1. 创建InpaintGenerator模型实例并装载到选择的设备上
        self.model = InpaintGenerator().to(self.device)
        # 2. 载入预训练模型的权重，转载模型的状态字典
        self.model.load_state_dict(torch.load(model_path, map_location='cpu')['netG'])
        # 3. # 将模型设置为评估模式
        self.model.eval()
        # 模型输入用的宽和高（env VSR_STTN_INPUT_WIDTH/HEIGHT 可调；
        # 受 STTN 注意力 patch 网格约束，仅允许 432/240 的整数倍且封顶 864x480）
        self.model_input_width, self.model_input_height = resolve_sttn_det_input_size()
        # 2. 设置相连帧数
        self.neighbor_stride = config.sttnNeighborStride.value
        self.ref_length = config.sttnReferenceLength.value
        # mask 驱动的 crop 带高余量（env VSR_STTN_BAND_MARGIN 可调）
        try:
            self.band_margin = max(0, min(200, int(os.environ.get("VSR_STTN_BAND_MARGIN", "40"))))
        except (TypeError, ValueError):
            self.band_margin = 40

    def __call__(self, input_frames: List[np.ndarray], input_mask: np.ndarray):
        """
        :param input_frames: 原视频帧
        :param mask: 字幕区域mask
        """
        mask = input_mask[:, :, None]
        H_ori, W_ori = mask.shape[:2]
        H_ori = int(H_ori + 0.5)
        W_ori = int(W_ori + 0.5)
        # 确定去字幕的垂直高度部分：按 mask 非零行的纵向范围决定 crop 高度，
        # 不再用固定比例（竖屏 1080x1920 旧逻辑 crop 高达 1066px，硬压到
        # 432x240 再放大贴回形变/降质严重）；空 mask 时回退旧比例逻辑。
        split_h = get_band_driven_split_h(mask, H_ori, W_ori, self.band_margin)
        inpaint_area = get_inpaint_area_by_mask(W_ori, H_ori, split_h, mask)
        # 初始化帧存储变量
        # 高分辨率帧存储列表（浅拷贝 + 逐帧 copy，避免 deepcopy 开销）
        frames_hr = [f.copy() for f in input_frames]
        frames_scaled = {}  # 存放缩放后帧的字典
        masks_scaled = {}  # 存放缩放后遮罩的字典
        comps = {}  # 存放补全后帧的字典
        # 存储最终的视频帧
        inpainted_frames = []
        for k in range(len(inpaint_area)):
            frames_scaled[k] = []  # 为每个去除部分初始化一个列表
            masks_scaled[k] = []  # 为每个去除部分初始化一个列表

        # 读取并缩放帧
        for j in range(len(frames_hr)):
            image = frames_hr[j]
            # 对每个去除部分进行切割和缩放
            for k in range(len(inpaint_area)):
                image_crop = image[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 切割
                mask_crop = mask[inpaint_area[k][0]:inpaint_area[k][1], :, :]  # 切割
                image_resize = cv2.resize(image_crop, (self.model_input_width, self.model_input_height))  # 缩放
                mask_resize = cv2.resize(mask_crop, (self.model_input_width, self.model_input_height))  # 缩放
                frames_scaled[k].append(image_resize)  # 将缩放后的帧添加到对应列表
                masks_scaled[k].append(mask_resize)  # 将缩放后的遮罩添加到对应列表

        # 处理每一个去除部分
        for k in range(len(inpaint_area)):
            # 调用inpaint函数进行处理
            comps[k] = self.inpaint(frames_scaled[k], masks_scaled[k])

        # 如果存在去除部分
        if inpaint_area:
            for j in range(len(frames_hr)):
                frame = frames_hr[j]  # 取出原始帧
                # 对于模式中的每一个段落
                for k in range(len(inpaint_area)):
                    y0, y1 = inpaint_area[k][0], inpaint_area[k][1]
                    crop_h = y1 - y0
                    comp_result = comps[k][j]
                    if comp_result is None:
                        continue
                    comp = cv2.resize(comp_result, (W_ori, crop_h))  # 将补全帧缩放回原大小
                    comp = cv2.cvtColor(comp.astype(np.uint8), cv2.COLOR_BGR2RGB)  # 转换颜色空间
                    # 只在（高斯羽化后的）mask 范围内与原始帧混合，mask 外的
                    # 画面保持原始帧不动——避免整条 crop 带被 432x240 低分辨率
                    # round-trip 覆盖导致整带发软。
                    mask_area = mask[y0:y1, :, :]  # 取出遮罩区域
                    alpha = cv2.GaussianBlur(
                        (mask_area[:, :, 0] > 127).astype(np.float32), (0, 0), FEATHER_SIGMA
                    )[:, :, None]
                    # 高斯羽化在 mask 边界内外都衰减，mask 核心只能到 ~0.98，
                    # 直接混合会让原字幕像素残留 ~2%。放大 1.02 倍再截断，
                    # 让 mask 核心饱和到 1（完全用补全帧），羽化过渡保持不变。
                    alpha = np.clip(alpha * 1.02, 0.0, 1.0)
                    frame[y0:y1, :, :] = np.clip(
                        alpha * comp.astype(np.float32)
                        + (1.0 - alpha) * frame[y0:y1, :, :].astype(np.float32),
                        0, 255,
                    ).astype(np.uint8)
                # 将最终帧添加到列表
                inpainted_frames.append(frame)
                # print(f'processing frame, {len(frames_hr) - j} left')
        else:
            inpainted_frames = frames_hr
        return inpainted_frames

    @staticmethod
    def read_mask(path):
        img = cv2.imread(path, 0)
        # 转为binary mask
        ret, img = cv2.threshold(img, 127, 1, cv2.THRESH_BINARY)
        img = img[:, :, None]
        return img

    def get_ref_index(self, neighbor_ids, length):
        """
        采样整个视频的参考帧
        """
        # 初始化参考帧的索引列表
        ref_index = []
        # 在视频长度范围内根据ref_length逐步迭代
        for i in range(0, length, self.ref_length):
            # 如果当前帧不在近邻帧中
            if i not in neighbor_ids:
                # 将它添加到参考帧列表
                ref_index.append(i)
        # 返回参考帧索引列表
        return ref_index

    def inpaint(self, frames: List[np.ndarray], masks: List[np.ndarray]):
        """
        使用STTN完成空洞填充（空洞即被遮罩的区域）
        """
        frame_length = len(frames)
        # 对帧进行预处理转换为张量，并进行归一化
        feats = _to_tensors(frames).unsqueeze(0) * 2 - 1

        binary_masks = [np.expand_dims((np.array(m) > 0.5).astype(np.uint8), 2) for m in masks]
        # 将掩码转换为张量
        masks_tensor = (_to_tensors(masks).unsqueeze(0) > 0.5).float()

        # 把特征张量转移到指定的设备（CPU或GPU）
        feats, masks_tensor = feats.to(self.device), masks_tensor.to(self.device)
        # 初始化一个与视频长度相同的列表，用于存储处理完成的帧
        comp_frames = [None] * frame_length
        # 统一关闭梯度计算，用于推理阶段节省内存并加速
        with torch.no_grad():
            # 将处理好的帧通过编码器，产生特征表示
            feats = self.model.encoder((feats*(1-masks_tensor).float()).view(frame_length, 3, self.model_input_height, self.model_input_width))
            # 获取特征维度信息
            _, c, feat_h, feat_w = feats.size()
            # 调整特征形状以匹配模型的期望输入
            feats = feats.view(1, frame_length, c, feat_h, feat_w)
            # 在设定的邻居帧步幅内循环处理视频
            for f in range(0, frame_length, self.neighbor_stride):
                # 计算邻近帧的ID
                neighbor_ids = [i for i in range(max(0, f - self.neighbor_stride), min(frame_length, f + self.neighbor_stride + 1))]
                # 获取参考帧的索引
                ref_ids = self.get_ref_index(neighbor_ids, frame_length)
                # 通过模型推断特征并传递给解码器以生成完成的帧
                pred_feat = self.model.infer(
                    feats[0, neighbor_ids + ref_ids, :, :, :], masks_tensor[0, neighbor_ids + ref_ids, :, :, :])

                # 将预测的特征通过解码器生成图片，并应用激活函数tanh
                pred_img = torch.tanh(self.model.decoder(pred_feat[:len(neighbor_ids), :, :, :]))
                # 将结果张量重新缩放到0到255的范围内（图像像素值）
                pred_img = (pred_img + 1) / 2
                # 将张量移动回CPU并转为NumPy数组
                pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
                # 遍历邻近帧
                for i in range(len(neighbor_ids)):
                    idx = neighbor_ids[i]
                    # 将预测的图片转换为无符号8位整数格式
                    img = pred_img[i].astype(np.uint8) * binary_masks[idx] + frames[idx] * (1 - binary_masks[idx])
                    if comp_frames[idx] is None:
                        comp_frames[idx] = img
                    else:
                        comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
        # 返回处理完成的帧序列
        return comp_frames
