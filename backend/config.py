
import json
import os
from enum import Enum
from pathlib import Path

from backend.tools.constant import InpaintMode, SubtitleDetectMode

# 项目版本号
VERSION = "1.4.0"

# 硬件加速选项开关
HARDWARD_ACCELERATION_OPTION = True


class ConfigItem:
    """轻量配置项，提供 qfluentwidgets.ConfigItem 的 .value 接口。"""

    def __init__(self, section, key, default):
        self.section = section
        self.key = key
        self.default = default
        self.value = default

    def __repr__(self):
        return f"ConfigItem({self.section}.{self.key}={self.value!r})"


def _serialize_value(value):
    if isinstance(value, Enum):
        return f"{type(value).__name__}.{value.name}"
    return value


def _deserialize_value(raw, default):
    if isinstance(default, Enum):
        if isinstance(raw, Enum):
            return raw
        if isinstance(raw, str):
            name = raw.split(".")[-1]
            # 向后兼容：旧的 SubtitleDetectMode 枚举值为中文，迁移为新值
            legacy_detect = {
                "快速": "PP_OCRv5_MOBILE", "Fast": "PP_OCRv5_MOBILE",
                "精准": "PP_OCRv5_SERVER", "Precise": "PP_OCRv5_SERVER",
            }
            name = legacy_detect.get(name, name)
            enum_cls = type(default)
            try:
                return enum_cls[name]
            except KeyError:
                try:
                    return enum_cls(raw)
                except ValueError:
                    return default
        return default
    return raw


class Config:
    """纯 Python 配置单例（原 qfluentwidgets.QConfig 的替代品）。

    启动时从 config/config.json 读取覆盖值（文件不存在则全用默认值）；
    set(..., save=True) 时写回该文件。
    """

    CONFIG_FILE = "config/config.json"

    def __init__(self):
        """
        MODE可选算法类型
        - InpaintMode.STTN_AUTO 智能擦除版
        - InpaintMode.STTN_DET 带字幕检测版, 无智能擦除
        - InpaintMode.LAMA 算法：对于动画类视频效果好，速度一般，不可以跳过字幕检测
        - InpaintMode.PROPAINTER 算法： 需要消耗大量显存，速度较慢，对运动非常剧烈的视频效果较好
        """
        # 【设置inpaint算法】
        self.inpaintMode = ConfigItem("Main", "InpaintMode", InpaintMode.STTN_AUTO)
        self.subtitleDetectMode = ConfigItem("Main", "SubtitleDetectMode", SubtitleDetectMode.PP_OCRv5_SERVER)

        # 【设置像素点偏差】
        # 用于判断是不是非字幕区域(一般认为字幕文本框的长度是要大于宽度的，如果字幕框的高大于宽，且大于的幅度超过指定像素点大小，则认为是错误检测)
        self.subtitleYXAxisDifferencePixel = ConfigItem("Main", "SubtitleYXAxisDifferencePixel", 10)
        # [deprecated] 已由 subtitleAreaDeviationPixelX / Y 替代。create_mask 不再读取本字段。
        # 保留仅用于兼容旧版 config.json 与 API 的 subtitle_area_deviation_pixel 选项。
        self.subtitleAreaDeviationPixel = ConfigItem("Main", "SubtitleAreaDeviationPixel", 10)
        # X / Y 方向独立的 mask 外扩像素。
        # X 保持 10 与旧 subtitleAreaDeviationPixel 一致；Y 提升至 22 以覆盖 3-8px 残 + 抗锯齿 + 安全余量。
        # 详见 docs/superpowers/specs/2026-06-10-subtitle-edge-residual-design.md
        self.subtitleAreaDeviationPixelX = ConfigItem("Main", "SubtitleAreaDeviationPixelX", 10)
        self.subtitleAreaDeviationPixelY = ConfigItem("Main", "SubtitleAreaDeviationPixelY", 22)
        # 同于判断两个文本框是否为同一行字幕，高度差距指定像素点以内认为是同一行
        self.subtitleAreaYAxisDifferencePixel = ConfigItem("Main", "SubtitleAreaYAxisDifferencePixel", 20)
        # 用于判断两个字幕文本的矩形框是否相似，如果X轴和Y轴偏差都在指定阈值内，则认为时同一个文本框
        self.subtitleAreaPixelToleranceYPixel = ConfigItem("Main", "SubtitleAreaPixelToleranceYPixel", 20)
        self.subtitleAreaPixelToleranceXPixel = ConfigItem("Main", "SubtitleAreaPixelToleranceXPixel", 20)
        self.subtitleTimelineBackwardFrameCount = ConfigItem("Main", "SubtitleTimelineBackwardFrameCount", 3)
        self.subtitleTimelineForwardFrameCount = ConfigItem("Main", "subtitleTimelineForwardFrameCount", 3)
        # 以下参数仅适用STTN算法时，才生效
        """
        1. STTN_SKIP_DETECTION
        含义：是否使用跳过检测
        效果：设置为True跳过字幕检测，会省去很大时间，但是可能误伤无字幕的视频帧或者会导致去除的字幕漏了

        2. STTN_NEIGHBOR_STRIDE
        含义：相邻帧数步长, 如果需要为第50帧填充缺失的区域，STTN_NEIGHBOR_STRIDE=5，那么算法会使用第45帧、第40帧等作为参照。
        效果：用于控制参考帧选择的密度，较大的步长意味着使用更少、更分散的参考帧，较小的步长意味着使用更多、更集中的参考帧。

        3. STTN_REFERENCE_LENGTH
        含义：参数帧数量，STTN算法会查看每个待修复帧的前后若干帧来获得用于修复的上下文信息
        效果：调大会增加显存占用，处理效果变好，但是处理速度变慢

        4. STTN_MAX_LOAD_NUM
        含义：STTN算法每次最多加载的视频帧数量
        效果：设置越大速度越慢，但效果越好
        注意：要保证STTN_MAX_LOAD_NUM大于STTN_NEIGHBOR_STRIDE和STTN_REFERENCE_LENGTH
        """
        # 参考帧步长
        self.sttnNeighborStride = ConfigItem("Sttn", "NeighborStride", 5)
        # 参考帧数量
        self.sttnReferenceLength = ConfigItem("Sttn", "ReferenceLength", 10)
        # 设置STTN算法最大同时处理的帧数量
        self.sttnMaxLoadNum = ConfigItem("Sttn", "MaxLoadNum", 50)

        # 以下参数仅适用PROPAINTER算法时，才生效
        # 【根据自己的GPU显存大小设置】最大同时处理的图片数量，设置越大处理效果越好，但是要求显存越高
        # 1280x720p视频设置80需要25G显存，设置50需要19G显存
        # 720x480p视频设置80需要8G显存，设置50需要7G显存
        self.propainterMaxLoadNum = ConfigItem("ProPainter", "MaxLoadNum", 70)

        # 是否使用硬件加速
        self.hardwareAcceleration = ConfigItem("Main", "HardwareAcceleration", HARDWARD_ACCELERATION_OPTION)

        self._items = [
            value for value in vars(self).values() if isinstance(value, ConfigItem)
        ]
        self._load()

    def getSttnMaxLoadNum(self):
        return max(self.sttnMaxLoadNum.value, self.sttnNeighborStride.value * self.sttnReferenceLength.value)

    def set(self, item, value, save=True):
        if isinstance(item.default, Enum):
            value = _deserialize_value(value, item.default)
        item.value = value
        if save:
            self.save()

    def _load(self):
        try:
            with open(self.CONFIG_FILE, encoding="utf-8") as f:
                data = json.load(f)
        except (OSError, ValueError):
            return
        if not isinstance(data, dict):
            return
        for item in self._items:
            section = data.get(item.section)
            if not isinstance(section, dict) or item.key not in section:
                continue
            item.value = _deserialize_value(section[item.key], item.default)

    def save(self):
        data = {}
        for item in self._items:
            data.setdefault(item.section, {})[item.key] = _serialize_value(item.value)
        try:
            os.makedirs(os.path.dirname(self.CONFIG_FILE), exist_ok=True)
            with open(self.CONFIG_FILE, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except OSError:
            pass


config = Config()

# 项目的base目录
BASE_DIR = str(Path(os.path.abspath(__file__)).parent)

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
