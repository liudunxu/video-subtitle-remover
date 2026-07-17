import json
import os
import stat
import subprocess

import platform
from .common_tools import merge_big_file_if_not_exists
from backend.config import BASE_DIR

class FFmpegCLI:
    
    """
    进程管理器类，用于管理子进程的生命周期
    使用弱引用避免内存泄漏
    """
    _instance = None
    
    @classmethod
    def instance(cls):
        """单例模式获取实例"""
        if cls._instance is None:
            cls._instance = FFmpegCLI()
        return cls._instance
    
    def __init__(self):
        os.chmod(self.ffmpeg_path, stat.S_IRWXU + stat.S_IRWXG + stat.S_IRWXO)
        
    @property
    def ffmpeg_path(self):
        system = platform.system()
        if system == "Windows":
            ffmpeg_dir = os.path.join(BASE_DIR, 'ffmpeg', 'win_x64')
            merge_big_file_if_not_exists(ffmpeg_dir, 'ffmpeg.exe')
            return os.path.join(ffmpeg_dir, 'ffmpeg.exe')
        elif system == "Linux":
            return os.path.join(BASE_DIR, 'ffmpeg',  'linux_x64', 'ffmpeg')
        else:
            return os.path.join(BASE_DIR, 'ffmpeg', 'macos', 'ffmpeg')

    @property
    def ffprobe_path(self):
        """优先使用与 vendored ffmpeg 同目录的 ffprobe，否则回退 PATH 上的 ffprobe。"""
        sibling = os.path.join(
            os.path.dirname(self.ffmpeg_path),
            'ffprobe.exe' if platform.system() == "Windows" else 'ffprobe',
        )
        if os.path.isfile(sibling):
            return sibling
        return 'ffprobe'


def probe_audio_codec(video_path):
    """探测视频首个音轨的 codec 名（如 'aac' / 'mp3'）。

    Returns:
        codec 名字符串；源视频没有音轨时返回 None；探测失败（ffprobe
        缺失/超时/输出异常）返回 'unknown'，调用方应按未知 codec 走
        安全的转码路径而不是静默放弃音轨。
    """
    try:
        output = subprocess.check_output(
            [
                FFmpegCLI.instance().ffprobe_path,
                '-v', 'error',
                '-select_streams', 'a:0',
                '-show_entries', 'stream=codec_name',
                '-of', 'json',
                str(video_path),
            ],
            stdin=subprocess.DEVNULL,
            timeout=60,
        )
        streams = json.loads(output.decode('utf-8')).get('streams') or []
        if not streams:
            return None
        return streams[0].get('codec_name') or 'unknown'
    except Exception:
        return 'unknown'