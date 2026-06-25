import os
import queue
import subprocess
import threading

import cv2
import numpy as np

from .ffmpeg_cli import FFmpegCLI


# 单帧解码正常 <100ms（4K 也 <500ms），EOF 挂起时 cap.read() 永不返回。
# 15s 是 30~100 倍余量，绝不在健康源上误触发，仅在 EOF 挂起时兜底打破死锁。
_EOF_TIMEOUT = 15


class FramePrefetcher:
    """
    后台线程预解码视频帧，使 I/O 与模型推理重叠。
    接口兼容 cv2.VideoCapture（read/release）。
    """

    def __init__(self, video_cap, buffer_size=10):
        self.cap = video_cap
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._stopped = False
        self._read_count = 0
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            self._buffer.put((ret, frame))
            if not ret:
                break
            self._read_count += 1

    def read(self):
        """读取下一帧，接口与 cv2.VideoCapture.read() 一致。

        带超时保护：cv2.VideoCapture.read() 在某些源（moov 声称的帧数
        大于实际可解码帧数、VFR、损坏尾部）的 EOF 处会挂起不返回，导致
        后台 _read_loop 永远不会把 (False, None) 哨兵入队，消费方在
        queue.get() 上永久死锁（表现为进度到 100% 后完全无活动）。
        超时后当作 EOF 返回，打破死锁；正常解码单帧远低于超时阈值，不受影响。
        """
        try:
            return self._buffer.get(timeout=_EOF_TIMEOUT)
        except queue.Empty:
            print(
                f"[phase] prefetcher: no frame for {_EOF_TIMEOUT}s after "
                f"{self._read_count} frames (cap.read() likely hung at EOF); "
                f"treating as EOF to break deadlock",
                flush=True,
            )
            return (False, None)

    def get(self, propId):
        return self.cap.get(propId)

    def stop(self):
        """停止预读取，不释放底层 video_cap。"""
        self._stopped = True
        try:
            while not self._buffer.empty():
                self._buffer.get_nowait()
        except queue.Empty:
            pass
        # 读线程若仍存活，说明它卡在 cap.read()（EOF 挂起）不会退出；
        # join 只会白等。它是 daemon，随进程退出即可，cap.release()
        # 在线程阻塞时调用也是安全的（运行时已验证）。
        if not self._thread.is_alive():
            self._thread.join(timeout=5)

    def release(self):
        self.stop()
        self.cap.release()


class FFmpegVideoWriter:
    """
    通过 FFmpeg 管道写入帧，使用 libx264 编码。
    接口兼容 cv2.VideoWriter（write/release）。
    """

    def __init__(self, output_path, fps, size):
        w, h = size
        cmd = [
            FFmpegCLI.instance().ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'fast',
            '-loglevel', 'error',
            output_path
        ]
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        """写入一帧（numpy BGR 数组）。"""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def release(self):
        """关闭管道并等待编码完成。"""
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)
