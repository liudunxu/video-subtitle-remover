FROM python:3.12

RUN --mount=type=cache,target=/root/.cache,sharing=private \
    apt update && \
    apt install -y libgl1-mesa-glx \
        # pyside6
        libegl1 libxkbcommon0 libdbus-1-3 && \
    true

ADD . /vsr
ARG CUDA_VERSION=12.6
ARG HARDWARD_ACCELERATOR="cuda"

# CUDA 12.x (cu126 / cu128). torch 2.8.0 has no cu118 wheel, so CUDA 11.8
# is no longer supported. cu126 is the shared target with OmniVoice.
RUN --mount=type=cache,target=/root/.cache,sharing=private \
    if [ "${HARDWARD_ACCELERATOR}" = "cuda" ] && [ "${CUDA_VERSION}" != "11.8" ]; then \
        pip install paddlepaddle==3.0 && \
        pip install torch==2.8.0 torchvision==0.23.0 --index-url https://download.pytorch.org/whl/cu$(echo ${CUDA_VERSION} | tr -d '.') && \
        pip install -r /vsr/requirements.txt; \
    fi

# 如果是 DirectML 版本，执行 DirectML 特定设置
RUN --mount=type=cache,target=/root/.cache,sharing=private \
    if [ "${HARDWARD_ACCELERATOR}" = "directml" ]; then \
        pip install paddlepaddle==3.0 && \
        pip install torch_directml==0.2.5.dev240914 && \
        pip install -r /vsr/requirements.txt; \
    fi

# 如果是 CPU 版本，执行 CPU 特定设置
RUN --mount=type=cache,target=/root/.cache,sharing=private \
    if [ "${HARDWARD_ACCELERATOR}" = "cpu" ]; then \
        pip install paddlepaddle==3.0 && \
        pip install -r /vsr/requirements.txt && \
        sed -i 's/HARDWARD_ACCELERATION_OPTION *= *.*/HARDWARD_ACCELERATION_OPTION = False/g' /vsr/backend/config.py; \
    fi

ENV LD_LIBRARY_PATH=/usr/local/lib/python3.12/site-packages/nvidia/cudnn/lib/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/cuda_runtime/lib/
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/lib/python3.12/site-packages/nvidia/cuda_nvrtc/lib/
# STTN_DET 画质调优旋钮（含义见 docker/Dockerfile.gpu 注释）：crop 带高
# 余量与模型输入尺寸；非法输入尺寸会被忽略并回退 432x240。
ENV VSR_STTN_BAND_MARGIN=40
ENV VSR_STTN_INPUT_WIDTH=432
ENV VSR_STTN_INPUT_HEIGHT=240
WORKDIR /vsr
CMD ["python", "/vsr/backend/main.py"]