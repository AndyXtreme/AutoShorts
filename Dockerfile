# =============================================================================
# Stage 1 — builder: compiles decord against CUDA and FFmpeg.
#
# Needs the -devel image for nvcc and the CUDA headers. Nothing from this stage
# ships except the finished decord wheel; the CUDA toolchain alone accounts for
# roughly 14GB and is useless at runtime.
#
# CUDA 12.8 is required for Blackwell (sm_120, RTX 50xx) - the 12.6 nvcc has no
# sm_120 codegen. Deliberately the 2.8.0 tag and not 2.10: the newer tag is
# Ubuntu 24.04 + Python 3.12, which renames the apt packages (t64 transition,
# libgl1-mesa-glx dropped) and makes the cp311 FlashAttention wheel unusable.
# =============================================================================
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    build-essential \
    cmake \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# FFmpeg 4.4.2 — decord requires the 4.x API. The runtime stage installs the
# same conda build, so the ABI decord is linked against matches there.
RUN conda install -y -c conda-forge "ffmpeg=4.4.2" && conda clean -afy

# NV codec headers for NVENC/NVDEC support.
# Mirrored from git.videolan.org, which drops HTTP/2 connections mid-clone.
RUN git -c http.version=HTTP/1.1 clone --depth 1 \
        https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install && \
    cd .. && rm -rf nv-codec-headers

# Driver stubs, needed only to satisfy the linker. At runtime the real
# libraries are injected by the NVIDIA container runtime.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvidia-decode-535 \
    libnvidia-encode-535 \
    && rm -rf /var/lib/apt/lists/* && \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so

# Build decord and package it as a wheel, so the runtime stage installs it
# like any other dependency instead of us guessing which files to copy.
#
# PKG_CONFIG_PATH is required: FFmpeg comes from conda, whose lib/pkgconfig is
# not on the default pkg-config search path, and decord's FFmpeg.cmake aborts
# with "Unable to find FFMPEG automatically" without it.
RUN git clone --recursive https://github.com/dmlc/decord && \
    cd decord && mkdir build && cd build && \
    PKG_CONFIG_PATH=/opt/conda/lib/pkgconfig \
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/conda \
    -DCUDA_nvcuvid_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    make -j$(nproc) && \
    cd ../python && \
    pip install --no-cache-dir wheel && \
    python setup.py bdist_wheel && \
    mkdir -p /wheels && cp dist/*.whl /wheels/


# =============================================================================
# Stage 2 — runtime: everything needed to run, nothing needed to build.
#
# The -runtime base already ships torch 2.8.0+cu128 whose kernels include
# sm_120, so there is no reason to reinstall PyTorch here. That keeps both the
# CUDA toolchain and a second copy of torch out of the image.
# =============================================================================
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive

# Runtime libraries. The font packages are not optional: without them the
# Chromium that PyCaps renders captions in has no usable font, every text node
# measures 0x0, and captions come out invisible while the pipeline still
# reports success. Noto Color Emoji covers the emoji the templates inject.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    sox \
    fontconfig \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto-color-emoji \
    # Playwright browser dependencies
    libnss3 \
    libnspr4 \
    libatk1.0-0 \
    libatk-bridge2.0-0 \
    libcups2 \
    libdrm2 \
    libdbus-1-3 \
    libxkbcommon0 \
    libatspi2.0-0 \
    libxcomposite1 \
    libxdamage1 \
    libxfixes3 \
    libxrandr2 \
    libgbm1 \
    libasound2 \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Same FFmpeg build decord was compiled against in the builder stage.
RUN conda install -y -c conda-forge "ffmpeg=4.4.2" && conda clean -afy

ENV FFMPEG_BINARY=/opt/conda/bin/ffmpeg
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV DECORD_EOF_RETRY_MAX=65536
ENV DECORD_SKIP_TAIL_FRAMES=0
ENV NVIDIA_DRIVER_CAPABILITIES=all

# FlashAttention for the TTS path. Matched to the base image's torch 2.8;
# a mismatched build fails to load with an undefined-symbol ImportError.
# Optional at runtime - tts_generator falls back to eager attention.
RUN pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.16/flash_attn-2.6.3%2Bcu128torch2.8-cp311-cp311-linux_x86_64.whl

# pycaps is installed straight from GitHub, so pip needs git here. Installed
# and removed within the same layer - purging it later would not shrink the
# image, since the files would still live in the earlier layer.
COPY requirements.txt requirements.txt
RUN apt-get update && apt-get install -y --no-install-recommends git && \
    pip install --no-cache-dir -r requirements.txt && \
    apt-get purge -y git && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN playwright install chromium

# decord, built in stage 1
COPY --from=builder /wheels/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# C++ library fix for compatibility
RUN rm -f /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

COPY . .

RUN mkdir -p gameplay generated assets

# Fail the build rather than ship an image with a broken GPU stack.
#
# Only what is checkable without a GPU. Two things are deliberately not
# asserted here, because a build has no device and no driver:
#   - torch.cuda.get_arch_list() returns [] without an initialised CUDA
#     context, so it cannot confirm the sm_120 kernels at build time.
#   - importing decord loads libcuda.so.1, which the NVIDIA container runtime
#     injects only at run time.
# The CUDA version the wheel was built against is the meaningful proxy: 12.8+
# is what carries sm_120. Both are verified for real on first run.
RUN python -c "\
import importlib.util, torch, flash_attn; \
major, minor = (int(p) for p in torch.version.cuda.split('.')[:2]); \
assert (major, minor) >= (12, 8), f'CUDA {torch.version.cuda} has no sm_120 support'; \
assert importlib.util.find_spec('decord'), 'decord not installed'; \
print(f'PyTorch {torch.__version__} (CUDA {torch.version.cuda}), FlashAttention {flash_attn.__version__}, decord present')"

# The sed strips CRLF so the script still runs when the build context comes
# from a Windows checkout.
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Streamlit web UI
EXPOSE 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]
