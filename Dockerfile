# Base image: CUDA 12.8 toolchain — required for Blackwell (sm_120, RTX 50xx);
# the 12.6 nvcc has no sm_120 codegen, which breaks the decord build below.
# Deliberately 2.8.0 and not 2.10.0: the 2.10 tag is Ubuntu 24.04 + Python 3.12,
# which renames the apt packages in step 1 (t64 transition, libgl1-mesa-glx dropped)
# and makes the cp311 FlashAttention wheel in step 7 uninstallable.
# Torch itself is upgraded to 2.10 from the cu128 index in step 6 anyway.
FROM pytorch/pytorch:2.8.0-cuda12.8-cudnn9-devel

ENV DEBIAN_FRONTEND=noninteractive

# 1. Install system tools and dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    build-essential \
    cmake \
    pkg-config \
    wget \
    sox \
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
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# 2. Install FFmpeg 4.4.2 (required for Decord compatibility)
RUN conda update -n base -c defaults conda -y && \
    conda install -y -c conda-forge "ffmpeg=4.4.2"

# 3. Set up environment
ENV FFMPEG_BINARY=/opt/conda/bin/ffmpeg
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV DECORD_EOF_RETRY_MAX=65536
ENV DECORD_SKIP_TAIL_FRAMES=0
ENV NVIDIA_DRIVER_CAPABILITIES=all

# 4. Install NV Codec Headers for NVENC support
# Mirrored from git.videolan.org, which drops HTTP/2 connections mid-clone.
# HTTP/1.1 + shallow clone keeps this step reliable.
RUN git -c http.version=HTTP/1.1 clone --depth 1 \
        https://github.com/FFmpeg/nv-codec-headers.git && \
    cd nv-codec-headers && \
    make install && \
    cd .. && rm -rf nv-codec-headers

# 5. Install NVIDIA driver libraries for linking
RUN apt-get update && apt-get install -y --no-install-recommends \
    libnvidia-decode-535 \
    libnvidia-encode-535 \
    && rm -rf /var/lib/apt/lists/*

# Create symlinks for CMake and Python
RUN ln -sf /usr/lib/x86_64-linux-gnu/libnvcuvid.so.1 /usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-encode.so.1 /usr/lib/x86_64-linux-gnu/libnvidia-encode.so

# 6. PyTorch from the cu128 wheels (cu126 wheels ship no sm_120 kernels).
# Versions are pinned deliberately: the cu128 index also carries 2.11, and an
# unpinned install resolves to it, which then fails to load the FlashAttention
# wheel below (built as cu128torch2.10) with an undefined-symbol ImportError.
RUN pip install --no-cache-dir \
        torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
        --index-url https://download.pytorch.org/whl/cu128 && \
    pip install --no-cache-dir torchcodec

# 7. Install FlashAttention 2 (prebuilt wheel for torch 2.10)
# Using manylinux wheel for broader compatibility
RUN pip install --no-cache-dir \
    https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.7.12/flash_attn-2.6.3+cu128torch2.10-cp311-cp311-linux_x86_64.whl

# 8. Install Python dependencies
COPY requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# 9. Install Playwright browsers for PyCaps
RUN playwright install chromium

# 9b. Fonts for the Chromium that PyCaps renders captions in.
# The base image ships no /usr/share/fonts, no /etc/fonts and no fontconfig,
# so Chromium has zero usable fonts: every text node measures 0x0, each word
# screenshot collapses to its CSS padding, and the captions come out invisible
# while the pipeline still reports success. The conda fonts in /opt/conda/fonts
# are not visible to Chromium — it uses the system fontconfig.
# Noto Color Emoji is needed for the emoji the caption templates inject.
RUN apt-get update && apt-get install -y --no-install-recommends \
    fontconfig \
    fonts-liberation \
    fonts-dejavu-core \
    fonts-noto-color-emoji \
    && fc-cache -f \
    && rm -rf /var/lib/apt/lists/*

# 10. Build Decord with CUDA support
# FFmpeg comes from conda, whose lib/pkgconfig is not on the default pkg-config
# search path — without this, decord's FFmpeg.cmake aborts with "Unable to find
# FFMPEG automatically". Set inline rather than as a global ENV so the cached
# layers above (FlashAttention, requirements, Playwright) stay valid.
RUN git clone --recursive https://github.com/dmlc/decord && \
    cd decord && \
    mkdir build && cd build && \
    PKG_CONFIG_PATH=/opt/conda/lib/pkgconfig \
    cmake .. -DUSE_CUDA=ON -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_PREFIX_PATH=/opt/conda \
    -DCUDA_nvcuvid_LIBRARY=/usr/lib/x86_64-linux-gnu/libnvcuvid.so && \
    make -j$(nproc) && \
    cd ../python && \
    python setup.py install && \
    cd /app && rm -rf decord

# 11. C++ library fix for compatibility
RUN rm -f /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

# 12. Cleanup build-time NVIDIA libraries (use host-mounted ones at runtime)
RUN apt-get purge -y libnvidia-decode-535 libnvidia-encode-535 && \
    rm -f /usr/lib/x86_64-linux-gnu/libnvcuvid.so /usr/lib/x86_64-linux-gnu/libnvidia-encode.so && \
    apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# 13. Copy application code
COPY . .

# Create directories for input/output
RUN mkdir -p gameplay generated assets

# 14. Pre-download TTS model (optional - comment out to download on first run)
# RUN python -c "from src.tts_generator import download_model; download_model()"

# Verify installations
RUN python -c "import torch; import flash_attn; print(f'PyTorch {torch.__version__}, FlashAttention {flash_attn.__version__}')"

# 15. Entrypoint. The sed strips CRLF so the script still runs when the build
# context comes from a Windows checkout.
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Streamlit web UI
EXPOSE 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]