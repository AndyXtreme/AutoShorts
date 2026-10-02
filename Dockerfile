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
# libgl1-mesa-glx dropped).
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

# A second, modern FFmpeg alongside the conda one. It is a static build, so
# it shares no libraries with the 4.4.2 above and cannot disturb the ABI
# decord is linked against - decord keeps the old libraries, everything that
# shells out to ffmpeg/ffprobe gets this one.
#
# The conda build is from 2021 and has neither libdav1d nor av1_nvenc, which
# makes AV1 unusable in both directions: decoding fails outright ("Cannot
# determine format of input stream 0:0 after EOF") and there is no AV1 encoder
# to write the finished shorts with. Pinned to a dated release rather than the
# rolling "latest" tag so the image stays reproducible - and to a month-end
# one: BtbN deletes the daily autobuilds after about two weeks (the 2026-09-09
# pin was gone by October and the build failed with a 404), while the last
# build of each month is kept.
ARG FFMPEG_BUILD=autobuild-2026-08-31-13-27
ARG FFMPEG_ARCHIVE=ffmpeg-n9.0.1-11-ge47273f4d9-linux64-gpl-9.0
RUN apt-get update && apt-get install -y --no-install-recommends curl xz-utils && \
    curl -fsSL -o /tmp/ffmpeg.tar.xz \
        "https://github.com/BtbN/FFmpeg-Builds/releases/download/${FFMPEG_BUILD}/${FFMPEG_ARCHIVE}.tar.xz" && \
    tar -xJf /tmp/ffmpeg.tar.xz -C /tmp && \
    mv "/tmp/${FFMPEG_ARCHIVE}/bin/ffmpeg" /usr/local/bin/ffmpeg && \
    mv "/tmp/${FFMPEG_ARCHIVE}/bin/ffprobe" /usr/local/bin/ffprobe && \
    chmod +x /usr/local/bin/ffmpeg /usr/local/bin/ffprobe && \
    rm -rf /tmp/ffmpeg.tar.xz "/tmp/${FFMPEG_ARCHIVE}" && \
    apt-get purge -y curl xz-utils && apt-get autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# /opt/conda/bin comes first in the base image's PATH, so the new binaries
# have to be put in front of it explicitly.
ENV PATH=/usr/local/bin:$PATH
ENV FFMPEG_BINARY=/usr/local/bin/ffmpeg
ENV FFPROBE_BINARY=/usr/local/bin/ffprobe
ENV LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64:$LD_LIBRARY_PATH
ENV NVIDIA_DRIVER_CAPABILITIES=all

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

# A C compiler, for Triton. It builds a small helper module at runtime, and
# without a compiler Whisper's CUDA DTW - which computes the word timings the
# captions are highlighted by - throws and falls back to a CPU implementation
# (0.75s per call instead of 0.001s), warning about a "missing CUDA toolkit".
# The toolkit is not the issue; ptxas ships with Triton. Costs ~200MB.
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libc6-dev \
    && rm -rf /var/lib/apt/lists/*

# C++ library fix for compatibility
RUN rm -f /opt/conda/lib/libstdc++.so.6 && \
    ln -s /usr/lib/x86_64-linux-gnu/libstdc++.so.6 /opt/conda/lib/libstdc++.so.6

COPY . .

RUN mkdir -p gameplay generated

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
import importlib.util, torch; \
major, minor = (int(p) for p in torch.version.cuda.split('.')[:2]); \
assert (major, minor) >= (12, 8), f'CUDA {torch.version.cuda} has no sm_120 support'; \
assert importlib.util.find_spec('decord'), 'decord not installed'; \
print(f'PyTorch {torch.__version__} (CUDA {torch.version.cuda}), decord present')"

# AV1 needs a decoder to read the source and an encoder to write the short,
# and neither exists in the conda FFmpeg. Assert the new one is what PATH
# resolves to and that it carries both - the alternative is finding out
# halfway through a render.
RUN ffmpeg -hide_banner -decoders | grep -q libdav1d && \
    ffmpeg -hide_banner -encoders | grep -q av1_nvenc && \
    ffmpeg -hide_banner -encoders | grep -q hevc_nvenc && \
    echo "FFmpeg $(ffmpeg -version | head -1 | cut -d' ' -f3): AV1 decode and encode present"

# The sed strips CRLF so the script still runs when the build context comes
# from a Windows checkout.
RUN sed -i 's/\r$//' /app/docker-entrypoint.sh && chmod +x /app/docker-entrypoint.sh

# Streamlit web UI
EXPOSE 8501

ENTRYPOINT ["/app/docker-entrypoint.sh"]
