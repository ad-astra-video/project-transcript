FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/home/user/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/:${LD_LIBRARY_PATH}"

# Default environment variables
ENV WHISPER_MODEL=large
ENV WHISPER_DEVICE=cuda
ENV SUBSCRIBE_URL=http://172.17.0.1:3389/sample
ENV PUBLISH_URL=http://172.17.0.1:3389/publish
ENV SEGMENT_DURATION=3.0
ENV MODEL_DIR=/models

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.10 \
    python3-pip \
    python3-dev \
    git \
    wget \
    build-essential \
    yasm \
    pkg-config \
    libass-dev \
    libfreetype6-dev \
    libsdl2-dev \
    libva-dev \
    libvdpau-dev \
    libvorbis-dev \
    libxcb1-dev \
    libxcb-shm0-dev \
    libxcb-xfixes0-dev \
    texinfo \
    zlib1g-dev \
    nasm \
    cmake \
    libunistring-dev \
    libaom-dev \
    libdav1d-dev \
    && rm -rf /var/lib/apt/lists/*

# Compile FFmpeg from source with subtitle filters
WORKDIR /tmp/ffmpeg_sources

# Download and extract FFmpeg source
RUN wget -O ffmpeg.tar.bz2 https://ffmpeg.org/releases/ffmpeg-6.0.tar.bz2 && \
    tar xjf ffmpeg.tar.bz2 && \
    rm ffmpeg.tar.bz2 && \
    cd ffmpeg-6.0 && \
    ./configure \
      --prefix=/usr/local \
      --pkg-config-flags="--static" \
      --enable-gpl \
      --enable-nonfree \
      --enable-libass \
      --enable-libfreetype \
      --enable-libvorbis \
      --enable-postproc \
      --enable-avfilter \
      --enable-libxcb \
      --enable-version3 \
      --disable-debug \
      --enable-shared \
      --disable-doc \
      --disable-htmlpages \
      --disable-manpages \
      --disable-podpages \
      --disable-txtpages \
      --enable-filter=subtitles \
      --enable-filter=drawtext && \
    make -j$(nproc) && \
    make install && \
    ldconfig && \
    cd .. && \
    rm -rf ffmpeg-6.0

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir --upgrade pip && \
    pip3 install torch==2.7.0+cu128 torchaudio==2.7.0+cu128 --index-url https://download.pytorch.org/whl/cu128 && \
    pip3 install --no-cache-dir -r requirements.txt --ignore-installed torch torchaudio

# Verify CUDA installation
RUN python3 -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('CUDA version:', torch.version.cuda); print('Device count:', torch.cuda.device_count())"

# Copy application code
COPY . .

# create it so permissions are correct
RUN mkdir -p ${MODEL_DIR}

# Set default command
CMD ["uvicorn", "src.server.app:app", "--host", "0.0.0.0", "--port", "8000"]