FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PATH="/usr/local/cuda/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda/lib64:/usr/local/cuda/extras/CUPTI/lib64:/home/user/.local/lib/python3.10/site-packages/nvidia/cudnn/lib/:${LD_LIBRARY_PATH}"

# Default environment variables
ENV WHISPER_MODEL=large
ENV WHISPER_DEVICE=cuda
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
    libx264-dev \
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

# Create app directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install uv

RUN uv pip install -r requirements.txt --system

COPY . .

# create it so permissions are correct
RUN mkdir -p ${MODEL_DIR}

EXPOSE 8000

# Set default command
CMD ["python3", "-m","src.pipeline.main"]