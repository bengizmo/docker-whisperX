# syntax=docker/dockerfile:1

FROM nvidia/cuda:12.6.0-runtime-ubuntu22.04 as python

ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    git-lfs \
    wget \
    curl \
    ffmpeg \
    build-essential \
    libssl-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    libncursesw5-dev \
    xz-utils \
    tk-dev \
    libxml2-dev \
    libxmlsec1-dev \
    libffi-dev \
    liblzma-dev \
    libsndfile-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

FROM python as prepare_base_amd64

WORKDIR /tmp

# Add NVIDIA repository for CUDA and install required packages
RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb && \
    dpkg -i cuda-keyring_1.1-1_all.deb && \
    apt-get update && \
    apt-get install -y --no-install-recommends --allow-change-held-packages \
        cuda-nvrtc-12-6 \
        cuda-cudart-12-6 \
        cuda-cupti-12-6 \
        libcublas-12-6 \
        cuda-compat-12-6 \
        libnccl2 \
        libcudnn9-cuda-12=9.8.0.87-1 \
        libcudnn9-dev-cuda-12=9.8.0.87-1 \
        cuda-libraries-12-6 \
        libcusparse-12-6 \
        libcusparselt0 \
        python3-pip && \
    rm -f cuda-keyring_1.1-1_all.deb && \
    rm -f /usr/lib/x86_64-linux-gnu/libcudnn.so && \
    ln -s /usr/lib/x86_64-linux-gnu/libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so && \
    ldconfig && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Set CUDA and CUDNN environment variables
ENV PATH="/usr/local/cuda-12.6/bin:${PATH}"
ENV LD_LIBRARY_PATH="/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64:${LD_LIBRARY_PATH}"

FROM prepare_base_amd64 as prepare_build_amd64

WORKDIR /app

# Copy uv from ghcr.io/astral-sh/uv:latest
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Set up uv environment
ENV UV_PROJECT_ENVIRONMENT=/venv
ENV VIRTUAL_ENV=/venv
ENV UV_LINK_MODE=copy
ENV UV_PYTHON_DOWNLOADS=0
ENV UV_INDEX=https://download.pytorch.org/whl/cu126

# Install big dependencies separately for layer caching
RUN --mount=type=cache,id=uv-amd64,sharing=locked,target=/root/.cache/uv \
    uv venv --python python3.11 --system-site-packages /venv && \
    uv pip install --no-deps \
    torch==2.6.0+cu126 \
    torchaudio==2.6.0+cu126 \
    pyannote.audio==3.3.2

# Install whisperX dependencies
RUN --mount=type=cache,id=uv-amd64,sharing=locked,target=/root/.cache/uv \
    --mount=type=bind,source=whisperX/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=whisperX/uv.lock,target=uv.lock \
    uv sync --frozen --no-dev --no-install-project --no-editable

# Install whisperX project
# Removed COPY and RUN for local whisperX submodule installation

FROM prepare_build_amd64 as build

FROM python as no_model

# Copy CUDA and CUDNN libraries from build stage
COPY --from=build /usr/lib/x86_64-linux-gnu/libcudnn* /usr/lib/x86_64-linux-gnu/libcusparse* /usr/lib/x86_64-linux-gnu/
COPY --from=build /usr/local/cuda-12.6 /usr/local/cuda-12.6
RUN ln -sf /usr/lib/x86_64-linux-gnu/libcudnn.so.9 /usr/lib/x86_64-linux-gnu/libcudnn.so && \
    ldconfig

# Create non-root user
RUN groupadd -g 1001 whisperx && \
    useradd -l -u 1001 -g whisperx -m -s /bin/sh -N whisperx

# Create directories
RUN install -d -m 775 -o whisperx -g 0 /licenses && \
    install -d -m 775 -o whisperx -g 0 /root && \
    install -d -m 775 -o whisperx -g 0 /.cache

# Copy ffmpeg and dumb-init
COPY --from=ghcr.io/jim60105/static-ffmpeg-upx:7.1 /ffmpeg /usr/local/bin/
COPY --from=ghcr.io/jim60105/static-ffmpeg-upx:7.1 /dumb-init /usr/local/bin/

# Copy licenses
COPY --chown=whisperx:0 --chmod=775 LICENSE /licenses/LICENSE
COPY --chown=whisperx:0 --chmod=775 whisperX/LICENSE /licenses/whisperX.LICENSE

# Copy venv
COPY --chown=whisperx:0 --chmod=775 --from=build /venv /venv

# Install and test whisperx in the final stage
RUN /venv/bin/python3 -m pip install whisperx==3.3.2 && \
    /venv/bin/python3 -c 'import whisperx; print("whisperX import successful")' && \
    /venv/bin/whisperx -h

WORKDIR /app

# Set environment variables
ENV HF_HOME=/tmp/cache/huggingface \
    TORCH_HOME=/tmp/cache/torch \
    TRANSFORMERS_CACHE=/tmp/cache/huggingface/transformers \
    PYTHONWARNINGS="ignore::UserWarning" \
    TORCH_WARN_ONCE=1 \
    PYTORCH_DISABLE_VERSION_WARNING=1 \
    CUDA_LAUNCH_BLOCKING=1 \
    PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512 \
    PYTORCH_ENABLE_MPS_FALLBACK=1 \
    PATH="/venv/bin:$PATH" \
    PYTHONPATH="/venv/lib/python3.11/site-packages" \
    LD_LIBRARY_PATH="/usr/lib/x86_64-linux-gnu:/usr/local/cuda-12.6/lib64:/usr/local/cuda/lib64${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

USER whisperx

ENTRYPOINT ["/usr/local/bin/dumb-init", "--", "/venv/bin/whisperx"]