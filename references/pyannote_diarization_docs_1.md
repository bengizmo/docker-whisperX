# Github Discussion: "Pyannote 3.1.0 still on CPU only? #1563"

## Source: https://github.com/pyannote/pyannote-audio/issues/1563#issuecomment-1833135019

## Reply
It seems that the problem was in my installation.

I used this as requirements.txt (found from here):

gradio==3.38.0
--extra-index-url https://download.pytorch.org/whl/cu113
torch==2.0.1
pyannote-audio==3.1.0
And this for Dockerfile.

FROM nvidia/cuda:11.7.1-cudnn8-devel-ubuntu22.04
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends \
    git \
    git-lfs \
    wget \
    curl \
    # python build dependencies \
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
    # gradio dependencies \
    ffmpeg \
    ca-certificates \
    # fairseq2 dependencies \
    libsndfile-dev && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:${PATH}

WORKDIR ${HOME}

RUN git clone https://github.com/yyuu/pyenv.git .pyenv

ENV PATH=${HOME}/.pyenv/shims:${HOME}/.pyenv/bin:${PATH}

ARG PYTHON_VERSION=3.10
RUN pyenv install ${PYTHON_VERSION} && \
    pyenv global ${PYTHON_VERSION} && \
    pyenv rehash && \
    pip install --no-cache-dir -U pip setuptools wheel

COPY --chown=1000 ./requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir --upgrade -r /tmp/requirements.txt

COPY --chown=1000 . ${HOME}/app
ENV PYTHONPATH=${HOME}/app \
    PYTHONUNBUFFERED=1 \
    GRADIO_ALLOW_FLAGGING=never \
    GRADIO_NUM_PORTS=1 \
    GRADIO_SERVER_NAME=0.0.0.0 \
    GRADIO_THEME=huggingface \
    SYSTEM=spaces \
    GRADIO_SERVER_PORT=7860
EXPOSE 7860
WORKDIR ${HOME}/app
CMD ["python", "app.py"]
I do not know if it is using GPU or not. But without this, It took around 90 minutes to process a 110 minute file. Now, It takes around 1~2 minutes.

this worked for me too.

specifically, what i did was create a requirements.txt file:

with the contents:

gradio==3.38.0
--extra-index-url https://download.pytorch.org/whl/cu113
torch==2.0.1
pyannote-audio==3.1.0
Then install it with pip install -r requirements.txt.

Now, I can run some simple code:

In [1]: from pyannote.audio import Pipeline

In [2]: import torch

In [3]: pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
torchvision is not available - cannot save figures

In [4]: pipeline.to(torch.device("cuda"))
Out[4]: <pyannote.audio.pipelines.speaker_diarization.SpeakerDiarization at 0x7f2ce8f143d0>

In [5]: diarization = pipeline("/tmp/tmphgpfklya.wav")
And $ nvidia-smi -l 1 shows:

image

It took me quite a while to find this solution. Should it be added to README? Why is this version of torch required for the GPU to be properly utilized?



whisper-large-v3-turbo #1099
Open
Open
whisper-large-v3-turbo
#1099
@MohannadEhabBarakat
Description
MohannadEhabBarakat
opened on Apr 6, 2025

Hi thanks for this great work. I'm curious if there is a plan to add turbo models
Activity
IlIlllIIllIIlll
IlIlllIIllIIlll commented on Apr 8, 2025
IlIlllIIllIIlll
on Apr 8, 2025

It seems to have already supported the Turbo model in fastwhisper

Example:

model = whisperx.load_model('large-v3-turbo', device='cuda',

The model in hugging face
eek
eek commented on Apr 9, 2025
eek
on Apr 9, 2025

It also works with whisperx audio_file --model turbo