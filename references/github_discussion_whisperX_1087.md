# Github Discussion: "Its a mess. (Sorted Out) #1087"

## Source: https://github.com/m-bain/whisperX/issues/1087

## Reply
nuxil
opened on Mar 24, 2025 · edited by nuxil

Its a mess alright.
install instructions does not work and causes all sort of incomplete or corrupt install where only parts of whisperX works or dont work at all because
stupid python libs.

Please sum up all needed deps to run this whisperX
I mean all of them, so i can freez the packages on the system and only get what is required.
Activity
ysdede
ysdede commented on Mar 24, 2025
ysdede
on Mar 24, 2025 · edited by ysdede

Hey,

I use WhisperX almost every day for benchmarking and label creation. Previously, I installed the latest version directly from the source using:

pip install git+https://github.com/m-bain/whisperx.git

along with some additional workarounds mentioned here: #1027 (comment)

Yesterday, it stopped working in my Colab environment after the latest UV-related commit, so I decided to lock the working library versions.

I log environment details with my results and used them to fix the versions:

{'torch': '2.5.1+cu121', 'torchaudio': '2.5.1+cu121', 'ctranslate2': '4.4.0', 'faster-whisper': '1.1.0', 'whisperx': '3.3.1'}

# Debian packages: {'libcudnn*': '8.9.2.26-1+cuda12.1, 9.2.1.18-1'}

Below is my installation cell from Colab. I launch new Colab sessions and install WhisperX from scratch using this setup, and it works. I hope this helps.

!pip uninstall torch torchvision torchaudio -y

# Workaround from: https://github.com/m-bain/whisperX/issues/1027#issuecomment-2627525081
!pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# WhisperX-related packages:
!pip install ctranslate2==4.4.0
!pip install faster-whisper==1.1.0
# !pip install git+https://github.com/m-bain/whisperx.git
!pip install whisperx==3.3.1

!apt-get update
!apt-get install libcudnn8=8.9.2.26-1+cuda12.1
!apt-get install libcudnn8-dev=8.9.2.26-1+cuda12.1

!python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True"

I hope the developers clarify whether these workarounds are still necessary. @m-bain @Barabazs
dklassic
dklassic commented on Mar 26, 2025
dklassic
on Mar 26, 2025 · edited by dklassic

@ysdede just want to say your version fix works flawlessly and you really saved me here, my WhisperX colab notebook has been dead for months. I was hoping to fix it last week but I can't find the proper dependency and steps needed to resolve the issue for days.
nuxil
nuxil commented on Mar 27, 2025
nuxil
on Mar 27, 2025 · edited by nuxil
Author

Well i appreciate you taking your time and trying to help.
but it would be nice to have a working install instructions for windows.
i am not using linux so these apt-get command does jack for me.

i did do the pip commands. no go. I even tried to created a new enviorment in miniconda and tried to do a fresh install there.
Still no go.

If i just try to run whisper from the cmd. like: whisperx Voice-Referances\vocal1_normalized.wav
i get this output:

(wx) D:\Coding\python\tts>whisperx Voice-Referances\vocal1_normalized.wav
INFO:speechbrain.utils.quirks:Applied quirks (see `speechbrain.utils.quirks`): [disable_jit_profiling, allow_tf32]
INFO:speechbrain.utils.quirks:Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated list): []
No language specified, language will be first be detected for each audio file (increases inference time).
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.5.1. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint C:\Users\me\miniconda3\envs\wx\lib\site-packages\whisperx\assets\pytorch_model.bin`
Model was trained with pyannote.audio 0.0.1, yours is 3.3.2. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.5.1+cu121. Bad things might happen unless you revert torch to 1.x.
>>Performing transcription...
C:\Users\me\miniconda3\envs\wx\lib\site-packages\pyannote\audio\utils\reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.
It can be re-enabled by calling
   >>> import torch
   >>> torch.backends.cuda.matmul.allow_tf32 = True
   >>> torch.backends.cudnn.allow_tf32 = True
See https://github.com/pyannote/pyannote-audio/issues/1370 for more details.

  warnings.warn(
Warning: audio is shorter than 30s, language detection may be inaccurate.
Could not locate cudnn_ops_infer64_8.dll. Please make sure it is in your library path!

i have the CUDA sdk installed on my system and i have setup the system enviorment paths for it.
CUDA_PATH
CUDA_PATH_V12_8
im not sure if this is the cause. mismatch of cuda version. Im at a loss here.

python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True"
umm so?
python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True" && python -m whisperx audio.wav

looks like there is some issues with the cuda libs.
not sure how to resolve this.

Currently i have moved on to MFA, but i would like to have whisperX to work.
SeBL4RD
SeBL4RD commented on Mar 30, 2025
SeBL4RD
on Mar 30, 2025 · edited by SeBL4RD

SOLUTION FOR WINDOWS :
(Activate your venv to avoid dep conflict)
pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
pip install ctranslate2==4.4.0 faster-whisper==1.1.0 whisperx==3.3.1

download one of the archives for CUDA 12 :
https://github.com/Purfview/whisper-standalone-win/releases/tag/libs

put the missing dlls in :
yourvenv\Lib\site-packages\torch\lib\here

It works !

PS: You don't need to install CUDA or libcudnn or anything else on your host machine. All dependencies are in your venv with these commands.
rasber-3
rasber-3 commented on Mar 31, 2025
rasber-3
on Mar 31, 2025

Yeah it's a mess alright. Been trying to get it running for hours, and i've followed every instructions i could find and with help from various chatbots and i cannot get past this error.

Unable to load any of {libcudnn_ops.so.9.1.0, libcudnn_ops.so.9.1, libcudnn_ops.so.9, libcudnn_ops.so}
RomainGehrig
mentioned this on Apr 1, 2025

    Unable to load any of {libcudnn_cnn.so.9.1.0, libcudnn_cnn.so.9.1, libcudnn_cnn.so.9, libcudnn_cnn.so} jim60105/docker-whisperX#67

tzvc
tzvc commented on Apr 1, 2025
tzvc
on Apr 1, 2025

Same issue here. If anyone has managed to get it working in Docker, I'd love to see the config so I can reproduce
nuxil
nuxil commented on Apr 1, 2025
nuxil
on Apr 1, 2025 · edited by nuxil
Author

    SOLUTION FOR WINDOWS : (Activate your venv to avoid dep conflict) pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121 pip install ctranslate2==4.4.0 faster-whisper==1.1.0 whisperx==3.3.1

    download one of the archives for CUDA 12 : https://github.com/Purfview/whisper-standalone-win/releases/tag/libs

    put the missing dlls in : yourvenv\Lib\site-packages\torch\lib\here

    It works !

    PS: You don't need to install CUDA or libcudnn or anything else on your host machine. All dependencies are in your venv with these commands.

Hi..

I gave it another shot and followed your steps.
But it still gave me this error:

Could not locate cudnn_ops_infer64_8.dll. Please make sure it is in your library path!			

When i unziped the cuBLAS.and.cuDNN_CUDA12_win_v3.7z i saw no cudnn_ops_infer64_8.dll in there.
So i tried to download the CUDA11 zip instead. and wouldnt you know it, there it was, the missing cudnn_ops_infer64_8.dll.

Im not sure if the CUDA12 is required, But i did update the dll files before adding in the cudnn_ops_infer64_8.dll from the Cuda 11 Zip

So you need do download both Cuda 11 and Cuda 12:

    cuBLAS.and.cuDNN_CUDA11_win_v2.7z
    cuBLAS.and.cuDNN_CUDA12_win_v3.7z

I Summed up my steps for a fresh envuiourment using miniconda.

Windows With Miniconda,
Open Anaconda Prompt and enter.

1 * conda create -n whisperx_env python=3.10 -y
2 * conda activate whisperx_env

3 * pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
4 * pip install ctranslate2==4.4.0 faster-whisper==1.1.0 whisperx==3.3.1

Download one of the archives for CUDA 11 & 12: https://github.com/Purfview/whisper-standalone-win/releases/tag/libs
 
CuBLAS.and.cuDNN_CUDA11_win_v2.7z 
cuBLAS.and.cuDNN_CUDA12_win_v3.7z

Extract CuBLAS.and.cuDNN_CUDA11_win_v2.7z 
Extract cuBLAS.and.cuDNN_CUDA12_win_v3.7z

Put the missing dlls in : Your-Env\Lib\site-packages\torch\lib\here

In my case:
C:\Users\%USERNAME%\miniconda3\envs\whisperx\Lib\site-packages\torch\lib\here


It works Now!

@rasber-3 @tzvc maybe you need to download the linux drivers from the link and do the same i had to do on windows?
nuxil
changed the title [-]Its a mess.[/-] [+]Its a mess. (Sorted Out)[/+] on Apr 1, 2025
tzvc
tzvc commented on Apr 2, 2025
tzvc
on Apr 2, 2025

Still trying to implement this fix in Docker, I've implemented @ysdede's fix but I still get the error.

Here's my current Dockerfile

# Start from an NVIDIA CUDA base image (supports GPU with CUDA 12.1)
FROM nvidia/cuda:12.1.1-cudnn8-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH=${CUDA_HOME}/bin:${PATH}
ENV LD_LIBRARY_PATH=${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}
ENV NVIDIA_VISIBLE_DEVICES=all
# Increase max split size and add garbage collection settings
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512,garbage_collection_threshold:0.8
# Add CUDA debugging
ENV CUDA_LAUNCH_BLOCKING=1
# Add PyTorch debugging
ENV TORCH_SHOW_CPP_STACKTRACES=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    wget \
    ffmpeg \
    git \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    pkg-config \
    libsndfile1 \
    cuda-command-line-tools-12-1 \
    nvidia-cuda-toolkit \
    && rm -rf /var/lib/apt/lists/* \
    && ln -sf /usr/bin/python3 /usr/bin/python \
    && python -m pip install --upgrade pip setuptools wheel

RUN pip uninstall torch torchvision torchaudio -y

RUN pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121

# Step 1: Install ctranslate2 dependency
RUN pip install --no-cache-dir ctranslate2==4.4.0

# Step 2: Install faster-whisper
RUN pip install --no-cache-dir faster-whisper==1.1.0

# Step 3: Install whisperx
RUN pip install --no-cache-dir whisperx==3.3.1

# Step 4: Install runpod dependency
RUN pip install --no-cache-dir runpod==1.7.7

# Step 1: Update package lists
RUN apt-get update

# Step 2: Install cuDNN libraries
RUN apt-get install -y --allow-change-held-packages libcudnn8=8.9.2.26-1+cuda12.1
RUN apt-get install -y --allow-change-held-packages libcudnn8-dev=8.9.2.26-1+cuda12.1

# Step 3: Configure PyTorch to use TF32
RUN python -c "import torch; torch.backends.cuda.matmul.allow_tf32 = True; torch.backends.cudnn.allow_tf32 = True"

# Copy scripts that download/cache models and run them
COPY build.sh .
RUN chmod +x build.sh && ./build.sh

# Finally, copy application code
COPY . .

CMD ["python", "-u", "handler.py"]

ysdede
ysdede commented on Apr 2, 2025
ysdede
on Apr 2, 2025 · edited by ysdede

I have a Dockerfile with WhisperX that I hadn’t touched for two weeks since last changes. I updated it by specifying a fixed version for pip install whisperx, and it worked. I also realized there’s no need to install ctranslate2 and faster-whisper separately before WhisperX—version 3.3.1 handles them perfectly. Additionally, we can skip the uninstall torch line.

The main differences I noticed are the base Docker images and the fact that I use a non-root user instead of running as root. This setup was aimed at making the Docker image compatible with Hugging Face Spaces.

It’s not a perfect Dockerfile—I’m still new to Docker—but it might be helpful:

FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y \
    python3-pip \
    build-essential \
    git \
    net-tools \
    wget \
    nano \
    htop \
    && rm -rf /var/lib/apt/lists/*

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user
ENV PATH="/home/user/.local/bin:$PATH"
ENV PIP_CACHE_DIR=/home/user/.cache/pip

WORKDIR $HOME/app

RUN pip install --upgrade pip

COPY --chown=user ./pyproject.toml pyproject.toml

RUN pip install "setuptools>=64.0.0" wheel "setuptools_scm>=8.0" --upgrade
RUN pip install torch==2.5.1+cu121 torchvision==0.20.1 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

# RUN pip install git+https://github.com/m-bain/whisperx.git
RUN pip install whisperx==3.3.1

COPY --chown=user ./requirements.txt requirements.txt

RUN pip install -r requirements.txt

COPY --chown=user . $HOME/app
COPY --chown=user webapp/ $HOME/app/webapp/

# for debugging
# CMD ["tail", "-f", "/dev/null"]

RUN pip install -e .

CMD ["uvicorn", "--factory", "whisperx_server.main:create_app", "--host", "0.0.0.0", "--port", "8000", "--ws", "websockets", "--timeout-keep-alive", "60"]

nuxil
nuxil commented on Apr 4, 2025
nuxil
on Apr 4, 2025
Author

No ideas how docker works, never used it.
But you could try miniconda instead. its availeble for linux as well.
i guess in they somewhat do the same task.
use the link in @SeBL4RD's post to grab the drivers in combination with my post.
it worked on windows might work on linux aswell, no idea tho, its just an suggestion.