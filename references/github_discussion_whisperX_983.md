# Github Discussion: "Can't get cuda to work. #983"

## Source: https://github.com/m-bain/whisperX/issues/983

## Reply
manus693
opened on Jan 8, 2025 · edited by manus693

Whatever I do I get Torch not compiled with CUDA, I've followed the instructions and installed as written.

miniconda3\envs\whisperx\lib\site-packages\torch\cuda_init_.py", line 310, in _lazy_init
raise AssertionError("Torch not compiled with CUDA enabled")
AssertionError: Torch not compiled with CUDA enabled

When running pip install whisperx it installs torch without cuda enabled. I'm running this inside the conda environment.
I'm not really sure how the get this to work, been trying for ages now.
It also install torch 2.5.0, but the conda install is 2.0.0 before the "pip install whisperx" in the description.

Is Setup in description outdated?

Is there any difference except maybe speed? CPU or Cuda?
I ran whisperx on a movie with int8 enabled, and it was almost correct, some timestamps where completely wrong, just random places.
I could not get float 16 to work, trying CPU and float 32 now.

I have 4060ti.
Activity
mukhituly
mukhituly commented on Jan 11, 2025
mukhituly
on Jan 11, 2025

+1, experiencing this issue as well
NefariousC
NefariousC commented on Jan 11, 2025
NefariousC
on Jan 11, 2025 · edited by NefariousC

Ran into the same issue, idk but there seems to be a pip bug where installing/updating whisperx automatically pulls the CPU version of torch instead of the CUDA version, regardless of what you had before.

You can try this (bear in mind that i'm on CUDA 12)
This will force reinstall torch and torchaudio with CUDA support along with their dependencies:

pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121 --force-reinstall --no-cache-dir

    After that, you might run into another error when using whisperx :

Warning

Could not locate cudnn_ops_infer64_8.dll. Please make sure it is in your library path!

    Download the cuDNN library from nvidia website (for windows)

Important

Since CTranslate 4.4.0 only supports up to cuDNN 8.x, make sure to grab the 8.x version compatible with CUDA 12

    Locate the files inside the downloaded zip bin folder then extract them to your whisperx environment bin folder ...\envs\whisperx\bin

I think there's a simpler solution somewhere but after searching for hours this is what I got.
ruze00
ruze00 commented on Jan 29, 2025
ruze00
on Jan 29, 2025

So CUDA 12.1 works? I've been holding off on even trying to install whisperx since it says it supports 11.8 and other versions are YMMV. I'm not going to downgrade my whole env for 1 component.
Matagon
Matagon commented on Jan 29, 2025
Matagon
on Jan 29, 2025

This works, FYI, the kernel will die if ran from a Jupyter Notebook.
NefariousC
NefariousC commented on Feb 2, 2025
NefariousC
on Feb 2, 2025 · edited by NefariousC

    @ruze00 So CUDA 12.1 works? I've been holding off on even trying to install whisperx since it says it supports 11.8 and other versions are YMMV. I'm not going to downgrade my whole env for 1 component.

yes, it works. i've been using it with no problem.
ruze00
ruze00 commented on Feb 2, 2025
ruze00
on Feb 2, 2025

        @ruze00 So CUDA 12.1 works? I've been holding off on even trying to install whisperx since it says it supports 11.8 and other versions are YMMV. I'm not going to downgrade my whole env for 1 component.

    yes, it works. i've been using it with no problem.

Thanks, appreciate it.
zos474
zos474 commented on Feb 2, 2025
zos474
on Feb 2, 2025

Thanks @NefariousC your solution worked. I have CUDA 12.5 installed on my Windows 11 machine and had a perfectly working WhisperX 3.1 Conda environment until I decided stupidly to do an upgrade to 3.3.1. I ran into one problem after another, eventually deciding to trash the environment and create a new one, but of course the standard install doesnt work, because even though you have installed torch 2.0.0, the install of whisperx from PyPI overwrites it with 2.6.0 and it is the cpu version... But your steps worked exactly. I made two slight mods: Installed 12.4  pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu124 --force-reinstall --no-cache-dir and copied the cudnn dll files directly into .\miniconda3\envs\whisperx not into a bin subdirectory. Thanks so much. I had wasted most of Sunday trying to get it to all work again. I guess the lesson is dont upgrade what aint broke.
NefariousC
NefariousC commented on Feb 7, 2025
NefariousC
on Feb 7, 2025 · edited by NefariousC

@zos474 No worries, I was in the same boat as you. I encountered some issues like OP after blindly upgrading WhisperX. After running into errors even after reinstalling, I had to dig into Google and see what this error was about:

    miniconda3\envs\whisperx\lib\site-packages\torch\cuda_init_.py", line 310, in _lazy_init
    raise AssertionError("Torch not compiled with CUDA enabled")
    AssertionError: Torch not compiled with CUDA enabled

I was so confused because I thought I DID install the CUDA version, per what the setup docs said:

        Install PyTorch, e.g. for Linux and Windows CUDA11.8:
        conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

After surfing through forums, I found out about the pip bug.

Then I remembered WhisperX uses faster-whisper backend, which needs CTranslate2. Went back to the WhisperX docs and found this important bit:

Important

GPU execution requires the NVIDIA libraries cuBLAS 11.x and cuDNN 8.x to be installed on the system. Please refer to the CTranslate2 documentation

Looks like I missed that tidbit when I was just yolo-ing the update. Lesson learned for me too. Have to thoroughly read the documentation before updating I guess 😂.
jfsbra
jfsbra commented on Feb 17, 2025
jfsbra
on Feb 17, 2025

Hi All.
I have been trying to get WhisperX to run on my Win11 PC for days now... Simply no joy!
Whatever i do seems to break things because of dependency incompatibilites.
I'm wondering if someone who has whisperx running correctly could provide a list of their installation (e.g. pip freeze) so I could try to replicate the install - preferably in a Docker container or conda environment.
Thanks a million!
protik09
added a commit that references this issue on Feb 17, 2025

Add automatic CUDNN install fix for whisperx for Windows
636c8fa
UdinaDev
UdinaDev commented on Feb 22, 2025
UdinaDev
on Feb 22, 2025

@jfsbra

Had similar dependencies issue yesterday.
Now it's working but my solution is for whisperx==3.1.1 as it has all the features I need

Create venv with your python 3.10 :

python3.10 -m venv whisperx311

Then install whisperx==3.1.1 :

E:\Codes\virtual_envs\whisperx311\Scripts\python -m pip install whisperx==3.1.1

Create following requirements.txt :

--extra-index-url https://download.pytorch.org/whl/cu118
torch==2.0.0+cu118
torchaudio==2.0.1+cu118
torchvision==0.15.1+cu118
faster-whisper==0.10.1
transformers
ctranslate2<4.5.0
pandas
setuptools==65.6.3
nltk
pytorch_lightning==2.0.1
lightning==2.0.1
numpy==1.26.4

Then

E:\Codes\virtual_envs\whisperx311\Scripts\python -m pip install -r requirements.txt 

At this point you should have no dependency resolver errors and a working whisperx venv (whisperx===3.1.1)
SomalianCoolhacker
SomalianCoolhacker commented on Mar 17, 2025
SomalianCoolhacker
on Mar 17, 2025 · edited by SomalianCoolhacker

    At this point you should have no dependency resolver errors and a working whisperx venv (whisperx===3.1.1)

Thanks, but I'm still getting error, unfortunately.
Traceback (most recent call last):
File "...\miniconda3\envs\whisperx311\lib\runpy.py", line 196, in _run_module_as_main
return _run_code(code, main_globals, None,
File "...\miniconda3\envs\whisperx311\lib\runpy.py", line 86, in run_code
exec(code, run_globals)
File "...\miniconda3\envs\whisperx311\Scripts\whisperx.exe_main.py", line 7, in
File "...\miniconda3\envs\whisperx311\lib\site-packages\whisperx\transcribe.py", line 170, in cli
model = load_model(model_name, device=device, device_index=device_index, download_root=model_dir, compute_type=compute_type, language=args['language'], asr_options=asr_options, vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, task=task, threads=faster_whisper_threads)
File "...\miniconda3\envs\whisperx311\lib\site-packages\whisperx\asr.py", line 345, in load_model
vad_model = load_vad_model(torch.device(device), use_auth_token=None, **default_vad_options)
File "...\miniconda3\envs\whisperx311\lib\site-packages\whisperx\vad.py", line 29, in load_vad_model
with urllib.request.urlopen(VAD_SEGMENTATION_URL) as source, open(model_fp, "wb") as output:
File "...\miniconda3\envs\whisperx311\lib\urllib\request.py", line 216, in urlopen
return opener.open(url, data, timeout)
File "...\miniconda3\envs\whisperx311\lib\urllib\request.py", line 525, in open
response = meth(req, response)
File "...\miniconda3\envs\whisperx311\lib\urllib\request.py", line 634, in http_response
response = self.parent.error(
File "...\miniconda3\envs\whisperx311\lib\urllib\request.py", line 563, in error
return self._call_chain(*args)
File "...\miniconda3\envs\whisperx311\lib\urllib\request.py", line 496, in _call_chain
result = func(*args)
File "...\miniconda3\envs\whisperx311\lib\urllib\request.py", line 643, in http_error_default
raise HTTPError(req.full_url, code, msg, hdrs, fp)
urllib.error.HTTPError: HTTP Error 301: Moved Permanently
SomalianCoolhacker
SomalianCoolhacker commented on Mar 17, 2025
SomalianCoolhacker
on Mar 17, 2025 · edited by SomalianCoolhacker

After fresh install of whisperX 3.3.1, @NefariousC answer finally worked for me, thanks man; even with the latest nightly cu128 builds. It works fine if you put the cuDNN files (whisperx asked for cudnn_cnn_infer64_8.dll, not sure if cudnn_ops_infer64_8.dll is needed in newer versions, I had it installed beforehand) directly in the root folder (...\envs\whisperx).

Something like this should do the trick (for now):
pip install torch torchaudio torchvision --index-url https://download.pytorch.org/whl/nightly/cu128 --force-reinstall --no-cache-dir