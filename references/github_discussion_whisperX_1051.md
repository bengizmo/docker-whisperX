# Github Discussion: "pip install whisperx results in installation of torch >2.0.0 #1051"

## Source: https://github.com/m-bain/whisperX/issues/1051

## Reply
ymednis
opened on Feb 17, 2025 · edited by ymednis

Description:
For proper functionality, WhisperX requires torch version 2.0.0. However, after running:

pip install whisperx

the installed torch version is 2.6.0—even if torch 2.0.0 was previously installed using:

conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

The issue appears to be due to the following dependency chain:

    whisperx depends on pyannote.audio==3.3.2
    pyannote.audio depends on lightning>=2.0.1
    lightning depends on pytorch-lightning
    pytorch-lightning requires torch>=2.1.0

This chain forces the installation of a torch version higher than 2.0.0.

Steps to Reproduce:

Install torch 2.0.0 with:
conda install pytorch==2.0.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia

Run:
pip install whisperx

Verify that torch is upgraded (e.g., using pip show torch), confirming that version 2.6.0 is installed.

System Information:

lsb_release -a
No LSB modules are available. 
Distributor ID: Ubuntu  
Description:    Ubuntu 22.04.3 LTS  
Release:        22.04  
Codename:       jammy

Expected Behavior:
WhisperX should work with torch 2.0.0 as intended without forcing an upgrade.

Actual Behavior:
Torch is upgraded to 2.6.0 due to the dependency requirements of lightning/pytorch-lightning.

Even in case of downgrading the pyannote.audio to version 3.1.1 and 3.0.0 it still uses lightning which requires torch>=2.1.0

3.0.0.txt
3.1.1.txt
Activity
Rhugan
Rhugan commented on Feb 28, 2025
Rhugan
on Feb 28, 2025

As a temporary workaround, I am able to install whisperx==3.3.1 in a fresh environment like this:

pip install whipserx==3.3.1
in isolation

followed by:
pip install torch==2.0.1 torchaudio==2.0.2 lightning==2.3.0 pytorch-lightning==2.3.0 pyannote.audio==3.1.1 numpy==1.26.4
You will get a pip error to say the version of pyannote.audio is not compatible with WhisperX but it will still install them.

Then WhisperX is functioning as expected for me.
RuenMeteoric
RuenMeteoric commented on Mar 23, 2025
RuenMeteoric
on Mar 23, 2025

    As a temporary workaround, I am able to install whisperx==3.3.1 in a fresh environment like this:

    pip install whipserx==3.3.1 in isolation

    followed by: You will get a pip error to say the version of pyannote.audio is not compatible with WhisperX but it will still install them.pip install torch==2.0.1 torchaudio==2.0.2 lightning==2.3.0 pytorch-lightning==2.3.0 pyannote.audio==3.1.1 numpy==1.26.4

    Then WhisperX is functioning as expected for me.

Thanks a lot for sharing, but I followed your method and my whisperx still doesn't work：
Traceback (most recent call last):
File "D:\anaconda3\envs\whisperx\lib\site-packages\transformers\utils\import_utils.py", line 1968, in get_module
return importlib.import_module("." + module_name, self.name)
File "D:\anaconda3\envs\whisperx\lib\importlib_init.py", line 126, in import_module
return _bootstrap._gcd_import(name[level:], package, level)
File "", line 1050, in _gcd_import
File "", line 1027, in _find_and_load
File "", line 1006, in _find_and_load_unlocked
File "", line 688, in _load_unlocked
File "", line 883, in exec_module
File "", line 241, in _call_with_frames_removed
File "D:\anaconda3\envs\whisperx\lib\site-packages\transformers\models\wav2vec2\modeling_wav2vec2.py", line 40, in
from ...modeling_utils import PreTrainedModel
File "D:\anaconda3\envs\whisperx\lib\site-packages\transformers\modeling_utils.py", line 55, in
from .integrations.flex_attention import flex_attention_forward
File "D:\anaconda3\envs\whisperx\lib\site-packages\transformers\integrations\flex_attention.py", line 46, in
class WrappedFlexAttention:
File "D:\anaconda3\envs\whisperx\lib\site-packages\transformers\integrations\flex_attention.py", line 61, in WrappedFlexAttention
@torch.compiler.disable(recursive=False)
AttributeError: module 'torch' has no attribute 'compiler'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
File "D:\anaconda3\envs\whisperx\lib\runpy.py", line 196, in _run_module_as_main
return run_code(code, main_globals, None,
File "D:\anaconda3\envs\whisperx\lib\runpy.py", line 86, in run_code
exec(code, run_globals)
File "D:\anaconda3\envs\whisperx\Scripts\whisperx.exe_main.py", line 4, in
File "D:\anaconda3\envs\whisperx\lib\site-packages\whisperx_init.py", line 1, in
from .alignment import load_align_model, align
File "D:\anaconda3\envs\whisperx\lib\site-packages\whisperx\alignment.py", line 14, in
from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
File "", line 1075, in _handle_fromlist
File "D:\anaconda3\envs\whisperx\lib\site-packages\transformers\utils\import_utils.py", line 1957, in getattr
value = getattr(module, name)
File "D:\anaconda3\envs\whisperx\lib\site-packages\transformers\utils\import_utils.py", line 1956, in getattr
module = self._get_module(self._class_to_module[name])
File "D:\anaconda3\envs\whisperx\lib\site-packages\transformers\utils\import_utils.py", line 1970, in _get_module
raise RuntimeError(
RuntimeError: Failed to import transformers.models.wav2vec2.modeling_wav2vec2 because of the following error (look up to see its traceback):
module 'torch' has no attribute 'compiler'
SeBL4RD
SeBL4RD commented on Mar 28, 2025
SeBL4RD
on Mar 28, 2025

    pip install torch==2.0.1 torchaudio==2.0.2 lightning==2.3.0 pytorch-lightning==2.3.0 pyannote.audio==3.1.1 numpy==1.26.4

ERROR: pip's dependency resolver does not currently take into account all the packages that are installed. This behaviour is the source of the following dependency conflicts.
whisperx 3.3.1 requires pyannote.audio==3.3.2, but you have pyannote-audio 3.1.1 which is incompatible.

and if I install pyannote.audio==3.3.2 :

(whisperxtest) F:\IA\whisperX>whisperx video_0_vocals_tts.wav --model large-v3-turbo --diarize --hf_token hf_*my_key*

INFO:speechbrain.utils.quirks:Applied quirks (see `speechbrain.utils.quirks`): [disable_jit_profiling, allow_tf32]
INFO:speechbrain.utils.quirks:Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated list): []
Traceback (most recent call last):
  File "C:\Users\Seb\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\Users\Seb\AppData\Local\Programs\Python\Python310\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "F:\IA\whisperX\whisperxtest\Scripts\whisperx.exe\__main__.py", line 7, in <module>
  File "F:\IA\whisperX\whisperxtest\lib\site-packages\whisperx\transcribe.py", line 178, in cli
    model = load_model(model_name, device=device, device_index=device_index, download_root=model_dir, compute_type=compute_type, language=args['language'], asr_options=asr_options, vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, task=task, threads=faster_whisper_threads)
  File "F:\IA\whisperX\whisperxtest\lib\site-packages\whisperx\asr.py", line 325, in load_model
    model = model or WhisperModel(whisper_arch,
  File "F:\IA\whisperX\whisperxtest\lib\site-packages\faster_whisper\transcribe.py", line 634, in __init__
    self.model = ctranslate2.models.Whisper(
ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.

Its a brand new venv.
xandark
xandark commented on Mar 29, 2025
xandark
on Mar 29, 2025

Same goes when I use the super-cool, super-fast uv Python package and environment management tool which also installs torch=2.6.0 and therefore whisperx is failing to run on Ubuntu 24.04 x86_64:

uv tool install whisperx --python=3.10
whisperx --model small  --language en  --diarize  "test.mp3"

INFO:speechbrain.utils.quirks:Applied quirks (see `speechbrain.utils.quirks`): [allow_tf32, disable_jit_profiling]
INFO:speechbrain.utils.quirks:Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated list): []
Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.5.1. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../../.local/share/uv/tools/whisperx/lib/python3.10/site-packages/whisperx/assets/pytorch_model.bin`
Model was trained with pyannote.audio 0.0.1, yours is 3.3.2. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.6.0+cu124. Bad things might happen unless you revert torch to 1.x.
>>Performing transcription...
/home/michael/.local/share/uv/tools/whisperx/lib/python3.10/site-packages/pyannote/audio/utils/reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.
It can be re-enabled by calling
   >>> import torch
   >>> torch.backends.cuda.matmul.allow_tf32 = True
   >>> torch.backends.cudnn.allow_tf32 = True
See https://github.com/pyannote/pyannote-audio/issues/1370 for more details.

  warnings.warn(
Could not load library libcudnn_ops_infer.so.8. Error: libcudnn_ops_infer.so.8: cannot open shared object file: No such file or directory
fish: Job 1, 'whisperx --model small --langua…' terminated by signal SIGABRT (Abort)

FWIW, I can install vanilla Whisper and run it just fine.