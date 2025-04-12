# Github Discussion: "List indices must be integers or slices, not tuple #1001"

## Source: https://github.com/m-bain/whisperX/issues/1020

## Reply
Charlie11-Model
opened on Jan 16, 2025

Hi guys. I have been trying to use the latest model version 3.3.1 but the diarization step keeps giving me this error. Can anyone help me solve it? Or does anyone have a successful implementation of WhisperX 3.3.1 for transcription, alignment and diarization.
Activity
Barabazs
Barabazs commented on Jan 16, 2025
Barabazs
on Jan 16, 2025
Collaborator

Please post the full stack trace
Charlie11-Model
Charlie11-Model commented on Jan 16, 2025
Charlie11-Model
on Jan 16, 2025 · edited by Barabazs
Author

Here's the full stack trace:

/usr/local/lib/python3.10/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning: 
The secret `HF_TOKEN` does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
  warnings.warn(
tokenizer.json: 100%
 2.20M/2.20M [00:00<00:00, 7.09MB/s]
config.json: 100%
 2.80k/2.80k [00:00<00:00, 30.8kB/s]
vocabulary.txt: 100%
 460k/460k [00:00<00:00, 2.34MB/s]
model.bin: 100%
 3.09G/3.09G [01:12<00:00, 42.2MB/s]
No language specified, language will be first be detected for each audio file (increases inference time).
INFO:pytorch_lightning.utilities.migration.utils:Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.5.0.post0. To apply the upgrade to your files permanently, run `python -m pytorch_lightning.utilities.upgrade_checkpoint ../usr/local/lib/python3.10/dist-packages/whisperx/assets/pytorch_model.bin`
Model was trained with pyannote.audio 0.0.1, yours is 3.3.2. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.5.1+cu121. Bad things might happen unless you revert torch to 1.x.
/usr/local/lib/python3.10/dist-packages/pyannote/audio/utils/reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.
It can be re-enabled by calling
   >>> import torch
   >>> torch.backends.cuda.matmul.allow_tf32 = True
   >>> torch.backends.cudnn.allow_tf32 = True
See https://github.com/pyannote/pyannote-audio/issues/1370 for more details.

  warnings.warn(
Detected language: en (0.98) in first 30s of audio...
Downloading: "https://download.pytorch.org/torchaudio/models/wav2vec2_fairseq_base_ls960_asr_ls960.pth" to /root/.cache/torch/hub/checkpoints/wav2vec2_fairseq_base_ls960_asr_ls960.pth
100%|██████████| 360M/360M [00:01<00:00, 227MB/s]
WARNING:WhisperXModule:Word segments returned as a list. Defaulting sample rate to 16000.
config.yaml: 100%
 469/469 [00:00<00:00, 27.6kB/s]
pytorch_model.bin: 100%
 5.91M/5.91M [00:00<00:00, 34.2MB/s]
config.yaml: 100%
 399/399 [00:00<00:00, 31.2kB/s]
pytorch_model.bin: 100%
 26.6M/26.6M [00:00<00:00, 42.0MB/s]
config.yaml: 100%
 221/221 [00:00<00:00, 13.4kB/s]
ERROR:WhisperXModule:Diarization failed: list indices must be integers or slices, not tuple
An error occurred: list indices must be integers or slices, not tuple

Barabazs
Barabazs commented on Jan 16, 2025
Barabazs
on Jan 16, 2025
Collaborator

Unfortunately not enough information in that output.

Can you tell me where you're running it and how (which command and which options)?
Can you also try to use a hf_token and see if that makes a difference?
Charlie11-Model
Charlie11-Model commented on Jan 16, 2025
Charlie11-Model
on Jan 16, 2025
Author

i'm running it in google colab. Here's the colab template for installing and passing the token and audio file: ```# Install required packages
!pip install whisperx
!pip install pyannote.audio==0.0.1
!pip install torch==1.10.0+cu102 torchaudio==0.10.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102
Python script

import whisperx
Hugging Face Token

HF_TOKEN = "MY_TOKEN" # Replace with your actual token
Initialize the WhisperX model

model = whisperx.load_model("medium", device="cuda")
Transcribe and align

audio_file = "/content/small_audio.wav" # Update to your actual file path
result = model.transcribe(audio_file)
Diarization

try:
model_diarization = whisperx.DiarizationPipeline(
"pyannote/speaker-diarization-3.1",
device="cuda",
use_auth_token=HF_TOKEN,
)
diarized_segments = model_diarization(result["segments"], audio_file)
result["segments"] = diarized_segments
print(result)
except Exception as e:
print(f"An error occurred during diarization: {e}")```

Then it is giving me this error
` ` `/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_auth.py:94: UserWarning:
The secret HF_TOKEN does not exist in your Colab secrets.
To authenticate with the Hugging Face Hub, create a token in your settings tab (https://huggingface.co/settings/tokens), set it as secret in your Google Colab and restart your session.
You will be able to reuse this secret in all of your notebooks.
Please note that authentication is recommended but still optional to access public models or datasets.
warnings.warn(
model.bin: 100%
 1.53G/1.53G [00:36<00:00, 42.1MB/s]
config.json: 100%
 2.26k/2.26k [00:00<00:00, 73.4kB/s]
tokenizer.json: 100%
 2.20M/2.20M [00:00<00:00, 8.42MB/s]
vocabulary.txt: 100%
 460k/460k [00:00<00:00, 3.47MB/s]
No language specified, language will be first be detected for each audio file (increases inference time).
INFO:pytorch_lightning.utilities.migration.utils:Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.5.0.post0. To apply the upgrade to your files permanently, run python -m pytorch_lightning.utilities.upgrade_checkpoint ../usr/local/lib/python3.11/dist-packages/whisperx/assets/pytorch_model.bin
Model was trained with pyannote.audio 0.0.1, yours is 3.3.2. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.5.1+cu121. Bad things might happen unless you revert torch to 1.x.
/usr/local/lib/python3.11/dist-packages/pyannote/audio/utils/reproducibility.py:74: ReproducibilityWarning: TensorFloat-32 (TF32) has been disabled as it might lead to reproducibility issues and lower accuracy.
It can be re-enabled by calling

            import torch
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            See pyannote/pyannote-audio#1370 for more details.

warnings.warn(
Detected language: en (0.97) in first 30s of audio...
config.yaml: 100%
 469/469 [00:00<00:00, 41.2kB/s]
pytorch_model.bin: 100%
 5.91M/5.91M [00:00<00:00, 66.2MB/s]
config.yaml: 100%
 399/399 [00:00<00:00, 20.0kB/s]
pytorch_model.bin: 100%
 26.6M/26.6M [00:00<00:00, 41.0MB/s]
config.yaml: 100%
 221/221 [00:00<00:00, 19.1kB/s]
An error occurred during diarization: list indices must be integers or slices, not tuple` ` `
Barabazs
Barabazs commented on Jan 16, 2025
Barabazs
on Jan 16, 2025
Collaborator

    !pip install pyannote.audio==0.0.1
    !pip install torch==1.10.0+cu102 torchaudio==0.10.0+cu102 --extra-index-url https://download.pytorch.org/whl/cu102

All of this isn't needed in Google Colab and the versions you're using aren't compatible with whisperX.

fwiw, I can't even install it with what you're running
Charlie11-Model
Charlie11-Model commented on Jan 16, 2025
Charlie11-Model
on Jan 16, 2025
Author

I've also tried using the .whl file to install whisperX with all its dependencies. It still gave me the same error. I'm still a beginner in programming and I've been using ChatGPT to aid me here and there. Can you help me set it up in google colab?
Barabazs
Barabazs commented on Jan 16, 2025
Barabazs
on Jan 16, 2025
Collaborator

    I've also tried using the .whl file to install whisperX with all its dependencies. It still gave me the same error. I'm still a beginner in programming and I've been using ChatGPT to aid me here and there. Can you help me set it up in google colab?

First of all, read this for how to get a valid HF_TOKEN.

Then to install in Colab, you only need this !pip install whisperx

Then I would suggest to use the whisperX CLI, as this will let you run everything without having to write your own code. For example:
!whisperx "YOUR_FILE_PATH" --model medium --diarize --hf_token "YOUR_TOKEN"
Charlie11-Model
Charlie11-Model commented on Jan 16, 2025
Charlie11-Model
on Jan 16, 2025
Author

Thanks. It has run successfully.
Where can I find a list of all the possible command line arguments that I can use in this CLI?
Also where does it save the aligned speaker text output?
Lastly, this was the output:

>>Performing diarization...
config.yaml: 100% 469/469 [00:00<00:00, 3.85MB/s]
pytorch_model.bin: 100% 5.91M/5.91M [00:00<00:00, 48.4MB/s]
config.yaml: 100% 399/399 [00:00<00:00, 3.66MB/s]
pytorch_model.bin: 100% 26.6M/26.6M [00:00<00:00, 41.0MB/s]
config.yaml: 100% 221/221 [00:00<00:00, 2.24MB/s]
**/usr/local/lib/python3.11/dist-packages/pyannote/audio/models/blocks/pooling.py:104: UserWarning: std(): degrees of freedom is <= 0. Correction should be strictly less than the reduction factor (input numel divided by output numel). (Triggered internally at ../aten/src/ATen/native/ReduceOps.cpp:1823.)
  std = sequences.std(dim=-1, correction=1)**```

Should I worry about this last line?
