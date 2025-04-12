# Github Discussion: "TypeError: FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'initial_prompt' #1020"

## Source: https://github.com/m-bain/whisperX/issues/1020

## Reply
Rumeysakeskin
opened on Jan 26, 2025

import whisperx
import gc

device = "cuda"
audio_file = "/content/6501-out.wav"
batch_size = 16 # reduce if low on GPU mem
compute_type = "float16" # change to "int8" if low on GPU mem (may reduce accuracy)

model = whisperx.load_model("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda",download_root='models', vad_method="silero", language="tr")
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size, initial_prompt="Bu bir müşteri hizmetleri görüşme kaydıdır. Bazı kelimelerin doğru yazımı şu şekilde: Limited Şirketi")

model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)
print(result["segments"]) # after alignment

I get
TypeError: FasterWhisperPipeline.transcribe() got an unexpected keyword argument 'initial_prompt'
Activity
Rumeysakeskin
mentioned this on Jan 26, 2025

    Enable initial_prompt handling in transcribe method fixes #1020 #1021

Barabazs
Barabazs commented on Feb 12, 2025
Barabazs
on Feb 12, 2025
Collaborator

You're supposed to pass it as an asr_option in load_model, not in transcribe.

model = whisperx.load_model("deepdml/faster-whisper-large-v3-turbo-ct2", device="cuda",download_root='models', vad_method="silero", language="tr", asr_options={"initial_prompt":"Bu bir müşteri hizmetleri görüşme kaydıdır. Bazı kelimelerin doğru yazımı şu şekilde: Limited Şirketi"})
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)