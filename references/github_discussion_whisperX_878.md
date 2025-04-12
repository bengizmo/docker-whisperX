# Github Discussion: "ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation. #878"

## Source: https://github.com/m-bain/whisperX/issues/878

## Reply
kc01-8
opened on Sep 9, 2024

PS F:\whisperX-main> whisperx audio.mp4 --model large-v2 --diarize --highlight_words True --min_speakers 5 --max_speakers 5 --hf_token hf_x
C:\Users\kc01\AppData\Roaming\Python\Python310\site-packages\pyannote\audio\core\io.py:43: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
  torchaudio.set_audio_backend("soundfile")
Traceback (most recent call last):
  File "C:\Program Files\Python310\lib\runpy.py", line 196, in _run_module_as_main
    return _run_code(code, main_globals, None,
  File "C:\Program Files\Python310\lib\runpy.py", line 86, in _run_code
    exec(code, run_globals)
  File "C:\Users\kc01\AppData\Roaming\Python\Python310\Scripts\whisperx.exe\__main__.py", line 7, in <module>
  File "C:\Users\kc01\AppData\Roaming\Python\Python310\site-packages\whisperx\transcribe.py", line 170, in cli
    model = load_model(model_name, device=device, device_index=device_index, download_root=model_dir, compute_type=compute_type, language=args['language'], asr_options=asr_options, vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, task=task, threads=faster_whisper_threads)
  File "C:\Users\kc01\AppData\Roaming\Python\Python310\site-packages\whisperx\asr.py", line 288, in load_model
    model = model or WhisperModel(whisper_arch,
  File "C:\Users\kc01\AppData\Roaming\Python\Python310\site-packages\faster_whisper\transcribe.py", line 133, in __init__
    self.model = ctranslate2.models.Whisper(
ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.

Happens using a 3080ti, which works flawlessly with NVidia NeMo. Completely fresh install of whisperx.
Activity
Hasan-Naseer
Hasan-Naseer commented on Sep 10, 2024
Hasan-Naseer
on Sep 10, 2024 · edited by Hasan-Naseer
Contributor

    PS F:\whisperX-main> whisperx audio.mp4 --model large-v2 --diarize --highlight_words True --min_speakers 5 --max_speakers 5 --hf_token hf_x
    C:\Users\kc01\AppData\Roaming\Python\Python310\site-packages\pyannote\audio\core\io.py:43: UserWarning: torchaudio._backend.set_audio_backend has been deprecated. With dispatcher enabled, this function is no-op. You can remove the function call.
      torchaudio.set_audio_backend("soundfile")
    Traceback (most recent call last):
      File "C:\Program Files\Python310\lib\runpy.py", line 196, in _run_module_as_main
        return _run_code(code, main_globals, None,
      File "C:\Program Files\Python310\lib\runpy.py", line 86, in _run_code
        exec(code, run_globals)
      File "C:\Users\kc01\AppData\Roaming\Python\Python310\Scripts\whisperx.exe\__main__.py", line 7, in <module>
      File "C:\Users\kc01\AppData\Roaming\Python\Python310\site-packages\whisperx\transcribe.py", line 170, in cli
        model = load_model(model_name, device=device, device_index=device_index, download_root=model_dir, compute_type=compute_type, language=args['language'], asr_options=asr_options, vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, task=task, threads=faster_whisper_threads)
      File "C:\Users\kc01\AppData\Roaming\Python\Python310\site-packages\whisperx\asr.py", line 288, in load_model
        model = model or WhisperModel(whisper_arch,
      File "C:\Users\kc01\AppData\Roaming\Python\Python310\site-packages\faster_whisper\transcribe.py", line 133, in __init__
        self.model = ctranslate2.models.Whisper(
    ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.

    Happens using a 3080ti, which works flawlessly with NVidia NeMo. Completely fresh install of whisperx.

What is the device you are passing? Are you sure it's 'GPU' and not 'CPU'. If I recall correctly this was a CPU only problem not with whisperx but faster-whisper under the hood. See for example this issue here SYSTRAN/faster-whisper#65

If you are indeed sending the correct params for GPU use then I recommend running faster-whisper directly first to narrow down the problem. Make a .py file import the necessary starter code, you find it on faster-whisper's github and run with the verbose flag set

CT2_VERBOSE=1 time python3 main.py

should give more console output for debugging.

We can proceed from there to see what's wrong.
iSevenDays
iSevenDays commented on Jan 7, 2025
iSevenDays
on Jan 7, 2025

I have a similar issue on mac mini Intel 2018.

INFO:speechbrain.utils.quirks:Applied quirks (see `speechbrain.utils.quirks`): [allow_tf32, disable_jit_profiling]
INFO:speechbrain.utils.quirks:Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated list): []
[2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info] CPU: GenuineIntel (SSE4.1=true, AVX=true, AVX2=true, AVX512=false)
[2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - Selected ISA: AVX2
[2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - Use Intel MKL: true
[2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - SGEMM backend: MKL (packed: false)
[2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - GEMM_S16 backend: MKL (packed: false)
[2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - GEMM_S8 backend: MKL (packed: false, u8s8 preferred: true)
Traceback (most recent call last):
  File "/Users/seven/opt/anaconda3/envs/whisperx/bin/whisperx", line 8, in <module>
    sys.exit(cli())
  File "/Users/seven/opt/anaconda3/envs/whisperx/lib/python3.10/site-packages/whisperx/transcribe.py", line 171, in cli
    model = load_model(model_name, device=device, device_index=device_index, download_root=model_dir, compute_type=compute_type, language=args['language'], asr_options=asr_options, vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, task=task, threads=faster_whisper_threads)
  File "/Users/seven/opt/anaconda3/envs/whisperx/lib/python3.10/site-packages/whisperx/asr.py", line 292, in load_model
    model = model or WhisperModel(whisper_arch,
  File "/Users/seven/opt/anaconda3/envs/whisperx/lib/python3.10/site-packages/faster_whisper/transcribe.py", line 634, in __init__
    self.model = ctranslate2.models.Whisper(
ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.
(whisperx) seven@Mac-mini-real ai % 

Hasan-Naseer
Hasan-Naseer commented on Jan 21, 2025
Hasan-Naseer
on Jan 21, 2025
Contributor

    I have a similar issue on mac mini Intel 2018.

    INFO:speechbrain.utils.quirks:Applied quirks (see `speechbrain.utils.quirks`): [allow_tf32, disable_jit_profiling]
    INFO:speechbrain.utils.quirks:Excluded quirks specified by the `SB_DISABLE_QUIRKS` environment (comma-separated list): []
    [2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info] CPU: GenuineIntel (SSE4.1=true, AVX=true, AVX2=true, AVX512=false)
    [2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - Selected ISA: AVX2
    [2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - Use Intel MKL: true
    [2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - SGEMM backend: MKL (packed: false)
    [2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - GEMM_S16 backend: MKL (packed: false)
    [2025-01-07 13:33:29.075] [ctranslate2] [thread 4591897] [info]  - GEMM_S8 backend: MKL (packed: false, u8s8 preferred: true)
    Traceback (most recent call last):
      File "/Users/seven/opt/anaconda3/envs/whisperx/bin/whisperx", line 8, in <module>
        sys.exit(cli())
      File "/Users/seven/opt/anaconda3/envs/whisperx/lib/python3.10/site-packages/whisperx/transcribe.py", line 171, in cli
        model = load_model(model_name, device=device, device_index=device_index, download_root=model_dir, compute_type=compute_type, language=args['language'], asr_options=asr_options, vad_options={"vad_onset": vad_onset, "vad_offset": vad_offset}, task=task, threads=faster_whisper_threads)
      File "/Users/seven/opt/anaconda3/envs/whisperx/lib/python3.10/site-packages/whisperx/asr.py", line 292, in load_model
        model = model or WhisperModel(whisper_arch,
      File "/Users/seven/opt/anaconda3/envs/whisperx/lib/python3.10/site-packages/faster_whisper/transcribe.py", line 634, in __init__
        self.model = ctranslate2.models.Whisper(
    ValueError: Requested float16 compute type, but the target device or backend do not support efficient float16 computation.
    (whisperx) seven@Mac-mini-real ai % 

Like I said, this is potentially an issue with faster-whisper's implementation that doesn't let CPU architectures run float16. We run float32, that results in the same (if not more) level of accuracy just more memory consumed.

I want to believe that a mac mini (not being a dedicated cuda gpu like nvidia) runs into the same problem when running float16. Try changing to float32 and see what happens.

Again, refer to SYSTRAN/faster-whisper#65 for a better explanation.
adityaraute
adityaraute commented on Apr 11, 2025
adityaraute
on Apr 11, 2025

    What is the device you are passing? Are you sure it's 'GPU' and not 'CPU'. If I recall correctly this was a CPU only problem not with whisperx but faster-whisper under the hood.

Image

This is my entire console output

I'm facing the same issue

I am not passing anything regarding GPU or CPU in the package. What can I do to make this work?