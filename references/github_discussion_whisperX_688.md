# Github Discussion: "pyannote/speaker-diarization-3.0 runs slower than pyannote/speaker-diarization@2.1 #499"

## Source: https://github.com/m-bain/whisperX/issues/688

## Reply
metheofanis
opened on Feb 4, 2024

Running diarization is extremely slow.
I have NVIDIA 3060 with 12GB VRAM

It looks like it is using the pyannote default embedding_batch_size: 32

If I run it locally, offline, where I can edit the SpeakerDiarization.yaml file and give embedding_batch_size: 8, the performance is more than 37X.

Is there any way to pass the embedding_batch_size as parameter to the DiarizationPipeline?
If not, I suggest to allow this!
I'm not expert to make a PR.
Do I miss something?
Thanks.
Activity
raulpetru
raulpetru commented on Mar 31, 2024
raulpetru
on Mar 31, 2024 · edited by raulpetru

Yes you can but you have to modify the pipeline.py file (located at whisperx\Lib\site-packages\pyannote\audio\core).

A way to overwrite embedding_batch_size default value:

params = config["pipeline"].get("params", {})
params.setdefault("use_auth_token", use_auth_token)
# Overwrite embedding_batch_size
params["embedding_batch_size"] = 8
pipeline = Klass(**params)

SeeknnDestroy
SeeknnDestroy commented on Apr 25, 2024
SeeknnDestroy
on Apr 25, 2024

@raulpetru why is this the case? lowering embedding_batch_size is better for performance? will it degrade the quality tho?
raulpetru
raulpetru commented on May 3, 2024
raulpetru
on May 3, 2024 · edited by raulpetru

@SeeknnDestroy might be a bug? It isn't clear, read here.
For me lowering the embedding_batch_size to 8 did increase the diarization performance significantly.
Actually thanks to @metheofanis by opening this issue I found out this performance fix.

I haven't tested the accuracy, but I believe there is no quality degradation.
If you do test, please let me know!
techjp
techjp commented on Jul 22, 2024
techjp
on Jul 22, 2024 · edited by techjp

I have an RTX 3080 10GB card, and thought I was going insane trying to get the diarization to work.

I have a 1hour 45minute long meeting recording that I am trying to get transcribed. The original transcription takes about 84 seconds, alignment about 44seconds. Then diarization would run forever. I let it run for over an hour with no results. Tried splitting the file into chunks, still never finished, even with a 20minute chunk.

Most of the GPU memory was being used so I suspect there was some sort of crazy memory swapping going on, but I'm not sure.

After making the change suggested by @raulpetru above, creation if the diarize segments finishes in 95 seconds (this ran for an hour before without finishing!), and assigning speaker IDs took 12 seconds. Even more, GPU memory use was around 3GB instead of being 9.5GB to 9.7GB before.

I'm not sure if this setting impacts diarization quality, but wow, for someone with a smaller amount of GPU memory, it allows the system to actually work!!

I hope this setting can be integrated into a future release of whisperX, I'm sure there are many people out there with 10GB, 12GB (or smaller!) GPUs who are having the same problem.

Thank you to @metheofanis for creating the issue & suggesting the fix, and to @raulpetru for explaining how to change the setting in whisperX!

Edit: And for anyone using miniconda like me, the pipeline.py file is here (assuming your environment is named whisperx and you are using Python 3.10, of course):
~/miniconda3/envs/whisperx/lib/python3.10/site-packages/pyannote/audio/core
raulpetru
raulpetru commented on Jan 18, 2025
raulpetru
on Jan 18, 2025 · edited by raulpetru

I just updated to the latest whisperx version and I also had to change the segmentation_batch_size to 8 (default is 32) in order to obtain a good diarization time.

params = config["pipeline"].get("params", {})
params.setdefault("use_auth_token", use_auth_token)
# Overwrite embedding_batch_size and segmentation_batch_size
params["embedding_batch_size"] = 8
params["segmentation_batch_size"] = 8
pipeline = Klass(**params)

raulpetru
mentioned this on Jan 19, 2025

    pyannote/speaker-diarization-3.0 runs slower than pyannote/speaker-diarization@2.1 #499

dinopio
dinopio commented on Mar 7, 2025
dinopio
on Mar 7, 2025

    I just updated to the latest whisperx version and I also had to change the segmentation_batch_size to 8 (default is 32) in order to obtain a good diarization time.

    params = config["pipeline"].get("params", {})
    params.setdefault("use_auth_token", use_auth_token)
    # Overwrite embedding_batch_size and segmentation_batch_size
    params["embedding_batch_size"] = 8
    params["segmentation_batch_size"] = 8
    pipeline = Klass(**params)

is it running on the GPU though or CPU?
zckrs
zckrs commented on Mar 7, 2025
zckrs
on Mar 7, 2025

Related / Duplicate
#274 (comment)