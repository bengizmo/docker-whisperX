# Github Discussion: "pyaDiarization too slow #274"

## Source: https://github.com/m-bain/whisperX/issues/274

## Reply
MitPitt
opened on May 25, 2023

1 hour 30 minutes of audio were processing for over 1 hour in the diarization... stage. I'm using an RTX 3090.

I'm guessing --batch_size doesn't affect pyannote. A setting for pyannote's batch size would be very nice to have.
Activity
jzeller2011
jzeller2011 commented on May 25, 2023
jzeller2011
on May 25, 2023

I'm having the same issue. From what i'm reading, the pyannote/speaker-diarization model is slow, but word-level segmentation may be slowing it down even more. I assume there are factors that impact this more than others (i think number of speakers or number of segments influences this the most, but that's just a guess). Looking at hardware usage during runtime, looks like it's batching either one segment at a time or one word at a time (this would make sense, since we're chasing word-level timestamps with whisperx. The pyannote model reports a 2.5% realtime factor, which is definitely NOT been my experience, but may be the case if you ran the raw audio through without segmentation). Maybe there's a way to count individual calls to the GPU to verify. I haven't found a workaround yet, let me know if you find something out.
moritzbrantner
moritzbrantner commented on May 25, 2023
moritzbrantner
on May 25, 2023
Contributor

I have the same issue.
DigilConfianz
DigilConfianz commented on May 26, 2023
DigilConfianz
on May 26, 2023

#159 (comment)
m-bain
m-bain commented on May 26, 2023
m-bain
on May 26, 2023
Owner

    1 hour 30 minutes of audio were processing for over 1 hour in the diarization... stage. I'm using an RTX 3090.

That's very strange, it should not be that long, I would expect 5-10mins max. I suspect some bug here.

    I'm guessing --batch_size doesn't affect pyannote. A setting for pyannote's batch size would be very nice to have.

I would assume most of the time is the clustering step, which can be recursive and can take long if its not finding satisfactory cluster sizes.

    From what i'm reading, the pyannote/speaker-diarization model is slow, but word-level segmentation may be slowing it down even more.

Nah the ASR and word-level segmentation is ran independently of the diarization. The diarization is just running a standard pyannote pipeline. So word-level segmentation / whisperx batching shouldnt effect this
geoglrb
geoglrb commented on May 29, 2023
geoglrb
on May 29, 2023 · edited by geoglrb

@m-bain I'm also having extremely slow diarization. Using CLI.

Just now, to explore further, I also tried setting the --threads parameter to 50 to see if that would do something (I would prefer GPU!) and it is now making use of a variable number of threads, but well about four, which is what it had seemed to be limited to by default. There is still some GPU memory allocated even in the diarization stage, but not a ton. Very naive question--could things be slow because all of us have pyannote using CPU for some reason? Is there a way to specify that whisperx's pyannote must use GPU?

For reference, in case it helps:

>>> import torch
>>> torch.cuda.is_available()
True
>>> torch.cuda.device_count()
2
>>> torch.version.cuda
'11.7'

sorgfresser
sorgfresser commented on Jun 2, 2023
sorgfresser
on Jun 2, 2023
Contributor

There is an issue regarding pyannote not using GPU, but it should not occur with whisperx. To read more on this, see pyannote/pyannote-audio#1354.
It might have something to do with the device index though. Are both of your GPUs the same size? We're currently not passing device_index to the diarization, so we will simply do to('cuda') on loading the diarization model. This might be a problem when multiple GPUs are available.
goneill
goneill commented on Jun 7, 2023
goneill
on Jun 7, 2023

I am also having an extremely long, ie overnight, diarization on the command line. The transcription occurs, I get two failures in the align segment and then diarization occurs, and I get the following errors:

Lightning automatically upgraded your loaded checkpoint from v1.5.4 to v2.0.2. To apply the upgrade to your files permanently, run python -m pytorch_lightning.utilities.upgrade_checkpoint --file ../.cache/torch/pyannote/models--pyannote--segmentation/snapshots/c4c8ceafcbb3a7a280c2d357aee9fbc9b0be7f9b/pytorch_model.bin
Model was trained with pyannote.audio 0.0.1, yours is 2.1.1. Bad things might happen unless you revert pyannote.audio to 0.x.
Model was trained with torch 1.10.0+cu102, yours is 2.0.1. Bad things might happen unless you revert torch to 1.x.

and then I left it running overnight and still in the same state.
davidas1
davidas1 commented on Aug 1, 2023
davidas1
on Aug 1, 2023
Contributor

Please try my suggestion in #399 and see if it helps you too.
I'm getting around 30sec for diarization of 30 minute video using the standard model in the pyannote/speaker-diarization pipeline (speechbrain/spkrec-ecapa-voxceleb), and around 15sec if I change the embedding model to pyannote/embedding
DigilConfianz
DigilConfianz commented on Aug 1, 2023
DigilConfianz
on Aug 1, 2023

@davidas1 There is speed improvement when changing to whisper loaded audio from the raw audio file as you suggested. Thanks for that. How to change the embedding model in code?
davidas1
davidas1 commented on Aug 1, 2023
davidas1
on Aug 1, 2023
Contributor

Changing the pyannote pipeline is a bit more involved - I'm using an offline pipeline like described in https://github.com/pyannote/pyannote-audio/blob/develop/tutorials/applying_a_pipeline.ipynb
I had to patch whisperx a bit to allow working with a custom local pipeline.
Using this method you can customize the pipeline by editing the config.yaml (change the "embedding" configuration to the desired model).
datacurse
datacurse commented on Aug 2, 2023
datacurse
on Aug 2, 2023

    Please try my suggestion in #399 and see if it helps you too. I'm getting around 30sec for diarization of 30 minute video using the standard model in the pyannote/speaker-diarization pipeline (speechbrain/spkrec-ecapa-voxceleb), and around 15sec if I change the embedding model to pyannote/embedding

what??? thats crazy! here is my timings for 30 minute long mp3:
transcribe time: 69 seconds
align time: 10 seconds
diarization: 24 seconds
around 90 seconds in total, like 3 times longer than yours, and thats excluding the initial model loadings.

could you please suggest something like a checklist for speeding things up? i also updated to get your recet patch and it did speed up my diarization exponentially
davidas1
davidas1 commented on Aug 3, 2023
davidas1
on Aug 3, 2023
Contributor

I wrote that diarization takes 30sec, not the entire pipeline - before the change the diarization took almost 2 minutes.
Your timing looks great, other than the transcribe step that is faster on my setup, but that's probably due to the GPU you're using.
datacurse
datacurse commented on Aug 6, 2023
datacurse
on Aug 6, 2023

oooh i see that clears things. i got 4090 tho
dantheman0207
dantheman0207 commented on Aug 28, 2023
dantheman0207
on Aug 28, 2023

I'm looking for some help or insight into why diarization is so slow for me.

I have a recording that is 1 minute and 14 seconds with two native English speakers and diarization takes 11 minutes and 49 seconds (transcription took 6 seconds). I'm running on a Mac mini with an M2 chip and 8GB of RAM. I assume in this case it's running on CPU although I'm not sure with the Apple silicon. I'm basically using the default example on the README for transcribing and diarizing a file.

With a longer file (27 minutes and 39 seconds), with multiple speakers, it takes 2 minutes and 47 seconds to transcribe, 1 minute and 6 seconds to align but 12 hours, 48 minutes to diarize!
awhillas
awhillas commented on Nov 27, 2023
awhillas
on Nov 27, 2023

Same here. I'm getting 2-3% GPU utilization 0.9 GB of GPU memory?
SergeiKarulin
SergeiKarulin commented on Apr 9, 2024
SergeiKarulin
on Apr 9, 2024

same issue. Almost no GPU utilization and 1.5 hour of diarization per 60 minutes audio.
eplinux
eplinux commented on Apr 11, 2024
eplinux
on Apr 11, 2024

    same issue. Almost no GPU utilization and 1.5 hour of diarization per 60 minutes audio.

same here
eplinux
eplinux commented on Apr 16, 2024
eplinux
on Apr 16, 2024

I also noticed that there seems to be some throttling affecting the GPU utilization on Windows 11. As soon as the terminal window is in the background, the GPU utilization drops dramatically
prkumar112451
prkumar112451 commented on May 14, 2024
prkumar112451
on May 14, 2024

@m-bain Diarization is a key aspect where multiple speakers are having a conversation. I've been exploring different ways to speed up transcription & diarization pipeline.

Can see lots of different options for speeding up transcription like : CTranslate2, Batching, Flash Attention, Distil-Whisper, ComputeTime (float32,16)

but finding very limited options for diarization speedup.

for a 20 minutes audio, with optimizations we are able to get transcriptions in around 35 seconds.
But diarizing a 20 minute audio is taking roughly 1 minute via Nemo and around 45 seconds via Pyannote.

Could you please share if there is any direction which we can follow to speedup diarization process?
UdinaDev
UdinaDev commented on Feb 22, 2025
UdinaDev
on Feb 22, 2025 · edited by UdinaDev

I had the same issue with a RTX 3070 with 8 Go VRAM.

Without changing configuration for the pyannote/speaker-diarization-3.1 pipeline I don't have enough VRAM :
Image
Solution 1 : change config file locally

To change this go to https://github.com/pyannote/hf-speaker-diarization-3.1/tree/main and download config.yaml then put it in your local machine in a directory like C:\Users[PATH]\models--pyannote--speaker-diarization-3.1

Then in the file config.yaml change parameters embedding_batch_size and segmentation_batch_size from 32 to 4 (for exemple), resulting in :

version: 3.1.0

pipeline:
  name: pyannote.audio.pipelines.SpeakerDiarization
  params:
    clustering: AgglomerativeClustering
    embedding: pyannote/wespeaker-voxceleb-resnet34-LM
    embedding_batch_size: 4
    embedding_exclude_overlap: true
    segmentation: pyannote/segmentation-3.0
    segmentation_batch_size: 4

params:
  clustering:
    method: centroid
    min_cluster_size: 12
    threshold: 0.7045654963945799
  segmentation:
    min_duration_off: 0.0

Then when you load your whisperx.DiarizationPipeline refer to the above config.yaml :

path_to_config = r"C:\[PATH]\models--pyannote--speaker-diarization-3.1\config.yaml"
diarize_model = whisperx.DiarizationPipeline(model_name=path_to_config, use_auth_token="[HF_TOKEN]", device=device)

This will now not use all VRAM and speed up the process.

Image

For exemple for a 6min11 video it goes from 10min diarization to 10 s !

Code to try this made from https://huggingface.co/pyannote/speaker-diarization-3.1 , https://colab.research.google.com/gist/agorman/38ef94e8b4ae7fd3fef474e49c5b212a/mre_template.ipynb and whisperx\diarize.py :

# instantiate the pipeline
from pyannote.audio import Pipeline
import torch
from pyannote.audio.pipelines.utils.hook import ProgressHook
import whisperx
SAMPLE_RATE = 16000

audio_file = r"[PATH].mp4"
path_to_config = r"C:\[PATH]\models--pyannote--speaker-diarization-3.1\config.yaml"

# pipeline = Pipeline.from_pretrained(
#   checkpoint_path="pyannote/speaker-diarization-3.1",
#   use_auth_token="[HF_TOKEN]")
# pipeline.to(torch.device("cuda"))

pipeline = Pipeline.from_pretrained(
   checkpoint_path=path_to_config,
   use_auth_token="[HF_TOKEN]")
pipeline.to(torch.device("cuda"))

# run the pipeline on an audio file
audio = whisperx.load_audio(audio_file)

with ProgressHook() as hook:
  diarization = pipeline({"waveform": torch.from_numpy(audio).unsqueeze(0), "sample_rate": SAMPLE_RATE}, hook=hook) #, num_speakers=2)

# dump the diarization output to disk using RTTM format
with open("audio.rttm", "w") as rttm:
  diarization.write_rttm(rttm)

Image

Hope it will help people here, I struggled myself a lot with this issue !

@m-bain could this become a parameter in a futur update that will change the config.yaml automatically ?
Edit - Solution 2 : change parameters in code

The easiest way is (code from https://github.com/m-bain/whisperX/ ) :

device = "cuda" 
diarize_model = whisperx.DiarizationPipeline(use_auth_token="[HF_TOKEN]", device=device) 
diarize_model.model.embedding_batch_size = 4
diarize_model.model.segmentation_batch_size = 4
diarize_segments = diarize_model(audio) #, min_speakers=2, max_speakers=2)
result = whisperx.assign_word_speakers(diarize_segments, result)

segmentation_batch_size doesn't seems to have impact on performances, with :

diarize_model.model.embedding_batch_size = 12 
diarize_model.model.segmentation_batch_size = 32

memory usage is quite the same :

Image*

diarize_model.model.embedding_batch_size = 16 exceed VRAM capacity in my GPU