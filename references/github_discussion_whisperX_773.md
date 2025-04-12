# Github Discussion: "pyannote/speaker-diarization-3.0 runs slower than pyannote/speaker-diarization@2.1 #499"

## Source: https://github.com/m-bain/whisperX/issues/499

## Reply
kaihe-stori
opened on Sep 29, 2023 · edited by kaihe-stori

whisperX/setup.py

Line 22 in 07fafa3
 ] + ["pyannote.audio @ git+https://github.com/pyannote/pyannote-audio@db24eb6c60a26804b1f07a6c2b39055716beb852"], 

Currently pyannote.audio is pinned to 3.0.0, but it has been reported that it performed slower because the embeddings model ran on CPU. As a result a new release 3.0.1 fixed it by replacingonnxruntime with onnxruntime-gpu.

It makes sense for whisperX to update pyannote.audio to 3.0.1, however, there is a conflict with faster_whisper on onnxruntime, as discussed here. Until it is resolved on the faster_whisper side, installing both will end up onnxruntime still in CPU mode and thus slower performance.

My current workaround is running the following commands post installation

pip install pyannote.audio==3.0.1
pip uninstall onnxruntime
pip install --force-reinstall onnxruntime-gpu

Alternative, use the old 2.1 model.

model = whisperx.DiarizationPipeline(model_name='pyannote/speaker-diarization@2.1', use_auth_token=YOUR_AUTH_TOKEN, device='cuda')

Activity
kaihe-stori
mentioned this on Sep 29, 2023

    Huggingface Authentication Issues probably related to pyannote #498

sam1am
sam1am commented on Sep 29, 2023
sam1am
on Sep 29, 2023 · edited by sam1am

Brilliant, thank you. I thought I was crazy. Your fix worked for me. Went from around 8 minutes to 30 second for diarization on 2 speaker ~45minute audio file.
Asofwar
Asofwar commented on Oct 2, 2023
Asofwar
on Oct 2, 2023

thank you!
Sing303
Sing303 commented on Oct 2, 2023
Sing303
on Oct 2, 2023

Oh, man, thank you. I thought I was going crazy too, not long ago everything was working fast and now it's very very slow ....
9throok
9throok commented on Oct 3, 2023
9throok
on Oct 3, 2023

hey @kaihe-stori, I tried to use your approach, but I still get the error for onnxruntime

pkg_resources.DistributionNotFound: The 'onnxruntime<2,>=1.14' distribution was not found and is required by faster-whisper

Any suggestions how can I deal with that?
m-bain
m-bain commented on Oct 5, 2023
m-bain
on Oct 5, 2023
Owner

Great find @kaihe-stori, you could send PR to README if you want
kaihe-stori
kaihe-stori commented on Oct 5, 2023
kaihe-stori
on Oct 5, 2023
ContributorAuthor

    hey @kaihe-stori, I tried to use your approach, but I still get the error for onnxruntime

    pkg_resources.DistributionNotFound: The 'onnxruntime<2,>=1.14' distribution was not found and is required by faster-whisper

    Any suggestions how can I deal with that?

Did you get the error during package installation or running code?

Run my commands only after you install whisperx (thus faster-whisper).
kaihe-stori
kaihe-stori commented on Oct 5, 2023
kaihe-stori
on Oct 5, 2023
ContributorAuthor

    Great find @kaihe-stori, you could send PR to README if you want

Sure, happy to do it. Is "Limitations" a good section to put this in?
m-bain
m-bain commented on Oct 5, 2023
m-bain
on Oct 5, 2023
Owner

I think setup is. best! https://github.com/m-bain/whisperX#setup-%EF%B8%8F
kaihe-stori
mentioned this on Oct 12, 2023

    Add a special note about Speaker-Diarization-3.0 in readme #521

9throok
9throok commented on Oct 12, 2023
9throok
on Oct 12, 2023

        hey @kaihe-stori, I tried to use your approach, but I still get the error for onnxruntime

        pkg_resources.DistributionNotFound: The 'onnxruntime<2,>=1.14' distribution was not found and is required by faster-whisper

        Any suggestions how can I deal with that?

    Did you get the error during package installation or running code?

    Run my commands only after you install whisperx (thus faster-whisper).

thanks.. that worked :)
remic33
remic33 commented on Oct 12, 2023
remic33
on Oct 12, 2023
Contributor

Thanks ! I was investing it and did not get what was happening.
Trying your solution now.
Could we maybe add somethin in the setup file to correct it?
dylorr
dylorr commented on Oct 20, 2023
dylorr
on Oct 20, 2023 · edited by dylorr

Hey! Noticed this problem while executing and monitoring GPU usage. Tried this approach and still am getting 0% GPU usages when it comes to the diarization step - can you further explain at which point you are executing the 3 lines of code you mentioned?

pip install pyannote.audio==3.0.1 pip uninstall onnxruntime pip install --force-reinstall onnxruntime-gpu

I was able to get the old 2.1 model working fine w/ the GPU but for whatever reason, using the workaround for the newer model isn't working. Ty for bringing light to this issue!

Context/TLDR:

    Mac using Google Colab w/ GPU
    I figured adding these install/uninstalls after pulling an updated install of whisperx would do the trick, but no luck
    Assume that I'm using the Python starter code from the readme
    any way related to this issue (

    Sadly onnxruntime-gpu dependency kills Mac support pyannote/pyannote-audio#1505)

7k50
7k50 commented on Oct 20, 2023
7k50
on Oct 20, 2023 · edited by 7k50

Off-the-cuff question, but is there any reason to believe that the newer "3.0" versions of pyannote segmentation/diarization are worse than "2.1" for WhisperX diarization quality (not speed, in this case)? I just made a couple of transcripts with 3.0 for the first time, and I wasn't happy with the quality of the speaker segmentation and thus speaker recognition. I've been quite pleased in the past with the previous models with WhisperX. Just anecdotal, I haven't investigated this.
remic33
remic33 commented on Oct 20, 2023
remic33
on Oct 20, 2023
Contributor

    Off-the-cuff question, but is there any reason to believe that the newer "3.0" versions of pyannote segmentation/diarization are worse than "2.1" for WhisperX diarization quality (not speed, in this case)? I just made a couple of transcripts with 3.0 for the first time, and I wasn't happy with the quality of the speaker segmentation and thus speaker recognition. I've been quite pleased in the past with the previous models with WhisperX. Just anecdotal, I haven't investigated this.

It should not, pyannote 3.0 integrate a new model that is supposed to get better results especially on overlapping discussions. You can see their results on public database on the release note here.
But, it is research oriented. It is possible that your dataset or your data are not like the , and you can have worst result.
Another thing could be a problem on the whisperX process.
Too be sure you should compare manually. (And... it is not the easiest thing to do)

On the other subject, uninstall reinstall do not work for me either. And that is a big problem.
17 remaining items
m-bain
closed this as completedin #586on Nov 17, 2023
jimmy6DOF
mentioned this on Dec 26, 2023

    Update pyannote to v3.1.1 to fix a diarization problem (and diarize.py) #646

eplinux
eplinux commented on Apr 11, 2024
eplinux
on Apr 11, 2024

Unfortunately, this issue seems to be back for me. I had no problems whatsoever but then upgraded to the latest version this week and now diarization takes ages to complete with high CPU and RAM load. Maybe this is related to this issue?
grazder
grazder commented on Apr 11, 2024
grazder
on Apr 11, 2024

Try increasing OMP_NUM_THREADS, it works for me
eplinux
eplinux commented on Apr 11, 2024
eplinux
on Apr 11, 2024

    Try increasing OMP_NUM_THREADS, it works for me

thanks fo the suggestion. to what value did you increase it? currently trying 4
eplinux
eplinux commented on Apr 11, 2024
eplinux
on Apr 11, 2024

it's just weird because it seemed to run seemlessly and complete within a few minutes before - now it's pretty much stuck - using similar files.
danxvv
danxvv commented on Apr 13, 2024
danxvv
on Apr 13, 2024

Same here, two days ago it became too slow.
remic33
remic33 commented on Apr 15, 2024
remic33
on Apr 15, 2024
Contributor

That's weird, the only thing that did change is the usage of torchaudio>=2.2
What is the version of torchaudio in your project?
eplinux
eplinux commented on Apr 15, 2024
eplinux
on Apr 15, 2024

    That's weird, the only thing that did change is the usage of torchaudio>=2.2 What is the version of torchaudio in your project?

So after discovering that it didn't work, I just cleaned my whole environment and started from scratch - following the instructions in the readme file:

pip show torchaudio Name: torchaudio
Version: 2.0.1+cu118

eplinux
eplinux commented on Apr 15, 2024
eplinux
on Apr 15, 2024

    That's weird, the only thing that did change is the usage of torchaudio>=2.2 What is the version of torchaudio in your project?

Awesome, thanks for the hint! It seemed like I fixed it by reinstalling torchaudio et al., so now my GPU is running at full capacity again during diarization. It still takes very long, though. Is the new diarization model so much more resource hungry? Also, maybe we should edit the readme, then? I ran the following to reinstall torch-2.0.0+cu118 & torchaudio-2.0.1+cu118 ((Windows, CUDA 11.8):

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
remic33
remic33 commented on Apr 16, 2024
remic33
on Apr 16, 2024
Contributor

We can upgrade the requirment.txt file to upgrade to avoid that.
Lucidology
mentioned this on May 1, 2024

    404 Client Error for speaker-embedding.onnx #666

astubbs
astubbs commented on Sep 17, 2024
astubbs
on Sep 17, 2024 · edited by astubbs

Fresh install, diarization painfully slow (1 hour for for 1h12m of audio on M2 pro). Not sure how to debug this further...

pipx list
venvs are in /Users/astubbs/.local/pipx/venvs
apps are exposed on your $PATH at /Users/astubbs/.local/bin
manual pages are exposed at /Users/astubbs/.local/share/man
<snip>
   package whisperx 3.1.1, installed using Python 3.12.6
    - whisperx

raulpetru
raulpetru commented on Jan 19, 2025
raulpetru
on Jan 19, 2025

I'm not sure if this is the same issue, but if I change embedded_batch_size and segmentation_batch_size both to 8 (default was 32 for both), I get a very fast diarization.

Take a look at this issue, it might be the same: #688
adityaraute
adityaraute commented on Apr 11, 2025
adityaraute
on Apr 11, 2025

    pip install pyannote.audio==3.0.1
    pip uninstall onnxruntime
    pip install --force-reinstall onnxruntime-gpu
    Alternative, use the old 2.1 model.

Pyannote.audio's latest version is 3.3.x now
Is this fix still valid?
