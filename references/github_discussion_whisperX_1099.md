# Github Discussion: "whisper-large-v3-turbo #1099"

## Source: https://github.com/m-bain/whisperX/issues/1099

## Reply
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