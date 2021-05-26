from transformers import Wav2Vec2Tokenizer, Wav2Vec2ForCTC

import moviepy.editor as mp
import torch
import librosa
import os

tokenizer = Wav2Vec2Tokenizer.from_pretrained("facebook/wav2vec2-base-960h")
model = Wav2Vec2ForCTC.from_pretrained("facebook/wav2vec2-base-960h")
path = '../input/audio.mp3'

cc = ""

for path in clip_paths:
    # Load the audio with the librosa library
    input_audio, _ = librosa.load(path, 
                                sr=16000)

    # Tokenize the audio
    input_values = tokenizer(input_audio, return_tensors="pt", padding="longest").input_values

    # Feed it through Wav2Vec & choose the most probable tokens
    with torch.no_grad():
      logits = model(input_values).logits
      predicted_ids = torch.argmax(logits, dim=-1)

    # Decode & add to our caption string
    transcription = tokenizer.batch_decode(predicted_ids)[0]
    cc += transcription + " "