from os import path
from pydub import AudioSegment
import os
from scipy.io import wavfile



audio_data = []
path = "songs"
for r, d, f in os.walk(path):
    for file in f:
        if file.endswith('.mp3'):
            filepath = str(r)+ '/' + str(file)
            print(filepath)
            sound = AudioSegment.from_mp3(filepath)
            sound.export("songs/test.wav", format="wav")

fs, data = wavfile.read("songs/test.wav")
