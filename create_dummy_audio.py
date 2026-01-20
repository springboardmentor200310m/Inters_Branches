import numpy as np
import soundfile as sf
import os

# Create a 30 second dummy audio file (sine waves to simulate sound)
sr = 22050
duration = 30
t = np.linspace(0, duration, int(sr * duration))
freq = 440 # A4 note
audio = 0.5 * np.sin(2 * np.pi * freq * t)

# Add some noise
audio += 0.1 * np.random.normal(0, 1, audio.shape)

sf.write('jazz_song.wav', audio, sr)
print("Created dummy jazz_song.wav")
