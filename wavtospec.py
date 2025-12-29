import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import os

input_folder = "TinySOL/"          
output_folder = "mel_spectrograms/"    


for root, dirs, files in os.walk(input_folder):

    for filename in files:
        if filename.lower().endswith((".wav")):

            audio_path = os.path.join(root, filename)
            print("Processing:", audio_path)

            try:
                y, sr = librosa.load(audio_path, sr=None)
            except Exception as e:
                print("Error loading:", filename, e)
                continue

            mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
            mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

            plt.figure(figsize=(8,4))
            librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
            plt.tight_layout()

            out_name = os.path.splitext(filename)[0] + ".png"
            out_path = os.path.join(output_folder, out_name)

            plt.savefig(out_path)
            plt.close()

            print("Saved to:", out_path)
