import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm


INPUT_DIR = "irmas_train"
OUTPUT_DIR = "data/spectrograms"

os.makedirs(OUTPUT_DIR, exist_ok=True)


def create_melspectrogram(audio_path, output_path):
    y, sr = librosa.load(audio_path, sr=22050, mono=True)


   
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)




    plt.figure(figsize=(3, 3))
    plt.axis("off")
    librosa.display.specshow(mel_db, sr=sr)
    plt.savefig(output_path, bbox_inches='tight', pad_inches=0)
    plt.close()


def process_dataset():
    for root, dirs, files in os.walk(INPUT_DIR):
        for file in tqdm(files, desc="Processing audio"):
            if file.lower().endswith((".wav", ".mp3")):
                label = root.split("\\")[-1]  
                label_folder = os.path.join(OUTPUT_DIR, label)
                os.makedirs(label_folder, exist_ok=True)


                audio_path = os.path.join(root, file)
                output_path = os.path.join(label_folder, file.replace(".wav", ".png").replace(".mp3", ".png"))


                create_melspectrogram(audio_path, output_path)


if __name__ == "__main__":
    process_dataset()
