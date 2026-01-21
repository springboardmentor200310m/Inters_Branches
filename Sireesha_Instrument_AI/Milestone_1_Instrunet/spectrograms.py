import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

#DATASET_PATH = r"C:\Users\siri reddy\OneDrive\Documents\Desktop\Label\archive_orginal_backup\IRMAS-TrainingData"
#OUTPUT_PATH = "spectrograms"

#to process more than one file and image
DATASET_PATH = r"C:\Users\siri reddy\OneDrive\Documents\Desktop\Label\archive_full_data\IRMAS-TrainingData"
OUTPUT_PATH = "spectrograms_full_data"


os.makedirs(OUTPUT_PATH, exist_ok=True)

for label in os.listdir(DATASET_PATH):
    label_path = os.path.join(DATASET_PATH, label)

    if not os.path.isdir(label_path):
        continue

    output_label_path = os.path.join(OUTPUT_PATH, label)
    os.makedirs(output_label_path, exist_ok=True)

    for file in os.listdir(label_path):
        if file.lower().endswith((".wav", ".mp3")):
            file_path = os.path.join(label_path, file)
            print("Processing:", file_path)

            # Load audio
            y, sr = librosa.load(file_path, mono=True)

            # Safe normalization
            if np.max(np.abs(y)) != 0:
                y = y / np.max(np.abs(y))

            # Generate Mel spectrogram
            mel = librosa.feature.melspectrogram(y=y, sr=sr)
            mel_db = librosa.power_to_db(mel, ref=np.max)

            # Plot & save
            plt.figure(figsize=(3, 3))
            librosa.display.specshow(mel_db, sr=sr)
            plt.axis("off")

            save_path = os.path.join(
                output_label_path,
                file.rsplit(".", 1)[0] + ".png"
            )

            plt.savefig(save_path, bbox_inches="tight", pad_inches=0)            
            plt.close()

            print("Saved:", save_path)
            #exit()

print("✅ Spectrogram generation completed successfully!")