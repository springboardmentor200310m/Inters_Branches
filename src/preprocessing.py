import os
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt

RAW_AUDIO_DIR = "data/raw_audio"
SPEC_DIR = "data/spectrograms"

SAMPLE_RATE = 22050
N_MELS = 128
HOP_LENGTH = 512
DURATION = 3   # seconds (uniform length)


def generate_mel_spec(audio_path, save_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True, duration=DURATION)

    # Normalize
    y = y / (np.max(np.abs(y)) + 1e-9)

    # Mel spectrogram
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=N_MELS,
        hop_length=HOP_LENGTH
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Plot
    plt.figure(figsize=(3, 3))
    librosa.display.specshow(mel_db)
    plt.axis('off')

    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()


# Process ALL instrument folders automatically
for instrument in os.listdir(RAW_AUDIO_DIR):
    input_folder = os.path.join(RAW_AUDIO_DIR, instrument)
    output_folder = os.path.join(SPEC_DIR, instrument)

    if not os.path.isdir(input_folder):
        continue

    os.makedirs(output_folder, exist_ok=True)

    for file in os.listdir(input_folder):
        if file.endswith(".wav"):
            audio_file = os.path.join(input_folder, file)
            image_file = os.path.join(
                output_folder,
                file.replace(".wav", ".png")
            )

            generate_mel_spec(audio_file, image_file)

    print(f"✅ Completed spectrograms for: {instrument}")
