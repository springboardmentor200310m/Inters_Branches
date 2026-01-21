import os
import json
import numpy as np
import librosa
import tensorflow as tf

# -------------------------------
# PATHS
# -------------------------------
MODEL_PATH = "final_audio_cnn_model.h5"
LABEL_MAP_PATH = "label_mapping.json"
IMG_SIZE = 128

# -------------------------------
# LOAD LABELS
# -------------------------------
with open(LABEL_MAP_PATH, "r") as f:
    label_map = json.load(f)

index_to_label = {int(v): k for k, v in label_map.items()}

# -------------------------------
# LOAD MODEL
# -------------------------------
model = tf.keras.models.load_model(MODEL_PATH)

# -------------------------------
# AUDIO → GRAYSCALE SPECTROGRAM
# -------------------------------
def audio_to_spectrogram(audio_path):
    y, sr = librosa.load(audio_path, sr=22050)

    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=128, fmax=8000
    )

    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize 0–1
    mel_db = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min())

    # Resize
    mel_db = tf.image.resize(
        mel_db[..., np.newaxis],  # <-- ADD CHANNEL HERE
        (IMG_SIZE, IMG_SIZE)
    ).numpy()

    return mel_db, mel

# -------------------------------
# PREDICT FUNCTION
# -------------------------------
def predict_audio(audio_path):
    img, intensity_map = audio_to_spectrogram(audio_path)

    img = np.expand_dims(img, axis=0).astype("float32")
    # shape = (1, 128, 128, 1) ✅

    preds = model.predict(img, verbose=0)[0]

    class_idx = int(np.argmax(preds))
    confidence = float(np.max(preds))
    label = index_to_label.get(class_idx, "UNKNOWN")

    return label, confidence, preds, intensity_map
