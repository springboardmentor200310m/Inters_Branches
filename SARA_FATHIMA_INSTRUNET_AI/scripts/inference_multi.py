# inference_multi.py
import os
import numpy as np
import tensorflow as tf
import librosa
import pickle

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "instrunet_multilabel_audio_cnn_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "outputs", "label_mapping.pkl")

IMG_SIZE = (128, 128)
SAMPLE_RATE = 22050
SEGMENT_DURATION = 2  # seconds
THRESHOLD = 0.3  # sigmoid threshold for multi-label

# ---------------- LOAD MODEL & LABELS ----------------
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABEL_MAP_PATH, "rb") as f:
    label_to_index = pickle.load(f)
index_to_label = {v: k for k, v in label_to_index.items()}

print(f"✅ Model loaded: {MODEL_PATH}")
print(f"✅ Labels loaded: {label_to_index}")

# ---------------- FUNCTIONS ----------------
def audio_to_mel_array(file_path, duration=SEGMENT_DURATION):
    """Convert audio file to mel spectrogram array suitable for CNN input."""
    y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    max_len = duration * sr
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

    # Resize and repeat channels
    mel_spec_db = tf.image.resize(mel_spec_db[..., np.newaxis], IMG_SIZE)
    mel_spec_db = tf.repeat(mel_spec_db, repeats=3, axis=-1)

    mel_spec_db = mel_spec_db / 255.0
    return np.expand_dims(mel_spec_db, axis=0)

def predict_audio(file_path):
    """
    Predict instruments for a given audio file.
    Returns:
        - predicted_instruments: list of predicted labels above threshold
        - pred_labels: dict of all labels and their probabilities
    """
    mel_array = audio_to_mel_array(file_path)
    preds = model.predict(mel_array)[0]

    pred_labels = {index_to_label[i]: float(preds[i]) for i in range(len(preds))}
    predicted_instruments = [label for label, prob in pred_labels.items() if prob >= THRESHOLD]

    return predicted_instruments, pred_labels

# ---------------- MAIN (standalone testing) ----------------
if __name__ == "__main__":
    audio_file = input("Enter path to audio file: ").strip()
    if not os.path.isfile(audio_file):
        print(f"❌ File not found: {audio_file}")
    else:
        predicted, probs = predict_audio(audio_file)
        print(f"\n🔹 Audio file: {audio_file}")
        print(f"🔹 Predicted Instruments (threshold {THRESHOLD}): {predicted}")
        print("🔹 Probabilities per instrument:")
        for label, prob in probs.items():
            print(f"   {label} : {prob:.3f}")
