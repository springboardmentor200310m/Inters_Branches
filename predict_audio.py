import os
import json
import numpy as np
import librosa
from tensorflow.keras.models import load_model

# --------------------
# CONFIG
# --------------------
MODEL_PATH = "instrument_cnn_model.h5"
LABELS_PATH = "label_classes.npy"

SAMPLE_RATE = 22050
SEGMENT_DURATION = 2.0   # seconds
TARGET_WIDTH = 128

# --------------------
# LOAD MODEL & LABELS
# --------------------
model = load_model(MODEL_PATH)
class_names = np.load(LABELS_PATH)

print("Loaded model")
print("Classes:", class_names)

# --------------------
# PREPROCESS AUDIO SEGMENT
# --------------------
def preprocess_segment(y, sr):
    spec = librosa.feature.melspectrogram(
        y=y,
        sr=sr,
        n_mels=128,
        fmax=8000
    )
    spec = librosa.power_to_db(spec, ref=np.max)

    # Fix width
    if spec.shape[1] < TARGET_WIDTH:
        spec = np.pad(spec, ((0, 0), (0, TARGET_WIDTH - spec.shape[1])))
    else:
        spec = spec[:, :TARGET_WIDTH]

    # Normalize
    spec = spec / np.max(np.abs(spec))

    # Add batch + channel dims
    spec = spec[np.newaxis, ..., np.newaxis]
    return spec

# --------------------
# MAIN PREDICTION PIPELINE
# --------------------
def predict_audio(audio_path, output_json="prediction_output.json"):
    y, sr = librosa.load(audio_path, sr=SAMPLE_RATE, mono=True)

    segment_samples = int(SEGMENT_DURATION * sr)
    total_segments = int(len(y) / segment_samples)

    results = []

    for i in range(total_segments):
        start = i * segment_samples
        end = start + segment_samples
        segment = y[start:end]

        if len(segment) < segment_samples:
            continue

        X = preprocess_segment(segment, sr)
        probs = model.predict(X, verbose=0)[0]

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])

        results.append({
            "start_time_sec": round(i * SEGMENT_DURATION, 2),
            "end_time_sec": round((i + 1) * SEGMENT_DURATION, 2),
            "predicted_instrument": class_names[pred_idx],
            "confidence": round(confidence, 4)
        })

    output = {
        "audio_file": os.path.basename(audio_path),
        "segment_duration_sec": SEGMENT_DURATION,
        "predictions": results
    }

    with open(output_json, "w") as f:
        json.dump(output, f, indent=4)

    print(f"\nPrediction saved to {output_json}")

# --------------------
# RUN
# --------------------
if __name__ == "__main__":
    predict_audio("test_audio/test_guitar.wav")
