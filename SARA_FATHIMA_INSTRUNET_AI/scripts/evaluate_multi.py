import os
import numpy as np
import tensorflow as tf
import librosa
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import multilabel_confusion_matrix, classification_report
from tqdm import tqdm

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "irmas_train", "IRMAS-TrainingData")
MODEL_PATH = os.path.join(BASE_DIR, "outputs", "models", "instrunet_multilabel_audio_cnn_model.h5")
LABEL_MAP_PATH = os.path.join(BASE_DIR, "outputs", "label_mapping.pkl")
OUT_DIR = os.path.join(BASE_DIR, "outputs", "confusion_matrices")

os.makedirs(OUT_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
SAMPLE_RATE = 22050
SEGMENT_DURATION = 2
N_MELS = 128
THRESHOLD = 0.3

# ---------------- LOAD MODEL & LABELS ----------------
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABEL_MAP_PATH, "rb") as f:
    label_to_index = pickle.load(f)

index_to_label = {v: k for k, v in label_to_index.items()}
class_names = list(label_to_index.keys())

print("✅ Model & labels loaded")

# ---------------- FUNCTIONS ----------------
def parse_labels(filename):
    labels = [p.strip("[]") for p in filename.split("__")[0].split("][")]
    vec = np.zeros(len(label_to_index))
    for l in labels:
        if l in label_to_index:
            vec[label_to_index[l]] = 1
    return vec

def audio_to_mel(file_path):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except:
        return None

    max_len = SEGMENT_DURATION * sr
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max)

    mel = tf.image.resize(mel[..., np.newaxis], IMG_SIZE)
    mel = tf.repeat(mel, repeats=3, axis=-1)
    mel = mel / 255.0

    return np.expand_dims(mel.numpy(), axis=0)

# ---------------- LOAD DATA ----------------
X, y_true = [], []

for folder in os.listdir(AUDIO_DIR):
    folder_path = os.path.join(AUDIO_DIR, folder)
    if not os.path.isdir(folder_path):
        continue

    for file in tqdm(os.listdir(folder_path), desc=f"Processing {folder}"):
        if not file.endswith(".wav"):
            continue

        path = os.path.join(folder_path, file)
        mel = audio_to_mel(path)
        if mel is None:
            continue

        X.append(mel)
        y_true.append(parse_labels(file))

X = np.vstack(X)
y_true = np.array(y_true)

# ---------------- PREDICTION ----------------
y_probs = model.predict(X, batch_size=32)
y_pred = (y_probs >= THRESHOLD).astype(int)

# ---------------- CONFUSION MATRICES ----------------
cms = multilabel_confusion_matrix(y_true, y_pred)

print("\n📊 PER-INSTRUMENT CONFUSION MATRICES\n")

for i, cm in enumerate(cms):
    tn, fp, fn, tp = cm.ravel()
    label = class_names[i]

    print(f"{label}: TP={tp}, FP={fp}, FN={fn}, TN={tn}")

    plt.figure(figsize=(4, 3))
    plt.imshow([[tp, fp], [fn, tn]], cmap="Blues")
    plt.title(f"{label} Confusion Matrix")
    plt.xticks([0, 1], ["Predicted Yes", "Predicted No"])
    plt.yticks([0, 1], ["Actual Yes", "Actual No"])
    plt.colorbar()

    plt.text(0, 0, f"TP\n{tp}", ha="center", va="center")
    plt.text(1, 0, f"FP\n{fp}", ha="center", va="center")
    plt.text(0, 1, f"FN\n{fn}", ha="center", va="center")
    plt.text(1, 1, f"TN\n{tn}", ha="center", va="center")

    save_path = os.path.join(OUT_DIR, f"{label}_confusion_matrix.png")
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

print("\n✅ All confusion matrices saved in outputs/confusion_matrices/")

# ---------------- CLASSIFICATION REPORT ----------------
print("\n📊 MULTI-LABEL CLASSIFICATION METRICS\n")
report = classification_report(y_true, y_pred, target_names=class_names, zero_division=0)
print(report)

# ---------------- MISCLASSIFIED COUNT ----------------
misclassified = np.sum(y_true != y_pred)
print(f"Total misclassified elements: {misclassified}")
 