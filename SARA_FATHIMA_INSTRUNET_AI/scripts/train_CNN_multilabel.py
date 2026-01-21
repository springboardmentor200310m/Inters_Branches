import os
import numpy as np
import librosa
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten,
    Dense, Dropout, BatchNormalization
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import pickle
from tqdm import tqdm
import random

# ---------------- CONFIG ----------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUDIO_DIR = os.path.join(BASE_DIR, "irmas_train", "IRMAS-TrainingData")
MODEL_DIR = os.path.join(BASE_DIR, "outputs", "models")
os.makedirs(MODEL_DIR, exist_ok=True)

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 50
SAMPLE_RATE = 22050
SEGMENT_DURATION = 2
N_MELS = 128

# ---------------- LABEL MAPPING ----------------
class_names = sorted([
    d for d in os.listdir(AUDIO_DIR)
    if os.path.isdir(os.path.join(AUDIO_DIR, d))
])

label_to_index = {name: idx for idx, name in enumerate(class_names)}

with open(os.path.join(BASE_DIR, "outputs", "label_mapping.pkl"), "wb") as f:
    pickle.dump(label_to_index, f)

print("✅ Label mapping saved:", label_to_index)

# ---------------- FUNCTIONS ----------------
def parse_labels(filename):
    labels = [p.strip("[]") for p in filename.split("__")[0].split("][")]
    vec = np.zeros(len(label_to_index))
    for l in labels:
        if l in label_to_index:
            vec[label_to_index[l]] = 1
    return vec

def augment_audio(y, sr):
    if random.random() < 0.5:
        rate = random.uniform(0.9, 1.1)
        y = librosa.effects.time_stretch(y=y, rate=rate)

    if random.random() < 0.5:
        steps = random.randint(-2, 2)
        y = librosa.effects.pitch_shift(y=y, sr=sr, n_steps=steps)

    noise = np.random.normal(0, 0.003, y.shape)
    return y + noise

def audio_to_mel(file_path, augment=True):
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except:
        return None

    max_len = SEGMENT_DURATION * sr
    if len(y) < max_len:
        y = np.pad(y, (0, max_len - len(y)))
    else:
        y = y[:max_len]

    if augment:
        y = augment_audio(y, sr)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=N_MELS)
    mel = librosa.power_to_db(mel, ref=np.max)

    mel = tf.image.resize(mel[..., np.newaxis], IMG_SIZE)
    mel = tf.repeat(mel, repeats=3, axis=-1)
    mel = mel / 255.0

    return mel.numpy()

# ---------------- DATA LOAD ----------------
X, y = [], []

for label in class_names:
    label_dir = os.path.join(AUDIO_DIR, label)
    for file in tqdm(os.listdir(label_dir), desc=f"Processing {label}"):
        if not file.endswith(".wav"):
            continue

        path = os.path.join(label_dir, file)
        mel = audio_to_mel(path, augment=True)
        if mel is None:
            continue

        X.append(mel)
        y.append(parse_labels(file))

X = np.array(X)
y = np.array(y)

from sklearn.model_selection import train_test_split
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=42
)

# ---------------- MODEL (TUNED) ----------------
model = Sequential([
    Conv2D(32, (3,3), activation="relu", input_shape=(128,128,3)),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(64, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Conv2D(128, (3,3), activation="relu"),
    BatchNormalization(),
    MaxPooling2D(),

    Flatten(),
    Dense(256, activation="relu"),
    Dropout(0.5),
    Dense(len(class_names), activation="sigmoid")
])

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

# ---------------- CALLBACKS ----------------
callbacks = [
    EarlyStopping(monitor="val_loss", patience=6, restore_best_weights=True),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
]

# ---------------- TRAIN ----------------
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    callbacks=callbacks
)

# ---------------- SAVE ----------------
model.save(os.path.join(MODEL_DIR, "instrunet_multilabel_v2.keras"))

with open(os.path.join(BASE_DIR, "outputs", "history_multilabel_v2.pkl"), "wb") as f:
    pickle.dump(history.history, f)

print("✅ V2 training complete")
