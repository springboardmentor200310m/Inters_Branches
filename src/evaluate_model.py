import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report

# -----------------------------
# CONFIG
# -----------------------------
DATASET_PATH = "data/spectrograms"
MODEL_PATH = "model/instrunet_cnn_baseline_clean.h5"

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
COLOR_MODE = "grayscale"

# -----------------------------
# LOAD MODEL
# -----------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# -----------------------------
# DATA GENERATOR (VALIDATION)
# -----------------------------
datagen = ImageDataGenerator(
    rescale=1./255,
    validation_split=0.2
)

val_gen = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    color_mode=COLOR_MODE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(val_gen.class_indices.keys())
print("Classes:", class_names)

# -----------------------------
# PREDICTIONS
# -----------------------------
y_true = val_gen.classes
y_pred_probs = model.predict(val_gen)
y_pred = np.argmax(y_pred_probs, axis=1)

# -----------------------------
# CLASSIFICATION REPORT
# -----------------------------
print("\n📊 Classification Report:\n")
print(classification_report(y_true, y_pred, target_names=class_names))

# -----------------------------
# CONFUSION MATRIX
# -----------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="magma",
    xticklabels=class_names,
    yticklabels=class_names
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix – Baseline CNN")
plt.tight_layout()
plt.show()
