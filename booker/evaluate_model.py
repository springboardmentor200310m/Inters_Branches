import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score
)
import seaborn as sns

# ================================
# Load trained CNN model
# ================================
MODEL_PATH = "cnn_spectrogram_model.h5"
model = load_model(MODEL_PATH)
print("Model loaded successfully")

# ==========================================
# Load validation/test dataset
# ==========================================
DATASET_PATH = "spectrogram_output2"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

test_ds = tf.keras.utils.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = test_ds.class_names
num_classes = len(class_names)

print("Classes:", class_names)
print("Number of classes:", num_classes)

# Normalize images
test_ds = test_ds.map(lambda x, y: (x / 255.0, y))

# ======================================
# Get true & predicted labels
# ======================================
y_true = []
y_pred = []

for images, labels in test_ds:
    predictions = model.predict(images)
    y_pred.extend(np.argmax(predictions, axis=1))
    y_true.extend(labels.numpy())

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# =====================================
# Record final metrics
# =====================================
final_accuracy = accuracy_score(y_true, y_pred)
print("\nFinal Test Accuracy:", final_accuracy)

# =====================================
# Calculate accuracy gap
# =====================================
train_accuracy = max(model.history.history['accuracy']) if hasattr(model, 'history') else None

if train_accuracy:
    accuracy_gap = train_accuracy - final_accuracy
    print("Accuracy Gap (Train - Test):", accuracy_gap)
else:
    print("Training accuracy not available (trained in different script)")

# =====================================
# Plot accuracy curves
# =====================================
if hasattr(model, 'history'):
    plt.figure()
    plt.plot(model.history.history['accuracy'], label='Train Accuracy')
    plt.plot(model.history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.title("Accuracy Curve")
    plt.legend()
    plt.show()

# =====================================
# Plot loss curves
# =====================================
if hasattr(model, 'history'):
    plt.figure()
    plt.plot(model.history.history['loss'], label='Train Loss')
    plt.plot(model.history.history['val_loss'], label='Validation Loss')
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.title("Loss Curve")
    plt.legend()
    plt.show()

# =====================================
# Generate confusion matrix
# =====================================
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(10, 8))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=class_names,
    yticklabels=class_names
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.show()

# =================================================
# Class-wise performance analysis
# =================================================
print("\nClassification Report:\n")
print(classification_report(
    y_true,
    y_pred,
    labels=range(num_classes),
    target_names=class_names,
    zero_division=0
))

# =================================================
# Identify misclassified spectrograms
# =================================================
misclassified_indices = np.where(y_true != y_pred)[0]

print("Total Misclassified Samples:", len(misclassified_indices))

for i in misclassified_indices[:10]:  # show only first 10
    print(
        f"True: {class_names[y_true[i]]}, "
        f"Predicted: {class_names[y_pred[i]]}"
    )

# =====================================
# Check overfitting
# =====================================
if train_accuracy:
    if accuracy_gap > 0.1:
        print("\nModel may be OVERFITTING")
    else:
        print("\nNo significant overfitting detected")
else:
    print("\nOverfitting check skipped")

print("\nTASK 4 COMPLETED SUCCESSFULLY")
