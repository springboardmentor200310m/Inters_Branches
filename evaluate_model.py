import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# -------------------------
# Configuration
# -------------------------
DATASET_PATH = "spectrogram_output2"
MODEL_PATH = "cnn_spectrogram_model.keras"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# -------------------------
# Load trained model
# -------------------------
model = tf.keras.models.load_model(MODEL_PATH)
print("\nModel loaded successfully")

# -------------------------
# Load dataset
# -------------------------
dataset = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False
)

class_names = dataset.class_names
num_classes = len(class_names)

print(f"\nClasses: {class_names}")
print(f"Number of classes: {num_classes}")

# -------------------------
# Final evaluation metrics
# -------------------------
loss, accuracy = model.evaluate(dataset)

print("\nFinal Model Performance:")
print(f"Accuracy : {accuracy:.4f}")
print(f"Loss     : {loss:.4f}")

with open("final_metrics.txt", "w") as f:
    f.write(f"Accuracy: {accuracy}\n")
    f.write(f"Loss: {loss}\n")

# -------------------------
# Generate predictions
# -------------------------
y_true = []
y_pred = []

for images, labels in dataset:
    predictions = model.predict(images)
    y_true.extend(labels.numpy())
    y_pred.extend(np.argmax(predictions, axis=1))

y_true = np.array(y_true)
y_pred = np.array(y_pred)

# -------------------------
# Confusion matrix
# -------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(12, 10))
sns.heatmap(
    cm,
    xticklabels=class_names,
    yticklabels=class_names,
    cmap="Blues",
    annot=False
)
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()

# -------------------------
# Class-wise performance
# -------------------------
report = classification_report(y_true, y_pred, target_names=class_names)
print("\nClassification Report:\n")
print(report)

with open("classification_report.txt", "w") as f:
    f.write(report)

# -------------------------
# Misclassified samples
# -------------------------
misclassified_indices = np.where(y_true != y_pred)[0]
print(f"\nTotal misclassified spectrograms: {len(misclassified_indices)}")

# -------------------------
# Overfitting check note
# -------------------------
print("\nOverfitting check:")
print("Compare training accuracy with validation accuracy.")
print("Large gap indicates overfitting.")

print("\nEvaluation completed successfully.")


# command
# python evaluate_model.py

