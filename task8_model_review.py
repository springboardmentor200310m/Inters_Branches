# =========================================
# Task 8: Review Final Model & Analysis
# =========================================

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

# -----------------------------------------
# Step 1: Load trained model
# -----------------------------------------
MODEL_PATH = "improved_cnn_model.keras"
model = load_model(MODEL_PATH)

print("✅ Model loaded successfully")
model.summary()

# -----------------------------------------
# Step 2: Load dataset (IMPORTANT FIX HERE)
# -----------------------------------------
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

test_datagen = ImageDataGenerator(rescale=1.0 / 255)

test_data = test_datagen.flow_from_directory(
    "spectrogram_output2",
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="sparse",   # ✅ FIXED
    shuffle=False
)

# -----------------------------------------
# Step 3: Evaluate model
# -----------------------------------------
test_loss, test_accuracy = model.evaluate(test_data)

print("\n📊 Evaluation Results")
print("Test Accuracy:", test_accuracy)
print("Test Loss:", test_loss)

# -----------------------------------------
# Step 4: Predictions
# -----------------------------------------
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)

y_true = test_data.classes
class_labels = list(test_data.class_indices.keys())

# -----------------------------------------
# Step 5: Confusion Matrix
# -----------------------------------------
cm = confusion_matrix(y_true, y_pred_classes)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    xticklabels=class_labels,
    yticklabels=class_labels
)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")

# ✅ SAVE IMAGE
plt.savefig("confusion_matrix_task8.png", dpi=300)

plt.tight_layout()
plt.show()


# -----------------------------------------
# Step 6: Classification Report
# -----------------------------------------
print("\n📄 Classification Report:")
print(classification_report(y_true, y_pred_classes, target_names=class_labels))

print("\n✅ Task 8 Completed Successfully")
