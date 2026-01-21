# =========================================================
# FILE NAME : final_evaluation.py
# TASK 6 : FINAL MODEL EVALUATION AND ANALYSIS
# DATA FOLDER : spectrogram_output2
# =========================================================

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

# ---------------------------------------------------------
# STEP 1: LOAD MODEL
# ---------------------------------------------------------
model_path = "improved_cnn_model.keras"
best_model = load_model(model_path)
print("Model loaded successfully")

# ---------------------------------------------------------
# STEP 2: LOAD DATA
# ---------------------------------------------------------
data_dir = "spectrogram_output2"

img_height = 128
img_width = 128
batch_size = 32

datagen = ImageDataGenerator(rescale=1.0 / 255)

data_generator = datagen.flow_from_directory(
    data_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode="categorical",
    shuffle=False
)

# ---------------------------------------------------------
# STEP 3: PREDICTIONS
# ---------------------------------------------------------
y_true = data_generator.classes
y_pred_prob = best_model.predict(data_generator)
y_pred = np.argmax(y_pred_prob, axis=1)

# ---------------------------------------------------------
# STEP 4: ACCURACY (MANUAL)
# ---------------------------------------------------------
accuracy = accuracy_score(y_true, y_pred)

print("\n================================")
print("FINAL EVALUATION RESULTS")
print("Accuracy :", accuracy)
print("================================\n")

# ---------------------------------------------------------
# STEP 5: CONFUSION MATRIX
# ---------------------------------------------------------
cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=False, cmap="Blues")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Final Confusion Matrix")
plt.tight_layout()
plt.savefig("final_confusion_matrix.png")
plt.show()

# ---------------------------------------------------------
# STEP 6: CLASS-WISE METRICS
# ---------------------------------------------------------
class_labels = list(data_generator.class_indices.keys())

print("CLASS-WISE METRICS")
print("==============================")
print(classification_report(y_true, y_pred, target_names=class_labels))

# ---------------------------------------------------------
# STEP 7: ERROR ANALYSIS
# ---------------------------------------------------------
misclassified = np.where(y_true != y_pred)[0]
print("Total Misclassified Samples:", len(misclassified))

# ---------------------------------------------------------
# STEP 8: SAVE FINAL OBSERVATIONS
# ---------------------------------------------------------
with open("final_metrics.txt", "w") as f:
    f.write("TASK 6 - FINAL EVALUATION RESULTS\n")
    f.write("================================\n")
    f.write(f"Final Accuracy : {accuracy}\n")
    f.write(f"Total Samples  : {len(y_true)}\n")
    f.write(f"Misclassified  : {len(misclassified)}\n")
    f.write("Evaluation performed on spectrogram_output2 dataset.\n")

print("Final metrics saved to final_metrics.txt")
print("Final confusion matrix saved as final_confusion_matrix.png")

import pickle

with open("training_history.pkl", "wb") as f:
    pickle.dump(history.history, f)


