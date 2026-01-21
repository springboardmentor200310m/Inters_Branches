import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

MODEL_PATH = r"C:\Users\mjala\OneDrive\Desktop\cnn_dataset\improved_cnn_model.keras"
DATA_PATH = r"C:\Users\mjala\OneDrive\Desktop\cnn_dataset\spectrogram_output2"

BATCH_SIZE = 32

model = load_model(MODEL_PATH)
print("✅ Final model loaded successfully")

input_shape = model.input_shape
IMG_HEIGHT = input_shape[1]
IMG_WIDTH = input_shape[2]
print(f"📐 Model expects image size: {IMG_HEIGHT} x {IMG_WIDTH}")

test_datagen = ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    DATA_PATH,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    batch_size=BATCH_SIZE,
    class_mode='sparse',
    shuffle=False
)

loss, accuracy = model.evaluate(test_generator, verbose=1)

print("\n📊 FINAL MODEL METRICS")
print(f"Test Loss     : {loss:.4f}")
print(f"Test Accuracy : {accuracy:.4f}")

y_pred_probs = model.predict(test_generator)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

class_labels = list(test_generator.class_indices.keys())
labels = list(range(len(class_labels)))

print("\n📄 CLASSIFICATION REPORT")
print(
    classification_report(
        y_true,
        y_pred,
        labels=labels,
        target_names=class_labels,
        zero_division=0
    )
)

cm = confusion_matrix(y_true, y_pred, labels=labels)

plt.figure(figsize=(14, 10))
sns.heatmap(
    cm,
    xticklabels=class_labels,
    yticklabels=class_labels,
    cmap="Blues"
)
plt.title("Confusion Matrix - Final CNN Model")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.tight_layout()

# ✅ SAVE THE FIGURE
plt.savefig("confusion_matrix_final.png", dpi=300)

plt.show()

print("\n✅ TASK 7 COMPLETED SUCCESSFULLY")
