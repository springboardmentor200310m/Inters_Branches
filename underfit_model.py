import tensorflow as tf
import matplotlib.pyplot as plt
import os

# -----------------------------
# Paths & Parameters
# -----------------------------
DATASET_PATH = "spectrogram_output2"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 5   # VERY LOW → causes underfitting

# -----------------------------
# Load Dataset
# -----------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    DATASET_PATH,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=IMG_SIZE,
    batch_size=BATCH_SIZE
)

num_classes = len(train_ds.class_names)
print("Number of classes:", num_classes)

# -----------------------------
# UNDERFITTED CNN MODEL
# (Very small + few layers)
# -----------------------------
model = tf.keras.Sequential([
    tf.keras.layers.Rescaling(1./255, input_shape=(128, 128, 3)),
    tf.keras.layers.Conv2D(8, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(16, activation='relu'),
    tf.keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# -----------------------------
# Train Model
# -----------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS
)

# -----------------------------
# Save Model
# -----------------------------
model.save("underfit_cnn_model.keras")

# -----------------------------
# Plot Accuracy
# -----------------------------
plt.figure()
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Underfitting: Accuracy Curve')
plt.legend()
plt.savefig("underfit_accuracy.png")
plt.close()

# -----------------------------
# Plot Loss
# -----------------------------
plt.figure()
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Underfitting: Loss Curve')
plt.legend()
plt.savefig("underfit_loss.png")
plt.close()

print("Underfitting plots saved as PNG files")




#command 
#python underfit_model.py
